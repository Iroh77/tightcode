import { LayerNode } from "@opencode-ai/core/effect/layer-node"
import { InstanceState } from "@/effect/instance-state"
import { Provider } from "@/provider/provider"
import { Config, ConfigProvider, Context, Effect, Layer, Option } from "effect"
import type { Verdict } from "./tool-listing"

// The model a verdict question is scoped to. Matching is providerID equality
// plus model-id prefix, endpoint-independent (R12-010 Amendment 5): per-target
// verdicts are unstable on multi-backend routers, so the endpoint is not part
// of the question anymore.
export type Target = { model: Provider.Model }

// R12-013: how the verdict was reached — the originating cascade source,
// preserved through in-memory propagation (a memo hit reports the origin that
// populated it, never "memo").
export type VerdictProvenance = { origin: "pin" } | { origin: "static-table" } | { origin: "default" }

export type ResolvedVerdict = { verdict: Verdict; provenance: VerdictProvenance }

// Static capability rule: providerID equality AND modelID.startsWith(modelPrefix).
export type StaticRule = { providerID: string; modelPrefix: string; verdict: Verdict }

// Production static capability table (R12-010 Amendment 5). The conservative
// default already resolves binding; these entries document the intent for the
// GLM and DeepSeek families on OpenRouter and survive a future default change.
export const STATIC_TABLE: StaticRule[] = [
  { providerID: "openrouter", modelPrefix: "z-ai/glm", verdict: "binding" },
  { providerID: "openrouter", modelPrefix: "deepseek", verdict: "binding" },
]

export type Options = {
  // Test injection replaces the production table wholesale.
  staticTable?: StaticRule[]
  // R13-003 harness pin, cascade step 0: "binding" | "advisory"; anything else
  // is logged and ignored. Production feeds it via OPENCODE_PIN_BINDING_VERDICT;
  // tests inject the same raw input here.
  pin?: string
}

export interface Interface {
  readonly resolve: (input: Target) => Effect.Effect<ResolvedVerdict>
}

export class Service extends Context.Service<Service, Interface>()("@opencode/BindingVerdict") {}

const modelKey = (input: Target) => `${input.model.providerID}/${input.model.id}`

const PIN_ENV = "OPENCODE_PIN_BINDING_VERDICT"

const tableHit = (table: StaticRule[], input: Target): Verdict | undefined =>
  table.find((rule) => rule.providerID === input.model.providerID && input.model.id.startsWith(rule.modelPrefix))
    ?.verdict

export const layerWith = (options: Options = {}) =>
  Layer.effect(
    Service,
    Effect.gen(function* () {
      const table = options.staticTable ?? STATIC_TABLE
      const requested =
        options.pin ??
        Option.getOrUndefined(yield* Config.string(PIN_ENV).pipe(Config.option).parse(ConfigProvider.fromEnv()))
      const pin = requested === "binding" || requested === "advisory" ? requested : undefined
      if (requested !== undefined && pin === undefined) {
        yield* Effect.logWarning("ignoring invalid binding verdict pin, resolving unpinned", { env: PIN_ENV, value: requested })
      }

      const state = yield* InstanceState.make(
        Effect.fn("BindingVerdict.state")(function* () {
          return new Map<string, ResolvedVerdict>()
        }),
      )

      const resolve = Effect.fn("BindingVerdict.resolve")(function* (input: Target) {
        // The pin is cascade step 0 (R13-003 harness seam): ahead of the memo
        // and the static table, winning every resolve.
        if (pin !== undefined) return { verdict: pin, provenance: { origin: "pin" } } satisfies ResolvedVerdict
        const entries = yield* InstanceState.get(state)
        const key = modelKey(input)
        // The memo stores the full pair (R12-013): a hit reports the origin
        // that populated it, never "memo" — the memo is a performance detail,
        // not a cause.
        const memo = entries.get(key)
        if (memo !== undefined) return memo
        const hit = tableHit(table, input)
        const resolved: ResolvedVerdict =
          hit === undefined
            ? { verdict: "binding", provenance: { origin: "default" } }
            : { verdict: hit, provenance: { origin: "static-table" } }
        yield* Effect.logInfo("binding verdict resolved", { key, ...resolved })
        entries.set(key, resolved)
        return resolved
      })

      return Service.of({ resolve })
    }),
  )

export const node = LayerNode.make({
  service: Service,
  layer: layerWith(),
  deps: [],
})

export * as BindingVerdict from "./binding-verdict"
