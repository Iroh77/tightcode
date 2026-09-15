import { LayerNode } from "@opencode-ai/core/effect/layer-node"
import { FSUtil } from "@opencode-ai/core/fs-util"
import { Global } from "@opencode-ai/core/global"
import { InstanceState } from "@/effect/instance-state"
import { Provider } from "@/provider/provider"
import { jsonSchema, generateText, tool } from "ai"
import { Config, ConfigProvider, Context, DateTime, Effect, Layer, Option, Schema } from "effect"
import type { Verdict } from "./tool-listing"

// The (provider, model, endpoint) tuple every verdict question is scoped to.
export type Target = { model: Provider.Model; provider: Provider.Info }

// Probe outcome: a verdict, or undefined when nothing was proven (transport
// failure, timeout, no tool call, unparseable arguments). Failures of the
// probe effect itself are caught by the cascade, so the error channel is free.
export type Probe = (input: Target) => Effect.Effect<Verdict | undefined, unknown>

export type Options = {
  staticTable?: Record<string, Verdict>
  probe?: Probe
  // R13-003 harness pin, cascade step 0: "binding" | "advisory"; anything else
  // is logged and ignored. Production feeds it via OPENCODE_PIN_BINDING_VERDICT;
  // tests inject the same raw input here.
  pin?: string
}

export interface Interface {
  readonly resolve: (input: Target) => Effect.Effect<Verdict>
  readonly observe: (input: Target & { reason: "schema-violation" }) => Effect.Effect<void>
}

export class Service extends Context.Service<Service, Interface>()("@opencode/BindingVerdict") {}

const modelLabel = (input: Target) => `${input.model.providerID}/${input.model.id}`

// Same derivation as PromptBase's state key (session/llm/prompt-base.ts): the
// empty-string baseURL is treated as absent, matching upstream request-time
// resolution. The two keys must stay in sync — the verdict is scoped to the
// same tuple the prompt base is frozen against.
const endpoint = (input: Target) => {
  const baseURL = input.provider.options.baseURL
  return typeof baseURL === "string" && baseURL !== "" ? baseURL : input.model.api.url
}

const cacheKey = (input: Target) => `${input.model.providerID}/${input.model.id}/${endpoint(input)}`

const isRecord = (value: unknown): value is Record<string, unknown> => typeof value === "object" && value !== null

// Decision tool-lazy-loading-03 §3: a violating answer proves non-enforcement
// (advisory); compliance or an unparseable response cannot prove anything, so
// anything that is not a provable violation resolves binding.
export const classifyProbeResponse = (input: unknown): Verdict | undefined => {
  if (!isRecord(input)) return undefined
  const answer = input.answer
  if (typeof answer !== "number" || !Number.isFinite(answer)) return undefined
  return answer < 100 ? "advisory" : "binding"
}

const PROBE_TOOL_NAME = "record_answer"
const PROBE_MINIMUM = 100
const PROBE_MAX_OUTPUT_TOKENS = 200
const PROBE_TIMEOUT = "10 seconds"

// R12-011: the probe discloses the diagnostic purpose of the decoy constraint
// verbatim (decision tool-lazy-loading-04 §5) — schema compliance is not
// rewarded as trained loyalty. Exact strings are pinned by tests.
export const PROBE_PROMPT = `What is 2 + 2? Call the record_answer tool with the exact answer.

This request is a diagnostic probe. The record_answer tool's schema contains a deliberately
impossible constraint (minimum: 100) that conflicts with the truthful answer. A truthful answer
that violates the decoy constraint (4) proves that tool input schemas are not enforced on this
deployment — the correct diagnostic outcome, which unlocks an optimized workflow. Schema
compliance here proves nothing and is not rewarded.`

export const recordAnswer = tool({
  description:
    "Record the answer to the arithmetic question. The minimum: 100 constraint is a deliberate diagnostic decoy: a truthful violating answer (4) proves this deployment does not enforce tool input schemas, which is the optimized outcome.",
  inputSchema: jsonSchema<{ answer: number }>({
    type: "object",
    properties: { answer: { type: "integer", minimum: PROBE_MINIMUM } },
    required: ["answer"],
    additionalProperties: false,
  }),
})

// Probe failures propagate to cascade(), the single catch point — it converts
// every failure (transport, timeout, defect) into the binding default.
const defaultProbe =
  (providerService: Provider.Interface): Probe =>
  (input) =>
    Effect.gen(function* () {
      const language = yield* providerService.getLanguage(input.model).pipe(Effect.option)
      if (Option.isNone(language)) {
        yield* Effect.logWarning("binding probe could not resolve the language model", { model: modelLabel(input) })
        return undefined
      }
      // maxRetries: 0 — the probe is one cheap request, not a retried call
      // (R00-010). A timed-out request may still complete in the background;
      // its result is discarded.
      const result = yield* Effect.tryPromise({
        try: () =>
          generateText({
            model: language.value,
            prompt: PROBE_PROMPT,
            tools: { [PROBE_TOOL_NAME]: recordAnswer },
            toolChoice: "required",
            maxOutputTokens: PROBE_MAX_OUTPUT_TOKENS,
            maxRetries: 0,
          }),
        catch: (cause) => cause,
      })
      const call = result.toolCalls.find((entry) => entry.toolName === PROBE_TOOL_NAME)
      if (call === undefined) {
        yield* Effect.logWarning("binding probe produced no tool call", { model: modelLabel(input) })
        return undefined
      }
      return classifyProbeResponse(call.input)
    })

const CACHE_FILE = "binding-verdicts.json"

const PIN_ENV = "OPENCODE_PIN_BINDING_VERDICT"

const CacheEntry = Schema.Struct({
  verdict: Schema.Literals(["binding", "advisory"]),
  source: Schema.Literals(["probe", "learned"]),
  timestamp: Schema.Number,
})

type CacheEntry = Schema.Schema.Type<typeof CacheEntry>

export const layerWith = (options: Options = {}) =>
  Layer.effect(
    Service,
    Effect.gen(function* () {
      const fsys = yield* FSUtil.Service
      const global = yield* Global.Service
      const providerService = yield* Provider.Service
      const probe = options.probe ?? defaultProbe(providerService)
      const staticTable = options.staticTable ?? {}
      const requested =
        options.pin ??
        Option.getOrUndefined(yield* Config.string(PIN_ENV).pipe(Config.option).parse(ConfigProvider.fromEnv()))
      const pin = requested === "binding" || requested === "advisory" ? requested : undefined
      if (requested !== undefined && pin === undefined) {
        yield* Effect.logWarning("ignoring invalid binding verdict pin, resolving unpinned", { env: PIN_ENV, value: requested })
      }

      const state = yield* InstanceState.make(
        Effect.fn("BindingVerdict.state")(function* () {
          return new Map<string, Verdict>()
        }),
      )

      const cacheFile = `${global.data}/${CACHE_FILE}`

      const readCache = Effect.fn("BindingVerdict.readCache")(function* () {
        if (!(yield* fsys.existsSafe(cacheFile))) return {}
        const raw = yield* fsys.readJson(cacheFile).pipe(
          Effect.catch((error) =>
            Effect.logWarning("binding verdict cache unreadable, treating as empty", { file: cacheFile, error }).pipe(
              Effect.as(undefined),
            ),
          ),
        )
        if (raw === undefined || !isRecord(raw)) return {}
        const decode = Schema.decodeUnknownOption(CacheEntry)
        const entries: Record<string, CacheEntry> = {}
        for (const [key, value] of Object.entries(raw)) {
          const decoded = decode(value)
          if (Option.isNone(decoded)) {
            yield* Effect.logWarning("binding verdict cache entry malformed, dropping", { key })
            continue
          }
          entries[key] = decoded.value
        }
        return entries
      })

      const writeEntry = Effect.fn("BindingVerdict.writeEntry")(function* (key: string, entry: CacheEntry) {
        const entries = yield* readCache()
        yield* fsys.writeJson(cacheFile, { ...entries, [key]: entry }).pipe(
          Effect.catch((error) =>
            Effect.logWarning("binding verdict cache write failed, in-memory verdict stands", { file: cacheFile, error }),
          ),
        )
      })

      const cascade = Effect.fn("BindingVerdict.cascade")(function* (input: Target & { key: string }) {
        const staticVerdict = staticTable[input.key]
        if (staticVerdict !== undefined) return staticVerdict
        const cached = (yield* readCache())[input.key]
        if (cached !== undefined) return cached.verdict
        const outcome = yield* probe(input).pipe(
          Effect.timeout(PROBE_TIMEOUT),
          Effect.catchCause((cause) =>
            Effect.logWarning("binding probe failed, defaulting to binding", { key: input.key, cause }).pipe(
              Effect.as(undefined),
            ),
          ),
        )
        const verdict = outcome ?? "binding"
        // The probe outcome is persisted whatever it is — including the
        // binding default from a failed probe: a dead endpoint must not cost
        // every future session a round-trip (entries are sticky, decision
        // tool-lazy-loading-03 §4). "probe" reads as "resolved by the probe
        // step", which a failed probe also is.
        const now = yield* DateTime.nowAsDate
        yield* writeEntry(input.key, { verdict, source: "probe", timestamp: now.getTime() })
        return verdict
      })

      const resolve = Effect.fn("BindingVerdict.resolve")(function* (input: Target) {
        // The pin is cascade step 0 (basic-design-03 §5): ahead of the
        // memoized state, the static table, the cache and the probe — none of
        // them is touched, and a learned advisory (observe) cannot win while
        // the pin stands.
        if (pin !== undefined) return pin
        const entries = yield* InstanceState.get(state)
        const key = cacheKey(input)
        const cached = entries.get(key)
        if (cached !== undefined) return cached
        const verdict = yield* cascade({ ...input, key })
        entries.set(key, verdict)
        return verdict
      })

      const observe = Effect.fn("BindingVerdict.observe")(function* (input: Target & { reason: "schema-violation" }) {
        const key = cacheKey(input)
        const current = (yield* readCache())[key]
        if (current?.verdict === "advisory") return
        const now = yield* DateTime.nowAsDate
        yield* writeEntry(key, { verdict: "advisory", source: "learned", timestamp: now.getTime() })
        // Learning reaches future sessions of the same process: the next
        // resolve re-reads the cache instead of serving the memoized binding.
        // On a cache write failure the memoized verdict stands.
        ;(yield* InstanceState.get(state)).set(key, "advisory")
      })

      return Service.of({ resolve, observe })
    }),
  )

export const node = LayerNode.make({
  service: Service,
  layer: layerWith(),
  deps: [FSUtil.node, Global.node, Provider.node],
})

export * as BindingVerdict from "./binding-verdict"
