import { LayerNode } from "@opencode-ai/core/effect/layer-node"
import { FSUtil } from "@opencode-ai/core/fs-util"
import { Global } from "@opencode-ai/core/global"
import { InstanceState } from "@/effect/instance-state"
import { Provider } from "@/provider/provider"
import type { JSONSchema7 } from "@ai-sdk/provider"
import { jsonSchema, generateText, tool } from "ai"
import { Config, ConfigProvider, Context, DateTime, Effect, Layer, Option, Schema } from "effect"
import type { Verdict } from "./tool-listing"

// The (provider, model, endpoint) tuple every verdict question is scoped to.
export type Target = { model: Provider.Model; provider: Provider.Info }

// Probe outcome: a verdict plus the raw decoy answer as evidence (R12-013) —
// any emitted call, its absence, or a known failure class. undefined = nothing
// was learned (no language model, unexplained failure): the cascade's
// conservative default, without evidence. Failures of the probe effect itself
// are caught by the cascade, so the error channel is free.
export type ProbeOutcome = { verdict: Verdict; evidence: ProbeEvidence }

export type Probe = (input: Target) => Effect.Effect<ProbeOutcome | undefined, unknown>

// R12-013: the raw decoy answer, never only the collapsed verdict. `args` is
// the verbatim emitted arguments JSON, capped at 2048 chars.
export type ProbeEvidence =
  | { kind: "call"; args: string }
  | { kind: "no-tool-call" }
  | { kind: "failure"; class: "timeout" | "transport" | "unparseable" }

// R12-010 Amendment 4: evidence a learned advisory flip rests on — the
// violating tool and its verbatim emitted arguments (capped), the flip's
// offline-forensics trail. Distinct from ProbeEvidence: a learned entry never
// carries probe evidence and vice versa.
export type ViolationEvidence = { kind: "violation"; tool: string; args: string }

// R12-013: how the verdict was reached — the originating cascade source,
// preserved through in-memory propagation (a memo hit reports the origin that
// populated it, never "memo"). Cache provenance carries the entry's source and
// timestamp, plus its evidence when the entry carries one (probe evidence on
// probe-written entries, violation evidence on learned ones).
export type VerdictProvenance =
  | { origin: "pin" }
  | { origin: "static-table" }
  | { origin: "cache"; source: "probe" | "learned"; timestamp: number; evidence?: ProbeEvidence | ViolationEvidence }
  | { origin: "probe"; evidence: ProbeEvidence }
  | { origin: "default" }

export type ResolvedVerdict = { verdict: Verdict; provenance: VerdictProvenance }

export type Options = {
  staticTable?: Record<string, Verdict>
  probe?: Probe
  // R13-003 harness pin, cascade step 0: "binding" | "advisory"; anything else
  // is logged and ignored. Production feeds it via OPENCODE_PIN_BINDING_VERDICT;
  // tests inject the same raw input here.
  pin?: string
}

export interface Interface {
  readonly resolve: (input: Target) => Effect.Effect<ResolvedVerdict>
  readonly observe: (input: Target & { reason: "schema-violation"; tool: string; args: unknown }) => Effect.Effect<void>
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

// R12-010 Amendment 3 (decision tool-lazy-loading-05 §1): the decoy registers
// the advisory placeholder — the exact shape advisory sessions register — and
// the adversarial constraint moved into PROBE_PROMPT (pass the answer as an
// argument). On a binding stack the empty schema forces the call's arguments
// to {}, the observed misclassification signature the probe now predicts.
export const PROBE_DECOY_SCHEMA: JSONSchema7 = { type: "object", properties: {} }

// Decision tool-lazy-loading-03 §3 as amended: any emitted argument violates
// the empty decoy schema and proves non-enforcement (advisory); an empty call
// or absent arguments are compliance (binding, conservative — unchanged
// direction); a non-record response is unparseable (probe failure ⇒ binding
// downstream).
export const classifyProbeResponse = (input: unknown): Verdict | undefined => {
  if (input === undefined) return "binding"
  if (!isRecord(input)) return undefined
  return Object.keys(input).length === 0 ? "binding" : "advisory"
}

// R12-013 + Amendment 4: evidence carries the verbatim emitted arguments JSON
// (probe decoy or violating tool call), bounded so a pathological call cannot
// bloat the cache file.
const EVIDENCE_MAX_CHARS = 2048

// R12-013: the probe's answer becomes evidence — a missing call and unparseable
// arguments are their own classes, a record call carries its verbatim args
// (capped). Exported so the mapping is testable without the AI SDK; the
// transport class is defaultProbe's request-failure catch.
export const probeOutcomeFromCall = (call: { input: unknown } | undefined): ProbeOutcome => {
  if (call === undefined) return { verdict: "binding", evidence: { kind: "no-tool-call" } }
  const verdict = classifyProbeResponse(call.input)
  if (verdict === undefined) return { verdict: "binding", evidence: { kind: "failure", class: "unparseable" } }
  return {
    verdict,
    evidence: { kind: "call", args: JSON.stringify(call.input ?? {}).slice(0, EVIDENCE_MAX_CHARS) },
  }
}

const PROBE_TOOL_NAME = "record_answer"
const PROBE_MAX_OUTPUT_TOKENS = 200
const PROBE_TIMEOUT = "10 seconds"

// R12-011: the probe discloses the diagnostic purpose of the adversarial
// constraint verbatim (decision tool-lazy-loading-04 §5, strings re-pinned by
// round 3) — schema compliance is not rewarded as trained loyalty. Exact
// strings are pinned by tests.
export const PROBE_PROMPT = `What is 2 + 2? Call the record_answer tool and pass the exact answer as its argument.

This request is a diagnostic probe. The record_answer tool's input schema is deliberately
empty — it accepts no arguments at all — so a truthful answer passed as an argument violates
the schema. That violation proves tool input schemas are not enforced on this deployment,
the correct diagnostic outcome, which unlocks an optimized workflow. An empty call proves
nothing and is not rewarded.`

export const recordAnswer = tool({
  description:
    "Record the answer to the arithmetic question. Its input schema is deliberately empty (a diagnostic decoy): passing your answer as an argument violates the schema and proves this deployment does not enforce tool input schemas, which is the optimized outcome.",
  inputSchema: jsonSchema<Record<string, never>>(PROBE_DECOY_SCHEMA),
})

// Probe failures map to evidence-bearing binding outcomes (R12-013): a failed
// request is a transport failure, no tool call and unparseable arguments are
// their own classes — each logged loudly (R00-010). Only "nothing was learned"
// (no language model) returns undefined, which the cascade resolves as the
// conservative default without evidence.
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
      const attempted = yield* Effect.tryPromise({
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
      }).pipe(
        Effect.catchCause((cause) =>
          Effect.logWarning("binding probe request failed, treating as transport failure", {
            model: modelLabel(input),
            cause,
          }).pipe(Effect.as(undefined)),
        ),
      )
      if (attempted === undefined) {
        return { verdict: "binding", evidence: { kind: "failure", class: "transport" } }
      }
      const call = attempted.toolCalls.find((entry) => entry.toolName === PROBE_TOOL_NAME)
      const outcome = probeOutcomeFromCall(call)
      const warning =
        outcome.evidence.kind === "no-tool-call"
          ? "binding probe produced no tool call"
          : outcome.evidence.kind === "failure"
            ? "binding probe produced unparseable arguments, treating as failure"
            : undefined
      if (warning !== undefined) {
        yield* Effect.logWarning(warning, { model: modelLabel(input) })
      }
      return outcome
    })

const CACHE_FILE = "binding-verdicts.json"

const PIN_ENV = "OPENCODE_PIN_BINDING_VERDICT"

// R12-013: the probe's raw decoy answer rides the entry additively — a sticky
// stale entry stays diagnosable offline. Entries predating the field stay
// valid (their provenance reads as cache provenance without evidence).
const ProbeEvidenceSchema = Schema.Union([
  Schema.Struct({ kind: Schema.Literal("call"), args: Schema.String }),
  Schema.Struct({ kind: Schema.Literal("no-tool-call") }),
  Schema.Struct({
    kind: Schema.Literal("failure"),
    class: Schema.Literals(["timeout", "transport", "unparseable"]),
  }),
])

const ViolationEvidenceSchema = Schema.Struct({
  kind: Schema.Literal("violation"),
  tool: Schema.String,
  args: Schema.String,
})

const CacheEntry = Schema.Struct({
  verdict: Schema.Literals(["binding", "advisory"]),
  source: Schema.Literals(["probe", "learned"]),
  timestamp: Schema.Number,
  evidence: Schema.optional(Schema.Union([ProbeEvidenceSchema, ViolationEvidenceSchema])),
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
          return new Map<string, ResolvedVerdict>()
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
        if (staticVerdict !== undefined) {
          const resolved: ResolvedVerdict = { verdict: staticVerdict, provenance: { origin: "static-table" } }
          yield* Effect.logInfo("binding verdict resolved", { key: input.key, ...resolved })
          return resolved
        }
        const cached = (yield* readCache())[input.key]
        if (cached !== undefined) {
          const resolved: ResolvedVerdict = {
            verdict: cached.verdict,
            provenance: {
              origin: "cache",
              source: cached.source,
              timestamp: cached.timestamp,
              ...(cached.evidence !== undefined ? { evidence: cached.evidence } : {}),
            },
          }
          yield* Effect.logInfo("binding verdict resolved", { key: input.key, ...resolved })
          return resolved
        }
        // Failure first, timeout second: a probe that fails or dies is caught
        // into "nothing learnable" (Some(undefined), logged); only a probe
        // interrupted by the deadline resolves as Option.none — the timeout
        // failure class (R12-013).
        const probed = yield* probe(input).pipe(
          Effect.catchCause((cause) =>
            Effect.logWarning("binding probe failed, defaulting to binding", { key: input.key, cause }).pipe(
              Effect.as(undefined as ProbeOutcome | undefined),
            ),
          ),
          Effect.timeoutOption(PROBE_TIMEOUT),
        )
        const timedOut = Option.isNone(probed)
        if (timedOut) {
          yield* Effect.logWarning("binding probe timed out, treating as timeout failure", { key: input.key })
        }
        const outcome: ProbeOutcome | undefined = timedOut
          ? { verdict: "binding", evidence: { kind: "failure", class: "timeout" } }
          : probed.value
        const verdict = outcome?.verdict ?? "binding"
        const evidence = outcome?.evidence
        // The probe outcome is persisted whatever it is — including the
        // binding default from a failed probe: a dead endpoint must not cost
        // every future session a round-trip (entries are sticky, decision
        // tool-lazy-loading-03 §4). "probe" reads as "resolved by the probe
        // step", which a failed probe also is; the raw answer rides along as
        // evidence when one exists (R12-013).
        const now = yield* DateTime.nowAsDate
        yield* writeEntry(input.key, {
          verdict,
          source: "probe",
          timestamp: now.getTime(),
          ...(evidence !== undefined ? { evidence } : {}),
        })
        const resolved: ResolvedVerdict =
          evidence === undefined
            ? { verdict, provenance: { origin: "default" } }
            : { verdict, provenance: { origin: "probe", evidence } }
        yield* Effect.logInfo("binding verdict resolved", { key: input.key, ...resolved })
        return resolved
      })

      const resolve = Effect.fn("BindingVerdict.resolve")(function* (input: Target) {
        // The pin is cascade step 0 (basic-design-03 §5): ahead of the
        // memoized state, the static table, the cache and the probe — none of
        // them is touched, and a learned advisory (observe) cannot win while
        // the pin stands. The pin wins every resolve (round-2 edge 10): its
        // provenance stays { origin: "pin" } even after observe writes.
        if (pin !== undefined) return { verdict: pin, provenance: { origin: "pin" } } satisfies ResolvedVerdict
        const entries = yield* InstanceState.get(state)
        const key = cacheKey(input)
        // The memo stores the full pair (R12-013): a hit reports the origin
        // that populated it, never "memo" — the memo is a performance detail,
        // not a cause.
        const memo = entries.get(key)
        if (memo !== undefined) return memo
        const resolved = yield* cascade({ ...input, key })
        entries.set(key, resolved)
        return resolved
      })

      // R12-010 Amendment 4: the learning rule is gated upstream — the caller
      // (SessionTools) feeds observe only when the resolved verdict was
      // binding and the wrapper served no schema for the tool, the one shape
      // where a violation proves non-enforcement rather than a model slip
      // (schema-eager serving) or the serving shape itself (advisory
      // serving — the self-sealing loop that poisoned the 2026-09-16 cache).
      // Every accepted flip is logged loudly and persisted with its evidence.
      const observe = Effect.fn("BindingVerdict.observe")(
        function* (input: Target & { reason: "schema-violation"; tool: string; args: unknown }) {
          const key = cacheKey(input)
          const current = (yield* readCache())[key]
          if (current?.verdict === "advisory") return
          const now = yield* DateTime.nowAsDate
          const timestamp = now.getTime()
          const evidence: ViolationEvidence = {
            kind: "violation",
            tool: input.tool,
            args: JSON.stringify(input.args ?? {}).slice(0, EVIDENCE_MAX_CHARS),
          }
          yield* Effect.logWarning("binding verdict learned advisory from a schema violation", {
            key,
            tool: evidence.tool,
            args: evidence.args,
          })
          yield* writeEntry(key, { verdict: "advisory", source: "learned", timestamp, evidence })
          // Learning reaches future sessions of the same process: the next
          // resolve re-reads the cache instead of serving the memoized binding.
          // On a cache write failure the memoized verdict stands. The memoized
          // provenance mirrors the entry's (cache, learned, R12-013).
          ;(yield* InstanceState.get(state)).set(key, {
            verdict: "advisory",
            provenance: { origin: "cache", source: "learned", timestamp, evidence },
          })
        },
      )

      return Service.of({ resolve, observe })
    }),
  )

export const node = LayerNode.make({
  service: Service,
  layer: layerWith(),
  deps: [FSUtil.node, Global.node, Provider.node],
})

export * as BindingVerdict from "./binding-verdict"
