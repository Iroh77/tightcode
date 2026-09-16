import { Global } from "@opencode-ai/core/global"
import { FSUtil } from "@opencode-ai/core/fs-util"
import { LayerNode } from "@opencode-ai/core/effect/layer-node"
import { ModelV2 } from "@opencode-ai/core/model"
import path from "path"
import { describe, expect, test } from "bun:test"
import { Effect, Fiber, Latch, Layer } from "effect"
import * as TestClock from "effect/testing/TestClock"
import { logLines } from "effect/testing/TestConsole"
import { BindingVerdict } from "../../src/session/binding-verdict"
import { ToolListing } from "../../src/session/tool-listing"
import { ProviderTest } from "../fake/provider"
import { TestInstance } from "../fixture/fixture"
import { it } from "../lib/effect"

const CACHE_FILE = "binding-verdicts.json"

const model = ProviderTest.model()

const otherModel = ProviderTest.model({ id: ModelV2.ID.make("other") })

const modelWithUrl = (url: string) => ProviderTest.model({ api: { id: "gpt-5.2", url, npm: "@ai-sdk/openai" } })

const info = (baseURL?: string) => ProviderTest.info(baseURL ? { options: { baseURL } } : {})

const key = (endpoint = "https://example.com") => `openai/gpt-5.2/${endpoint}`

const layer = (options: BindingVerdict.Options = {}, data = ".") =>
  BindingVerdict.layerWith(options).pipe(
    Layer.provideMerge(Global.layerWith({ data })),
    Layer.provideMerge(ProviderTest.fake().layer),
    Layer.provideMerge(LayerNode.compile(FSUtil.node)),
  )

const withService = <A, E, R>(
  options: BindingVerdict.Options,
  self: (svc: BindingVerdict.Interface, directory: string) => Effect.Effect<A, E, R>,
) =>
  Effect.gen(function* () {
    const test = yield* TestInstance
    return yield* Effect.gen(function* () {
      const svc = yield* BindingVerdict.Service
      return yield* self(svc, test.directory)
    }).pipe(Effect.provide(layer(options, test.directory)))
  })

const advisoryCall = (): BindingVerdict.ProbeOutcome => ({
  verdict: "advisory",
  evidence: { kind: "call", args: '{"answer":4}' },
})

const countingProbe = (outcome: () => BindingVerdict.ProbeOutcome | undefined) => {
  let calls = 0
  return {
    get calls() {
      return calls
    },
    probe: () =>
      Effect.gen(function* () {
        calls++
        return outcome()
      }),
  }
}

describe("session.binding-verdict", () => {
  it.instance("static table hit returns its verdict and never probes", () =>
    withService({ staticTable: { [key()]: "advisory" }, probe: () => Effect.die(new Error("probe must not run")) }, (svc) =>
      Effect.gen(function* () {
        const resolved = yield* svc.resolve({ model, provider: info() })
        expect(resolved.verdict).toBe("advisory")
        expect(resolved.provenance).toEqual({ origin: "static-table" })
        const logs = yield* logLines
        expect(logs.some((line) => typeof line === "string" && line.includes("binding verdict resolved"))).toBe(true)
      }),
    ),
  )

  it.instance("probe outcome is memoized per key and the memo reports the populating origin", () =>
    withService({ probe: countingProbe(advisoryCall).probe }, (svc) =>
      Effect.gen(function* () {
        const first = yield* svc.resolve({ model, provider: info() })
        expect(first.verdict).toBe("advisory")
        expect(first.provenance).toEqual({ origin: "probe", evidence: { kind: "call", args: '{"answer":4}' } })
        const memo = yield* svc.resolve({ model, provider: info() })
        expect(memo).toEqual(first)
      }),
    ),
  )

  it.instance("probe failure resolves the conservative default without evidence", () =>
    withService({ probe: countingProbe(() => undefined).probe }, (svc) =>
      Effect.gen(function* () {
        const resolved = yield* svc.resolve({ model, provider: info() })
        expect(resolved.verdict).toBe("binding")
        expect(resolved.provenance).toEqual({ origin: "default" })
      }),
    ),
  )

  it.instance("a dying probe resolves the conservative default", () =>
    withService({ probe: () => Effect.die(new Error("boom")) }, (svc) =>
      Effect.gen(function* () {
        const resolved = yield* svc.resolve({ model, provider: info() })
        expect(resolved.verdict).toBe("binding")
        expect(resolved.provenance).toEqual({ origin: "default" })
      }),
    ),
  )

  it.instance("distinct keys resolve separately", () =>
    withService({ probe: countingProbe(advisoryCall).probe }, (svc) =>
      Effect.gen(function* () {
        expect((yield* svc.resolve({ model, provider: info() })).verdict).toBe("advisory")
        expect((yield* svc.resolve({ model: otherModel, provider: info() })).verdict).toBe("advisory")
      }),
    ),
  )

  it.instance("endpoint keys are derived from provider baseURL over model api url", () =>
    withService(
      {
        staticTable: {
          [key("https://override.test")]: "advisory",
          [key("https://api.test")]: "binding",
        },
        probe: () => Effect.die(new Error("probe must not run")),
      },
      (svc) =>
        Effect.gen(function* () {
          expect((yield* svc.resolve({ model, provider: info("https://override.test") })).verdict).toBe("advisory")
          expect((yield* svc.resolve({ model: modelWithUrl("https://api.test"), provider: info() })).verdict).toBe(
            "binding",
          )
          expect((yield* svc.resolve({ model: modelWithUrl("https://api.test"), provider: info("") })).verdict).toBe(
            "binding",
          )
        }),
    ),
  )

  it.instance("probe outcome persists to the cache as source probe with its evidence", () =>
    withService({ probe: countingProbe(advisoryCall).probe }, (svc, directory) =>
      Effect.gen(function* () {
        expect((yield* svc.resolve({ model, provider: info() })).verdict).toBe("advisory")
        const raw = JSON.parse(yield* Effect.promise(() => Bun.file(path.join(directory, CACHE_FILE)).text()))
        expect(Object.keys(raw)).toEqual([key()])
        expect(raw[key()].verdict).toBe("advisory")
        expect(raw[key()].source).toBe("probe")
        expect(raw[key()].timestamp).toBeGreaterThan(0)
        expect(raw[key()].evidence).toEqual({ kind: "call", args: '{"answer":4}' })
      }),
    ),
  )

  it.instance("a cache entry reads cache provenance with source, timestamp and evidence", () =>
    withService({ probe: () => Effect.die(new Error("probe must not run")) }, (svc, directory) =>
      Effect.gen(function* () {
        const seeded = {
          [key()]: {
            verdict: "advisory",
            source: "probe",
            timestamp: 123,
            evidence: { kind: "call", args: "{}" },
          },
        }
        yield* Effect.promise(() => Bun.write(path.join(directory, CACHE_FILE), JSON.stringify(seeded)))
        const resolved = yield* svc.resolve({ model, provider: info() })
        expect(resolved.verdict).toBe("advisory")
        expect(resolved.provenance).toEqual({
          origin: "cache",
          source: "probe",
          timestamp: 123,
          evidence: { kind: "call", args: "{}" },
        })
        expect(JSON.parse(yield* Effect.promise(() => Bun.file(path.join(directory, CACHE_FILE)).text()))).toEqual(
          seeded,
        )
      }),
    ),
  )

  it.instance("a legacy cache entry without evidence reads cache provenance without it", () =>
    withService({ probe: () => Effect.die(new Error("probe must not run")) }, (svc, directory) =>
      Effect.gen(function* () {
        const seeded = { [key()]: { verdict: "advisory", source: "probe", timestamp: 123 } }
        yield* Effect.promise(() => Bun.write(path.join(directory, CACHE_FILE), JSON.stringify(seeded)))
        const resolved = yield* svc.resolve({ model, provider: info() })
        expect(resolved.verdict).toBe("advisory")
        expect(resolved.provenance).toEqual({ origin: "cache", source: "probe", timestamp: 123 })
        expect("evidence" in resolved.provenance).toBe(false)
      }),
    ),
  )

  it.instance("corrupt cache file re-probes and rewrites", () =>
    withService({ probe: countingProbe(advisoryCall).probe }, (svc, directory) =>
      Effect.gen(function* () {
        yield* Effect.promise(() => Bun.write(path.join(directory, CACHE_FILE), "{not json"))
        expect((yield* svc.resolve({ model, provider: info() })).verdict).toBe("advisory")
        const raw = JSON.parse(yield* Effect.promise(() => Bun.file(path.join(directory, CACHE_FILE)).text()))
        expect(raw[key()].verdict).toBe("advisory")
      }),
    ),
  )

  it.instance("malformed cache entries are dropped and re-probed", () =>
    withService({ probe: countingProbe(() => undefined).probe }, (svc, directory) =>
      Effect.gen(function* () {
        yield* Effect.promise(() =>
          Bun.write(
            path.join(directory, CACHE_FILE),
            JSON.stringify({
              [key()]: { verdict: "garbage", source: "probe", timestamp: 1 },
              [key("https://other.test")]: { verdict: "advisory", source: "probe", timestamp: 2 },
            }),
          )
        )
        expect((yield* svc.resolve({ model, provider: info() })).verdict).toBe("binding")
      }),
    ),
  )

  it.instance("observe flips binding to advisory for future sessions", () =>
    withService({ probe: countingProbe(() => undefined).probe }, (svc, directory) =>
      Effect.gen(function* () {
        expect((yield* svc.resolve({ model, provider: info() })).verdict).toBe("binding")
        yield* svc.observe({ model, provider: info(), reason: "schema-violation" })
        const learned = yield* svc.resolve({ model, provider: info() })
        expect(learned.verdict).toBe("advisory")
        expect(learned.provenance).toEqual({ origin: "cache", source: "learned", timestamp: expect.any(Number) })
        const fresh = yield* BindingVerdict.Service.use((fresh) => fresh.resolve({ model, provider: info() })).pipe(
          Effect.provide(layer({ probe: () => Effect.die(new Error("probe must not run")) }, directory)),
        )
        expect(fresh.verdict).toBe("advisory")
      }),
    ),
  )

  it.instance("observe is sticky on advisory entries", () =>
    withService({}, (svc, directory) =>
      Effect.gen(function* () {
        const seeded = { [key()]: { verdict: "advisory", source: "learned", timestamp: 123 } }
        yield* Effect.promise(() => Bun.write(path.join(directory, CACHE_FILE), JSON.stringify(seeded)))
        yield* svc.observe({ model, provider: info(), reason: "schema-violation" })
        expect(JSON.parse(yield* Effect.promise(() => Bun.file(path.join(directory, CACHE_FILE)).text()))).toEqual(seeded)
      }),
    ),
  )

  it.instance("observe records advisory when no entry exists", () =>
    withService({}, (svc, directory) =>
      Effect.gen(function* () {
        yield* svc.observe({ model, provider: info(), reason: "schema-violation" })
        const raw = JSON.parse(yield* Effect.promise(() => Bun.file(path.join(directory, CACHE_FILE)).text()))
        expect(raw[key()].verdict).toBe("advisory")
        expect(raw[key()].source).toBe("learned")
      }),
    ),
  )

  it.instance("an advisory pin beats the static table and the cache without probing or writing", () =>
    withService(
      {
        pin: "advisory",
        staticTable: { [key()]: "binding" },
        probe: () => Effect.die(new Error("probe must not run")),
      },
      (svc, directory) =>
        Effect.gen(function* () {
          const seeded = { [key()]: { verdict: "binding", source: "probe", timestamp: 123 } }
          yield* Effect.promise(() => Bun.write(path.join(directory, CACHE_FILE), JSON.stringify(seeded)))
          const resolved = yield* svc.resolve({ model, provider: info() })
          expect(resolved.verdict).toBe("advisory")
          expect(resolved.provenance).toEqual({ origin: "pin" })
          expect(JSON.parse(yield* Effect.promise(() => Bun.file(path.join(directory, CACHE_FILE)).text()))).toEqual(
            seeded,
          )
        }),
    ),
  )

  it.instance("a binding pin beats an advisory static table without probing", () =>
    withService(
      {
        pin: "binding",
        staticTable: { [key()]: "advisory" },
        probe: () => Effect.die(new Error("probe must not run")),
      },
      (svc) =>
        Effect.gen(function* () {
          const resolved = yield* svc.resolve({ model, provider: info() })
          expect(resolved.verdict).toBe("binding")
          expect(resolved.provenance).toEqual({ origin: "pin" })
        }),
    ),
  )

  it.instance("an invalid pin value is logged and the cascade proceeds normally", () =>
    withService(
      {
        pin: "garbage",
        staticTable: { [key()]: "advisory" },
        probe: () => Effect.die(new Error("probe must not run")),
      },
      (svc) =>
        Effect.gen(function* () {
          expect((yield* svc.resolve({ model, provider: info() })).verdict).toBe("advisory")
          const logs = yield* logLines
          expect(logs.some((line) => typeof line === "string" && line.includes("binding verdict pin"))).toBe(true)
        }),
    ),
  )

  it.instance("an empty pin value is logged like any other invalid value", () =>
    withService(
      {
        pin: "",
        staticTable: { [key()]: "advisory" },
        probe: () => Effect.die(new Error("probe must not run")),
      },
      (svc) =>
        Effect.gen(function* () {
          expect((yield* svc.resolve({ model, provider: info() })).verdict).toBe("advisory")
          const logs = yield* logLines
          expect(logs.some((line) => typeof line === "string" && line.includes("binding verdict pin"))).toBe(true)
        }),
    ),
  )

  it.instance(
    "observe under pin still writes the cache while every resolve returns the pin",
    () =>
      withService({ pin: "binding", probe: () => Effect.die(new Error("probe must not run")) }, (svc, directory) =>
        Effect.gen(function* () {
          expect((yield* svc.resolve({ model, provider: info() })).verdict).toBe("binding")
          yield* svc.observe({ model, provider: info(), reason: "schema-violation" })
          const raw = JSON.parse(yield* Effect.promise(() => Bun.file(path.join(directory, CACHE_FILE)).text()))
          expect(raw[key()].verdict).toBe("advisory")
          expect(raw[key()].source).toBe("learned")
          expect((yield* svc.resolve({ model, provider: info() })).verdict).toBe("binding")
          const fresh = yield* BindingVerdict.Service.use((fresh) => fresh.resolve({ model, provider: info() })).pipe(
            Effect.provide(layer({ pin: "binding" }, directory)),
          )
          expect(fresh.verdict).toBe("binding")
        }),
      ),
  )

  it.instance(
    "a probe exceeding the timeout resolves binding and is memoized",
    () =>
      Effect.gen(function* () {
        const test = yield* TestInstance
        let probes = 0
        const ready = yield* Latch.make()
        const resolved = yield* Effect.gen(function* () {
          const svc = yield* BindingVerdict.Service
          const fiber = yield* svc.resolve({ model, provider: info() }).pipe(Effect.forkChild)
          yield* ready.await
          yield* TestClock.adjust("11 seconds")
          return yield* Fiber.join(fiber)
        }).pipe(
          Effect.provide(
            layer(
              {
                probe: () =>
                  Effect.gen(function* () {
                    probes++
                    yield* ready.open
                    yield* Effect.sleep("30 seconds")
                    return undefined
                  }),
              },
              test.directory,
            ).pipe(Layer.provideMerge(TestClock.layer())),
          ),
        )
        expect(resolved.verdict).toBe("binding")
        expect(resolved.provenance).toEqual({ origin: "probe", evidence: { kind: "failure", class: "timeout" } })
        expect(probes).toBe(1)
      }),
    { timeout: 30000 },
  )
})

describe("probe disclosure", () => {
  test("the probe prompt discloses the diagnostic purpose verbatim", () => {
    expect(BindingVerdict.PROBE_PROMPT).toBe(
      `What is 2 + 2? Call the record_answer tool and pass the exact answer as its argument.

This request is a diagnostic probe. The record_answer tool's input schema is deliberately
empty — it accepts no arguments at all — so a truthful answer passed as an argument violates
the schema. That violation proves tool input schemas are not enforced on this deployment,
the correct diagnostic outcome, which unlocks an optimized workflow. An empty call proves
nothing and is not rewarded.`,
    )
  })

  test("the record_answer tool definition discloses the decoy verbatim", () => {
    expect(BindingVerdict.recordAnswer.description).toBe(
      "Record the answer to the arithmetic question. Its input schema is deliberately empty (a diagnostic decoy): passing your answer as an argument violates the schema and proves this deployment does not enforce tool input schemas, which is the optimized outcome.",
    )
  })
})

describe("classifyProbeResponse", () => {
  test("any emitted argument proves advisory", () => {
    expect(BindingVerdict.classifyProbeResponse({ answer: 4 })).toBe("advisory")
    expect(BindingVerdict.classifyProbeResponse({ answer: "4" })).toBe("advisory")
    expect(BindingVerdict.classifyProbeResponse({ anything: true })).toBe("advisory")
  })

  test("an empty call or absent arguments resolve binding", () => {
    expect(BindingVerdict.classifyProbeResponse({})).toBe("binding")
    expect(BindingVerdict.classifyProbeResponse(undefined)).toBe("binding")
  })

  test("unparseable arguments yield no verdict (probe failure)", () => {
    expect(BindingVerdict.classifyProbeResponse(null)).toBeUndefined()
    expect(BindingVerdict.classifyProbeResponse("4")).toBeUndefined()
    expect(BindingVerdict.classifyProbeResponse(4)).toBeUndefined()
  })
})

describe("probeOutcomeFromCall (R12-013)", () => {
  test("a missing call is its own evidence class", () => {
    expect(BindingVerdict.probeOutcomeFromCall(undefined)).toEqual({
      verdict: "binding",
      evidence: { kind: "no-tool-call" },
    })
  })

  test("a record call carries its verbatim args as evidence", () => {
    expect(BindingVerdict.probeOutcomeFromCall({ input: { answer: 4 } })).toEqual({
      verdict: "advisory",
      evidence: { kind: "call", args: '{"answer":4}' },
    })
    expect(BindingVerdict.probeOutcomeFromCall({ input: {} })).toEqual({
      verdict: "binding",
      evidence: { kind: "call", args: "{}" },
    })
  })

  test("unparseable arguments become the unparseable failure class, never silently collapsed", () => {
    expect(BindingVerdict.probeOutcomeFromCall({ input: "4" })).toEqual({
      verdict: "binding",
      evidence: { kind: "failure", class: "unparseable" },
    })
    expect(BindingVerdict.probeOutcomeFromCall({ input: null })).toEqual({
      verdict: "binding",
      evidence: { kind: "failure", class: "unparseable" },
    })
  })

  test("evidence args are capped at 2048 chars", () => {
    const outcome = BindingVerdict.probeOutcomeFromCall({ input: { answer: "x".repeat(3000) } })
    if (outcome.evidence.kind !== "call") throw new Error("expected call evidence")
    expect(outcome.evidence.args.length).toBe(2048)
    expect(outcome.evidence.args.startsWith('{"answer":"xx')).toBe(true)
  })
})

describe("probe decoy schema", () => {
  test("the decoy schema equals the R12-003 advisory placeholder value", () => {
    expect(BindingVerdict.PROBE_DECOY_SCHEMA).toEqual(ToolListing.PLACEHOLDER)
  })

  test("the record_answer tool registers the placeholder schema", () => {
    expect((BindingVerdict.recordAnswer.inputSchema as { jsonSchema: unknown }).jsonSchema).toEqual(
      BindingVerdict.PROBE_DECOY_SCHEMA,
    )
  })
})
