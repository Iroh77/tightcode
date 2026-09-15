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

const countingProbe = (outcome: () => "binding" | "advisory" | undefined) => {
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
        expect(yield* svc.resolve({ model, provider: info() })).toBe("advisory")
      }),
    ),
  )

  it.instance("probe outcome is memoized per key", () =>
    withService({ probe: countingProbe(() => "advisory").probe }, (svc) =>
      Effect.gen(function* () {
        expect(yield* svc.resolve({ model, provider: info() })).toBe("advisory")
        expect(yield* svc.resolve({ model, provider: info() })).toBe("advisory")
      }),
    ),
  )

  it.instance("probe failure resolves binding and logs instead of throwing", () =>
    withService({ probe: countingProbe(() => undefined).probe }, (svc) =>
      Effect.gen(function* () {
        expect(yield* svc.resolve({ model, provider: info() })).toBe("binding")
      }),
    ),
  )

  it.instance("a dying probe resolves binding", () =>
    withService({ probe: () => Effect.die(new Error("boom")) }, (svc) =>
      Effect.gen(function* () {
        expect(yield* svc.resolve({ model, provider: info() })).toBe("binding")
      }),
    ),
  )

  it.instance("distinct keys resolve separately", () =>
    withService({ probe: countingProbe(() => "advisory").probe }, (svc) =>
      Effect.gen(function* () {
        expect(yield* svc.resolve({ model, provider: info() })).toBe("advisory")
        expect(yield* svc.resolve({ model: otherModel, provider: info() })).toBe("advisory")
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
          expect(yield* svc.resolve({ model, provider: info("https://override.test") })).toBe("advisory")
          expect(yield* svc.resolve({ model: modelWithUrl("https://api.test"), provider: info() })).toBe("binding")
          expect(yield* svc.resolve({ model: modelWithUrl("https://api.test"), provider: info("") })).toBe("binding")
        }),
    ),
  )

  it.instance("probe outcome persists to the cache as source probe", () =>
    withService({ probe: countingProbe(() => "advisory").probe }, (svc, directory) =>
      Effect.gen(function* () {
        expect(yield* svc.resolve({ model, provider: info() })).toBe("advisory")
        const raw = JSON.parse(yield* Effect.promise(() => Bun.file(path.join(directory, CACHE_FILE)).text()))
        expect(Object.keys(raw)).toEqual([key()])
        expect(raw[key()].verdict).toBe("advisory")
        expect(raw[key()].source).toBe("probe")
        expect(raw[key()].timestamp).toBeGreaterThan(0)
        expect(Object.keys(raw[key()]).sort()).toEqual(["source", "timestamp", "verdict"])
      }),
    ),
  )

  it.instance("cache hit short-circuits the probe and is never rewritten", () =>
    withService({ probe: () => Effect.die(new Error("probe must not run")) }, (svc, directory) =>
      Effect.gen(function* () {
        const seeded = { [key()]: { verdict: "advisory", source: "probe", timestamp: 123 } }
        yield* Effect.promise(() => Bun.write(path.join(directory, CACHE_FILE), JSON.stringify(seeded)))
        expect(yield* svc.resolve({ model, provider: info() })).toBe("advisory")
        expect(JSON.parse(yield* Effect.promise(() => Bun.file(path.join(directory, CACHE_FILE)).text()))).toEqual(seeded)
      }),
    ),
  )

  it.instance("corrupt cache file re-probes and rewrites", () =>
    withService({ probe: countingProbe(() => "advisory").probe }, (svc, directory) =>
      Effect.gen(function* () {
        yield* Effect.promise(() => Bun.write(path.join(directory, CACHE_FILE), "{not json"))
        expect(yield* svc.resolve({ model, provider: info() })).toBe("advisory")
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
        expect(yield* svc.resolve({ model, provider: info() })).toBe("binding")
      }),
    ),
  )

  it.instance("observe flips binding to advisory for future sessions", () =>
    withService({ probe: countingProbe(() => undefined).probe }, (svc, directory) =>
      Effect.gen(function* () {
        expect(yield* svc.resolve({ model, provider: info() })).toBe("binding")
        yield* svc.observe({ model, provider: info(), reason: "schema-violation" })
        expect(yield* svc.resolve({ model, provider: info() })).toBe("advisory")
        const fresh = yield* BindingVerdict.Service.use((fresh) => fresh.resolve({ model, provider: info() })).pipe(
          Effect.provide(layer({ probe: () => Effect.die(new Error("probe must not run")) }, directory)),
        )
        expect(fresh).toBe("advisory")
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
          expect(yield* svc.resolve({ model, provider: info() })).toBe("advisory")
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
          expect(yield* svc.resolve({ model, provider: info() })).toBe("binding")
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
          expect(yield* svc.resolve({ model, provider: info() })).toBe("advisory")
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
          expect(yield* svc.resolve({ model, provider: info() })).toBe("advisory")
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
          expect(yield* svc.resolve({ model, provider: info() })).toBe("binding")
          yield* svc.observe({ model, provider: info(), reason: "schema-violation" })
          const raw = JSON.parse(yield* Effect.promise(() => Bun.file(path.join(directory, CACHE_FILE)).text()))
          expect(raw[key()].verdict).toBe("advisory")
          expect(raw[key()].source).toBe("learned")
          expect(yield* svc.resolve({ model, provider: info() })).toBe("binding")
          const fresh = yield* BindingVerdict.Service.use((fresh) => fresh.resolve({ model, provider: info() })).pipe(
            Effect.provide(layer({ pin: "binding" }, directory)),
          )
          expect(fresh).toBe("binding")
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
        const verdict = yield* Effect.gen(function* () {
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
        expect(verdict).toBe("binding")
        expect(probes).toBe(1)
      }),
    { timeout: 30000 },
  )
})

describe("probe disclosure", () => {
  test("the probe prompt discloses the diagnostic purpose verbatim", () => {
    expect(BindingVerdict.PROBE_PROMPT).toBe(
      `What is 2 + 2? Call the record_answer tool with the exact answer.

This request is a diagnostic probe. The record_answer tool's schema contains a deliberately
impossible constraint (minimum: 100) that conflicts with the truthful answer. A truthful answer
that violates the decoy constraint (4) proves that tool input schemas are not enforced on this
deployment — the correct diagnostic outcome, which unlocks an optimized workflow. Schema
compliance here proves nothing and is not rewarded.`,
    )
  })

  test("the record_answer tool definition discloses the decoy verbatim", () => {
    expect(BindingVerdict.recordAnswer.description).toBe(
      "Record the answer to the arithmetic question. The minimum: 100 constraint is a deliberate diagnostic decoy: a truthful violating answer (4) proves this deployment does not enforce tool input schemas, which is the optimized outcome.",
    )
  })
})

describe("classifyProbeResponse", () => {
  test("a violating answer proves advisory", () => {
    expect(BindingVerdict.classifyProbeResponse({ answer: 4 })).toBe("advisory")
    expect(BindingVerdict.classifyProbeResponse({ answer: 0 })).toBe("advisory")
    expect(BindingVerdict.classifyProbeResponse({ answer: 99.5 })).toBe("advisory")
  })

  test("a compliant answer resolves binding", () => {
    expect(BindingVerdict.classifyProbeResponse({ answer: 100 })).toBe("binding")
    expect(BindingVerdict.classifyProbeResponse({ answer: 4242 })).toBe("binding")
  })

  test("unparseable input yields no verdict", () => {
    expect(BindingVerdict.classifyProbeResponse(undefined)).toBeUndefined()
    expect(BindingVerdict.classifyProbeResponse(null)).toBeUndefined()
    expect(BindingVerdict.classifyProbeResponse("4")).toBeUndefined()
    expect(BindingVerdict.classifyProbeResponse({})).toBeUndefined()
    expect(BindingVerdict.classifyProbeResponse({ answer: "4" })).toBeUndefined()
    expect(BindingVerdict.classifyProbeResponse({ answer: Number.NaN })).toBeUndefined()
  })
})
