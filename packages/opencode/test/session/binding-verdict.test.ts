import { ModelV2 } from "@opencode-ai/core/model"
import { ProviderV2 } from "@opencode-ai/core/provider"
import { describe, expect, test } from "bun:test"
import { Effect } from "effect"
import { logLines } from "effect/testing/TestConsole"
import { BindingVerdict } from "../../src/session/binding-verdict"
import { ProviderTest } from "../fake/provider"
import { TestInstance } from "../fixture/fixture"
import { it } from "../lib/effect"

const model = ProviderTest.model()

const openrouterGlm = ProviderTest.model({ providerID: ProviderV2.ID.make("openrouter"), id: ModelV2.ID.make("z-ai/glm-4.6") })
const openrouterDeepseek = ProviderTest.model({ providerID: ProviderV2.ID.make("openrouter"), id: ModelV2.ID.make("deepseek-chat") })
const otherModel = ProviderTest.model({ id: ModelV2.ID.make("other") })

const itInstance = it.instance

const withService = <A, E, R>(
  options: BindingVerdict.Options,
  self: (svc: BindingVerdict.Interface) => Effect.Effect<A, E, R>,
) =>
  Effect.gen(function* () {
    yield* TestInstance
    return yield* Effect.gen(function* () {
      const svc = yield* BindingVerdict.Service
      return yield* self(svc)
    }).pipe(Effect.provide(BindingVerdict.layerWith(options)))
  })

describe("session.binding-verdict", () => {
  itInstance("a pin beats the static table and wins every resolve", () =>
    withService({ pin: "advisory", staticTable: [{ providerID: "openai", modelPrefix: "gpt", verdict: "binding" }] }, (svc) =>
      Effect.gen(function* () {
        const resolved = yield* svc.resolve({ model })
        expect(resolved.verdict).toBe("advisory")
        expect(resolved.provenance).toEqual({ origin: "pin" })
        const memo = yield* svc.resolve({ model })
        expect(memo).toEqual(resolved)
      }),
    ),
  )

  itInstance("a binding pin beats an advisory static table", () =>
    withService({ pin: "binding", staticTable: [{ providerID: "openai", modelPrefix: "gpt", verdict: "advisory" }] }, (svc) =>
      Effect.gen(function* () {
        const resolved = yield* svc.resolve({ model })
        expect(resolved.verdict).toBe("binding")
        expect(resolved.provenance).toEqual({ origin: "pin" })
      }),
    ),
  )

  itInstance("an invalid pin value is logged and the cascade proceeds", () =>
    withService(
      { pin: "garbage", staticTable: [{ providerID: "openai", modelPrefix: "gpt", verdict: "advisory" }] },
      (svc) =>
        Effect.gen(function* () {
          expect((yield* svc.resolve({ model })).verdict).toBe("advisory")
          const logs = yield* logLines
          expect(logs.some((line) => typeof line === "string" && line.includes("binding verdict pin"))).toBe(true)
        }),
    ),
  )

  itInstance("an empty pin value is logged like any other invalid value", () =>
    withService({ pin: "", staticTable: [{ providerID: "openai", modelPrefix: "gpt", verdict: "advisory" }] }, (svc) =>
      Effect.gen(function* () {
        expect((yield* svc.resolve({ model })).verdict).toBe("advisory")
        const logs = yield* logLines
        expect(logs.some((line) => typeof line === "string" && line.includes("binding verdict pin"))).toBe(true)
      }),
    ),
  )

  itInstance("a static-table hit returns its verdict with static-table provenance", () =>
    withService(
      {
        staticTable: [
          { providerID: "openai", modelPrefix: "gpt-5", verdict: "advisory" },
          { providerID: "openai", modelPrefix: "gpt", verdict: "binding" },
        ],
      },
      (svc) =>
        Effect.gen(function* () {
          const resolved = yield* svc.resolve({ model })
          expect(resolved.verdict).toBe("advisory")
          expect(resolved.provenance).toEqual({ origin: "static-table" })
          const logs = yield* logLines
          expect(logs.some((line) => typeof line === "string" && line.includes("binding verdict resolved"))).toBe(true)
        }),
    ),
  )

  itInstance("table matching is providerID equality plus model-id prefix, first rule wins", () =>
    withService(
      {
        staticTable: [
          { providerID: "openai", modelPrefix: "gpt-5", verdict: "advisory" },
          { providerID: "openai", modelPrefix: "gpt", verdict: "binding" },
        ],
      },
      (svc) =>
        Effect.gen(function* () {
          // first rule wins on a model both prefixes match
          expect((yield* svc.resolve({ model })).verdict).toBe("advisory")
          // prefix semantics: "other" does not start with either prefix
          expect((yield* svc.resolve({ model: otherModel })).provenance).toEqual({ origin: "default" })
          // providerID equality: same prefix, different provider does not match
          expect(
            (yield* svc.resolve({
              model: ProviderTest.model({ providerID: ProviderV2.ID.make("anthropic") }),
            })).provenance,
          ).toEqual({ origin: "default" })
        }),
    ),
  )

  itInstance("a static-table miss resolves the conservative default binding", () =>
    withService({ staticTable: [{ providerID: "openai", modelPrefix: "nomatch", verdict: "advisory" }] }, (svc) =>
      Effect.gen(function* () {
        const resolved = yield* svc.resolve({ model })
        expect(resolved.verdict).toBe("binding")
        expect(resolved.provenance).toEqual({ origin: "default" })
      }),
    ),
  )

  itInstance("the production static table resolves binding for OpenRouter GLM and DeepSeek families", () =>
    withService({}, (svc) =>
      Effect.gen(function* () {
        for (const target of [openrouterGlm, openrouterDeepseek]) {
          const resolved = yield* svc.resolve({ model: target })
          expect(resolved.verdict).toBe("binding")
          expect(resolved.provenance).toEqual({ origin: "static-table" })
        }
      }),
    ),
  )

  itInstance("distinct keys resolve separately and the memo preserves the populating origin", () =>
    withService(
      {
        staticTable: [
          { providerID: "openai", modelPrefix: "gpt", verdict: "advisory" },
          { providerID: "openrouter", modelPrefix: "deepseek", verdict: "binding" },
        ],
      },
      (svc) =>
        Effect.gen(function* () {
          const first = yield* svc.resolve({ model })
          expect(first.provenance).toEqual({ origin: "static-table" })
          const second = yield* svc.resolve({ model: openrouterDeepseek })
          expect(second.verdict).toBe("binding")
          expect(second.provenance).toEqual({ origin: "static-table" })
          // memo hits report the origin that populated them, never "memo"
          expect(yield* svc.resolve({ model })).toEqual(first)
          expect(yield* svc.resolve({ model: openrouterDeepseek })).toEqual(second)
        }),
    ),
  )

  itInstance("the default origin memoizes as binding", () =>
    withService({}, (svc) =>
      Effect.gen(function* () {
        const resolved = yield* svc.resolve({ model })
        expect(resolved).toEqual({ verdict: "binding", provenance: { origin: "default" } })
        expect(yield* svc.resolve({ model })).toEqual(resolved)
      }),
    ),
  )
})

describe("STATIC_TABLE contract", () => {
  test("the production rules pin OpenRouter GLM and DeepSeek to binding", () => {
    expect(BindingVerdict.STATIC_TABLE).toEqual([
      { providerID: "openrouter", modelPrefix: "z-ai/glm", verdict: "binding" },
      { providerID: "openrouter", modelPrefix: "deepseek", verdict: "binding" },
    ])
  })
})
