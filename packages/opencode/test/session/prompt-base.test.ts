import { describe, expect, test } from "bun:test"
import { LayerNode } from "@opencode-ai/core/effect/layer-node"
import { ModelV2 } from "@opencode-ai/core/model"
import { ProviderV2 } from "@opencode-ai/core/provider"
import { Effect } from "effect"
import type { Provider } from "../../src/provider/provider"
import { PromptBase, type SystemBlock } from "../../src/session/llm/prompt-base"
import { testEffect } from "../lib/effect"
const it = testEffect(LayerNode.compile(PromptBase.node))

const model = (url = "https://api.test", id = "test-model"): Provider.Model => ({
  id: ModelV2.ID.make(id),
  providerID: ProviderV2.ID.make("test"),
  api: { id, url, npm: "@ai-sdk/test" },
  name: "Test Model",
  capabilities: {
    temperature: true,
    reasoning: false,
    attachment: false,
    toolcall: true,
    input: { text: true, audio: false, image: false, video: false, pdf: false },
    output: { text: true, audio: false, image: false, video: false, pdf: false },
    interleaved: false,
  },
  cost: { input: 0, output: 0, cache: { read: 0, write: 0 } },
  limit: { context: 128000, output: 8192 },
  status: "active",
  options: {},
  headers: {},
  release_date: "",
})

const provider = (baseURL?: string): Provider.Info => ({
  id: ProviderV2.ID.make("test"),
  name: "Test",
  source: "custom",
  env: [],
  options: baseURL ? { baseURL } : {},
  models: {},
})

const section = (name: string, ...lines: string[]) =>
  [`  <server name="${name}">`, ...lines.map((line) => `    ${line}`), "  </server>"].join("\n")

describe("session.prompt-base", () => {
  it.instance("freezes blocks on first write and re-serves identical bytes on later turns", () =>
    Effect.gen(function* () {
      const promptBase = yield* PromptBase.Service
      const base = { sessionID: "ses_freeze", model: model(), provider: provider() }
      const first = yield* promptBase.reconcileSystem({
        ...base,
        blocks: [
          { key: "environment", content: "env block" },
          { key: "instructions", content: "instructions v1" },
          { key: "mcp:srv", content: section("srv", "Use lookup first.") },
          { key: "skills", content: "skills block" },
        ],
      })
      expect(first.appended).toEqual(["environment", "instructions", "mcp:srv", "skills"])

      const second = yield* promptBase.reconcileSystem({
        ...base,
        blocks: [
          { key: "environment", content: "env block" },
          { key: "instructions", content: "instructions v2 CHANGED" },
          { key: "mcp:srv", content: section("srv", "Changed instructions are absorbed.") },
          { key: "skills", content: "skills block" },
        ],
      })
      expect(second.appended).toEqual([])
      expect(second.blocks).toEqual(first.blocks)
      expect(PromptBase.render(second.blocks).join("\n")).toEqual(PromptBase.render(first.blocks).join("\n"))
    }),
  )

  it.instance("appends new keys as one batch and keeps structured_output per-turn", () =>
    Effect.gen(function* () {
      const promptBase = yield* PromptBase.Service
      const base = { sessionID: "ses_append", model: model(), provider: provider() }
      yield* promptBase.reconcileSystem({
        ...base,
        blocks: [
          { key: "environment", content: "env" },
          { key: "instructions", content: "instr" },
        ],
      })

      const connect = yield* promptBase.reconcileSystem({
        ...base,
        blocks: [
          { key: "environment", content: "env" },
          { key: "instructions", content: "instr" },
          { key: "mcp:new", content: section("new", "New server.") },
          { key: "structured_output", content: "structured prompt" },
        ],
      })
      expect(connect.appended).toEqual(["mcp:new"])
      expect(connect.blocks.some((block) => block.key === "mcp:new")).toBe(true)
      expect(connect.blocks.some((block) => block.key === "structured_output")).toBe(true)

      const plain = yield* promptBase.reconcileSystem({
        ...base,
        blocks: [
          { key: "environment", content: "env" },
          { key: "instructions", content: "instr" },
          { key: "mcp:new", content: section("new", "New server.") },
        ],
      })
      expect(plain.appended).toEqual([])
      expect(plain.blocks.some((block) => block.key === "structured_output")).toBe(false)
      expect(plain.blocks).toEqual(connect.blocks.filter((block) => block.key !== "structured_output"))
    }),
  )

  it.instance("freezes per sessionID and provider/model/endpoint", () =>
    Effect.gen(function* () {
      const promptBase = yield* PromptBase.Service
      const blocks: SystemBlock[] = [
        { key: "environment", content: "env" },
        { key: "instructions", content: "instr" },
      ]
      const first = yield* promptBase.reconcileSystem({
        sessionID: "ses_a",
        model: model(),
        provider: provider(),
        blocks,
      })
      expect(first.appended).toEqual(["environment", "instructions"])

      const otherSession = yield* promptBase.reconcileSystem({
        sessionID: "ses_b",
        model: model(),
        provider: provider(),
        blocks,
      })
      expect(otherSession.appended).toEqual(["environment", "instructions"])

      const otherModel = yield* promptBase.reconcileSystem({
        sessionID: "ses_a",
        model: model("https://api.test", "test-model-2"),
        provider: provider(),
        blocks,
      })
      expect(otherModel.appended).toEqual(["environment", "instructions"])

      const otherEndpoint = yield* promptBase.reconcileSystem({
        sessionID: "ses_a",
        model: model("https://fallback.test"),
        provider: provider("https://override.test"),
        blocks,
      })
      expect(otherEndpoint.appended).toEqual(["environment", "instructions"])

      const sameEndpointAgain = yield* promptBase.reconcileSystem({
        sessionID: "ses_a",
        model: model("https://ignored-fallback.test"),
        provider: provider("https://override.test"),
        blocks,
      })
      expect(sameEndpointAgain.appended).toEqual([])
    }),
  )

  it.instance("fails with a typed error on duplicate keys in one batch", () =>
    Effect.gen(function* () {
      const promptBase = yield* PromptBase.Service
      const error = yield* Effect.flip(
        promptBase.reconcileSystem({
          sessionID: "ses_dup",
          model: model(),
          provider: provider(),
          blocks: [
            { key: "environment", content: "env" },
            { key: "environment", content: "env again" },
          ],
        }),
      )
      expect(error._tag).toBe("DuplicateSystemBlockError")
      expect(error.key).toBe("environment")
    }),
  )

  describe("render", () => {
    test("reproduces upstream input.system byte shape", () => {
      const env1 = "You are powered by the model named test."
      const env2 = "<available_references><reference>refs</reference></available_references>"
      const instr1 = "Instructions from: AGENTS.md\nA"
      const instr2 = "Instructions from: nested/B.md\nB"
      const alpha = section("alpha", "Alpha line one.", "Alpha line two.")
      const beta = section("beta", "Beta line.")
      const rendered = PromptBase.render([
        { key: "environment", content: [env1, env2].join("\n") },
        { key: "instructions", content: [instr1, instr2].join("\n") },
        { key: "mcp:beta", content: beta },
        { key: "mcp:alpha", content: alpha },
        { key: "skills", content: "skills" },
        { key: "structured_output", content: "structured" },
      ])
      expect(rendered.join("\n")).toEqual(
        [
          env1,
          env2,
          instr1,
          instr2,
          ["<mcp_instructions>", beta, alpha, "</mcp_instructions>"].join("\n"),
          "skills",
          "structured",
        ].join("\n"),
      )
    })

    test("omits the mcp wrapper when no mcp blocks exist", () => {
      const rendered = PromptBase.render([
        { key: "environment", content: "env" },
        { key: "skills", content: "skills" },
      ])
      expect(rendered).toEqual(["env", "skills"])
    })

    test("places the wrapper at the canonical position even when the mcp block arrives after skills", () => {
      const rendered = PromptBase.render([
        { key: "environment", content: "env" },
        { key: "skills", content: "skills" },
        { key: "mcp:late", content: section("late", "Late server.") },
      ])
      expect(rendered).toEqual([
        "env",
        ["<mcp_instructions>", section("late", "Late server."), "</mcp_instructions>"].join("\n"),
        "skills",
      ])
    })

    test("keeps the wrapper position stable when a second server appends", () => {
      const one = PromptBase.render([
        { key: "environment", content: "env" },
        { key: "instructions", content: "instr" },
        { key: "mcp:first", content: section("first", "First server.") },
        { key: "skills", content: "skills" },
      ])
      const two = PromptBase.render([
        { key: "environment", content: "env" },
        { key: "instructions", content: "instr" },
        { key: "mcp:first", content: section("first", "First server.") },
        { key: "skills", content: "skills" },
        { key: "mcp:second", content: section("second", "Second server.") },
      ])
      expect(two).toEqual([
        "env",
        "instr",
        ["<mcp_instructions>", section("first", "First server."), section("second", "Second server."), "</mcp_instructions>"].join("\n"),
        "skills",
      ])
      // the append only grows the group; the prefix before it stays byte-identical
      expect(one[0]).toEqual(two[0])
      expect(one[1]).toEqual(two[1])
      expect(one[3]).toEqual(two[3])
    })
  })
})
