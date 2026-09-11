import { describe, expect, test } from "bun:test"
import type { JSONSchema7 } from "@ai-sdk/provider"
import { LayerNode } from "@opencode-ai/core/effect/layer-node"
import { ModelV2 } from "@opencode-ai/core/model"
import { ProviderV2 } from "@opencode-ai/core/provider"
import { Effect } from "effect"
import type { Provider } from "../../src/provider/provider"
import { PromptBase, type SystemBlock } from "../../src/session/llm/prompt-base"
import { ToolListing, type ToolSeed } from "../../src/session/tool-listing"
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

const toolSchema = (label: string): JSONSchema7 => ({
  type: "object",
  properties: { label: { type: "string", const: label } },
  required: ["label"],
})

const seed = (name: string, kind: "eager" | "deferred" = "deferred", description?: string, server?: string): ToolSeed => ({
  name,
  kind,
  fullDescription: description ?? `${name} description`,
  jsonSchema: toolSchema(name),
  ...(server ? { server } : {}),
})

describe("session.prompt-base", () => {
  describe("reconcileTools", () => {
    it.instance("freezes tool entries on first write and absorbs per-turn recomputation", () =>
      Effect.gen(function* () {
        const promptBase = yield* PromptBase.Service
        const base = { sessionID: "ses_tools_freeze", model: model(), provider: provider(), mode: "advisory" as const }
        const first = yield* promptBase.reconcileTools({
          ...base,
          seeds: [seed("shell", "eager"), seed("glob")],
        })
        expect(first.appended).toEqual(["shell", "glob"])
        expect(first.mode).toBe("advisory")

        // Upstream per-turn rebuilds (describeTask list, tool.definition hook,
        // MCP def refresh) feed changed facts; the frozen entries absorb them.
        const second = yield* promptBase.reconcileTools({
          ...base,
          seeds: [seed("shell", "eager", "CHANGED description"), seed("glob", "deferred", "CHANGED description")],
        })
        expect(second.appended).toEqual([])
        expect(second.entries).toEqual(first.entries)
        expect(second.mode).toBe("advisory")
      }),
    )

    it.instance("appends new names as one batch in insertion order, existing entries immutable", () =>
      Effect.gen(function* () {
        const promptBase = yield* PromptBase.Service
        const base = { sessionID: "ses_tools_batch", model: model(), provider: provider(), mode: "advisory" as const }
        const first = yield* promptBase.reconcileTools({ ...base, seeds: [seed("shell", "eager"), seed("glob")] })

        const connect = yield* promptBase.reconcileTools({
          ...base,
          seeds: [
            seed("shell", "eager"),
            seed("glob"),
            seed("firecrawl_scrape", "deferred", "Scrapes a page", "firecrawl"),
            seed("firecrawl_search", "deferred", "Searches the web", "firecrawl"),
          ],
        })
        expect(connect.appended).toEqual(["firecrawl_scrape", "firecrawl_search"])
        expect(connect.entries.map((entry) => entry.name)).toEqual([
          "shell",
          "glob",
          "firecrawl_scrape",
          "firecrawl_search",
        ])
        expect(connect.entries.slice(0, 2)).toEqual(first.entries)

        const later = yield* promptBase.reconcileTools({
          ...base,
          seeds: [...connect.entries.slice(0, 2), seed("firecrawl_scrape", "deferred", "CHANGED", "firecrawl")],
        })
        expect(later.appended).toEqual([])
        expect(later.entries).toEqual(connect.entries)
      }),
    )

    it.instance("agent switch never removes frozen entries or re-evaluates permissibility", () =>
      Effect.gen(function* () {
        const promptBase = yield* PromptBase.Service
        const base = { sessionID: "ses_tools_agent", model: model(), provider: provider(), mode: "advisory" as const }
        const first = yield* promptBase.reconcileTools({
          ...base,
          seeds: [seed("shell", "eager"), seed("glob"), seed("task")],
        })

        // The switched-to agent's per-turn listing excludes task; the frozen
        // base keeps it (R12-007: revocation is enforced at execution time).
        const switched = yield* promptBase.reconcileTools({ ...base, seeds: [seed("shell", "eager"), seed("glob")] })
        expect(switched.appended).toEqual([])
        expect(switched.entries).toEqual(first.entries)
        expect(switched.entries.some((entry) => entry.name === "task")).toBe(true)
      }),
    )

    it.instance("freezes the mode at first write; a later verdict change never re-renders entries", () =>
      Effect.gen(function* () {
        const promptBase = yield* PromptBase.Service
        const base = { sessionID: "ses_tools_mode", model: model(), provider: provider(), mode: "advisory" as const }
        const first = yield* promptBase.reconcileTools({ ...base, seeds: [seed("glob")] })
        expect(first.mode).toBe("advisory")

        // BindingVerdict.observe can flip the per-turn verdict mid-session
        // (behavioral learning); the frozen base's mode is already written.
        const flipped = yield* promptBase.reconcileTools({
          ...base,
          mode: "binding" as const,
          seeds: [seed("glob", "deferred", "CHANGED")],
        })
        expect(flipped.mode).toBe("advisory")
        const rendered = ToolListing.render(flipped.entries, flipped.mode)
        expect(rendered[0].jsonSchema).toEqual({ type: "object", properties: {} })
      }),
    )

    it.instance("fails with a typed error on duplicate names in one batch", () =>
      Effect.gen(function* () {
        const promptBase = yield* PromptBase.Service
        const error = yield* Effect.flip(
          promptBase.reconcileTools({
            sessionID: "ses_tools_dup",
            model: model(),
            provider: provider(),
            mode: "advisory" as const,
            seeds: [seed("glob"), seed("glob")],
          }),
        )
        expect(error._tag).toBe("DuplicateToolEntryError")
        expect(error.name).toBe("glob")
      }),
    )
  })

  describe("entries", () => {
    it.instance("reads the frozen entries and frozen mode for the session key", () =>
      Effect.gen(function* () {
        const promptBase = yield* PromptBase.Service
        const base = { sessionID: "ses_entries", model: model(), provider: provider(), mode: "advisory" as const }
        yield* promptBase.reconcileTools({ ...base, seeds: [seed("shell", "eager"), seed("glob")] })

        const view = yield* promptBase.entries({ sessionID: "ses_entries", model: model(), provider: provider() })
        expect(view.mode).toBe("advisory")
        expect(view.entries.map((entry) => entry.name)).toEqual(["shell", "glob"])

        // load_tool reads the same base the request froze; another session
        // (or model/endpoint) sees nothing.
        const other = yield* promptBase.entries({
          sessionID: "ses_other",
          model: model(),
          provider: provider(),
        })
        expect(other.entries).toEqual([])
        expect(other.mode).toBeUndefined()
      }),
    )
  })

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
