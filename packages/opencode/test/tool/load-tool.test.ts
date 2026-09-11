import { describe, expect, test } from "bun:test"
import { LayerNode } from "@opencode-ai/core/effect/layer-node"
import { ModelV2 } from "@opencode-ai/core/model"
import { ProviderV2 } from "@opencode-ai/core/provider"
import { SessionV1 } from "@opencode-ai/core/v1/session"
import { Effect, Layer } from "effect"
import { Agent } from "../../src/agent/agent"
import { PromptBase } from "../../src/session/llm/prompt-base"
import { Provider } from "../../src/provider/provider"
import { SessionID, MessageID, PartID } from "../../src/session/schema"
import { Truncate } from "../../src/tool/truncate"
import { LoadTool, delivered } from "../../src/tool/load_tool"
import type { Tool } from "../../src/tool/tool"
import { testEffect } from "../lib/effect"

const sessionID = SessionID.make("ses_load-tool")
const messageID = MessageID.ascending()

const providerInfo: Provider.Info = {
  id: ProviderV2.ID.make("test"),
  name: "Test",
  source: "custom",
  env: [],
  options: {},
  models: {},
}

const providerStub = Layer.mock(Provider.Service, {
  getProvider: () => Effect.succeed(providerInfo),
})

const it = testEffect(
  LayerNode.compile(LayerNode.group([PromptBase.node, Provider.node, Agent.node, Truncate.node]), [
    [Provider.node, providerStub],
  ]),
)

const model: Provider.Model = {
  id: ModelV2.ID.make("test-model"),
  providerID: ProviderV2.ID.make("test"),
  api: { id: "test-model", url: "https://api.test", npm: "@ai-sdk/test" },
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
}

const ctx = (messages: SessionV1.WithParts[] = []): Tool.Context => ({
  sessionID,
  messageID,
  agent: "build",
  abort: new AbortController().signal,
  messages,
  extra: { model },
  metadata: () => Effect.void,
  ask: () => Effect.void,
})

// Freezes a base for the test session: eager shell, deferred glob and grep.
const freeze = (mode: "advisory" | "binding" = "advisory") =>
  Effect.gen(function* () {
    const promptBase = yield* PromptBase.Service
    return yield* promptBase.reconcileTools({
      sessionID,
      model,
      provider: providerInfo,
      mode,
      seeds: [
        {
          name: "shell",
          kind: "eager",
          fullDescription: "shell full description",
          jsonSchema: { type: "object", properties: {} },
        },
        {
          name: "glob",
          kind: "deferred",
          fullDescription: "glob full description",
          jsonSchema: {
            type: "object",
            properties: { pattern: { type: "string", description: "The glob pattern" } },
            required: ["pattern"],
          },
        },
        {
          name: "grep",
          kind: "deferred",
          fullDescription: "grep full description",
          jsonSchema: { type: "object", properties: { pattern: { type: "string" } }, required: ["pattern"] },
        },
      ],
    })
  })

const assistantInfo = (id: string): SessionV1.Assistant =>
  ({
    id,
    sessionID,
    role: "assistant",
    time: { created: 0 },
    parentID: "msg_parent",
    modelID: "test-model",
    providerID: "test",
    mode: "",
    agent: "build",
    path: { cwd: "/", root: "/" },
    cost: 0,
    tokens: { input: 0, output: 0, reasoning: 0, cache: { read: 0, write: 0 } },
  }) as unknown as SessionV1.Assistant

// A persisted load_tool part whose output delivered `tools`. Typed with the
// completed state so the fixtures can read/override its fields.
const loadPart = (tools: string[], compacted?: number): SessionV1.ToolPart & { state: SessionV1.ToolStateCompleted } => ({
  id: PartID.ascending(),
  sessionID,
  messageID,
  type: "tool",
  callID: "call_load",
  tool: "load_tool",
  state: {
    status: "completed",
    input: { tools },
    output: "full content",
    title: `Loaded ${tools.length} tools`,
    metadata: { load_tool: { tools }, truncated: false },
    time: { start: 0, end: 1, ...(compacted ? { compacted } : {}) },
  },
})

const history = (parts: SessionV1.Part[]): SessionV1.WithParts[] => [{ info: assistantInfo("msg_1"), parts }]

describe("tool.load-tool", () => {
  it.instance("serves the full description and schema on first load (advisory)", () =>
    Effect.gen(function* () {
      yield* freeze()
      const info = yield* LoadTool
      const tool = yield* info.init()

      const result = yield* tool.execute({ tools: ["glob"] }, ctx())

      expect(result.output).toContain("### glob")
      expect(result.output).toContain("glob full description")
      expect(result.output).toContain("Input schema:")
      expect(result.output).toContain('"pattern"')
      expect(result.output).not.toContain("already")
      expect(result.metadata.load_tool).toEqual({ tools: ["glob"] })
      expect(result.metadata.truncated).toBe(false)
    }),
  )

  it.instance("bulk loads deduplicated names in order, one section per name", () =>
    Effect.gen(function* () {
      yield* freeze()
      const info = yield* LoadTool
      const tool = yield* info.init()

      const result = yield* tool.execute({ tools: ["glob", "grep", "glob"] }, ctx())

      expect(result.output).toContain("### glob")
      expect(result.output).toContain("### grep")
      expect(result.metadata.load_tool).toEqual({ tools: ["glob", "grep"] })
      expect(result.title).toBe("Loaded 2 tools")
    }),
  )

  it.instance("returns a short confirmation when the load is still delivered, and marks nothing", () =>
    Effect.gen(function* () {
      yield* freeze()
      const info = yield* LoadTool
      const tool = yield* info.init()
      const messages = history([loadPart(["glob"])])

      const result = yield* tool.execute({ tools: ["glob"] }, ctx(messages))

      expect(result.output).toContain("glob: already loaded.")
      expect(result.output).not.toContain("### glob")
      expect(result.metadata.load_tool).toBeUndefined()
      // A mixed call marks only what it (re)served.
      const mixed = yield* tool.execute({ tools: ["glob", "grep"] }, ctx(messages))
      expect(mixed.output).toContain("glob: already loaded.")
      expect(mixed.output).toContain("### grep")
      expect(mixed.metadata.load_tool).toEqual({ tools: ["grep"] })
    }),
  )

  it.instance("re-serves the full content after the original load left the model-visible history", () =>
    Effect.gen(function* () {
      yield* freeze()
      const info = yield* LoadTool
      const tool = yield* info.init()

      // Synthetic compaction: the prune stamp makes the model see
      // "[Old tool result content cleared]" instead of the load output.
      const compacted = history([loadPart(["glob"], Date.now())])
      const result = yield* tool.execute({ tools: ["glob"] }, ctx(compacted))
      expect(result.output).toContain("### glob")
      expect(result.output).toContain("Input schema:")
      expect(result.metadata.load_tool).toEqual({ tools: ["glob"] })
    }),
  )

  it.instance("on binding sessions serves the description only — the schema is already listed", () =>
    Effect.gen(function* () {
      yield* freeze("binding")
      const info = yield* LoadTool
      const tool = yield* info.init()

      const result = yield* tool.execute({ tools: ["glob"] }, ctx())

      expect(result.output).toContain("### glob")
      expect(result.output).toContain("glob full description")
      expect(result.output).not.toContain("Input schema:")
      expect(result.output).toContain("already registered in the tool listing")
      expect(result.metadata.load_tool).toEqual({ tools: ["glob"] })
    }),
  )

  it.instance("unknown names error per name with loadable candidates; eager names refuse; empty input errors", () =>
    Effect.gen(function* () {
      yield* freeze()
      const info = yield* LoadTool
      const tool = yield* info.init()

      const unknown = yield* tool.execute({ tools: ["bogus"] }, ctx())
      expect(unknown.output).toContain("bogus: unknown tool. Loadable tools: glob, grep")

      const eager = yield* tool.execute({ tools: ["shell"] }, ctx())
      expect(eager.output).toContain("shell: already fully listed in the tool listing.")

      const empty = yield* tool.execute({ tools: [] }, ctx())
      expect(empty.output).toContain("No tool names provided.")
      expect(empty.metadata.load_tool).toBeUndefined()

      // Nothing fatal: errors are per-name output lines, the call completes.
      const mixed = yield* tool.execute({ tools: ["bogus", "shell", "glob"] }, ctx())
      expect(mixed.output).toContain("bogus: unknown tool")
      expect(mixed.output).toContain("shell: already fully listed")
      expect(mixed.output).toContain("### glob")
      expect(mixed.metadata.load_tool).toEqual({ tools: ["glob"] })
    }),
  )
})

describe("tool.load-tool delivered", () => {
  // The marker scan runs over the model-visible filtered history; no Effect
  // runtime needed — pure predicate over persisted parts.
  test("completed parts whose output delivered the name qualify", () => {
    expect(delivered("glob", history([loadPart(["glob"])]))).toBe(true)
  })

  test("compacted completed parts do not qualify — the model no longer sees the content", () => {
    expect(delivered("glob", history([loadPart(["glob"], Date.now())]))).toBe(false)
  })

  test("error parts qualify by marker alone", () => {
    const part = loadPart(["grep"])
    const errored: SessionV1.ToolPart = {
      ...part,
      tool: "grep",
      state: {
        status: "error",
        input: { pattern: "*" },
        error: "The grep tool failed...\n\nInput schema: ...",
        metadata: { load_tool: { tools: ["grep"] } },
        time: { start: 0, end: 1 },
      },
    }
    expect(delivered("grep", history([errored]))).toBe(true)
  })

  test("parts without the marker never qualify", () => {
    const base = loadPart(["glob"])
    const noMarker: SessionV1.ToolPart = {
      ...base,
      state: { ...base.state, metadata: {} },
    }
    expect(delivered("glob", history([noMarker]))).toBe(false)
    expect(delivered("glob", history([]))).toBe(false)
  })

  test("fallback-delivered parts of other tools qualify (marker over part tool)", () => {
    const fallback: SessionV1.ToolPart = {
      ...loadPart(["glob"]),
      tool: "grep",
      state: { ...loadPart(["glob"]).state, input: { pattern: "*" } },
    }
    expect(delivered("glob", history([fallback]))).toBe(true)
  })
})