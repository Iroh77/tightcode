import { describe, expect } from "bun:test"
import type { JSONSchema7 } from "@ai-sdk/provider"
import { LayerNode } from "@opencode-ai/core/effect/layer-node"
import { ModelV2 } from "@opencode-ai/core/model"
import { ProviderV2 } from "@opencode-ai/core/provider"
import { Effect } from "effect"
import { tool as aiTool, jsonSchema, type ModelMessage, type Tool } from "ai"
import type { Agent } from "../../src/agent/agent"
import { RuntimeFlags } from "../../src/effect/runtime-flags"
import { LLMRequestPrep } from "../../src/session/llm/request"
import { PromptBase, type SystemBlock } from "../../src/session/llm/prompt-base"
import type { Plugin } from "../../src/plugin"
import type { Provider } from "../../src/provider/provider"
import { MessageID, SessionID } from "../../src/session/schema"
import { SystemPrompt } from "../../src/session/system"
import type { ToolSeed, Verdict } from "../../src/session/tool-listing"
import { testEffect } from "../lib/effect"

const it = testEffect(LayerNode.compile(LayerNode.group([PromptBase.node, RuntimeFlags.node])))

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

const provider: Provider.Info = {
  id: ProviderV2.ID.make("test"),
  name: "Test",
  source: "custom",
  env: [],
  options: {},
  models: {},
}

const agent: Agent.Info = {
  name: "build",
  mode: "primary",
  permission: [],
  options: {},
}

const plugin: Plugin.Interface = {
  trigger: (_name, _input, output) => Effect.succeed(output),
  list: () => Effect.succeed([]),
  init: () => Effect.void,
}

const blocks = (instructions: string): SystemBlock[] => [
  { key: "environment", content: "env block" },
  { key: "instructions", content: instructions },
  { key: "skills", content: "skills block" },
]

const expectedSystem = (turnBlocks: SystemBlock[]) =>
  [[...SystemPrompt.provider(model), ...PromptBase.render(turnBlocks)].filter((x) => x).join("\n")]

const prepareWith = (input: {
  promptBase: PromptBase.Interface
  flags: RuntimeFlags.Info
  sessionID?: string
  system?: SystemBlock[]
  small?: boolean
  messages?: ModelMessage[]
  tools?: Record<string, Tool>
  toolSeeds?: ToolSeed[]
  toolVerdict?: Verdict
}) =>
  Effect.gen(function* () {
    return yield* LLMRequestPrep.prepare({
      user: {
        id: MessageID.ascending(),
        sessionID: SessionID.make(input.sessionID ?? "ses_prep"),
        role: "user",
        time: { created: 0 },
        agent: "build",
        model: { providerID: ProviderV2.ID.make("test"), modelID: ModelV2.ID.make("test-model") },
      },
      sessionID: input.sessionID ?? "ses_prep",
      model,
      agent,
      permission: [],
      system: input.system ?? blocks("instructions v1"),
      messages: input.messages ?? [],
      small: input.small,
      tools: input.tools ?? {},
      toolSeeds: input.toolSeeds ?? [],
      toolVerdict: input.toolVerdict,
      provider,
      auth: undefined,
      plugin,
      promptBase: input.promptBase,
      flags: input.flags,
      data: "/tmp",
      isWorkflow: false,
    })
  })

describe("session.llm-request-prep.prompt-freeze", () => {
  it.instance("freezes system blocks across turns; later turns re-serve the first write", () =>
    Effect.gen(function* () {
      const promptBase = yield* PromptBase.Service
      const base = yield* RuntimeFlags.Service
      const first = yield* prepareWith({ promptBase, flags: base })
      expect(first.system).toEqual(expectedSystem(blocks("instructions v1")))

      const second = yield* prepareWith({
        promptBase,
        flags: base,
        system: blocks("instructions v2 CHANGED"),
      })
      expect(second.system).toEqual(first.system)
    }),
  )

  it.instance("kill-switch restores upstream per-turn bytes", () =>
    Effect.gen(function* () {
      const promptBase = yield* PromptBase.Service
      const base = yield* RuntimeFlags.Service
      const frozen = yield* prepareWith({ promptBase, flags: base })
      const bypass = yield* prepareWith({
        promptBase,
        flags: { ...base, disableLazyTools: true },
        system: blocks("instructions v2 CHANGED"),
      })
      expect(bypass.system).toEqual(expectedSystem(blocks("instructions v2 CHANGED")))
      expect(bypass.system).not.toEqual(frozen.system)
    }),
  )

  it.instance("small turns bypass the freeze entirely", () =>
    Effect.gen(function* () {
      const promptBase = yield* PromptBase.Service
      const base = yield* RuntimeFlags.Service
      const small = yield* prepareWith({
        promptBase,
        flags: base,
        small: true,
        system: blocks("instructions small-turn"),
      })
      expect(small.system).toEqual(expectedSystem(blocks("instructions small-turn")))

      const normal = yield* prepareWith({ promptBase, flags: base, system: blocks("instructions v1") })
      expect(normal.system).toEqual(expectedSystem(blocks("instructions v1")))

      const stable = yield* prepareWith({ promptBase, flags: base, system: blocks("instructions v2 CHANGED") })
      expect(stable.system).toEqual(normal.system)
    }),
  )
})

describe("session.llm-request-prep.tool-freeze", () => {
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

  // The per-turn AITool record upstream rebuilds every turn: full facts. The
  // freeze must pin the payload to the frozen listing regardless.
  const fullTool = (name: string, description: string): Tool =>
    aiTool({
      description,
      inputSchema: jsonSchema(toolSchema(name)),
      execute: async () => ({ output: "", title: "", metadata: {} }),
    })

  const long = "find files by glob patterns " + "y".repeat(80)

  const seedsTurn = (description?: string): ToolSeed[] => [seed("load_tool", "eager", "Loads a deferred tool"), seed("glob", "deferred", description ?? long)]

  const toolsTurn = (description?: string): Record<string, Tool> => ({
    load_tool: fullTool("load_tool", "Loads a deferred tool"),
    glob: fullTool("glob", description ?? long),
  })

  it.instance("imposes the frozen listing at the payload: deferred entries keep the placeholder", () =>
    Effect.gen(function* () {
      const promptBase = yield* PromptBase.Service
      const base = yield* RuntimeFlags.Service
      const prepared = yield* prepareWith({
        promptBase,
        flags: base,
        sessionID: "ses_tools_payload",
        tools: toolsTurn(),
        toolSeeds: seedsTurn(),
        toolVerdict: "advisory",
      })
      expect(Object.keys(prepared.tools).toSorted()).toEqual(["glob", "load_tool"])
      // eager: full facts
      expect(prepared.tools.load_tool.description).toBe("Loads a deferred tool")
      expect(prepared.tools.load_tool.inputSchema).toEqual(jsonSchema(toolSchema("load_tool")))
      // deferred: truncated description + placeholder schema — by construction
      // at the payload, even though the per-turn record carried full facts
      expect(prepared.tools.glob.description).toBe(long.slice(0, long.lastIndexOf(" ")) + "...")
      expect(prepared.tools.glob.inputSchema).toEqual(jsonSchema({ type: "object", properties: {} }))
    }),
  )

  it.instance("payload bytes stay stable across turns; later recomputation is absorbed", () =>
    Effect.gen(function* () {
      const promptBase = yield* PromptBase.Service
      const base = yield* RuntimeFlags.Service
      const first = yield* prepareWith({
        promptBase,
        flags: base,
        sessionID: "ses_tools_stable",
        tools: toolsTurn(),
        toolSeeds: seedsTurn(),
        toolVerdict: "advisory",
      })

      const second = yield* prepareWith({
        promptBase,
        flags: base,
        sessionID: "ses_tools_stable",
        tools: toolsTurn("CHANGED description " + "z".repeat(90)),
        toolSeeds: seedsTurn("CHANGED description " + "z".repeat(90)),
      })
      expect(second.tools.glob.description).toEqual(first.tools.glob.description)
      expect(second.tools.glob.inputSchema).toEqual(first.tools.glob.inputSchema)
      expect(second.tools.load_tool.inputSchema).toEqual(first.tools.load_tool.inputSchema)
    }),
  )

  it.instance("R12-005 at the payload: the entry never gains the full schema mid-session, even after a load", () =>
    Effect.gen(function* () {
      const promptBase = yield* PromptBase.Service
      const base = yield* RuntimeFlags.Service
      const first = yield* prepareWith({
        promptBase,
        flags: base,
        sessionID: "ses_tools_load",
        tools: toolsTurn(),
        toolSeeds: seedsTurn(),
        toolVerdict: "advisory",
      })

      // load_tool output rides message history: the full description + schema
      // reach the model here, never through the tools array.
      const loaded = "### glob\n\n" + long + "\n\nInput schema:\n" + JSON.stringify(toolSchema("glob"))
      const afterLoad = yield* prepareWith({
        promptBase,
        flags: base,
        sessionID: "ses_tools_load",
        tools: toolsTurn(),
        toolSeeds: seedsTurn(),
        toolVerdict: "advisory",
        messages: [
          { role: "user", content: "find the files" },
          {
            role: "assistant",
            content: [{ type: "tool-call", toolCallId: "call_1", toolName: "load_tool", input: { tools: ["glob"] } }],
          },
          {
            role: "tool",
            content: [
              {
                type: "tool-result",
                toolCallId: "call_1",
                toolName: "load_tool",
                output: { type: "text", value: loaded },
              },
            ],
          },
        ],
      })
      expect(afterLoad.messages.at(-1)?.content).toEqual([
        { type: "tool-result", toolCallId: "call_1", toolName: "load_tool", output: { type: "text", value: loaded } },
      ])
      expect(afterLoad.tools.glob.inputSchema).toEqual(first.tools.glob.inputSchema)
      expect(afterLoad.tools.glob.inputSchema).toEqual(jsonSchema({ type: "object", properties: {} }))
    }),
  )

  it.instance("a new server's entries appear as one append batch; the prefix is written once", () =>
    Effect.gen(function* () {
      const promptBase = yield* PromptBase.Service
      const base = yield* RuntimeFlags.Service
      const first = yield* prepareWith({
        promptBase,
        flags: base,
        sessionID: "ses_tools_batch",
        tools: toolsTurn(),
        toolSeeds: seedsTurn(),
        toolVerdict: "advisory",
      })

      const connect = yield* prepareWith({
        promptBase,
        flags: base,
        sessionID: "ses_tools_batch",
        tools: {
          ...toolsTurn(),
          firecrawl_scrape: fullTool("firecrawl_scrape", "Scrapes a page"),
          firecrawl_search: fullTool("firecrawl_search", "Searches the web"),
        },
        toolSeeds: [
          ...seedsTurn(),
          seed("firecrawl_scrape", "deferred", "Scrapes a page", "firecrawl"),
          seed("firecrawl_search", "deferred", "Searches the web", "firecrawl"),
        ],
        toolVerdict: "advisory",
      })
      expect(connect.tools.firecrawl_scrape.description).toBe("firecrawl: Scrapes a page")
      expect(connect.tools.firecrawl_search.description).toBe("Searches the web")
      // existing entries byte-identical across the append; the prefix stays on
      // the first-written entry (append stability is asserted at the unit level
      // in tool-listing.test.ts, the frozen insertion order in prompt-base.test.ts)
      expect(connect.tools.glob.description).toEqual(first.tools.glob.description)
      expect(connect.tools.glob.inputSchema).toEqual(first.tools.glob.inputSchema)
      expect(connect.tools.firecrawl_scrape.inputSchema).toEqual(jsonSchema({ type: "object", properties: {} }))
    }),
  )

  it.instance("agent switch: frozen entries stay listed; revocation is execution-time", () =>
    Effect.gen(function* () {
      const promptBase = yield* PromptBase.Service
      const base = yield* RuntimeFlags.Service
      const first = yield* prepareWith({
        promptBase,
        flags: base,
        sessionID: "ses_tools_agent",
        tools: { load_tool: fullTool("load_tool", "Loads a deferred tool"), glob: fullTool("glob", long), task: fullTool("task", "Delegate to an agent") },
        toolSeeds: [seed("load_tool", "eager"), seed("glob"), seed("task")],
        toolVerdict: "advisory",
      })

      // The switched-to agent's per-turn listing drops task; the frozen base
      // keeps it listed (R12-007: nothing is ever removed mid-session).
      const switched = yield* prepareWith({
        promptBase,
        flags: base,
        sessionID: "ses_tools_agent",
        tools: { load_tool: fullTool("load_tool", "Loads a deferred tool"), glob: fullTool("glob", long) },
        toolSeeds: [seed("load_tool", "eager"), seed("glob")],
        toolVerdict: "advisory",
      })
      expect(switched.tools.task).toBeDefined()
      expect(switched.tools.task.description).toEqual(first.tools.task.description)
      expect(switched.tools.task.inputSchema).toEqual(first.tools.task.inputSchema)
    }),
  )

  it.instance("a frozen name the per-turn record no longer produces fails its call with a clear error", () =>
    Effect.gen(function* () {
      const promptBase = yield* PromptBase.Service
      const base = yield* RuntimeFlags.Service
      yield* prepareWith({
        promptBase,
        flags: base,
        sessionID: "ses_tools_dead",
        tools: { glob: fullTool("glob", long) },
        toolSeeds: [seed("glob", "deferred", long)],
        toolVerdict: "advisory",
      })

      // MCP server died mid-session: the per-turn record drops the closure;
      // the frozen base keeps the entry listed with a clearly-failing execute
      // (R12-007: "their calls fail with a clear error").
      const dead = yield* prepareWith({
        promptBase,
        flags: base,
        sessionID: "ses_tools_dead",
        tools: {},
        toolSeeds: [],
      })
      expect(dead.tools.glob).toBeDefined()
      expect(dead.tools.glob.description).toBe(long.slice(0, long.lastIndexOf(" ")) + "...")
      const failed = yield* Effect.flip(
        Effect.tryPromise({
          try: () => (dead.tools.glob as Tool).execute!({}, { toolCallId: "call_x", messages: [] }),
          catch: (error) => error,
        }),
      )
      expect(failed).toBeInstanceOf(Error)
      expect((failed as Error).message).toMatch(/no longer available/)
    }),
  )

  it.instance("unfrozen per-message tools pass through untouched", () =>
    Effect.gen(function* () {
      const promptBase = yield* PromptBase.Service
      const base = yield* RuntimeFlags.Service
      const structuredOutput = fullTool("StructuredOutput", "Return structured output")
      const prepared = yield* prepareWith({
        promptBase,
        flags: base,
        sessionID: "ses_tools_pass",
        tools: { ...toolsTurn(), StructuredOutput: structuredOutput },
        toolSeeds: seedsTurn(),
        toolVerdict: "advisory",
      })
      expect(prepared.tools.StructuredOutput.description).toBe("Return structured output")
      expect(prepared.tools.StructuredOutput.inputSchema).toEqual(jsonSchema(toolSchema("StructuredOutput")))
    }),
  )

  it.instance("kill-switch: upstream per-turn bytes, freeze not engaged", () =>
    Effect.gen(function* () {
      const promptBase = yield* PromptBase.Service
      const base = yield* RuntimeFlags.Service
      const bypass = yield* prepareWith({
        promptBase,
        flags: { ...base, disableLazyTools: true },
        sessionID: "ses_tools_kill",
        tools: toolsTurn(),
        toolSeeds: seedsTurn(),
        toolVerdict: "advisory",
      })
      expect(bypass.tools.glob.description).toBe(long)
      expect(bypass.tools.glob.inputSchema).toEqual(jsonSchema(toolSchema("glob")))

      const changed = yield* prepareWith({
        promptBase,
        flags: { ...base, disableLazyTools: true },
        sessionID: "ses_tools_kill",
        tools: toolsTurn("upstream rebuilt description"),
        toolSeeds: seedsTurn("upstream rebuilt description"),
        toolVerdict: "advisory",
      })
      expect(changed.tools.glob.description).toBe("upstream rebuilt description")
    }),
  )

  it.instance("small turns bypass the tool freeze entirely", () =>
    Effect.gen(function* () {
      const promptBase = yield* PromptBase.Service
      const base = yield* RuntimeFlags.Service
      const small = yield* prepareWith({
        promptBase,
        flags: base,
        sessionID: "ses_tools_small",
        small: true,
        tools: toolsTurn(),
        toolSeeds: seedsTurn(),
        toolVerdict: "advisory",
      })
      expect(small.tools.glob.description).toBe(long)
      expect(small.tools.glob.inputSchema).toEqual(jsonSchema(toolSchema("glob")))
    }),
  )

  it.instance("binding verdict: deferred entries carry the full schema at the payload, description stays truncated", () =>
    Effect.gen(function* () {
      const promptBase = yield* PromptBase.Service
      const base = yield* RuntimeFlags.Service
      const prepared = yield* prepareWith({
        promptBase,
        flags: base,
        sessionID: "ses_tools_binding",
        tools: toolsTurn(),
        toolSeeds: seedsTurn(),
        toolVerdict: "binding",
      })
      // R12-010 amendment 1: schema-eager, description-deferred
      expect(prepared.tools.glob.description).toBe(long.slice(0, long.lastIndexOf(" ")) + "...")
      expect(prepared.tools.glob.inputSchema).toEqual(jsonSchema(toolSchema("glob")))
      // eager entries are mode-independent
      expect(prepared.tools.load_tool.inputSchema).toEqual(jsonSchema(toolSchema("load_tool")))
    }),
  )

  it.instance("the mode is frozen with the first write; a later verdict flip never re-renders entries", () =>
    Effect.gen(function* () {
      const promptBase = yield* PromptBase.Service
      const base = yield* RuntimeFlags.Service
      const first = yield* prepareWith({
        promptBase,
        flags: base,
        sessionID: "ses_tools_flip",
        tools: toolsTurn(),
        toolSeeds: seedsTurn(),
        toolVerdict: "binding",
      })
      // a mid-session BindingVerdict.observe flip changes the per-turn verdict…
      const flipped = yield* prepareWith({
        promptBase,
        flags: base,
        sessionID: "ses_tools_flip",
        tools: toolsTurn(),
        toolSeeds: seedsTurn(),
        toolVerdict: "advisory",
      })
      // …but frozen entries keep the bytes they were written with (R12-007)
      expect(flipped.tools.glob.inputSchema).toEqual(first.tools.glob.inputSchema)
      expect(flipped.tools.glob.inputSchema).toEqual(jsonSchema(toolSchema("glob")))
    }),
  )

  it.instance("a missing verdict resolves conservative binding (R12-010: missing sources ⇒ binding)", () =>
    Effect.gen(function* () {
      const promptBase = yield* PromptBase.Service
      const base = yield* RuntimeFlags.Service
      const prepared = yield* prepareWith({
        promptBase,
        flags: base,
        sessionID: "ses_tools_missing",
        tools: toolsTurn(),
        toolSeeds: seedsTurn(),
      })
      expect(prepared.tools.glob.inputSchema).toEqual(jsonSchema(toolSchema("glob")))
    }),
  )
})
