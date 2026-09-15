import { describe, expect } from "bun:test"
import type { JSONSchema7 } from "@ai-sdk/provider"
import { LayerNode } from "@opencode-ai/core/effect/layer-node"
import { ModelV2 } from "@opencode-ai/core/model"
import { ProviderV2 } from "@opencode-ai/core/provider"
import { Effect } from "effect"
import { tool as aiTool, jsonSchema, type ModelMessage, type Tool } from "ai"
import fs from "fs/promises"
import os from "os"
import path from "path"
import type { Agent } from "../../src/agent/agent"
import { RuntimeFlags } from "../../src/effect/runtime-flags"
import { LLMRequestPrep } from "../../src/session/llm/request"
import { PromptBase, type SystemBlock } from "../../src/session/llm/prompt-base"
import type { Plugin } from "../../src/plugin"
import type { Provider } from "../../src/provider/provider"
import { MessageID, SessionID } from "../../src/session/schema"
import { SystemPrompt } from "../../src/session/system"
import type { ToolSeed, Verdict } from "../../src/session/tool-listing"
import { DEFERRED_TOOL_DESCRIPTION, DEFERRED_TOOL_SCHEMA } from "../../src/tool/deferred_tool"
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
  toolWrapper?: boolean
  data?: string
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
      toolWrapper: input.toolWrapper,
      provider,
      auth: undefined,
      plugin,
      promptBase: input.promptBase,
      flags: input.flags,
      data: input.data ?? "/tmp",
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

  const seed = (
    name: string,
    kind: "eager" | "deferred" = "deferred",
    description?: string,
    server?: string,
    source: ToolSeed["source"] = "builtin",
  ): ToolSeed => ({
    name,
    kind,
    fullDescription: description ?? `${name} description`,
    jsonSchema: toolSchema(name),
    source,
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

describe("session.llm-request-prep.wrapper-payload (R12-012, ticket 17)", () => {
  const toolSchema = (label: string): JSONSchema7 => ({
    type: "object",
    properties: { label: { type: "string", const: label } },
    required: ["label"],
  })

  const seed = (
    name: string,
    kind: "eager" | "deferred" = "deferred",
    description?: string,
    server?: string,
    source: ToolSeed["source"] = "builtin",
  ): ToolSeed => ({
    name,
    kind,
    fullDescription: description ?? `${name} description`,
    jsonSchema: toolSchema(name),
    source,
    ...(server ? { server } : {}),
  })

  const fullTool = (name: string, description: string): Tool =>
    aiTool({
      description,
      inputSchema: jsonSchema(toolSchema(name)),
      execute: async () => ({ output: "", title: "", metadata: {} }),
    })

  const long = "find files by glob patterns " + "y".repeat(80)

  // The per-turn record mirrors what SessionTools.resolve produces on a
  // wrapper session: eager load_tool + deferred_tool seeds, glob as a deferred
  // seed, and a per-turn AITool record that still carries glob's full facts.
  const wrapperSeeds = (): ToolSeed[] => [
    seed("load_tool", "eager"),
    seed("deferred_tool", "eager", DEFERRED_TOOL_DESCRIPTION),
    seed("glob", "deferred", long),
  ]
  const wrapperTools = (): Record<string, Tool> => ({
    load_tool: fullTool("load_tool", "Loads a deferred tool"),
    glob: fullTool("glob", long),
  })

  // Round-1 seeds (no meta tool): what SessionTools.resolve produces on
  // advisory or kill-switch sessions — deferred_tool never enters the universe.
  const round1Seeds = (): ToolSeed[] => [seed("load_tool", "eager"), seed("glob", "deferred", long)]

  it.instance("wrapper session: frozen deferred entries are dropped from the payload; eager entries stay", () =>
    Effect.gen(function* () {
      const promptBase = yield* PromptBase.Service
      const base = yield* RuntimeFlags.Service
      const prepared = yield* prepareWith({
        promptBase,
        flags: base,
        sessionID: "ses_wrapper_drop",
        tools: wrapperTools(),
        toolSeeds: wrapperSeeds(),
        toolVerdict: "binding",
        toolWrapper: true,
      })
      // R12-012: glob is not in the tools array in any form — the per-turn
      // record's full facts must not leak through impose.
      expect(Object.keys(prepared.tools).toSorted()).toEqual(["deferred_tool", "load_tool"])
      // the eager meta-tool seed rides the payload; this ticket ships the seed
      // only — no live closure exists yet, so impose's clearly-failing fallback
      // carries it until ticket 19 constructs the meta-tool at the confluence
      expect(prepared.tools.deferred_tool.description).toBe(DEFERRED_TOOL_DESCRIPTION)
      expect(prepared.tools.deferred_tool.execute).toBeDefined()
      expect(prepared.tools.load_tool.inputSchema).toEqual(jsonSchema(toolSchema("load_tool")))
    }),
  )

  it.instance("the frozen pair decides: a mid-session wrapper flip cannot re-add dropped entries", () =>
    Effect.gen(function* () {
      const promptBase = yield* PromptBase.Service
      const base = yield* RuntimeFlags.Service
      const first = yield* prepareWith({
        promptBase,
        flags: base,
        sessionID: "ses_wrapper_frozen",
        tools: wrapperTools(),
        toolSeeds: wrapperSeeds(),
        toolVerdict: "binding",
        toolWrapper: true,
      })
      expect(first.tools.glob).toBeUndefined()

      // a later turn reports wrapper=false (kill-switch flipped mid-session);
      // the frozen wrapper=true still governs the written shapes
      const after = yield* prepareWith({
        promptBase,
        flags: base,
        sessionID: "ses_wrapper_frozen",
        tools: wrapperTools(),
        toolSeeds: wrapperSeeds(),
        toolVerdict: "binding",
        toolWrapper: false,
      })
      expect(after.tools.glob).toBeUndefined()
      expect(after.tools.deferred_tool).toBeDefined()

      // the mirror: a session that started wrapper=false keeps glob listed
      // even when a later turn reports wrapper=true
      yield* prepareWith({
        promptBase,
        flags: base,
        sessionID: "ses_wrapper_frozen_off",
        tools: wrapperTools(),
        toolSeeds: round1Seeds(),
        toolVerdict: "binding",
        toolWrapper: false,
      })
      const late = yield* prepareWith({
        promptBase,
        flags: base,
        sessionID: "ses_wrapper_frozen_off",
        tools: wrapperTools(),
        toolSeeds: round1Seeds(),
        toolVerdict: "binding",
        toolWrapper: true,
      })
      expect(late.tools.glob.inputSchema).toEqual(jsonSchema(toolSchema("glob")))
    }),
  )

  it.instance("non-frozen per-turn tools (StructuredOutput, _noop) pass through on wrapper sessions", () =>
    Effect.gen(function* () {
      const promptBase = yield* PromptBase.Service
      const base = yield* RuntimeFlags.Service
      const structuredOutput = fullTool("StructuredOutput", "Return structured output")
      const noop = fullTool("_noop", "Do not call this tool.")
      const prepared = yield* prepareWith({
        promptBase,
        flags: base,
        sessionID: "ses_wrapper_pass",
        tools: { ...wrapperTools(), StructuredOutput: structuredOutput, _noop: noop },
        toolSeeds: wrapperSeeds(),
        toolVerdict: "binding",
        toolWrapper: true,
      })
      expect(prepared.tools.StructuredOutput.description).toBe("Return structured output")
      expect(prepared.tools.StructuredOutput.inputSchema).toEqual(jsonSchema(toolSchema("StructuredOutput")))
      expect(prepared.tools._noop.description).toBe("Do not call this tool.")
    }),
  )

  it.instance("advisory payload is unchanged by the wrapper axis (defensive advisory-wins)", () =>
    Effect.gen(function* () {
      const promptBase = yield* PromptBase.Service
      const base = yield* RuntimeFlags.Service
      const prepared = yield* prepareWith({
        promptBase,
        flags: base,
        sessionID: "ses_wrapper_advisory",
        tools: wrapperTools(),
        toolSeeds: round1Seeds(),
        toolVerdict: "advisory",
        toolWrapper: true,
      })
      expect(Object.keys(prepared.tools).toSorted()).toEqual(["glob", "load_tool"])
      expect(prepared.tools.glob.inputSchema).toEqual(jsonSchema({ type: "object", properties: {} }))
      expect(prepared.tools.glob.description).toBe(long.slice(0, long.lastIndexOf(" ")) + "...")
    }),
  )

  it.instance("binding with the wrapper kill-switch: schema-eager round-1 payload bytes", () =>
    Effect.gen(function* () {
      const promptBase = yield* PromptBase.Service
      const base = yield* RuntimeFlags.Service
      const prepared = yield* prepareWith({
        promptBase,
        flags: base,
        sessionID: "ses_wrapper_kill",
        tools: wrapperTools(),
        toolSeeds: round1Seeds(),
        toolVerdict: "binding",
        toolWrapper: false,
      })
      expect(Object.keys(prepared.tools).toSorted()).toEqual(["glob", "load_tool"])
      expect(prepared.tools.glob.inputSchema).toEqual(jsonSchema(toolSchema("glob")))
      expect(prepared.tools.glob.description).toBe(long.slice(0, long.lastIndexOf(" ")) + "...")

      // a missing toolWrapper resolves round-1 semantics (false) as well
      const missing = yield* prepareWith({
        promptBase,
        flags: base,
        sessionID: "ses_wrapper_kill_missing",
        tools: wrapperTools(),
        toolSeeds: round1Seeds(),
        toolVerdict: "binding",
      })
      expect(missing.tools.glob.inputSchema).toEqual(jsonSchema(toolSchema("glob")))
    }),
  )
})

describe("session.llm-request-prep.capture-tool-servers (R10-006, ticket 23)", () => {
  const toolSchema = (label: string): JSONSchema7 => ({
    type: "object",
    properties: { label: { type: "string", const: label } },
    required: ["label"],
  })

  const fullTool = (name: string, description: string): Tool =>
    aiTool({
      description,
      inputSchema: jsonSchema(toolSchema(name)),
      execute: async () => ({ output: "", title: "", metadata: {} }),
    })

  const seed = (name: string, kind: "eager" | "deferred" = "deferred", server?: string, source: ToolSeed["source"] = "builtin"): ToolSeed => ({
    name,
    kind,
    fullDescription: `${name} description`,
    jsonSchema: toolSchema(name),
    source,
    ...(server ? { server } : {}),
  })

  const tools = (): Record<string, Tool> => ({
    load_tool: fullTool("load_tool", "Loads a deferred tool"),
    glob: fullTool("glob", "find files"),
    bash: fullTool("bash", "Runs a shell command"),
  })

  // One MCP-attributed deferred tool amid inherent ones — the map carries the
  // attributed entry only (server !== undefined), keyed by tool name.
  const seeds = (): ToolSeed[] => [seed("load_tool", "eager"), seed("glob", "deferred", "firecrawl", "mcp"), seed("bash")]

  // Capture dump target for the unit-level seam: prepare with the capture flag
  // on and a real data dir, then read the dumped file back.
  const readCaptureMeta = async (data: string, sessionID: string) => {
    const file = await Bun.file(path.join(data, "prompt-captures", sessionID, "0000.json")).json()
    return file as { meta: Record<string, unknown> }
  }

  const captureData = Effect.acquireRelease(
    Effect.promise(() => fs.mkdtemp(path.join(os.tmpdir(), "opencode-prep-capture-"))),
    (dir) => Effect.promise(() => fs.rm(dir, { recursive: true, force: true })),
  )

  it.instance("lazy dump carries meta.toolServers for MCP-attributed frozen entries", () =>
    Effect.gen(function* () {
      const data = yield* captureData
      const promptBase = yield* PromptBase.Service
      const base = yield* RuntimeFlags.Service
      yield* prepareWith({
        promptBase,
        flags: { ...base, enablePromptCapture: true },
        sessionID: "ses_capture_servers",
        data,
        tools: tools(),
        toolSeeds: seeds(),
        toolVerdict: "advisory",
      })
      const file = yield* Effect.promise(() => readCaptureMeta(data, "ses_capture_servers"))
      expect(file.meta.toolServers).toEqual({ glob: "firecrawl" })
    }),
  )

  it.instance("bypass (kill-switch) and small dumps omit the field entirely", () =>
    Effect.gen(function* () {
      const data = yield* captureData
      const promptBase = yield* PromptBase.Service
      const base = yield* RuntimeFlags.Service
      yield* prepareWith({
        promptBase,
        flags: { ...base, enablePromptCapture: true, disableLazyTools: true },
        sessionID: "ses_capture_bypass",
        data,
        tools: tools(),
        toolSeeds: seeds(),
        toolVerdict: "advisory",
      })
      yield* prepareWith({
        promptBase,
        flags: { ...base, enablePromptCapture: true },
        sessionID: "ses_capture_small",
        data,
        small: true,
        tools: tools(),
        toolSeeds: seeds(),
        toolVerdict: "advisory",
      })
      const bypass = yield* Effect.promise(() => readCaptureMeta(data, "ses_capture_bypass"))
      const small = yield* Effect.promise(() => readCaptureMeta(data, "ses_capture_small"))
      expect("toolServers" in bypass.meta).toBe(false)
      expect("toolServers" in small.meta).toBe(false)
    }),
  )
})
