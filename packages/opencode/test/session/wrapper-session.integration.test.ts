import { Database } from "@opencode-ai/core/database/database"
import { LayerNode } from "@opencode-ai/core/effect/layer-node"
import { CrossSpawnSpawner } from "@opencode-ai/core/cross-spawn-spawner"
import { FSUtil } from "@opencode-ai/core/fs-util"
import { Global } from "@opencode-ai/core/global"
import { Ripgrep } from "@opencode-ai/core/ripgrep"
import { ConfigV1 } from "@opencode-ai/core/v1/config/config"
import { SessionProjector } from "@opencode-ai/core/session/projector"
import { SessionV1 } from "@opencode-ai/core/v1/session"
import { expect, describe } from "bun:test"
import { Effect, Layer } from "effect"
import { Agent as AgentSvc } from "../../src/agent/agent"
import { BindingVerdict } from "../../src/session/binding-verdict"
import { BackgroundJob } from "@/background/job"
import { Command } from "../../src/command"
import { Config } from "@/config/config"
import { Env } from "../../src/env"
import { EventV2Bridge } from "@/event-v2-bridge"
import { Format } from "../../src/format"
import { Git } from "../../src/git"
import { Image } from "../../src/image/image"
import { Instruction } from "../../src/session/instruction"
import { LLM } from "../../src/session/llm"
import { LSP } from "@/lsp/lsp"
import { MCP } from "../../src/mcp"
import { MessageV2 } from "../../src/session/message-v2"
import { Permission } from "../../src/permission"
import { Plugin } from "../../src/plugin"
import { Provider as ProviderSvc } from "@/provider/provider"
import { Question } from "../../src/question"
import { RuntimeFlags } from "@/effect/runtime-flags"
import { SessionID } from "../../src/session/schema"
import { Session } from "@/session/session"
import { SessionCompaction } from "../../src/session/compaction"
import { SessionProcessor } from "../../src/session/processor"
import { SessionPrompt } from "../../src/session/prompt"
import { SessionRevert } from "../../src/session/revert"
import { SessionRunState } from "../../src/session/run-state"
import { SessionStatus } from "../../src/session/status"
import { SessionSummary } from "../../src/session/summary"
import { Skill } from "../../src/skill"
import { Snapshot } from "../../src/snapshot"
import { SystemPrompt } from "../../src/session/system"
import { Todo } from "../../src/session/todo"
import { DEFERRED_TOOL_DESCRIPTION, DEFERRED_TOOL_SCHEMA } from "../../src/tool/deferred_tool"
import { ToolRegistry } from "@/tool/registry"
import { Truncate } from "@/tool/truncate"
import { TestInstance } from "../fixture/fixture"
import { testEffect } from "../lib/effect"
import { TestLLMServer } from "../lib/llm-server"
import GLOB_DESCRIPTION from "../../src/tool/glob.txt"
import GREP_DESCRIPTION from "../../src/tool/grep.txt"
import LOAD_TOOL_DESCRIPTION from "../../src/tool/load_tool.txt"

const summary = Layer.succeed(
  SessionSummary.Service,
  SessionSummary.Service.of({
    summarize: () => Effect.void,
    diff: () => Effect.succeed([]),
    computeDiff: () => Effect.succeed([]),
  }),
)

const lsp = Layer.succeed(
  LSP.Service,
  LSP.Service.of({
    init: () => Effect.void,
    status: () => Effect.succeed([]),
    hasClients: () => Effect.succeed(false),
    touchFile: () => Effect.void,
    diagnostics: () => Effect.succeed({}),
    hover: () => Effect.succeed(undefined),
    definition: () => Effect.succeed([]),
    references: () => Effect.succeed([]),
    implementation: () => Effect.succeed([]),
    documentSymbol: () => Effect.succeed([]),
    workspaceSymbol: () => Effect.succeed([]),
    prepareCallHierarchy: () => Effect.succeed([]),
    incomingCalls: () => Effect.succeed([]),
    outgoingCalls: () => Effect.succeed([]),
  }),
)

const mcp = Layer.mock(MCP.Service, {
  status: () => Effect.succeed({}),
  clients: () => Effect.succeed({}),
  instructions: () => Effect.succeed([]),
  tools: () => Effect.succeed({}),
  prompts: () => Effect.succeed({}),
  resources: () => Effect.succeed({}),
  resourceTemplates: () => Effect.succeed({}),
  add: () => Effect.succeed({ status: { status: "disabled" as const } }),
  connect: () => Effect.void,
  disconnect: () => Effect.void,
  startAuth: () => Effect.die("unexpected MCP auth in wrapper-session integration tests"),
  authenticate: () => Effect.die("unexpected MCP auth in wrapper-session integration tests"),
  finishAuth: () => Effect.die("unexpected MCP auth in wrapper-session integration tests"),
  removeAuth: () => Effect.void,
  supportsOAuth: () => Effect.succeed(false),
  hasStoredTokens: () => Effect.succeed(false),
  getAuthStatus: () => Effect.succeed("not_authenticated" as const),
})

const testLLMServerNode = LayerNode.make({ service: TestLLMServer, layer: TestLLMServer.layer, deps: [] })

const promptRoot = LayerNode.group([
  SessionPrompt.node,
  Session.node,
  SessionProjector.node,
  MessageV2.node,
  Snapshot.node,
  LLM.node,
  Env.node,
  AgentSvc.node,
  Command.node,
  Permission.node,
  Plugin.node,
  Config.node,
  ProviderSvc.node,
  LSP.node,
  MCP.node,
  FSUtil.node,
  BackgroundJob.node,
  SessionStatus.node,
  SessionRunState.node,
  Database.node,
  EventV2Bridge.node,
  Question.node,
  Todo.node,
  ToolRegistry.node,
  Skill.node,
  Git.node,
  Ripgrep.node,
  Format.node,
  Truncate.node,
  SessionProcessor.node,
  Image.node,
  SessionCompaction.node,
  SessionRevert.node,
  Instruction.node,
  SystemPrompt.node,
  CrossSpawnSpawner.node,
  RuntimeFlags.node,
])

// Forced-verdict injection, ticket 08's node-replacement pattern: the
// binding sessions pin the verdict through the cascade's step-0 pin seam
// (R13-003, landed by ticket 16 — the static table cannot be used here
// because its key embeds the server's dynamic port, unknowable at
// layer-compile time), so the assertions never depend on the fixture
// provider's probe behavior. The advisory session injects the verdict
// through the probe seam instead (the fixture provider cannot answer the
// probe with a violating tool call).
const forcedVerdictNode = (options: BindingVerdict.Options) =>
  LayerNode.make({
    service: BindingVerdict.Service,
    layer: BindingVerdict.layerWith(options),
    deps: [FSUtil.node, Global.node, ProviderSvc.node],
  })

const makeEnv = (verdict?: BindingVerdict.Options, flags?: Partial<RuntimeFlags.Info>) => {
  const root = LayerNode.group([promptRoot, testLLMServerNode])
  const replacements: LayerNode.Replacement[] = [
    [SessionSummary.node, summary],
    [LSP.node, lsp],
    [MCP.node, mcp],
    [RuntimeFlags.node, RuntimeFlags.layer({ experimentalEventSystem: true, ...flags })],
  ]
  if (verdict) replacements.push([BindingVerdict.node, forcedVerdictNode(verdict)])
  return LayerNode.compile(root, replacements)
}

// Wrapper axis on by default (R12-012): pinned binding verdict + wrapper active.
const wrapper = testEffect(makeEnv({ pin: "binding" }))
// R00-013 wrapper kill-switch: binding reverts to schema-eager round-1.
const killSwitch = testEffect(makeEnv({ pin: "binding" }, { disableToolWrapper: true }))
const advisory = testEffect(makeEnv({ probe: () => Effect.succeed("advisory") }))

const cfg = {
  provider: {
    test: {
      name: "Test",
      id: "test",
      env: [],
      npm: "@ai-sdk/openai-compatible",
      models: {
        "test-model": {
          id: "test-model",
          name: "Test Model",
          attachment: false,
          reasoning: false,
          temperature: false,
          tool_call: true,
          release_date: "2025-01-01",
          limit: { context: 100000, output: 10000 },
          cost: { input: 0, output: 0 },
          options: {},
        },
      },
      options: {
        apiKey: "test-key",
        baseURL: "http://localhost:1/v1",
      },
    },
  },
}

function providerCfg(url: string) {
  return {
    ...cfg,
    provider: {
      ...cfg.provider,
      test: {
        ...cfg.provider.test,
        options: {
          ...cfg.provider.test.options,
          baseURL: url,
        },
      },
    },
  }
}

const useServerConfig = Effect.fn("test.useServerConfig")(function* (config: (url: string) => Partial<ConfigV1.Info>) {
  const { directory: dir } = yield* TestInstance
  const fs = yield* FSUtil.Service
  const llm = yield* TestLLMServer
  yield* fs.writeWithDirs(
    `${dir}/opencode.json`,
    JSON.stringify({ $schema: "https://opencode.ai/config.json", ...config(llm.url) }),
  )
  return llm
})

type ChatBody = {
  messages?: Array<{
    role: string
    content: unknown
    tool_calls?: Array<{ function?: { name?: string; arguments?: string } }>
  }>
  tools?: Array<{ function?: { name?: string; description?: string; parameters?: unknown } }>
}

const bodyOf = (hit: { body: Record<string, unknown> }) => hit.body as ChatBody
const systemBytes = (hit: { body: Record<string, unknown> }) =>
  JSON.stringify((bodyOf(hit).messages ?? []).filter((message) => message.role === "system"))
const toolBytes = (hit: { body: Record<string, unknown> }) => JSON.stringify(bodyOf(hit).tools ?? [])
const fnOf = (hit: { body: Record<string, unknown> }, name: string) =>
  (bodyOf(hit).tools ?? []).find((entry) => entry.function?.name === name)?.function
const toolCallOf = (hit: { body: Record<string, unknown> }, name: string) =>
  (bodyOf(hit).messages ?? []).flatMap((message) => message.tool_calls ?? []).find(
    (call) => call.function?.name === name,
  )
const propertiesOf = (fn: { parameters?: unknown }) =>
  Object.keys(((fn.parameters ?? {}) as { properties?: Record<string, unknown> }).properties ?? {})

const PLACEHOLDER = { type: "object", properties: {} }

type CompletedToolPart = SessionV1.ToolPart & { state: SessionV1.ToolStateCompleted }
type ErrorToolPart = SessionV1.ToolPart & { state: SessionV1.ToolStateError }

const sessionWithPrompt = Effect.fn("test.sessionWithPrompt")(function* (title: string, text: string) {
  const prompt = yield* SessionPrompt.Service
  const sessions = yield* Session.Service
  const session = yield* sessions.create({
    title,
    permission: [{ permission: "*", pattern: "*", action: "allow" }],
  })
  yield* prompt.prompt({
    sessionID: session.id,
    agent: "build",
    noReply: true,
    parts: [{ type: "text", text }],
  })
  return { session, prompt }
})

const toolPartsOf = (sessionID: SessionID) =>
  Effect.gen(function* () {
    const messages = yield* MessageV2.filterCompactedEffect(sessionID)
    return messages
      .flatMap((message) => message.parts)
      .filter((part): part is SessionV1.ToolPart => part.type === "tool")
  })

describe("binding+wrapper session end-to-end", () => {
  wrapper.instance(
    "wrapper session: eager-only listing, catalog discovery, dispatch, schema-in-error, full load_tool serve, byte stability",
    () =>
      Effect.gen(function* () {
        const llm = yield* useServerConfig(providerCfg)
        const { session, prompt } = yield* sessionWithPrompt("Wrapper session", "find text files")

        // Step 2: the wrapper call executes the inner tool.
        yield* llm.tool("deferred_tool", { name: "glob", args: JSON.stringify({ pattern: "*.json" }) })
        // Step 3: unparseable args on a known name → schema-in-error.
        yield* llm.tool("deferred_tool", { name: "glob", args: "{oops" })
        // Step 3: the marker re-opens the short confirmation.
        yield* llm.tool("load_tool", { tools: ["glob"] })
        // Step 4: load_tool on a fresh deferred name in wrapper mode.
        yield* llm.tool("load_tool", { tools: ["grep"] })
        yield* llm.text("done")

        const result = yield* prompt.loop({ sessionID: session.id })
        expect(result.info.role).toBe("assistant")
        if (result.info.role === "assistant") {
          expect(result.info.finish).toBe("stop")
          expect(result.parts.some((part) => part.type === "text" && part.text === "done")).toBe(true)
        }

        const hits = yield* llm.hits
        expect(hits).toHaveLength(5)

        // Step 1: the tools array is eager only — no deferred entry in any
        // form (truncated descriptions never render on a wrapper session).
        const toolNames = (bodyOf(hits[0]!).tools ?? []).map((entry) => entry.function?.name)
        for (const eager of ["bash", "read", "load_tool", "deferred_tool"]) {
          expect(toolNames).toContain(eager)
        }
        expect(fnOf(hits[0]!, "glob")).toBeUndefined()
        expect(fnOf(hits[0]!, "grep")).toBeUndefined()
        for (const entry of bodyOf(hits[0]!).tools ?? []) {
          expect(entry.function?.description?.endsWith("...")).toBe(false)
        }
        // The meta-tool's closed, deployment-uniform schema (R12-009).
        const meta = fnOf(hits[0]!, "deferred_tool")!
        expect(meta.description).toBe(DEFERRED_TOOL_DESCRIPTION)
        expect(Object.keys(((meta.parameters ?? {}) as { properties?: Record<string, unknown> }).properties ?? {})).toEqual(["name", "args"])
        expect(((meta.parameters ?? {}) as { required?: string[] }).required).toEqual(["name", "args"])
        expect((meta.parameters as { additionalProperties?: boolean }).additionalProperties).toBe(false)
        expect(meta.parameters).toEqual(DEFERRED_TOOL_SCHEMA)

        // Step 1: the catalog block rides the system string — name +
        // truncated description bullets.
        const system = systemBytes(hits[0]!)
        expect(system).toContain("<deferred_tools>")
        expect(system).toContain("</deferred_tools>")
        expect(system).toContain("- glob: ")
        expect(system).toContain(GLOB_DESCRIPTION.slice(0, 40))
        expect(system).toContain("- grep: ")
        expect(system).toContain(GREP_DESCRIPTION.slice(0, 40))

        // Step 5: turn 2's system bytes + tool-entry shapes equal turn 1's
        // (no append event, R12-008).
        for (const later of [1, 2, 3, 4]) {
          expect(systemBytes(hits[later]!)).toBe(systemBytes(hits[0]!))
          expect(toolBytes(hits[later]!)).toBe(toolBytes(hits[0]!))
        }

        const parts = yield* toolPartsOf(session.id)

        // Step 2: the wrapper call executed the inner tool; the unwrap
        // metadata rides the part; history stays verbatim.
        const globPart = parts.find(
          (part): part is CompletedToolPart => part.tool === "deferred_tool" && part.state.status === "completed",
        )
        expect(globPart).toBeDefined()
        expect(globPart!.state.input).toEqual({ name: "glob", args: '{"pattern":"*.json"}' })
        expect(globPart!.state.output).toContain("opencode.json")
        expect(globPart!.state.metadata?.deferred_tool).toEqual({ tool: "glob" })
        const wrapperCall = toolCallOf(hits[1]!, "deferred_tool")
        expect(wrapperCall).toBeDefined()
        expect(JSON.parse(wrapperCall!.function!.arguments!)).toEqual({ name: "glob", args: '{"pattern":"*.json"}' })
        const toolResult = (bodyOf(hits[1]!).messages ?? []).find((message) => message.role === "tool")
        expect(JSON.stringify(toolResult)).toContain("opencode.json")

        // Step 3: schema-in-error — the error carries the inner full schema
        // and the load_tool marker.
        const errorPart = parts.find(
          (part): part is ErrorToolPart => part.tool === "deferred_tool" && part.state.status === "error",
        )
        expect(errorPart).toBeDefined()
        expect(errorPart!.state.error).toContain("The glob tool has not been loaded. Its full input schema is:")
        expect(errorPart!.state.error).toContain('"pattern"')
        expect(errorPart!.state.metadata?.load_tool).toEqual({ tools: ["glob"] })
        expect(errorPart!.state.metadata?.deferred_tool).toEqual({ tool: "glob" })

        // Step 3: the marker's short confirmation.
        const confirmation = parts.filter(
          (part): part is CompletedToolPart => part.tool === "load_tool" && part.state.status === "completed",
        )
        expect(confirmation).toHaveLength(2)
        expect(confirmation[0]!.state.output).toContain("glob: already loaded.")
        expect(confirmation[0]!.state.metadata?.load_tool).toBeUndefined()

        // Step 4: wrapper-mode load_tool serves description AND schema.
        expect(confirmation[1]!.state.input).toEqual({ tools: ["grep"] })
        expect(confirmation[1]!.state.output).toContain("### grep")
        expect(confirmation[1]!.state.output).toContain(GREP_DESCRIPTION)
        expect(confirmation[1]!.state.output).toContain("Input schema:")
        expect(confirmation[1]!.state.output).toContain('"pattern"')
        expect(confirmation[1]!.state.metadata?.load_tool).toEqual({ tools: ["grep"] })
      }),
    60000,
  )

  killSwitch.instance(
    "kill-switch binding session reverts to schema-eager listing without the wrapper",
    () =>
      Effect.gen(function* () {
        const llm = yield* useServerConfig(providerCfg)
        const { session, prompt } = yield* sessionWithPrompt("Wrapper kill-switch", "find text files")

        yield* llm.tool("load_tool", { tools: ["glob"] })
        yield* llm.text("done")

        const result = yield* prompt.loop({ sessionID: session.id })
        expect(result.info.role).toBe("assistant")
        expect(result.parts.some((part) => part.type === "text" && part.text === "done")).toBe(true)

        const hits = yield* llm.hits
        expect(hits).toHaveLength(2)

        // Schema-eager listing (R12-010 without R12-012): no meta tool, no
        // catalog block.
        const toolNames = (bodyOf(hits[0]!).tools ?? []).map((entry) => entry.function?.name)
        expect(toolNames).not.toContain("deferred_tool")
        expect(systemBytes(hits[0]!)).not.toContain("<deferred_tools>")
        const glob = fnOf(hits[0]!, "glob")!
        expect(glob.description?.endsWith("...")).toBe(true)
        expect(propertiesOf(glob)).toContain("pattern")

        const parts = yield* toolPartsOf(session.id)
        const load = parts.find(
          (part): part is CompletedToolPart => part.tool === "load_tool" && part.state.status === "completed",
        )
        expect(load).toBeDefined()
        expect(load!.state.output).toContain("### glob")
        expect(load!.state.output).toContain(GLOB_DESCRIPTION)
        expect(load!.state.output).toContain("The full input schema is already registered in the tool listing.")
        expect(load!.state.output).not.toContain("Input schema:")
      }),
    30000,
  )

  advisory.instance(
    "advisory session keeps round-1 bytes: placeholder entries, no meta tool, no catalog",
    () =>
      Effect.gen(function* () {
        const llm = yield* useServerConfig(providerCfg)
        const { session, prompt } = yield* sessionWithPrompt("Wrapper advisory", "find text files")

        yield* llm.tool("load_tool", { tools: ["glob"] })
        yield* llm.text("done")

        const result = yield* prompt.loop({ sessionID: session.id })
        expect(result.info.role).toBe("assistant")
        expect(result.parts.some((part) => part.type === "text" && part.text === "done")).toBe(true)

        const hits = yield* llm.hits
        expect(hits).toHaveLength(2)

        // Round-1 advisory regression guard: placeholder listing entries, no
        // wrapper artifacts anywhere.
        const toolNames = (bodyOf(hits[0]!).tools ?? []).map((entry) => entry.function?.name)
        expect(toolNames).not.toContain("deferred_tool")
        expect(systemBytes(hits[0]!)).not.toContain("<deferred_tools>")
        const glob = fnOf(hits[0]!, "glob")!
        expect(glob.parameters).toEqual(PLACEHOLDER)
        expect(glob.description?.endsWith("...")).toBe(true)
        expect(glob.description).not.toBe(GLOB_DESCRIPTION)
      }),
    30000,
  )
})
