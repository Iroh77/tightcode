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
import { ToolRegistry } from "@/tool/registry"
import { Truncate } from "@/tool/truncate"
import { TestInstance } from "../fixture/fixture"
import { testEffect } from "../lib/effect"
import { TestLLMServer } from "../lib/llm-server"
import GLOB_DESCRIPTION from "../../src/tool/glob.txt"
import { loadToolDescriptions } from "../../src/tool/load_tool"

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

const mcp = Layer.succeed(
  MCP.Service,
  MCP.Service.of({
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
    getPrompt: () => Effect.succeed(undefined),
    readResource: () => Effect.succeed(undefined),
    startAuth: () => Effect.die("unexpected MCP auth in lazy-tools integration tests"),
    authenticate: () => Effect.die("unexpected MCP auth in lazy-tools integration tests"),
    finishAuth: () => Effect.die("unexpected MCP auth in lazy-tools integration tests"),
    removeAuth: () => Effect.void,
    supportsOAuth: () => Effect.succeed(false),
    hasStoredTokens: () => Effect.succeed(false),
    getAuthStatus: () => Effect.succeed("not_authenticated" as const),
  }),
)

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

// The fixture provider cannot answer the adversarial binding probe with a
// violating tool call (the server answers probes off the books), so the
// advisory session injects the verdict through the probe seam instead. The
// binding session keeps the real cascade: probe request → no tool call →
// conservative binding. The wrapper axis is killed off so the round-1
// schema-eager binding bytes stay asserted end-to-end (R12-012: flag set ⇒
// round-1 binding); the wrapper path is covered by ticket 21's integration test.
const forcedVerdictNode = (probe: BindingVerdict.Probe) =>
  LayerNode.make({
    service: BindingVerdict.Service,
    layer: BindingVerdict.layerWith({ probe }),
    deps: [FSUtil.node, Global.node, ProviderSvc.node],
  })

const makeEnv = (probe?: BindingVerdict.Probe, flags?: Partial<RuntimeFlags.Info>) => {
  const root = LayerNode.group([promptRoot, testLLMServerNode])
  const replacements: LayerNode.Replacement[] = [
    [SessionSummary.node, summary],
    [LSP.node, lsp],
    [MCP.node, mcp],
    [RuntimeFlags.node, RuntimeFlags.layer({ experimentalEventSystem: true, ...flags })],
  ]
  if (probe) replacements.push([BindingVerdict.node, forcedVerdictNode(probe)])
  return LayerNode.compile(root, replacements)
}

const advisoryProbe = (): BindingVerdict.ProbeOutcome => ({
  verdict: "advisory",
  evidence: { kind: "call", args: '{"answer":4}' },
})
const advisory = testEffect(makeEnv(() => Effect.succeed(advisoryProbe())))
const binding = testEffect(makeEnv(undefined, { disableToolWrapper: true }))

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
  messages?: Array<{ role: string; content: unknown }>
  tools?: Array<{ function?: { name?: string; description?: string; parameters?: unknown } }>
}

const bodyOf = (hit: { body: Record<string, unknown> }) => hit.body as ChatBody
const systemBytes = (hit: { body: Record<string, unknown> }) =>
  JSON.stringify((bodyOf(hit).messages ?? []).filter((message) => message.role === "system"))
const toolBytes = (hit: { body: Record<string, unknown> }) => JSON.stringify(bodyOf(hit).tools ?? [])
const fnOf = (hit: { body: Record<string, unknown> }, name: string) =>
  (bodyOf(hit).tools ?? []).find((entry) => entry.function?.name === name)?.function

const PLACEHOLDER = { type: "object", properties: {} }

const deferredEntryAssertions = (fn: { description?: string; parameters?: unknown }, fullDescription: string) => {
  expect(fn.parameters).toEqual(PLACEHOLDER)
  expect(fn.description).not.toBe(fullDescription)
  expect(fn.description?.endsWith("...")).toBe(true)
  expect(fn.description!.length).toBeLessThanOrEqual(103)
  expect(fullDescription.startsWith(fn.description!.slice(0, -3))).toBe(true)
}

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

describe("lazy-loading session end-to-end", () => {
  advisory.instance(
    "advisory session keeps the prompt base byte-stable and delivers schemas via history",
    () =>
      Effect.gen(function* () {
        const llm = yield* useServerConfig(providerCfg)
        const { session, prompt } = yield* sessionWithPrompt("Lazy advisory", "find text files")

        yield* llm.tool("load_tool", { tools: ["glob"] })
        yield* llm.tool("grep", {})
        yield* llm.tool("load_tool", { tools: ["glob", "grep"] })
        yield* llm.text("done")

        const result = yield* prompt.loop({ sessionID: session.id })
        expect(result.info.role).toBe("assistant")
        if (result.info.role === "assistant") {
          expect(result.info.finish).toBe("stop")
          expect(result.parts.some((part) => part.type === "text" && part.text === "done")).toBe(true)
        }

        const hits = yield* llm.hits
        expect(hits).toHaveLength(4)

        // R12-008 at the payload: the system-plus-tools prefix is byte-stable
        // across every consecutive turn.
        for (const later of [1, 2, 3]) {
          expect(systemBytes(hits[later]!)).toBe(systemBytes(hits[0]!))
          expect(toolBytes(hits[later]!)).toBe(toolBytes(hits[0]!))
        }

        // R12-002 at the payload: the eager set is fully listed from the first
        // turn. The shell tool's exposed id is "bash" (ShellID.ToolID — kept
        // for plugin/permission compatibility, rename planned upstream), so
        // that is the name the requirement's "shell" resolves to in the
        // payload; fullness is asserted structurally because its description
        // is rendered per shell/platform.
        const shell = fnOf(hits[0]!, "bash")
        const read = fnOf(hits[0]!, "read")
        const loadTool = fnOf(hits[0]!, "load_tool")
        const propertiesOf = (fn: { parameters?: unknown }) =>
          Object.keys(((fn.parameters ?? {}) as { properties?: Record<string, unknown> }).properties ?? {})
        expect(shell?.description?.endsWith("...")).toBe(false)
        expect(shell!.description!.length).toBeGreaterThan(103)
        expect(propertiesOf(shell!)).toContain("command")
        expect(read?.description?.endsWith("...")).toBe(false)
        expect(propertiesOf(read!)).toContain("filePath")
        expect(loadTool?.description).toBe(loadToolDescriptions.advisory)
        expect(propertiesOf(loadTool!)).toContain("tools")

        // R12-003: the deferred entry is name + truncated description + placeholder.
        deferredEntryAssertions(fnOf(hits[0]!, "glob")!, GLOB_DESCRIPTION)
        // R12-005 at the payload: the entry stays the placeholder even after the
        // schema reached the model through load_tool output in history.
        deferredEntryAssertions(fnOf(hits[1]!, "glob")!, GLOB_DESCRIPTION)

        const parts = yield* toolPartsOf(session.id)
        const loads = parts.filter(
          (part): part is CompletedToolPart => part.tool === "load_tool" && part.state.status === "completed",
        )
        expect(loads).toHaveLength(2)
        const [firstLoad, confirmationLoad] = loads
        expect(firstLoad.state.input).toEqual({ tools: ["glob"] })
        expect(firstLoad.state.output).toContain(GLOB_DESCRIPTION)
        expect(firstLoad.state.output).toContain("Input schema:")
        expect(firstLoad.state.metadata?.load_tool?.tools).toEqual(["glob"])
        expect(confirmationLoad.state.input).toEqual({ tools: ["glob", "grep"] })
        expect(confirmationLoad.state.output).toContain("glob: already loaded.")
        expect(confirmationLoad.state.output).toContain("grep: already loaded.")
        expect(confirmationLoad.state.metadata?.load_tool).toBeUndefined()

        // R12-006: the unloaded failing call recovers at the protocol level —
        // the error output carries the full schema and the part is marked loaded.
        const grepPart = parts.find((part): part is ErrorToolPart => part.tool === "grep" && part.state.status === "error")
        expect(grepPart).toBeDefined()
        expect(grepPart!.state.error).toContain("The grep tool has not been loaded. Its full input schema is:")
        expect(grepPart!.state.error).toContain('"pattern"')
        expect(grepPart!.state.metadata?.load_tool?.tools).toEqual(["grep"])
      }),
    60000,
  )

  binding.instance(
    "binding session lists full schemas and load_tool serves the description only",
    () =>
      Effect.gen(function* () {
        const llm = yield* useServerConfig(providerCfg)
        const { session, prompt } = yield* sessionWithPrompt("Lazy binding", "find text files")

        yield* llm.tool("load_tool", { tools: ["glob"] })
        yield* llm.text("done")

        const result = yield* prompt.loop({ sessionID: session.id })
        expect(result.info.role).toBe("assistant")
        expect(result.parts.some((part) => part.type === "text" && part.text === "done")).toBe(true)

        const hits = yield* llm.hits
        expect(hits).toHaveLength(2)
        expect(systemBytes(hits[1]!)).toBe(systemBytes(hits[0]!))
        expect(toolBytes(hits[1]!)).toBe(toolBytes(hits[0]!))

        // R12-010: no schema deferral on a binding session — the deferred entry
        // replaces the placeholder with the full input schema, only the
        // description stays deferred.
        const glob = fnOf(hits[0]!, "glob")!
        expect(glob.description?.endsWith("...")).toBe(true)
        expect(glob.description).not.toBe(GLOB_DESCRIPTION)
        const parameters = glob.parameters as { properties?: Record<string, unknown> }
        expect(Object.keys(parameters.properties ?? {})).toContain("pattern")

        const parts = yield* toolPartsOf(session.id)
        const load = parts.find(
          (part): part is CompletedToolPart => part.tool === "load_tool" && part.state.status === "completed",
        )
        expect(load).toBeDefined()
        expect(load!.state.output).toContain("### glob")
        expect(load!.state.output).toContain(GLOB_DESCRIPTION)
        expect(load!.state.output).toContain("The full input schema is already registered in the tool listing.")
        expect(load!.state.output).not.toContain("Input schema:")
        expect(load!.state.metadata?.load_tool?.tools).toEqual(["glob"])
      }),
    30000,
  )
})
