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
import fs from "fs/promises"
import fsSync from "node:fs"
import os from "os"
import path from "path"
import { Agent as AgentSvc } from "../../src/agent/agent"
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
import { Session } from "@/session/session"
import { SessionCompaction } from "../../src/session/compaction"
import { SessionProcessor } from "../../src/session/processor"
import { SessionPrompt } from "../../src/session/prompt"
import { SessionRevert } from "../../src/session/revert"
import { SessionRunState } from "../../src/session/run-state"
import { SessionStatus } from "../../src/session/status"
import { SessionSummary } from "../../src/session/summary"
import { SessionID } from "../../src/session/schema"
import { Skill } from "../../src/skill"
import { Snapshot } from "../../src/snapshot"
import { SystemPrompt } from "../../src/session/system"
import { Todo } from "../../src/session/todo"
import { ToolRegistry } from "@/tool/registry"
import { Truncate } from "@/tool/truncate"
import { TestInstance } from "../fixture/fixture"
import { testEffect } from "../lib/effect"
import { reply, TestLLMServer } from "../lib/llm-server"
import { logLines } from "effect/testing/TestConsole"
import { loadToolDescriptions } from "../../src/tool/load_tool"
import type { InboundToolCallLine } from "../../src/session/llm/inbound-tool-log"

// One shared data dir per log mode: the flag-on runs share one append-only
// JSONL file (line-level assertions are per sessionID), the flag-off run gets
// its own dir so "no file at all" is assertable independently.
const fixtureData = await fs.mkdtemp(path.join(os.tmpdir(), "opencode-inbound-log-it-"))
const fixtureDataOff = await fs.mkdtemp(path.join(os.tmpdir(), "opencode-inbound-log-it-off-"))
// Write-failure target: a regular file where a directory is required — mkdir
// fails ENOTDIR for every user, including root (chmod-based dirs don't).
const blockedData = path.join(fixtureData, "not-a-directory")
await Bun.write(blockedData, "file")
process.on("exit", () => {
  for (const dir of [fixtureData, fixtureDataOff]) fsSync.rmSync(dir, { recursive: true, force: true })
})

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
    startAuth: () => Effect.die("unexpected MCP auth in prompt-capture integration tests"),
    authenticate: () => Effect.die("unexpected MCP auth in prompt-capture integration tests"),
    finishAuth: () => Effect.die("unexpected MCP auth in prompt-capture integration tests"),
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

// Global.data is overridden to the fixture dir so captures land somewhere the
// test can inspect; only `data` changes (config/state stay on the real dirs).
const makeEnv = (data: string, enabled: boolean) =>
  LayerNode.compile(
    LayerNode.group([promptRoot, testLLMServerNode, Global.node]),
    [
      [SessionSummary.node, summary],
      [LSP.node, lsp],
      [MCP.node, mcp],
      [RuntimeFlags.node, RuntimeFlags.layer({ experimentalEventSystem: true, enableInboundToolLog: enabled })],
      [Global.node, Global.layerWith({ data })],
    ],
  )


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
        apiKey: "test-key-not-a-credential",
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
  const fsUtil = yield* FSUtil.Service
  const llm = yield* TestLLMServer
  yield* fsUtil.writeWithDirs(
    `${dir}/opencode.json`,
    JSON.stringify({ $schema: "https://opencode.ai/config.json", ...config(llm.url) }),
  )
  return llm
})

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


const inboundOn = testEffect(makeEnv(fixtureData, true))
const inboundOff = testEffect(makeEnv(fixtureDataOff, false))

const logFile = path.join(fixtureData, "inbound-tool-calls.jsonl")

const readLines = async (): Promise<InboundToolCallLine[]> => {
  const raw = await Bun.file(logFile).text()
  return raw
    .split("\n")
    .filter((line) => line !== "")
    .map((line) => JSON.parse(line))
}

const toolPartsOf = (sessionID: SessionID) =>
  Effect.gen(function* () {
    const messages = yield* MessageV2.filterCompactedEffect(sessionID)
    return messages
      .flatMap((message) => message.parts)
      .filter((part): part is SessionV1.ToolPart => part.type === "tool")
  })

describe("inbound tool-call log end-to-end", () => {
  inboundOn.instance(
    "flag on: raw reassembles byte-identical across deltas and matches the stored part",
    () =>
      Effect.gen(function* () {
        const llm = yield* useServerConfig(providerCfg)
        const { session, prompt } = yield* sessionWithPrompt("Inbound log on", "find typescript files")

        // The argument JSON is emitted in three deltas; the middle one carries
        // an escaped quote so byte-identity is distinguishable from any
        // re-serialization.
        const chunks = ['{"pattern":', '"*.ts\", \"path\"', ':"/tmp"}']
        yield* llm.push(reply().toolChunks("glob", chunks))
        yield* prompt.loop({ sessionID: session.id })

        const lines = yield* Effect.promise(readLines)
        const mine = lines.filter((line) => line.sessionID === session.id)
        expect(mine).toHaveLength(1)
        const line = mine[0]!
        // byte-identical verbatim argument string (R10-010)
        expect(line.raw).toBe(chunks.join(""))
        // input equals the parsed receipt the tool-call event carried
        expect(line.input).toEqual(JSON.parse(chunks.join("")))
        expect(line.tool).toBe("glob")
        expect(line.providerID).toBe("test")
        expect(line.modelID).toBe("test-model")
        expect(typeof line.time).toBe("number")

        // part fields match: the stored part's tool/callID/session/message
        const parts = yield* toolPartsOf(session.id)
        const glob = parts.find((part) => part.tool === "glob")
        expect(glob).toBeDefined()
        if (!glob) throw new Error("glob part missing")
        expect(line.callID).toBe(glob.callID)
        expect(line.sessionID).toBe(glob.sessionID)
        expect(line.messageID).toBe(glob.messageID)
      }),
    60000,
  )

  inboundOff.instance(
    "flag off (default): no log file, the turn completes normally",
    () =>
      Effect.gen(function* () {
        const llm = yield* useServerConfig(providerCfg)
        const { session, prompt } = yield* sessionWithPrompt("Inbound log off", "say hi")
        yield* llm.text("done")
        const result = yield* prompt.loop({ sessionID: session.id })
        expect(result.parts.some((part) => part.type === "text" && part.text === "done")).toBe(true)
        const exists = yield* Effect.promise(() =>
          fs
            .access(path.join(fixtureDataOff, "inbound-tool-calls.jsonl"))
            .then(() => true)
            .catch(() => false),
        )
        expect(exists).toBe(false)
      }),
    60000,
  )

  inboundOn.instance(
    "malformed arguments: the line lands verbatim and the session error part still forms",
    () =>
      Effect.gen(function* () {
        const llm = yield* useServerConfig(providerCfg)
        const { session, prompt } = yield* sessionWithPrompt("Inbound log malformed", "break the json")

        const invalid = '{"pattern": oops}'
        yield* llm.push(reply().toolChunks("glob", [invalid]))
        const result = yield* prompt.loop({ sessionID: session.id })

        const lines = yield* Effect.promise(readLines)
        const mine = lines.filter((line) => line.sessionID === session.id)
        expect(mine).toHaveLength(1)
        expect(mine[0]!.raw).toBe(invalid)
        // the capture never interferes: the session still forms its part
        const parts = yield* toolPartsOf(session.id)
        const glob = parts.find((part) => part.tool === "glob")
        expect(glob).toBeDefined()
        void result
      }),
    60000,
  )
})
