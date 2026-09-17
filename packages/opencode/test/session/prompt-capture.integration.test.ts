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
import { TestLLMServer } from "../lib/llm-server"
import { logLines } from "effect/testing/TestConsole"
import { loadToolDescriptions } from "../../src/tool/load_tool"
import type { CaptureFile } from "../../src/session/llm/prompt-capture"

// One shared data dir per capture mode: captures are namespaced per session,
// and every session in every test has a fresh ID, so dirs never collide. The
// flag-off run gets its own dir so "no prompt-captures dir at all" is
// assertable independently of the capture-on runs.
const fixtureData = await fs.mkdtemp(path.join(os.tmpdir(), "opencode-prompt-capture-"))
const fixtureDataOff = await fs.mkdtemp(path.join(os.tmpdir(), "opencode-prompt-capture-off-"))
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
const makeEnv = (data: string, capture: boolean) =>
  LayerNode.compile(
    LayerNode.group([promptRoot, testLLMServerNode, Global.node]),
    [
      [SessionSummary.node, summary],
      [LSP.node, lsp],
      [MCP.node, mcp],
      [RuntimeFlags.node, RuntimeFlags.layer({ experimentalEventSystem: true, enablePromptCapture: capture })],
      [Global.node, Global.layerWith({ data })],
    ],
  )

const captureOn = testEffect(makeEnv(fixtureData, true))
const captureOff = testEffect(makeEnv(fixtureDataOff, false))
const captureBlocked = testEffect(makeEnv(blockedData, true))

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

type ChatBody = {
  messages?: Array<{ role: string; content: unknown }>
  tools?: Array<{ function?: { name?: string; description?: string; parameters?: unknown } }>
}

const bodyOf = (hit: { body: Record<string, unknown> }) => hit.body as ChatBody
const wireToolsOf = (hit: { body: Record<string, unknown> }) =>
  (bodyOf(hit).tools ?? []).flatMap((entry) =>
    entry.function?.name ? [{ name: entry.function.name, ...entry.function }] : [],
  )
const wireSystem = (hit: { body: Record<string, unknown> }) =>
  (bodyOf(hit).messages ?? [])
    .filter((message) => message.role === "system")
    .map((message) => message.content as string)
    .join("\n")

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

const userMessageIDs = (sessionID: SessionID) =>
  Effect.gen(function* () {
    const messages = yield* MessageV2.filterCompactedEffect(sessionID)
    return messages.filter((message) => message.info.role === "user").map((message) => message.info.id)
  })

const capturesDir = (sessionID: SessionID) => path.join(fixtureData, "prompt-captures", sessionID)
const readCapture = async (sessionID: SessionID, seq: number): Promise<CaptureFile> =>
  JSON.parse(await Bun.file(path.join(capturesDir(sessionID), `${String(seq).padStart(4, "0")}.json`)).text())

describe("prompt capture end-to-end", () => {
  captureOn.instance(
    "capture-enabled turns dump the exact prepared payload, credentials excluded",
    () =>
      Effect.gen(function* () {
        const llm = yield* useServerConfig(providerCfg)
        const { session, prompt } = yield* sessionWithPrompt("Capture first", "say hi")

        yield* llm.text("first done")
        yield* prompt.loop({ sessionID: session.id })
        yield* llm.text("second done")
        yield* prompt.prompt({
          sessionID: session.id,
          agent: "build",
          noReply: true,
          parts: [{ type: "text", text: "again" }],
        })
        yield* prompt.loop({ sessionID: session.id })

        const hits = yield* llm.hits
        expect(hits).toHaveLength(2)

        // 1. turn 1 wrote 0000.json, turn 2 wrote 0001.json (seq advances)
        const first = yield* Effect.promise(() => readCapture(session.id, 0))
        const second = yield* Effect.promise(() => readCapture(session.id, 1))
        expect(Object.keys(first).toSorted()).toEqual(["meta", "payload"])
        expect(Object.keys(first.payload).toSorted()).toEqual(["messages", "system", "tools"])
        expect(first.meta.version).toBe(1)
        expect(first.meta.sessionID).toBe(session.id)
        expect(first.meta.providerID).toBe("test")
        expect(first.meta.modelID).toBe("test-model")
        expect(first.meta.agent).toBe("build")
        expect(first.meta.small).toBe(false)
        expect(first.meta.optimized).toEqual({ lazyTools: true, staticSlimming: true })

        // 2. dump ↔ wire cross-check on the fields the AI SDK does not rewrite:
        // joined system text, tool name set, one entry's facts, message count + roles.
        expect(first.payload.system.join("\n")).toBe(wireSystem(hits[0]!))
        // "invalid" is in the prepared payload but the transport drops it from
        // activeTools (llm.ts) — the dump keeps it, the wire never shows it.
        const dumpToolNames = Object.keys(first.payload.tools)
          .filter((name) => name !== "invalid")
          .toSorted()
        expect(dumpToolNames).toEqual(wireToolsOf(hits[0]!).map((fn) => fn.name).toSorted())
        const loadTool = first.payload.tools.load_tool!
        expect(loadTool.description).toBe(loadToolDescriptions.wrapper)
        expect(loadTool.inputSchema).toEqual(wireToolsOf(hits[0]!).find((fn) => fn.name === "load_tool")?.parameters)
        expect(first.payload.messages.length).toBe((hits[0]!.body.messages as unknown[]).length)
        expect(first.payload.messages.map((message) => message.role as string)).toEqual(
          (hits[0]!.body.messages as Array<{ role: string }>).map((message) => message.role),
        )

        // 3. requestID per turn matches that turn's prompt admission
        const prompts = yield* userMessageIDs(session.id)
        expect(prompts).toHaveLength(2)
        expect([first.meta.requestID, second.meta.requestID].toSorted()).toEqual(prompts.toSorted())

        // 4. credential scan: the fake API key and any Authorization never enter
        const raw = yield* Effect.promise(() => Bun.file(path.join(capturesDir(session.id), "0000.json")).text())
        expect(raw).not.toContain("test-key-not-a-credential")
        const visit = (value: unknown) => {
          if (Array.isArray(value)) return value.forEach(visit)
          if (value && typeof value === "object") {
            for (const [key, entry] of Object.entries(value)) {
              expect(key).not.toBe("Authorization")
              visit(entry)
            }
          }
        }
        visit(first)
        visit(second)
      }),
    60000,
  )

  captureOff.instance(
    "flag off (default): no prompt-captures dir is created at all",
    () =>
      Effect.gen(function* () {
        const llm = yield* useServerConfig(providerCfg)
        const { session, prompt } = yield* sessionWithPrompt("Capture off", "say hi")
        yield* llm.text("done")
        yield* prompt.loop({ sessionID: session.id })

        const hits = yield* llm.hits
        expect(hits).toHaveLength(1)
        const exists = yield* Effect.promise(() =>
          fs
            .access(path.join(fixtureDataOff, "prompt-captures"))
            .then(() => true)
            .catch(() => false),
        )
        expect(exists).toBe(false)
      }),
    60000,
  )

  captureBlocked.instance(
    "unwritable data dir: the turn completes and the dump failure is swallowed",
    () =>
      Effect.gen(function* () {
        const llm = yield* useServerConfig(providerCfg)
        const { session, prompt } = yield* sessionWithPrompt("Capture blocked", "say hi")
        yield* llm.text("done")
        const result = yield* prompt.loop({ sessionID: session.id })

        expect(result.info.role).toBe("assistant")
        if (result.info.role === "assistant") expect(result.info.finish).toBe("stop")
        expect(result.parts.some((part) => part.type === "text" && part.text === "done")).toBe(true)
        // the failure was logged (R00-010 at this seam), not silently swallowed
        const logs = yield* logLines
        expect(
          logs.some((line) => typeof line === "string" && line.includes("prompt capture dump failed")),
        ).toBe(true)
        // nothing was written: the blocked data path cannot hold captures
        const entries = yield* Effect.promise(() =>
          fs
            .readdir(path.join(blockedData, "prompt-captures"))
            .catch((error: NodeJS.ErrnoException) => (error.code === "ENOTDIR" ? [] : Promise.reject(error))),
        )
        expect(entries).toEqual([])
      }),
    60000,
  )
})
