import { Database } from "@opencode-ai/core/database/database"
import { LayerNode } from "@opencode-ai/core/effect/layer-node"
import { CrossSpawnSpawner } from "@opencode-ai/core/cross-spawn-spawner"
import { FSUtil } from "@opencode-ai/core/fs-util"
import { Ripgrep } from "@opencode-ai/core/ripgrep"
import { SessionProjector } from "@opencode-ai/core/session/projector"
import { expect, describe } from "bun:test"
import { Effect, Layer } from "effect"
import fs from "fs/promises"
import path from "path"
import { Agent } from "../../src/agent/agent"
import { BackgroundJob } from "@/background/job"
import { Command } from "../../src/command"
import { Config } from "@/config/config"
import { Env } from "../../src/env"
import { EventV2Bridge } from "@/event-v2-bridge"
import { RuntimeFlags } from "@/effect/runtime-flags"
import { Format } from "../../src/format"
import { Git } from "../../src/git"
import { Image } from "../../src/image/image"
import { LSP } from "@/lsp/lsp"
import { MCP } from "../../src/mcp"
import { MessageV2 } from "../../src/session/message-v2"
import { Permission } from "../../src/permission"
import { Plugin } from "../../src/plugin"
import { Provider } from "@/provider/provider"
import { ProviderV2 } from "@opencode-ai/core/provider"
import { ModelV2 } from "@opencode-ai/core/model"
import { Question } from "../../src/question"
import { Instruction } from "../../src/session/instruction"
import { LLM } from "../../src/session/llm"
import { Session } from "@/session/session"
import { SessionCompaction } from "../../src/session/compaction"
import { SessionID } from "../../src/session/schema"
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
import { provideInstance, testInstanceStoreLayer, tmpdirScoped } from "../fixture/fixture"
import { testEffect } from "../lib/effect"
import { TestLLMServer } from "../lib/llm-server"
import BASE_TEMPLATE from "../../src/session/prompt/default.txt"

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

const fixtureServer = { name: "fixture-mcp", instructions: "Fixture MCP server instructions.", tools: [] }

const fixtureSkill = {
  name: "fixture-skill",
  description: "Fixture skill description for the prompt-override integration test.",
  location: "/fixture/skill/SKILL.md",
  content: "Fixture skill body for the prompt-override integration test.",
}

const mcp = Layer.mock(MCP.Service, {
  instructions: () => Effect.succeed([fixtureServer]),
  clients: () => Effect.succeed({}),
  tools: () => Effect.succeed({}),
})

const skills = Layer.mock(Skill.Service, {
  dirs: () => Effect.succeed([]),
  all: () => Effect.succeed([fixtureSkill]),
  available: () => Effect.succeed([fixtureSkill]),
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
  Agent.node,
  Command.node,
  Permission.node,
  Plugin.node,
  Config.node,
  Provider.node,
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

const env = testEffect(
  LayerNode.compile(LayerNode.group([promptRoot, testLLMServerNode]), [
    [SessionSummary.node, summary],
    [LSP.node, lsp],
    [MCP.node, mcp],
    [Skill.node, skills],
    [RuntimeFlags.node, RuntimeFlags.layer({ experimentalEventSystem: true })],
  ]),
)

const baseProvider = {
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
    ...baseProvider,
    provider: {
      ...baseProvider.provider,
      test: {
        ...baseProvider.provider.test,
        options: {
          ...baseProvider.provider.test.options,
          baseURL: url,
        },
      },
    },
  }
}

type Fixture = { worktree: string; directory: string; llm: TestLLMServer["Service"] }

// Git-rooted worktree with the session cwd at sub/deep. The provider config
// lives at the worktree root (discovery walks up from the cwd); file-ref
// overrides must resolve against the instance working directory, not the root.
const runFixture = <A, E, R>(config: Record<string, unknown>, body: (fixture: Fixture) => Effect.Effect<A, E, R>) =>
  Effect.gen(function* () {
    const fsys = yield* FSUtil.Service
    const llm = yield* TestLLMServer
    const worktree = yield* tmpdirScoped({ git: true })
    const directory = path.join(worktree, "sub", "deep")
    yield* Effect.promise(() => fs.mkdir(directory, { recursive: true }))
    yield* fsys.writeWithDirs(
      path.join(worktree, "opencode.json"),
      JSON.stringify({ $schema: "https://opencode.ai/config.json", ...providerCfg(llm.url), ...config }),
    )
    return yield* body({ worktree, directory, llm }).pipe(provideInstance(directory))
  }).pipe(Effect.provide(testInstanceStoreLayer))

const sessionWithFixture = (title: string) =>
  Effect.gen(function* () {
    const sessions = yield* Session.Service
    return yield* sessions.create({
      title,
      permission: [{ permission: "*", pattern: "*", action: "allow" }],
    })
  })

const turn = (sessionID: SessionID, text: string, agent = "build") =>
  Effect.gen(function* () {
    const prompt = yield* SessionPrompt.Service
    return yield* prompt.prompt({ sessionID, agent, parts: [{ type: "text", text }] })
  })

type ChatBody = { messages?: Array<{ role: string; content: unknown }> }

const systemEntries = (hit: { body: Record<string, unknown> }) =>
  ((hit.body as ChatBody).messages ?? [])
    .filter((message) => message.role === "system")
    .map((message) => message.content as string)

// Round-1 invariants downstream of system[0]: the slimmed skills markdown
// still renders after the base text, whatever the cascade put at position 0.
const skillsBlock = [
  "Skills provide specialized instructions and workflows for specific tasks.",
  "Use the skill tool to load a skill when a task matches its description.",
  "## Available Skills",
  `- **fixture-skill**: ${fixtureSkill.description}`,
].join("\n")

describe("override cascade end-to-end", () => {
  env.live(
    "inline override replaces the base; round-1 invariants hold downstream",
    () =>
      runFixture({ system_prompt: { default: "FORK OVERRIDE PROMPT" } }, ({ llm }) =>
        Effect.gen(function* () {
          const session = yield* sessionWithFixture("Inline override")
          yield* turn(session.id, "one turn")

          const hits = yield* llm.hits
          expect(hits).toHaveLength(1)
          const entries = systemEntries(hits[0]!)
          expect(entries).toHaveLength(1)
          const system0 = entries[0]!

          expect(system0.startsWith("FORK OVERRIDE PROMPT")).toBe(true)
          expect(system0).not.toContain(BASE_TEMPLATE)
          expect(system0).toContain("Today's date: ")
          expect(system0).toContain(skillsBlock)
          expect(system0.indexOf("FORK OVERRIDE PROMPT")).toBeLessThan(system0.indexOf(skillsBlock))
        }),
      ),
    60000,
  )

  env.live(
    "file-ref override reads the file relative to the instance working directory",
    () =>
      runFixture({ system_prompt: { default: { file: "./prompts/slim.md" } } }, ({ directory, llm }) =>
        Effect.gen(function* () {
          const fsys = yield* FSUtil.Service
          yield* fsys.writeWithDirs(path.join(directory, "prompts", "slim.md"), "FILE OVERRIDE PROMPT")

          const session = yield* sessionWithFixture("File-ref override")
          yield* turn(session.id, "one turn")

          const hits = yield* llm.hits
          expect(hits).toHaveLength(1)
          const entries = systemEntries(hits[0]!)
          expect(entries).toHaveLength(1)
          const system0 = entries[0]!

          expect(system0.startsWith("FILE OVERRIDE PROMPT")).toBe(true)
          expect(system0).not.toContain(BASE_TEMPLATE)
          expect(system0).toContain(skillsBlock)
        }),
      ),
    60000,
  )

  env.live(
    "file-ref to a missing path fails the turn with the resolved path named",
    () =>
      runFixture({ system_prompt: { default: { file: "./prompts/missing.md" } } }, ({ directory, llm }) =>
        Effect.gen(function* () {
          const session = yield* sessionWithFixture("Missing file-ref override")
          const result = yield* turn(session.id, "one turn")

          expect(result.info.role).toBe("assistant")
          if (result.info.role !== "assistant") return
          const error = JSON.stringify(result.info.error)
          expect(error).toContain(path.join(directory, "prompts", "missing.md"))
          expect(yield* llm.hits).toHaveLength(0)
        }),
      ),
    60000,
  )

  env.live(
    "config agent prompt wins over the override (cascade top)",
    () =>
      runFixture(
        {
          system_prompt: { default: "FORK OVERRIDE PROMPT" },
          agent: { "cascade-agent": { prompt: "AGENT PROMPT", mode: "primary" } },
        },
        ({ llm }) =>
          Effect.gen(function* () {
            const session = yield* sessionWithFixture("Cascade top")
            yield* turn(session.id, "one turn", "cascade-agent")

            const hits = yield* llm.hits
            expect(hits).toHaveLength(1)
            const entries = systemEntries(hits[0]!)
            expect(entries).toHaveLength(1)
            const system0 = entries[0]!

            expect(system0.startsWith("AGENT PROMPT")).toBe(true)
            expect(system0).not.toContain("FORK OVERRIDE PROMPT")
          }),
      ),
    60000,
  )

  env.live(
    "record without the selected template's key: the built-in base renders",
    () =>
      runFixture({ system_prompt: { anthropic: "WRONG TEMPLATE" } }, ({ llm }) =>
        Effect.gen(function* () {
          const session = yield* sessionWithFixture("Inert key")
          yield* turn(session.id, "one turn")

          const hits = yield* llm.hits
          expect(hits).toHaveLength(1)
          const entries = systemEntries(hits[0]!)
          expect(entries).toHaveLength(1)
          expect(entries[0]!.startsWith(BASE_TEMPLATE)).toBe(true)
        }),
      ),
    60000,
  )

  env.live("R11-005 re-assertion: base() selection and provider output are behavior-preserving", () =>
    Effect.gen(function* () {
      const fixtureModel = {
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
        status: "active" as const,
        options: {},
        headers: {},
        release_date: "",
      }

      const selected = SystemPrompt.base(fixtureModel)
      expect(selected.template).toBe("default")
      expect(selected.name).toBeUndefined()
      expect(selected.raw).toBe(BASE_TEMPLATE)
      expect(SystemPrompt.provider(fixtureModel)).toEqual([BASE_TEMPLATE])
    }),
  )
})
