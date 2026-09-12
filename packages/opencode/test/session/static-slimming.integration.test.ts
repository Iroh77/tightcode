import { Database } from "@opencode-ai/core/database/database"
import { LayerNode } from "@opencode-ai/core/effect/layer-node"
import { CrossSpawnSpawner } from "@opencode-ai/core/cross-spawn-spawner"
import { FSUtil } from "@opencode-ai/core/fs-util"
import { Ripgrep } from "@opencode-ai/core/ripgrep"
import { SessionProjector } from "@opencode-ai/core/session/projector"
import { expect, describe } from "bun:test"
import { Effect, Layer } from "effect"
import path from "path"
import { Agent as AgentSvc } from "../../src/agent/agent"
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
import { Provider as ProviderSvc } from "@/provider/provider"
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

const MCP_SERVER_INSTRUCTIONS = "Fixture MCP server instructions. ".repeat(10).slice(0, 300)
const fixtureServer = { name: "fixture-mcp", instructions: MCP_SERVER_INSTRUCTIONS, tools: [] }

const fixtureSkill = {
  name: "fixture-skill",
  description: "Fixture skill description for the static-slimming integration test.",
  location: "/fixture/skill/SKILL.md",
  content: "Fixture skill body for the static-slimming integration test.",
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

const makeEnv = (flags: Partial<RuntimeFlags.Info> = {}) =>
  LayerNode.compile(LayerNode.group([promptRoot, testLLMServerNode]), [
    [SessionSummary.node, summary],
    [LSP.node, lsp],
    [MCP.node, mcp],
    [Skill.node, skills],
    [RuntimeFlags.node, RuntimeFlags.layer({ experimentalEventSystem: true, ...flags })],
  ])

const slimmed = testEffect(makeEnv())
const upstream = testEffect(makeEnv({ disableStaticSlimming: true }))

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

const AGENTS_FILES = {
  "AGENTS.md": "# Root",
  "sub/AGENTS.md": "# Mid",
  "sub/deep/AGENTS.md": "# Deep",
}

type Fixture = { worktree: string; directory: string; llm: TestLLMServer["Service"] }

// Git-rooted worktree with the session cwd at sub/deep, so instruction
// discovery has an ancestor strictly between cwd and worktree. The provider
// config lives at the worktree root (discovery walks up from the cwd).
const runFixture = <A, E, R>(body: (fixture: Fixture) => Effect.Effect<A, E, R>) =>
  Effect.gen(function* () {
    const fsys = yield* FSUtil.Service
    const llm = yield* TestLLMServer
    const worktree = yield* tmpdirScoped({ git: true })
    const directory = path.join(worktree, "sub", "deep")
    yield* fsys.writeWithDirs(
      path.join(worktree, "opencode.json"),
      JSON.stringify({ $schema: "https://opencode.ai/config.json", ...providerCfg(llm.url) }),
    )
    yield* Effect.forEach(
      Object.entries(AGENTS_FILES),
      ([file, content]) => fsys.writeWithDirs(path.join(worktree, file), content),
      { discard: true },
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

const turn = (sessionID: SessionID, text: string) =>
  Effect.gen(function* () {
    const prompt = yield* SessionPrompt.Service
    return yield* prompt.prompt({ sessionID, agent: "build", parts: [{ type: "text", text }] })
  })

type ChatBody = { messages?: Array<{ role: string; content: unknown }> }

// The payload's rendered system blocks: upstream prepare joins the position-0
// base template, the SC-2 blocks and user.system into one system message when
// no plugin adds more entries.
const systemEntries = (hit: { body: Record<string, unknown> }) =>
  ((hit.body as ChatBody).messages ?? [])
    .filter((message) => message.role === "system")
    .map((message) => message.content as string)

const maskDate = (payload: string) => payload.replace(/Today's date: .*/, "Today's date: <masked>")

const envBlock = (worktree: string, directory: string) =>
  [
    "Here is some useful information about the environment you are running in:",
    "<env>",
    `  Working directory: ${directory}`,
    `  Workspace root folder: ${worktree}`,
    "  Is directory a git repo: yes",
    `  Platform: ${process.platform}`,
    "  Today's date: <masked>",
    "</env>",
  ].join("\n")

describe("slimmed static blocks end-to-end", () => {
  slimmed.live(
    "slimmed static blocks reach the payload and freeze across turns",
    () =>
      runFixture(({ worktree, directory, llm }) =>
        Effect.gen(function* () {
          expect(MCP_SERVER_INSTRUCTIONS.length).toBe(300)

          const session = yield* sessionWithFixture("Slimmed static blocks")
          yield* turn(session.id, "turn one")

          // Design edge 9: discovery inputs mutate after turn 1 — the frozen
          // instructions block must absorb the change instead of recomputing.
          const fsys = yield* FSUtil.Service
          yield* fsys.writeWithDirs(path.join(directory, "AGENTS.md"), "# Deep\nmutated after turn one")

          yield* turn(session.id, "turn two")

          const hits = yield* llm.hits
          expect(hits).toHaveLength(2)
          const first = systemEntries(hits[0]!)
          const second = systemEntries(hits[1]!)
          expect(second).toEqual(first)

          expect(first).toHaveLength(1)
          const system0 = first[0]!

          // R11-005: the upstream base template for the fixture model leads the payload.
          expect(system0.startsWith(`${BASE_TEMPLATE}\nYou are powered by the model named `)).toBe(true)

          // R11-004: the environment block keeps upstream's structure, date masked.
          expect(maskDate(system0)).toContain(envBlock(worktree, directory))

          // R11-001: root wins — cwd-level + worktree-root instructions, intermediate dropped.
          expect(system0).toContain(`Instructions from: ${path.join(directory, "AGENTS.md")}\n# Deep`)
          expect(system0).toContain(`Instructions from: ${path.join(worktree, "AGENTS.md")}\n# Root`)
          expect(system0).not.toContain(path.join(worktree, "sub", "AGENTS.md"))
          expect(system0).not.toContain("# Mid")

          // R11-002: the server's instruction text is cut inside the ≤250 budget.
          const open = '<mcp_instructions>\n  <server name="fixture-mcp">\n'
          const close = "\n  </server>"
          const start = system0.indexOf(open)
          expect(start).toBeGreaterThanOrEqual(0)
          const closeAt = system0.indexOf(close, start)
          expect(closeAt).toBeGreaterThan(start)
          const line = system0.slice(start + open.length, closeAt)
          const text = line.replace(/^ {4}/, "")
          expect(text.length).toBeLessThanOrEqual(250)
          expect(text.endsWith("...")).toBe(true)
          expect(MCP_SERVER_INSTRUCTIONS.startsWith(text.slice(0, -3))).toBe(true)
          expect(system0).not.toContain(MCP_SERVER_INSTRUCTIONS)

          // R11-003: non-verbose markdown skills block, no XML wrapper, no location.
          expect(system0).toContain(
            [
              "Skills provide specialized instructions and workflows for specific tasks.",
              "Use the skill tool to load a skill when a task matches its description.",
              "## Available Skills",
              `- **fixture-skill**: ${fixtureSkill.description}`,
            ].join("\n"),
          )
          expect(system0).not.toContain("<available_skills>")
          expect(system0).not.toContain("<location>")
        }),
      ),
    60000,
  )

  upstream.live(
    "flag set restores upstream shapes",
    () =>
      runFixture(({ worktree, directory, llm }) =>
        Effect.gen(function* () {
          const session = yield* sessionWithFixture("Upstream shapes (flag set)")
          yield* turn(session.id, "one turn")

          const hits = yield* llm.hits
          expect(hits).toHaveLength(1)
          const entries = systemEntries(hits[0]!)
          expect(entries).toHaveLength(1)
          const system0 = entries[0]!

          // R11-001 with the flag: upstream ancestor stacking incl. intermediate.
          expect(system0).toContain(`Instructions from: ${path.join(worktree, "AGENTS.md")}\n# Root`)
          expect(system0).toContain(`Instructions from: ${path.join(worktree, "sub", "AGENTS.md")}\n# Mid`)
          expect(system0).toContain(`Instructions from: ${path.join(directory, "AGENTS.md")}\n# Deep`)

          // R11-002 with the flag: the full instruction text reaches the section.
          expect(system0).toContain(`    ${MCP_SERVER_INSTRUCTIONS}`)

          // R11-003 with the flag: verbose XML skills block with location.
          expect(system0).toContain("<available_skills>")
          expect(system0).toContain("    <name>fixture-skill</name>")
          expect(system0).toContain(`    <location>${fixtureSkill.location}</location>`)
          expect(system0).not.toContain("## Available Skills")
        }),
      ),
    60000,
  )
})
