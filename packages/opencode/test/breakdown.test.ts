import { describe, expect, spyOn, test } from "bun:test"
import type { CaptureFile, CaptureMeta } from "../src/session/llm/prompt-capture"
import { forModel, parseSkillsListing, parseSystemBlocks } from "../script/estimator"
import { breakdown } from "../script/breakdown"
import type { CaptureRecord, RunManifest, StepFinishRecord } from "../script/measure-usage"

// Breakdown core projection (R10-006, ticket 24): fixture captures over the
// o200k reference estimator (no tokenizer assets needed), covering attribution
// map present/absent, wrapper catalog bullets, pairing and rollups.

const estimator = forModel("claude-breakdown-test")

const meta = (over: Partial<CaptureMeta>): CaptureMeta => ({
  version: 1,
  sessionID: "ses_a",
  providerID: "test",
  modelID: "claude-breakdown-test",
  modelApi: "@ai-sdk/test",
  agent: "build",
  small: false,
  requestID: "msg_1",
  createdAt: "2026-09-15T00:00:00.000Z",
  optimized: { lazyTools: true, staticSlimming: true },
  ...over,
})

const ENV_BLOCK = [
  "You are powered by the model named claude-breakdown-test. The exact model ID is test/claude-breakdown-test",
  "Here is some useful information about the environment you are running in:",
  "<env>",
  "  Working directory: /tmp/fixture-repo",
  "</env>",
].join("\n")

const MCP_GROUP = [
  "<mcp_instructions>",
  '  <server name="github">',
  "    GitHub instructions.",
  "  </server>",
  '  <server name="linear">',
  "    Linear instructions.",
  "  </server>",
  "</mcp_instructions>",
].join("\n")

const SKILLS_BLOCK = [
  "Skills provide specialized instructions and workflows for specific tasks.",
  "Use the skill tool to load a skill when a task matches its description.",
  "## Available Skills",
  "- **tdd**: Test-driven development guidance",
].join("\n")

const catalogBlock = (server: string | undefined, bullets: string[]) =>
  [`<deferred_tools${server ? ` server="${server}"` : ""}>`, ...bullets, "</deferred_tools>"].join("\n")

const tool = (description: string): CaptureFile["payload"]["tools"][string] => ({
  description,
  inputSchema: { type: "object", properties: {} },
})

const toolServers = { mcp_github_create_issue: "github", mcp_linear_create_issue: "linear" }

const captureA0: CaptureFile = {
  meta: meta({ requestID: "msg_a0", toolServers }),
  payload: {
    system: ["Base template text.", ENV_BLOCK, MCP_GROUP, SKILLS_BLOCK].join("\n").split("\n"),
    tools: {
      bash: tool("Run shell commands"),
      load_tool: tool("Load a deferred tool"),
      deferred_tool: tool("Execute a deferred tool"),
      mcp_github_create_issue: tool("Create an issue"),
      mcp_linear_create_issue: tool("Create an issue"),
    },
    messages: [
      { role: "system", content: "Leading system message." },
      { role: "user", content: "hello" },
    ],
  },
}

const captureA1: CaptureFile = {
  meta: meta({ requestID: "msg_a1", toolServers: {} }),
  payload: {
    system: [
      "Base template text.",
      ENV_BLOCK,
      SKILLS_BLOCK,
      catalogBlock("github", ["- mcp_github_create_issue: Create an issue", "- mcp_github_list_prs"]),
      catalogBlock(undefined, ["- shellrun: Run a shell command"]),
    ].join("\n").split("\n"),
    tools: {
      bash: tool("Run shell commands"),
      deferred_tool: tool("Execute a deferred tool"),
      load_tool: tool("Load a deferred tool"),
    },
    messages: [{ role: "user", content: "again" }],
  },
}

const captureB0: CaptureFile = {
  // Old capture: no toolServers field — scoped prefix fallback.
  meta: meta({ sessionID: "ses_b", requestID: "msg_b0" }),
  payload: {
    system: ["Base template text.", MCP_GROUP].join("\n").split("\n"),
    tools: {
      bash: tool("Run shell commands"),
      github_create_issue: tool("Create an issue"),
    },
    messages: [{ role: "user", content: "legacy" }],
  },
}

const usageA0: StepFinishRecord = {
  sessionID: "ses_a",
  tokens: { input: 1000, output: 77, reasoning: 0, cacheRead: 0, cacheWrite: 0 },
  cost: 0.01,
}
const usageB0: StepFinishRecord = {
  sessionID: "ses_b",
  tokens: { input: 500, output: 20, reasoning: 0, cacheRead: 0, cacheWrite: 0 },
  cost: 0.005,
}

const manifest: RunManifest = {
  bin: "fork",
  modelID: "claude-breakdown-test",
  providerID: "test",
  promptsDigest: "abc",
  capture: true,
  proxyMode: false,
  dbPath: "data/opencode.db",
  capturesDir: "data/prompt-captures",
  startedAt: "2026-09-15T00:00:00.000Z",
  endedAt: "2026-09-15T00:01:00.000Z",
}

const record = (capture: CaptureFile): CaptureRecord => ({ file: `data/prompt-captures/${capture.meta.sessionID}/0000.json`, capture })

const fixture = { manifest, captures: [record(captureA0), record(captureA1), record(captureB0)], usage: [usageA0, usageB0] }

describe("breakdown", () => {
  test("emits schema v1 with the run block and table-ordered sessions", () => {
    const result = breakdown(fixture)
    expect(result.schema).toBe(1)
    expect(result.run.modelID).toBe("claude-breakdown-test")
    expect(result.run.providerID).toBe("test")
    expect(result.run.modelApi).toBe("@ai-sdk/test")
    expect(result.sessions.map((session) => session.sessionID)).toEqual(["ses_a", "ses_b"])
    expect(result.sessions[0].turns.map((turn) => turn.requestID)).toEqual(["msg_a0", "msg_a1"])
  })

  test("turn 0: block keys in byte order, absent categories omitted, never zero-filled", () => {
    const turn = breakdown(fixture).sessions[0].turns[0]
    expect(turn.index).toBe(0)
    expect(turn.small).toBe(false)
    expect(turn.estimator).toEqual({ modelID: "claude-breakdown-test", adapter: "o200k" })
    expect(turn.system.blocks.map((block) => block.key)).toEqual([
      "base",
      "environment",
      "mcp:github",
      "mcp:linear",
      "skills",
    ])
    expect(turn.system.mcpInstructionsTags).toBeGreaterThan(0)
    expect(turn.skills.perSkill.map((skill) => skill.name)).toEqual(["tdd"])
  })

  test("turn 0: tool attribution rides meta.toolServers; inherent tools unattributed", () => {
    const turn = breakdown(fixture).sessions[0].turns[0]
    const byName = new Map(turn.tools.entries.map((entry) => [entry.name, entry]))
    expect(byName.get("mcp_github_create_issue")?.server).toBe("github")
    expect(byName.get("mcp_linear_create_issue")?.server).toBe("linear")
    expect(byName.get("bash")?.server).toBeNull()
    expect(turn.mcp.unattributed.map((entry) => entry.name)).toEqual(["bash", "load_tool", "deferred_tool"])
    const github = turn.mcp.servers.find((server) => server.name === "github")
    expect(github).toBeDefined()
    expect(github!.instructionsBlock).toBeGreaterThan(0)
    expect(github!.catalogBlock).toBeNull()
    expect(github!.tools).toEqual([{ name: "mcp_github_create_issue", tokens: byName.get("mcp_github_create_issue")!.tokens }])
    expect(turn.system.mcpInstructionsTags).toBe(turn.mcp.instructionsTags)
  })

  test("turn 1 (wrapper): catalog bullets become per-tool rows, each MCP tool once", () => {
    const turn = breakdown(fixture).sessions[0].turns[1]
    expect(turn.system.blocks.map((block) => block.key)).toEqual(["base", "environment", "skills", "catalog:github", "catalog"])
    const github = turn.mcp.servers.find((server) => server.name === "github")
    expect(github!.instructionsBlock).toBe(0)
    expect(github!.catalogBlock).toBeGreaterThan(0)
    expect(github!.tools.map((entry) => entry.name)).toEqual(["mcp_github_create_issue", "mcp_github_list_prs"])
    expect(github!.tools.every((entry) => entry.tokens > 0)).toBe(true)
    // shellrun lives in the plain catalog block — not an MCP tool.
    expect(turn.mcp.servers.every((server) => !server.tools.some((entry) => entry.name === "shellrun"))).toBe(true)
  })

  test("record entries take precedence over catalog bullets with the same name", () => {
    const withBoth: CaptureFile = {
      meta: meta({ requestID: "msg_x", toolServers: { mcp_github_create_issue: "github" } }),
      payload: {
        system: ["Base template text.", catalogBlock("github", ["- mcp_github_create_issue: Create an issue"])].join("\n").split("\n"),
        tools: { mcp_github_create_issue: tool("Create an issue") },
        messages: [],
      },
    }
    const turn = breakdown({ manifest, captures: [record(withBoth)], usage: [] }).sessions[0].turns[0]
    const github = turn.mcp.servers.find((server) => server.name === "github")
    expect(github!.tools).toHaveLength(1)
    expect(github!.tools[0].tokens).toBe(turn.tools.entries[0].tokens)
  })

  test("old captures: scoped prefix fallback against servers visible in the capture", () => {
    const turn = breakdown(fixture).sessions[1].turns[0]
    const byName = new Map(turn.tools.entries.map((entry) => [entry.name, entry]))
    expect(byName.get("github_create_issue")?.server).toBe("github")
    expect(byName.get("bash")?.server).toBeNull()
    expect(turn.mcp.servers.find((server) => server.name === "github")!.tools).toEqual([
      { name: "github_create_issue", tokens: byName.get("github_create_issue")!.tokens },
    ])
  })

  test("no-double-count: inputEstimate = Σ categories, mcp.total = Σ servers + tags, grouping free", () => {
    const turn = breakdown(fixture).sessions[0].turns[0]
    expect(turn.inputEstimate).toBe(turn.system.total + turn.tools.total + turn.history.total)
    expect(turn.mcp.total).toBe(turn.mcp.servers.reduce((sum, server) => sum + server.total, 0) + turn.mcp.instructionsTags)
    const joined = turn.system.blocks.reduce((sum, block) => sum + block.tokens, 0) + (turn.system.mcpInstructionsTags ?? 0)
    expect(turn.system.total).toBe(estimator.estimate(captureA0.payload.system.join("\n")))
    expect(joined).toBe(turn.system.total)
  })

  test("attribution is grouping-only: totals identical with map, prefix fallback, or nothing", () => {
    const totalsOf = (toolServers: CaptureMeta["toolServers"]) => {
      const capture: CaptureFile = {
        meta: meta({ requestID: "msg_g", ...(toolServers === undefined ? {} : { toolServers }) }),
        payload: {
          system: ["Base template text.", MCP_GROUP, catalogBlock("github", ["- mcp_github_create_issue: Create an issue"])].join("\n").split("\n"),
          tools: { bash: tool("Run shell commands"), mcp_github_create_issue: tool("Create an issue") },
          messages: [],
        },
      }
      const turn = breakdown({ manifest, captures: [record(capture)], usage: [] }).sessions[0].turns[0]
      return { inputEstimate: turn.inputEstimate, system: turn.system.total, tools: turn.tools.total, history: turn.history.total }
    }
    const withMap = totalsOf({ mcp_github_create_issue: "github" })
    const fallback = totalsOf(undefined)
    expect(withMap).toEqual(fallback)
    // Same totals even though attribution differs (map hits, prefix hits, and
    // the unattributed bash row never enters a category total).
    expect(withMap.tools).toBeGreaterThan(0)
  })

  test("tiling: system rows, skills sub-view, tool rows, history rows each sum to their total", () => {
    const turn = breakdown(fixture).sessions[0].turns[0]
    const blockSum = turn.system.blocks.reduce((sum, block) => sum + block.tokens, 0) + (turn.system.mcpInstructionsTags ?? 0)
    expect(blockSum).toBe(turn.system.total)
    expect(turn.skills.headers + turn.skills.perSkill.reduce((sum, skill) => sum + skill.tokens, 0)).toBe(turn.skills.total)
    expect(turn.tools.entries.reduce((sum, entry) => sum + entry.tokens, 0)).toBe(turn.tools.total)
    expect(turn.history.messages.reduce((sum, message) => sum + message.tokens, 0)).toBe(turn.history.total)
  })

  test("history starts after leading role:system messages (round-1 convention)", () => {
    const turn = breakdown(fixture).sessions[0].turns[0]
    expect(turn.history.messages).toEqual([expect.objectContaining({ index: 1, role: "user" })])
  })

  test("output pairing: matched turns carry usage, unmatched turns nulls", () => {
    const turns = breakdown(fixture).sessions[0].turns
    expect(turns[0].output).toBe(77)
    expect(turns[0].inputReported).toBe(1000)
    expect(turns[1].output).toBeNull()
    expect(turns[1].inputReported).toBeNull()
  })

  test("rollups: session totals sum turns, run totals sum sessions", () => {
    const result = breakdown(fixture)
    const session = result.sessions[0]
    expect(session.totals.system).toBe(session.turns.reduce((sum, turn) => sum + turn.system.total, 0))
    expect(session.totals.inputEstimate).toBe(session.turns.reduce((sum, turn) => sum + turn.inputEstimate, 0))
    expect(session.totals.output).toBe(77)
    expect(result.totals.output).toBe(97)
    expect(result.totals.system).toBe(result.sessions.reduce((sum, s) => sum + s.totals.system, 0))
  })

  test("unpaired usage rows are warned and keep an empty slot, never folded into a turn", () => {
    const log = spyOn(console, "error")
    try {
      const result = breakdown({
        manifest,
        captures: [record(captureA0)],
        usage: [usageA0, { sessionID: "ses_c", tokens: usageA0.tokens, cost: 0 }],
      })
      expect(result.sessions.map((session) => session.sessionID)).toEqual(["ses_a", "ses_c"])
      expect(result.sessions[1].turns).toEqual([])
      expect(log).toHaveBeenCalled()
    } finally {
      log.mockRestore()
    }
  })

  test("parses the skills sub-view through the shared parser", () => {
    const turn = breakdown(fixture).sessions[0].turns[0]
    const parsed = parseSystemBlocks(captureA0.payload.system)
    const skills = parsed.segments.find((segment) => segment.label === "skills")
    const listing = parseSkillsListing(skills!.text)
    expect(turn.skills.total).toBe(estimator.estimate(skills!.text))
    expect(turn.skills.headers).toBe(estimator.estimate(listing.headers))
  })
})
