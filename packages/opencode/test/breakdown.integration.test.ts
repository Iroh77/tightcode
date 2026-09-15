import { describe, expect, spyOn, test } from "bun:test"
import { Database } from "bun:sqlite"
import fs from "fs/promises"
import path from "path"
import { ToolListing } from "../src/session/tool-listing"
import type { CaptureFile, CaptureMeta } from "../src/session/llm/prompt-capture"
import { Skill } from "../src/skill"
import { forModel } from "../script/estimator"
import { breakdown, main, type Breakdown } from "../script/breakdown"
import { loadRun, type RunManifest, type StepFinishRecord } from "../script/measure-usage"
import { tmpdir } from "./fixture/fixture"

// Feature-10 round-2 integration test (R10-006/008/009, ticket 26): one
// end-to-end path run dir → loadRun → breakdown() → v1 JSON → CLI emit over a
// fixture run dir built in-test. Fixture systems use the real render producers
// (Skill.fmt, ToolListing.catalogBlocks) so drift fails here directly; the
// estimator rides the o200k reference (no tokenizer assets needed outside
// scenario 7's committed bound fixtures).
// Contract: ARCHITECTURE/detailed/context-observability.md round 2 §Integration test.

const meta = (over: Partial<CaptureMeta>): CaptureMeta => ({
  version: 1,
  sessionID: "ses_main",
  providerID: "test",
  modelID: "claude-integration-test",
  modelApi: "@ai-sdk/test",
  agent: "build",
  small: false,
  requestID: "msg_0",
  createdAt: "2026-09-15T00:00:00.000Z",
  optimized: { lazyTools: true, staticSlimming: true },
  ...over,
})

const BASE_BLOCK = ["You are a helpful coding agent.", "Follow the user's instructions."].join("\n")

const ENV_BLOCK = [
  "You are powered by the model named claude-integration-test. The exact model ID is test/claude-integration-test",
  "Here is some useful information about the environment you are running in:",
  "<env>",
  "  Working directory: /tmp/fixture-repo",
  "  Workspace root folder: /tmp/fixture-repo",
  "  Is directory a git repo: yes",
  "  Platform: linux",
  "  Today's date: Tue Sep 15 2026",
  "</env>",
].join("\n")

const INSTRUCTIONS_BLOCK = [
  "Instructions from: /tmp/fixture-repo/AGENTS.md",
  "Root instructions, line one.",
  "Root instructions, line two.",
  "Instructions from: /home/u/.config/opencode/AGENTS.md",
  "Global instructions.",
].join("\n")

const MCP_GROUP = [
  "<mcp_instructions>",
  '  <server name="github">',
  "    GitHub instructions.",
  "    More GitHub detail.",
  "  </server>",
  '  <server name="linear">',
  "    Linear instructions.",
  "  </server>",
  "</mcp_instructions>",
].join("\n")

const SKILLS_INTRO = [
  "Skills provide specialized instructions and workflows for specific tasks.",
  "Use the skill tool to load a skill when a task matches its description.",
]

const skillsBlock = (verbose: boolean) =>
  [
    ...SKILLS_INTRO,
    Skill.fmt(
      [
        { name: "tdd", description: "Test-driven development guidance", location: "/tmp/skills/tdd/SKILL.md", content: "" },
        { name: "grill-me", description: "A relentless interview", location: "/tmp/skills/grill-me/SKILL.md", content: "" },
      ],
      { verbose },
    ),
  ].join("\n")

const STRUCTURED_OUTPUT_BLOCK =
  "IMPORTANT: The user has requested structured output. You MUST use the StructuredOutput tool to provide your final response. Do NOT respond with plain text - you MUST call the StructuredOutput tool with your answer formatted according to the schema."

const catalogBlocks = ToolListing.catalogBlocks({
  seeds: [
    { name: "mcp_github_create_issue", kind: "deferred", fullDescription: "Create an issue", jsonSchema: { type: "object" }, server: "github", source: "mcp" },
    { name: "mcp_github_list_prs", kind: "deferred", fullDescription: "", jsonSchema: { type: "object" }, server: "github", source: "mcp" },
    { name: "shellrun", kind: "deferred", fullDescription: "Run a shell command", jsonSchema: { type: "object" }, source: "builtin" },
  ],
})

const tool = (description: string): CaptureFile["payload"]["tools"][string] => ({
  description,
  inputSchema: { type: "object", properties: {} },
})

const toolServers = { mcp_github_create_issue: "github", mcp_linear_create_issue: "linear" }

// Turn 0 — schema-eager binding turn: every anchor, eager MCP entries.
const captureMain0: CaptureFile = {
  meta: meta({ requestID: "msg_0", toolServers }),
  payload: {
    system: [BASE_BLOCK, ENV_BLOCK, INSTRUCTIONS_BLOCK, MCP_GROUP, skillsBlock(false), STRUCTURED_OUTPUT_BLOCK, "Closing guidance supplied by the user."].join("\n").split("\n"),
    tools: {
      bash: tool("Run shell commands"),
      read: tool("Read a file"),
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

// Turn 1 — wrapper session: verbose skills, deferred MCP tools reachable only
// via the catalog blocks, no MCP listing entries.
const captureMain1: CaptureFile = {
  meta: meta({ requestID: "msg_1", toolServers: {} }),
  payload: {
    system: [BASE_BLOCK, ENV_BLOCK, skillsBlock(true), ...catalogBlocks.map((block) => block.content), "Late user guidance."].join("\n").split("\n"),
    tools: {
      bash: tool("Run shell commands"),
      read: tool("Read a file"),
      load_tool: tool("Load a deferred tool"),
      deferred_tool: tool("Execute a deferred tool"),
    },
    messages: [{ role: "user", content: "again" }],
  },
}

const usageMain0: StepFinishRecord = {
  sessionID: "ses_main",
  tokens: { input: 1000, output: 77, reasoning: 0, cacheRead: 0, cacheWrite: 0 },
  cost: 0.01,
}

const manifest: RunManifest = {
  bin: "fork",
  modelID: "claude-integration-test",
  providerID: "test",
  promptsDigest: "integration",
  capture: true,
  proxyMode: false,
  dbPath: "data/opencode.db",
  capturesDir: "data/prompt-captures",
  startedAt: "2026-09-15T00:00:00.000Z",
  endedAt: "2026-09-15T00:01:00.000Z",
}

// Fixture run dir on disk: manifest + tiny SQLite DB with one step-finish row
// (turn 0 paired; turn 1 unmatched) + capture files. bun:sqlite in-test, the
// same shape loadRun reads.
const writeRun = async (dir: string) => {
  await Bun.write(path.join(dir, "manifest.json"), JSON.stringify(manifest))
  const dbPath = path.join(dir, manifest.dbPath)
  await fs.mkdir(path.dirname(dbPath), { recursive: true })
  const db = new Database(dbPath)
  db.exec("CREATE TABLE session (id TEXT PRIMARY KEY)")
  db.exec("CREATE TABLE part (session_id TEXT, data TEXT)")
  db.run("INSERT INTO session (id) VALUES (?)", ["ses_main"])
  db.run(
    "INSERT INTO part (session_id, data) VALUES (?, ?)",
    [
      "ses_main",
      JSON.stringify({
        type: "step-finish",
        tokens: { input: usageMain0.tokens.input, output: usageMain0.tokens.output, reasoning: 0, cache: { read: 0, write: 0 } },
        cost: usageMain0.cost,
      }),
    ],
  )
  db.close()
  for (const [seq, file] of [
    [0, captureMain0],
    [1, captureMain1],
  ] as const) {
    const capturesDir = path.join(dir, manifest.capturesDir ?? "", "ses_main")
    await fs.mkdir(capturesDir, { recursive: true })
    await Bun.write(path.join(capturesDir, `${String(seq).padStart(4, "0")}.json`), JSON.stringify(file))
  }
}

const captureStdout = async (run: () => Promise<number>) => {
  const chunks: string[] = []
  const write = spyOn(process.stdout, "write").mockImplementation((chunk) => {
    chunks.push(String(chunk))
    return true
  })
  try {
    return { code: await run(), stdout: chunks.join("") }
  } finally {
    write.mockRestore()
  }
}

describe("breakdown end-to-end over a fixture run dir (ticket 26)", () => {
  const setup = async () => {
    await using tmp = await tmpdir()
    await writeRun(tmp.path)
    const run = await loadRun(tmp.path)
    const result = breakdown({ manifest: run.manifest, captures: run.captures, usage: run.usage })
    return { run, result }
  }

  test("scenario 1 — breakdown() emits schema v1 with per-turn taxonomy and inputEstimate = Σ categories", async () => {
    const { result } = await setup()
    expect(result.schema).toBe(1)
    expect(result.run).toMatchObject({ modelID: "claude-integration-test", providerID: "test", modelApi: "@ai-sdk/test" })
    expect(result.sessions.map((session) => session.sessionID)).toEqual(["ses_main"])
    const [turn0, turn1] = result.sessions[0].turns
    expect(turn0.system.blocks.map((block) => block.key)).toEqual([
      "base",
      "environment",
      "instructions",
      "mcp:github",
      "mcp:linear",
      "skills",
      "structured_output",
      "user_system",
    ])
    expect(turn0.system.mcpInstructionsTags).toBeGreaterThan(0)
    expect(turn1.system.blocks.map((block) => block.key)).toEqual(["base", "environment", "skills", "catalog:github", "catalog", "user_system"])
    for (const turn of [turn0, turn1]) {
      expect(turn.inputEstimate).toBe(turn.system.total + turn.tools.total + turn.history.total)
      expect(turn.estimator).toEqual({ modelID: "claude-integration-test", adapter: "o200k" })
    }
    // The skills sub-view rides the block row in both Skill.fmt modes
    // (the real producer sorts skills alphabetically).
    for (const turn of [turn0, turn1])
      expect(turn.skills.perSkill.map((skill) => skill.name)).toEqual(["grill-me", "tdd"])
    // Session rollups sum the turns.
    expect(result.sessions[0].totals.inputEstimate).toBe(turn0.inputEstimate + turn1.inputEstimate)
  })

  test("scenario 2 — tiling: block rows, skills, tools, history each partition their total exactly", async () => {
    const { result, run } = await setup()
    const est = forModel("claude-integration-test")
    for (const [turn, capture] of [
      [result.sessions[0].turns[0], run.captures[0].capture],
      [result.sessions[0].turns[1], run.captures[1].capture],
    ] as const) {
      const blockSum = turn.system.blocks.reduce((sum, block) => sum + block.tokens, 0) + (turn.system.mcpInstructionsTags ?? 0)
      expect(blockSum).toBe(turn.system.total)
      expect(turn.system.total).toBe(est.estimate(capture.payload.system.join("\n")))
      expect(turn.skills.headers + turn.skills.perSkill.reduce((sum, skill) => sum + skill.tokens, 0)).toBe(turn.skills.total)
      expect(turn.tools.entries.reduce((sum, entry) => sum + entry.tokens, 0)).toBe(turn.tools.total)
      expect(turn.history.messages.reduce((sum, message) => sum + message.tokens, 0)).toBe(turn.history.total)
    }
  })

  test("scenario 3 — MCP grouping: mcp.total = Σ servers + tags, totals grouping-free, wrapper bullets become per-tool rows", async () => {
    const { result } = await setup()
    const [turn0, turn1] = result.sessions[0].turns
    for (const turn of [turn0, turn1]) {
      expect(turn.mcp.total).toBe(turn.mcp.servers.reduce((sum, server) => sum + server.total, 0) + turn.mcp.instructionsTags)
      // Grouping never enters category totals.
      expect(turn.inputEstimate).toBe(turn.system.total + turn.tools.total + turn.history.total)
    }
    expect(turn0.mcp.servers.map((server) => server.name)).toEqual(["github", "linear"])
    expect(turn0.mcp.servers[0].instructionsBlock).toBeGreaterThan(0)
    expect(turn0.mcp.servers[0].catalogBlock).toBeNull()
    // Wrapper turn: deferred MCP tools appear as per-tool rows from catalog bullets.
    const github = turn1.mcp.servers.find((server) => server.name === "github")!
    expect(github.instructionsBlock).toBe(0)
    expect(github.catalogBlock).toBeGreaterThan(0)
    expect(github.tools.map((entry) => entry.name)).toEqual(["mcp_github_create_issue", "mcp_github_list_prs"])
    expect(github.tools.every((entry) => entry.tokens > 0)).toBe(true)
    // shellrun lives in the plain catalog block — not an MCP tool.
    expect(turn1.mcp.servers.every((server) => !server.tools.some((entry) => entry.name === "shellrun"))).toBe(true)
    // Attribution rides meta.toolServers on the eager turn; bash stays unattributed.
    const byName = new Map(turn0.tools.entries.map((entry) => [entry.name, entry]))
    expect(byName.get("mcp_github_create_issue")?.server).toBe("github")
    expect(byName.get("bash")?.server).toBeNull()
  })

  test("scenario 4 — output pairing: paired turn carries usage, unmatched turn nulls", async () => {
    const { result } = await setup()
    const [turn0, turn1] = result.sessions[0].turns
    expect(turn0.output).toBe(77)
    expect(turn0.inputReported).toBe(1000)
    expect(turn1.output).toBeNull()
    expect(turn1.inputReported).toBeNull()
  })

  test("scenario 5 — drill-down: --content keys resolve 1:1, selected nodes carry content, default carries none", async () => {
    const { run, result: counts } = await setup()
    expect(JSON.stringify(counts)).not.toContain('"content"')
    const result = breakdown(
      { manifest: run.manifest, captures: run.captures, usage: run.usage },
      {
        content: [
          "system.base",
          "system.instructions",
          "system.mcp:github",
          "system.skills:tdd",
          "system.structured_output",
          "system.user_system",
          "system.catalog:github",
          "tools.bash",
          "history.1",
        ],
      },
    )
    const [turn0, turn1] = result.sessions[0].turns
    const blocks0 = new Map(turn0.system.blocks.map((block) => [block.key, block]))
    const blocks1 = new Map(turn1.system.blocks.map((block) => [block.key, block]))
    expect(blocks0.get("base")?.content).toBe(BASE_BLOCK + "\n")
    expect(blocks0.get("structured_output")?.content).toBe(STRUCTURED_OUTPUT_BLOCK + "\n")
    expect(blocks0.get("environment")?.content).toBeUndefined()
    expect(turn0.skills.perSkill.find((skill) => skill.name === "tdd")?.content).toBeDefined()
    expect(turn0.tools.entries.find((entry) => entry.name === "bash")?.content).toEqual({
      description: "Run shell commands",
      inputSchema: { type: "object", properties: {} },
    })
    expect(turn0.history.messages[0]).toMatchObject({ index: 1, content: "hello" })
    // A key valid in any turn is accepted; blocks absent from a turn carry none.
    expect(blocks1.get("catalog:github")?.content).toContain("<deferred_tools")
    expect(blocks0.get("catalog:github")?.content).toBeUndefined()
  })

  test("scenario 6 — CLI: counts-only emit matches the pure core, --human prints, invalid run dir exits non-zero", async () => {
    const { result } = await setup()
    await using tmp = await tmpdir()
    await writeRun(tmp.path)
    const out = path.join(tmp.path, "out.json")
    const { code } = await captureStdout(() => main([tmp.path, "-o", out]))
    expect(code).toBe(0)
    const parsed: Breakdown = JSON.parse(await Bun.file(out).text())
    expect(parsed.schema).toBe(1)
    expect(JSON.stringify(parsed)).not.toContain('"content"')
    // generatedAt is the only non-deterministic field — compare the rest.
    const strip = ({ generatedAt: _, ...rest }: Breakdown["run"]) => rest
    expect(strip(parsed.run)).toEqual(strip(result.run))
    expect(parsed.totals).toEqual(result.totals)
    expect(parsed.sessions).toEqual(result.sessions)
    const human = await captureStdout(() => main([tmp.path, "--human"]))
    expect(human.code).toBe(0)
    expect(human.stdout).toContain("session ses_main")
    expect(human.stdout).toContain("turn 0")
    expect(human.stdout).toContain("%")
    const log = spyOn(console, "error")
    try {
      expect(await main([path.join(tmp.path, "nope")])).toBe(1)
      expect(log).toHaveBeenCalled()
    } finally {
      log.mockRestore()
    }
  })

  test.each(["glm", "deepseek"] as const)(
    "scenario 7 — R10-008 bound fixtures re-asserted through the full breakdown() path (%s)",
    async (name) => {
      const capture: CaptureFile = JSON.parse(await Bun.file(path.join(import.meta.dir, "fixture", "estimator", name, "capture.json")).text())
      const usage: { tokens: { input: number; output: number; reasoning: number; cache: { read: number; write: number } }; cost: number } = JSON.parse(
        await Bun.file(path.join(import.meta.dir, "fixture", "estimator", name, "usage.json")).text(),
      )
      const runManifest: RunManifest = {
        ...manifest,
        modelID: capture.meta.modelID,
        providerID: capture.meta.providerID,
      }
      const result = breakdown({
        manifest: runManifest,
        captures: [{ file: `data/prompt-captures/${capture.meta.sessionID}/0000.json`, capture }],
        usage: [
          {
            sessionID: capture.meta.sessionID,
            tokens: {
              input: usage.tokens.input,
              output: usage.tokens.output,
              reasoning: usage.tokens.reasoning,
              cacheRead: usage.tokens.cache.read,
              cacheWrite: usage.tokens.cache.write,
            },
            cost: usage.cost,
          },
        ],
      })
      const turn = result.sessions[0].turns[0]
      const reported = usage.tokens.input + usage.tokens.cache.read + usage.tokens.cache.write
      // Adapter engagement read post-estimate — a silent fallback cannot pass.
      expect(turn.estimator.adapter).toBe(name)
      expect(Math.abs(turn.inputEstimate - reported) / reported).toBeLessThanOrEqual(0.1)
      expect(turn.inputReported).toBe(usage.tokens.input)
      expect(turn.output).toBe(usage.tokens.output)
    },
  )
})
