import { describe, expect, spyOn, test } from "bun:test"
import { Database } from "bun:sqlite"
import { Schema } from "effect"
import fs from "fs/promises"
import path from "path"
import { tmpdir } from "./fixture/fixture"
import {
  cacheCollapseFlags,
  CACHE_COLLAPSE_RATIO,
  campaignReport,
  coldStartInput,
  deriveVerdicts,
  deriveVerdictProvenances,
  diff,
  loadRun,
  payloadChars,
  report,
  RunManifestSchema,
  sessionInput,
  type CaptureRecord,
  type Run,
  type RunManifest,
  type StepFinishRecord,
  type UsageReport,
} from "../script/measure-usage"
import type { CaptureFile } from "../src/session/llm/prompt-capture"
import type { VerdictProvenance } from "../src/session/binding-verdict"
import type { CampaignSpec } from "../script/reference-workload"

const manifest = (input?: Partial<RunManifest>): RunManifest => ({
  bin: "opencode",
  modelID: "test-model",
  providerID: "test",
  promptsDigest: "digest-1",
  capture: true,
  proxyMode: false,
  dbPath: "data/opencode.db",
  capturesDir: "data/prompt-captures",
  startedAt: "2026-09-12T00:00:00.000Z",
  endedAt: "2026-09-12T00:01:00.000Z",
  ...input,
})

const capture = (
  sessionID: string,
  requestID = "msg_1",
  payload?: Partial<CaptureFile["payload"]>,
  meta?: Partial<CaptureFile["meta"]>,
): CaptureFile => ({
  meta: {
    version: 1,
    sessionID,
    providerID: "test",
    modelID: "test-model",
    modelApi: "@ai-sdk/openai-compatible",
    agent: "build",
    small: false,
    requestID,
    createdAt: "2026-09-12T00:00:00.000Z",
    optimized: { lazyTools: true, staticSlimming: true },
    ...meta,
  },
  payload: {
    system: ["abcd"],
    tools: { read: { description: "abcd", inputSchema: { type: "object" } } },
    messages: [{ role: "user", content: "hello" }],
    ...payload,
  },
})

// Hand-built minimal run dir: the script only reads session.rowid/id and
// part.rowid/session_id/data — the driver-owned layout (design SC-4 + co-02).
const writeRun = async (
  dir: string,
  input: {
    manifest: RunManifest
    sessions: string[]
    parts: { sessionID: string; data: Record<string, unknown> }[]
    captures?: { sessionID: string; seq: number; file: CaptureFile }[]
    corrupt?: { sessionID: string; seq: number }[]
  },
) => {
  await Bun.write(path.join(dir, "manifest.json"), JSON.stringify(input.manifest))
  const dbPath = path.join(dir, input.manifest.dbPath)
  await fs.mkdir(path.dirname(dbPath), { recursive: true })
  const db = new Database(dbPath)
  db.exec("CREATE TABLE session (id TEXT PRIMARY KEY)")
  db.exec("CREATE TABLE part (session_id TEXT, data TEXT)")
  for (const id of input.sessions) db.run("INSERT INTO session (id) VALUES (?)", [id])
  for (const part of input.parts) {
    db.run("INSERT INTO part (session_id, data) VALUES (?, ?)", [part.sessionID, JSON.stringify(part.data)])
  }
  db.close()
  if (input.manifest.capturesDir !== null && input.captures) {
    for (const entry of input.captures) {
      const capturesDir = path.join(dir, input.manifest.capturesDir ?? "", entry.sessionID)
      await fs.mkdir(capturesDir, { recursive: true })
      const seq = String(entry.seq).padStart(4, "0")
      await Bun.write(path.join(capturesDir, `${seq}.json`), JSON.stringify(entry.file))
    }
  }
}

const usageRow = (sessionID: string, input: number, overrides?: Partial<StepFinishRecord>): StepFinishRecord => ({
  sessionID,
  tokens: { input, output: 2, reasoning: 0, cacheRead: 5, cacheWrite: 1 },
  cost: 0.5,
  ...overrides,
})

const stepFinish = (input: { input: number; output?: number; cost?: number }) => ({
  type: "step-finish",
  tokens: { input: input.input, output: input.output ?? 2, reasoning: 0, cache: { read: 5, write: 1 } },
  cost: input.cost ?? 0.5,
})

describe("usage-report.report", () => {
  test("pairs captures and step-finish rows by order within a session", async () => {
    await using tmp = await tmpdir()
    await writeRun(tmp.path, {
      manifest: manifest(),
      sessions: ["ses_a"],
      parts: [
        { sessionID: "ses_a", data: stepFinish({ input: 10 }) },
        { sessionID: "ses_a", data: stepFinish({ input: 20, output: 3, cost: 0.75 }) },
      ],
      captures: [
        { sessionID: "ses_a", seq: 1, file: capture("ses_a", "msg_a_1") },
        { sessionID: "ses_a", seq: 0, file: capture("ses_a", "msg_a_0") },
      ],
    })
    const run = await loadRun(tmp.path)
    const result = report(run)
    expect(result.sessions).toHaveLength(1)
    expect(result.sessions[0]?.sessionID).toBe("ses_a")
    expect(result.sessions[0]?.turns.map((turn) => turn.captureFile)).toEqual([
      "data/prompt-captures/ses_a/0000.json",
      "data/prompt-captures/ses_a/0001.json",
    ])
    expect(result.sessions[0]?.turns.map((turn) => turn.usage?.input)).toEqual([10, 20])
    expect(result.sessions[0]?.turns.map((turn) => turn.usage?.output)).toEqual([2, 3])
    expect(result.sessions[0]?.turns.map((turn) => turn.usage?.cost)).toEqual([0.5, 0.75])
    expect(result.sessions[0]?.totals).toEqual({ input: 30, output: 5, cost: 1.25 })
  })

  test("chars/4 estimator + proportional reconciliation (residual to history)", async () => {
    // system 4 chars -> 1; tools: name(4) + schema(17) + description(4) = 25 -> 7;
    // history: {"role":"user","content":"hello"} = 34 -> 9; total 17.
    // usage.input 10: k = 10/17 -> system round(0.588)=1, tools round(4.118)=4, residual 5.
    const input = {
      manifest: manifest(),
      sessions: ["ses_a"],
      captures: [{ file: "data/prompt-captures/ses_a/0000.json", capture: capture("ses_a") }],
      usage: [usageRow("ses_a", 10)],
    }
    const result = report(input)
    expect(result.sessions[0]?.turns[0]?.estimate).toEqual({
      systemTokens: 1,
      toolsTokens: 7,
      historyTokens: 9,
      totalTokens: 17,
    })
    expect(result.sessions[0]?.turns[0]?.reconciled).toEqual({ systemTokens: 1, toolsTokens: 4, historyTokens: 5 })
  })

  test("leading system-role messages count as system, not history", () => {
    // leading system message {"role":"system","content":"be good"} = 39 chars + 4 system = 43 -> 11;
    // history keeps only the user message -> 9.
    const file = capture("ses_a", "msg_1", {
      messages: [{ role: "system", content: "be good" }, { role: "user", content: "hello" }],
    })
    const result = report({
      manifest: manifest(),
      sessions: ["ses_a"],
      captures: [{ file: "data/prompt-captures/ses_a/0000.json", capture: file }],
      usage: [],
    })
    expect(result.sessions[0]?.turns[0]?.estimate).toEqual({
      systemTokens: 11,
      toolsTokens: 7,
      historyTokens: 9,
      totalTokens: 27,
    })
    // no usage row: reconciliation impossible
    expect(result.sessions[0]?.turns[0]?.reconciled).toBeNull()
  })

  test("unmatched rows are reported, never dropped", async () => {
    await using tmp = await tmpdir()
    await writeRun(tmp.path, {
      manifest: manifest(),
      sessions: ["ses_a"],
      parts: [
        { sessionID: "ses_a", data: stepFinish({ input: 10 }) },
        { sessionID: "ses_a", data: stepFinish({ input: 20 }) },
        { sessionID: "ses_a", data: stepFinish({ input: 30 }) },
      ],
      captures: [{ sessionID: "ses_a", seq: 0, file: capture("ses_a") }],
    })
    const result = report(await loadRun(tmp.path))
    const turns = result.sessions[0]?.turns ?? []
    expect(turns).toHaveLength(3)
    expect(turns[0]?.captureFile).toBe("data/prompt-captures/ses_a/0000.json")
    expect(turns[1]?.usage).toEqual({ input: 20, output: 2, reasoning: 0, cacheRead: 5, cacheWrite: 1, cost: 0.5 })
    expect(turns[1]?.estimate).toBeNull()
    expect(turns[1]?.reconciled).toBeNull()
    expect(Object.hasOwn(turns[1], "captureFile")).toBe(false)
  })

  test("extra capture beyond the usage rows stays visible (capture-only turn)", async () => {
    await using tmp = await tmpdir()
    await writeRun(tmp.path, {
      manifest: manifest(),
      sessions: ["ses_a"],
      parts: [{ sessionID: "ses_a", data: stepFinish({ input: 10 }) }],
      captures: [
        { sessionID: "ses_a", seq: 0, file: capture("ses_a") },
        { sessionID: "ses_a", seq: 1, file: capture("ses_a", "msg_retry") },
      ],
    })
    const result = report(await loadRun(tmp.path))
    const turns = result.sessions[0]?.turns ?? []
    expect(turns).toHaveLength(2)
    expect(turns[1]?.usage).toBeNull()
    expect(turns[1]?.captureFile).toBe("data/prompt-captures/ses_a/0001.json")
    expect(turns[1]?.estimate).toEqual({ systemTokens: 1, toolsTokens: 7, historyTokens: 9, totalTokens: 17 })
    expect(turns[1]?.reconciled).toBeNull()
  })

  test("upstream run (no captures): full usage rows, null attribution", async () => {
    await using tmp = await tmpdir()
    await writeRun(tmp.path, {
      manifest: manifest({ capture: false, capturesDir: null }),
      sessions: ["ses_u"],
      parts: [{ sessionID: "ses_u", data: stepFinish({ input: 42, cost: 0.25 }) }],
    })
    const result = report(await loadRun(tmp.path))
    const turn = result.sessions[0]?.turns[0]
    expect(turn).toEqual({ index: 0, usage: { input: 42, output: 2, reasoning: 0, cacheRead: 5, cacheWrite: 1, cost: 0.25 }, estimate: null, reconciled: null })
    expect(Object.hasOwn(turn ?? {}, "captureFile")).toBe(false)
  })

  test("sessions ordered by session table rowid; turn-less sessions keep their slot", async () => {
    await using tmp = await tmpdir()
    await writeRun(tmp.path, {
      manifest: manifest(),
      sessions: ["ses_a", "ses_b", "ses_empty"],
      parts: [
        { sessionID: "ses_b", data: stepFinish({ input: 20 }) },
        { sessionID: "ses_a", data: stepFinish({ input: 10 }) },
      ],
      captures: [{ sessionID: "ses_b", seq: 0, file: capture("ses_b") }],
    })
    const result = report(await loadRun(tmp.path))
    expect(result.sessions.map((session) => session.sessionID)).toEqual(["ses_a", "ses_b", "ses_empty"])
    expect(result.sessions[2]?.turns).toEqual([])
  })

  test("corrupt capture file is skipped with a warning, report continues", async () => {
    await using tmp = await tmpdir()
    await writeRun(tmp.path, {
      manifest: manifest(),
      sessions: ["ses_a"],
      parts: [{ sessionID: "ses_a", data: stepFinish({ input: 10 }) }],
      captures: [{ sessionID: "ses_a", seq: 0, file: capture("ses_a") }],
    })
    await Bun.write(path.join(tmp.path, "data", "prompt-captures", "ses_a", "0001.json"), "{not json")
    const error = spyOn(console, "error").mockImplementation(() => {})
    const result = report(await loadRun(tmp.path))
    expect(error).toHaveBeenCalled()
    error.mockRestore()
    // the corrupt file is skipped, not repaired: the run pairs the valid capture with the usage row
    expect(result.sessions[0]?.turns).toHaveLength(1)
    expect(result.sessions[0]?.turns[0]?.captureFile).toBe("data/prompt-captures/ses_a/0000.json")
  })

  test("fails loudly: missing manifest or missing usage DB", async () => {
    await using tmp = await tmpdir()
    await fs.mkdir(tmp.path, { recursive: true })
    await expect(loadRun(tmp.path)).rejects.toThrow(/manifest/)
    await Bun.write(path.join(tmp.path, "manifest.json"), JSON.stringify(manifest()))
    await expect(loadRun(tmp.path)).rejects.toThrow(/usage DB/)
  })

  test("empty tools record estimates zero tool tokens", () => {
    const file = capture("ses_a", "msg_1", { tools: {} })
    const result = report({
      manifest: manifest(),
      sessions: ["ses_a"],
      captures: [{ file: "data/prompt-captures/ses_a/0000.json", capture: file }],
      usage: [],
    })
    expect(result.sessions[0]?.turns[0]?.estimate?.toolsTokens).toBe(0)
  })

  test("tolerant readers: round-1 captures (no meta.toolServers) and toolServers-carrying captures both parse", () => {
    const result = report({
      manifest: manifest(),
      sessions: ["ses_a"],
      captures: [
        { file: "data/prompt-captures/ses_a/0000.json", capture: capture("ses_a", "msg_1") },
        {
          file: "data/prompt-captures/ses_a/0001.json",
          capture: capture("ses_a", "msg_2", undefined, { toolServers: { glob: "firecrawl" } }),
        },
      ],
      usage: [],
    })
    const turns = result.sessions[0]?.turns ?? []
    expect(turns).toHaveLength(2)
    expect(turns[0]?.estimate).toEqual({ systemTokens: 1, toolsTokens: 7, historyTokens: 9, totalTokens: 17 })
    expect(turns[1]?.estimate).toEqual({ systemTokens: 1, toolsTokens: 7, historyTokens: 9, totalTokens: 17 })
  })
})

describe("usage-report.diff", () => {
  const forkReport = (): UsageReport =>
    report({
      manifest: manifest(),
      sessions: ["ses_a"],
      captures: [{ file: "data/prompt-captures/ses_a/0000.json", capture: capture("ses_a") }],
      usage: [
        usageRow("ses_a", 10),
        usageRow("ses_a", 20, { tokens: { input: 20, output: 4, reasoning: 0, cacheRead: 0, cacheWrite: 0 }, cost: 0.75 }),
      ],
    })

  const upstreamReport = (): UsageReport =>
    report({
      manifest: manifest({ capture: false, capturesDir: null }),
      sessions: ["ses_u"],
      captures: [],
      usage: [
        usageRow("ses_u", 12, { tokens: { input: 12, output: 3, reasoning: 0, cacheRead: 4, cacheWrite: 2 }, cost: 0.25 }),
        usageRow("ses_u", 18, { tokens: { input: 18, output: 4, reasoning: 0, cacheRead: 0, cacheWrite: 0 }, cost: 0.5 }),
      ],
    })

  test("aligns by index, per-turn deltas plus fork reconciled components and totals", () => {
    const result = diff(forkReport(), upstreamReport())
    expect(result.warnings).toEqual([])
    const turns = result.sessions[0]?.turns ?? []
    expect(turns[0]?.delta).toEqual({ input: -2, output: -1, cacheRead: 1, cacheWrite: -1, cost: 0.25 })
    expect(turns[0]?.reconciled).toEqual({ systemTokens: 1, toolsTokens: 4, historyTokens: 5 })
    expect(turns[1]?.delta).toEqual({ input: 2, output: 0, cacheRead: 0, cacheWrite: 0, cost: 0.25 })
    expect(turns[1]?.reconciled).toBeNull()
    expect(result.sessions[0]?.totals).toEqual({ input: 0, output: -1, cost: 0.5 })
    expect(result.totals).toEqual({ input: 0, output: -1, cost: 0.5 })
  })

  test("refuses to align runs with different promptsDigest", () => {
    const fork = forkReport()
    const upstream = { ...upstreamReport(), manifest: manifest({ promptsDigest: "digest-2" }) }
    expect(() => diff(fork, upstream)).toThrow(/promptsDigest/)
  })

  test("turn-count mismatch is a warning, unpaired turns carry no delta", () => {
    const fork = forkReport()
    const upstream = upstreamReport()
    upstream.sessions[0].turns = upstream.sessions[0].turns.slice(0, 1)
    const result = diff(fork, upstream)
    expect(result.warnings.length).toBeGreaterThan(0)
    expect(result.sessions[0]?.turns).toHaveLength(2)
    expect(result.sessions[0]?.turns[1]?.delta).toBeNull()
  })

  test("session-count mismatch is a warning", () => {
    const fork = forkReport()
    const upstream = upstreamReport()
    fork.sessions = [...fork.sessions, { sessionID: "ses_b", turns: [], totals: { input: 0, output: 0, cost: 0 } }]
    const result = diff(fork, upstream)
    expect(result.warnings.length).toBeGreaterThan(0)
    expect(result.sessions).toHaveLength(2)
  })
})

describe("usage-report.cli", () => {
  test("report <runDir> prints the usage report JSON", async () => {
    await using tmp = await tmpdir()
    await writeRun(tmp.path, {
      manifest: manifest(),
      sessions: ["ses_a"],
      parts: [{ sessionID: "ses_a", data: stepFinish({ input: 10 }) }],
      captures: [{ sessionID: "ses_a", seq: 0, file: capture("ses_a") }],
    })
    const proc = Bun.spawn([process.execPath, path.join(import.meta.dir, "../script/measure-usage.ts"), "report", tmp.path], {
      cwd: path.join(import.meta.dir, ".."),
      stdout: "pipe",
      stderr: "pipe",
    })
    const [stdout, code] = await Promise.all([new Response(proc.stdout).text(), proc.exited])
    expect(code).toBe(0)
    const parsed = JSON.parse(stdout)
    expect(parsed.sessions).toHaveLength(1)
    expect(parsed.sessions[0]?.turns[0]?.reconciled).toEqual({ systemTokens: 1, toolsTokens: 4, historyTokens: 5 })
  })

  test("diff refusal exits non-zero with the reason on stderr", async () => {
    await using tmp = await tmpdir()
    await writeRun(tmp.path, {
      manifest: manifest({ promptsDigest: "a" }),
      sessions: ["ses_a"],
      parts: [{ sessionID: "ses_a", data: stepFinish({ input: 10 }) }],
    })
    await using tmp2 = await tmpdir()
    await writeRun(tmp2.path, {
      manifest: manifest({ promptsDigest: "b" }),
      sessions: ["ses_a"],
      parts: [{ sessionID: "ses_a", data: stepFinish({ input: 10 }) }],
    })
    const script = path.join(import.meta.dir, "../script/measure-usage.ts")
    const proc = Bun.spawn([process.execPath, script, "diff", tmp.path, tmp2.path], {
      cwd: path.join(import.meta.dir, ".."),
      stdout: "pipe",
      stderr: "pipe",
    })
    const [stderr, code] = await Promise.all([new Response(proc.stderr).text(), proc.exited])
    expect(code).toBe(1)
    expect(stderr).toMatch(/promptsDigest/)
  })
})

// Harness v2 seams (R13-001/003/005, ticket 28): the run manifest becomes the
// verdict/identity record and the comparison metrics are named pure functions.
// Contracts: ARCHITECTURE/detailed/comparison-testing.md §Module contracts.

describe("manifest v2 schema", () => {
  test("v1 manifests (absent v2 fields) validate — old run dirs stay loadable", () => {
    expect(Schema.is(RunManifestSchema)(manifest())).toBe(true)
  })

  test("v2 fields validate when present", () => {
    expect(
      Schema.is(RunManifestSchema)(manifest({ pin: "binding", verdicts: ["binding"], binary: "1.2.3" })),
    ).toBe(true)
    expect(Schema.is(RunManifestSchema)(manifest({ pin: null, verdicts: [], binary: null }))).toBe(true)
  })

  test("invalid v2 values fail validation", () => {
    expect(Schema.is(RunManifestSchema)(manifest({ pin: "bogus" as unknown as RunManifest["pin"] }))).toBe(false)
    expect(Schema.is(RunManifestSchema)(manifest({ verdicts: ["bogus"] as unknown as RunManifest["verdicts"] }))).toBe(false)
    expect(Schema.is(RunManifestSchema)(manifest({ binary: 42 as unknown as string }))).toBe(false)
  })

  test("old run dirs load and report unchanged", async () => {
    await using tmp = await tmpdir()
    await writeRun(tmp.path, {
      manifest: manifest(),
      sessions: ["ses_a"],
      parts: [{ sessionID: "ses_a", data: stepFinish({ input: 10 }) }],
      captures: [{ sessionID: "ses_a", seq: 0, file: capture("ses_a") }],
    })
    const run = await loadRun(tmp.path)
    const result = report(run)
    expect(result.sessions[0]?.turns[0]?.reconciled).toEqual({ systemTokens: 1, toolsTokens: 4, historyTokens: 5 })
  })
})

describe("deriveVerdicts", () => {
  const verdictCapture = (file: string, verdict?: "binding" | "advisory"): CaptureRecord => ({
    file,
    capture: capture("ses_a", file, undefined, verdict === undefined ? {} : { verdict }),
  })

  test("single verdict", () => {
    expect(deriveVerdicts([verdictCapture("a", "binding"), verdictCapture("b", "binding")])).toEqual(["binding"])
  })

  test("mixed verdicts: distinct, sorted, both recorded (never averaged away)", () => {
    expect(deriveVerdicts([verdictCapture("a", "binding"), verdictCapture("b", "advisory"), verdictCapture("c", "binding")])).toEqual([
      "advisory",
      "binding",
    ])
  })

  test("no captures or absent meta.verdict → []", () => {
    expect(deriveVerdicts([])).toEqual([])
    expect(deriveVerdicts([verdictCapture("a"), verdictCapture("b")])).toEqual([])
  })
})

describe("deriveVerdictProvenances", () => {
  const provenanceCapture = (file: string, provenance?: VerdictProvenance): CaptureRecord => ({
    file,
    capture: capture("ses_a", file, undefined, provenance === undefined ? {} : { verdictProvenance: provenance }),
  })

  test("distinct provenances are recorded, sorted by their JSON form", () => {
    expect(
      deriveVerdictProvenances([
        provenanceCapture("a", { origin: "pin" }),
        provenanceCapture("b", { origin: "default" }),
        provenanceCapture("c", { origin: "pin" }),
      ]),
    ).toEqual([{ origin: "default" }, { origin: "pin" }])
  })

  test("structurally equal origins with different facts stay distinct", () => {
    expect(
      deriveVerdictProvenances([
        provenanceCapture("a", { origin: "cache", source: "probe", timestamp: 1 }),
        provenanceCapture("b", { origin: "cache", source: "probe", timestamp: 2 }),
      ]),
    ).toEqual([
      { origin: "cache", source: "probe", timestamp: 1 },
      { origin: "cache", source: "probe", timestamp: 2 },
    ])
  })

  test("no captures or absent meta.verdictProvenance → []", () => {
    expect(deriveVerdictProvenances([])).toEqual([])
    expect(deriveVerdictProvenances([provenanceCapture("a"), provenanceCapture("b")])).toEqual([])
  })
})

describe("cache-immune metrics", () => {
  const runWith = (input?: Partial<Run>): Run => ({
    manifest: manifest(),
    sessions: ["ses_a"],
    captures: [],
    usage: [],
    ...input,
  })

  test("coldStartInput = first step-finish row's input; null without usage rows", () => {
    expect(coldStartInput(runWith({ usage: [usageRow("ses_a", 10), usageRow("ses_a", 20)] }))).toBe(10)
    expect(coldStartInput(runWith())).toBeNull()
  })

  test("sessionInput = Σ(input + cacheRead + cacheWrite); re-billing keeps the sum stable", () => {
    // Cache-population lag (GLM): turn billed 100 input + 50 cacheWrite on one
    // run, 150 flat input on another — the sum is invariant.
    const populated = runWith({
      usage: [
        usageRow("ses_a", 100, { tokens: { input: 100, output: 2, reasoning: 0, cacheRead: 0, cacheWrite: 50 } }),
        usageRow("ses_a", 30, { tokens: { input: 30, output: 2, reasoning: 0, cacheRead: 40, cacheWrite: 0 } }),
      ],
    })
    const flat = runWith({
      usage: [
        usageRow("ses_a", 150, { tokens: { input: 150, output: 2, reasoning: 0, cacheRead: 0, cacheWrite: 0 } }),
        usageRow("ses_a", 70, { tokens: { input: 30, output: 2, reasoning: 0, cacheRead: 40, cacheWrite: 0 } }),
      ],
    })
    expect(sessionInput(populated)).toBe(220)
    expect(sessionInput(flat)).toBe(220)
  })

  test("payloadChars = v1 estimator char accounting without ceil(/4); null without captures", () => {
    // fixture: system 4 + tools 4+17+4 = 25 + history 33 = 62 chars
    // (v1 estimate would ceil to 1+7+9 = 17 tokens)
    const run = runWith({ captures: [{ file: "c/0000.json", capture: capture("ses_a") }] })
    expect(payloadChars(run)).toBe(62)
    expect(coldStartInput(runWith())).toBeNull()
    expect(payloadChars(runWith())).toBeNull()
    const two = runWith({
      captures: [
        { file: "c/0000.json", capture: capture("ses_a") },
        { file: "c/0001.json", capture: capture("ses_a", "msg_2") },
      ],
    })
    expect(payloadChars(two)).toBe(124)
  })

  test("CACHE_COLLAPSE_RATIO is the documented 0.5 heuristic", () => {
    expect(CACHE_COLLAPSE_RATIO).toBe(0.5)
  })

  test("cacheCollapseFlags: fires below ratio × previous total, silent above, never on turn 0", () => {
    const rows = [
      usageRow("ses_a", 100, { tokens: { input: 100, output: 2, reasoning: 0, cacheRead: 0, cacheWrite: 0 } }),
      usageRow("ses_a", 20, { tokens: { input: 20, output: 2, reasoning: 0, cacheRead: 60, cacheWrite: 0 } }),
      usageRow("ses_a", 20, { tokens: { input: 20, output: 2, reasoning: 0, cacheRead: 10, cacheWrite: 0 } }),
    ]
    // turn 1: 60 >= 0.5×100 → silent; turn 2: 10 < 0.5×80 → flag
    expect(cacheCollapseFlags(rows)).toEqual([{ turn: 2, cacheRead: 10, expectedPrefix: 40 }])
    expect(cacheCollapseFlags([rows[0]])).toEqual([])
  })

  test("cacheCollapseFlags: per-session rows — a new session's first turn is never flagged", () => {
    const rows = [
      usageRow("ses_a", 100, { tokens: { input: 100, output: 2, reasoning: 0, cacheRead: 0, cacheWrite: 0 } }),
      usageRow("ses_b", 10, { tokens: { input: 10, output: 2, reasoning: 0, cacheRead: 0, cacheWrite: 0 } }),
      usageRow("ses_b", 0, { tokens: { input: 0, output: 2, reasoning: 0, cacheRead: 1, cacheWrite: 0 } }),
    ]
    expect(cacheCollapseFlags(rows)).toEqual([{ turn: 1, cacheRead: 1, expectedPrefix: 5 }])
  })
})

// Campaign aggregation (R13-002 medians, R13-003 gate, R13-006 report core;
// ticket 30): synthetic campaign dirs — campaign.json + per-run dirs written
// by the writeRun helper above. Contracts: ARCHITECTURE/detailed/comparison-testing.md
// §Data structures (ComparisonReport v1) + §Module contracts (campaign aggregation).

describe("campaign aggregation (v2 report)", () => {
  const forkManifest = (input?: Partial<RunManifest>): RunManifest =>
    manifest({ verdicts: ["binding"], pin: "binding", binary: "fork-1", ...input })

  const upstreamManifest = (input?: Partial<RunManifest>): RunManifest =>
    manifest({ capture: false, capturesDir: null, verdicts: [], pin: null, binary: "upstream-1", ...input })

  const spec = (phases: string[] = ["p1"], legs?: CampaignSpec["legs"]): CampaignSpec => ({
    phases: phases.map((name) => ({ name, model: "test/model" })),
    legs:
      legs ?? [
        { shape: "fork", bin: "fork-bin" },
        { shape: "upstream", bin: "upstream-bin" },
      ],
    runs: 2,
  })

  const doc = (runs: unknown[], input?: { spec?: unknown; endedAt?: string | null }) => ({
    spec: spec(),
    seed: 7,
    startedAt: "2026-09-15T00:00:00.000Z",
    endedAt: "2026-09-15T00:10:00.000Z",
    runs,
    ...input,
  })

  const dirOf = (index: number, shape: "fork" | "upstream", phase = "p1", proxy = false) =>
    `${String(index).padStart(2, "0")}-${phase}-${shape}${proxy ? "-proxy" : ""}`

  const scheduleRun = (index: number, phase: string, shape: "fork" | "upstream", proxy = false) => ({
    index,
    phase,
    shape,
    proxy,
    rep: index % 2,
    runDir: dirOf(index, shape, phase, proxy),
    status: "done",
  })

  type RunEntry = {
    dir: string
    manifest: RunManifest
    sessions: string[]
    parts: { sessionID: string; data: Record<string, unknown> }[]
    captures: { sessionID: string; seq: number; file: CaptureFile }[]
  }

  const writeCampaignDir = async (dir: string, document: object, runDirs: RunEntry[]) => {
    await fs.mkdir(dir, { recursive: true })
    await Bun.write(path.join(dir, "campaign.json"), JSON.stringify(document))
    for (const entry of runDirs) await writeRun(path.join(dir, entry.dir), entry)
  }

  // Hand-computed fixture legs (62 chars per capture — the ticket-28 fixture):
  // fork coldStart [10, 30] → median 20; sessionInput [16, 36] → median 26
  // (input+cacheRead+cacheWrite = +5+1); payloadChars [62, 124] → median 93.
  // upstream coldStart [12, 14] → median 13; sessionInput [18, 20] → median 19.
  const forkRep = (index: number, input?: Partial<RunManifest>): RunEntry => ({
    dir: dirOf(index, "fork"),
    manifest: forkManifest(input),
    sessions: ["ses_a"],
    parts: [{ sessionID: "ses_a", data: stepFinish({ input: index % 2 === 0 ? 10 : 30 }) }],
    captures: [
      { sessionID: "ses_a", seq: 0, file: capture("ses_a") },
      ...(index % 2 === 1 ? [{ sessionID: "ses_a", seq: 1, file: capture("ses_a", "msg_2") }] : []),
    ],
  })

  const upstreamRep = (index: number, input?: Partial<RunManifest>): RunEntry => ({
    dir: dirOf(index, "upstream"),
    manifest: upstreamManifest(input),
    sessions: ["ses_u"],
    parts: [{ sessionID: "ses_u", data: stepFinish({ input: index % 2 === 0 ? 12 : 14 }) }],
    captures: [],
  })

  const p1DocRuns = (): object[] => [
    scheduleRun(0, "p1", "fork"),
    scheduleRun(1, "p1", "fork"),
    scheduleRun(2, "p1", "upstream"),
    scheduleRun(3, "p1", "upstream"),
  ]

  const p1Doc = (input?: { spec?: unknown; endedAt?: string | null }): object =>
    doc(p1DocRuns(), input)

  test("happy path: per-leg dispersions match hand-computed medians; upstream legs payloadChars/verdict null, gate-exempt", async () => {
    await using tmp = await tmpdir()
    const entries = [
      forkRep(0),
      forkRep(1),
      upstreamRep(2),
      upstreamRep(3),
      { ...forkRep(4), dir: dirOf(4, "fork", "p2") },
      { ...forkRep(5), dir: dirOf(5, "fork", "p2") },
      { ...upstreamRep(6), dir: dirOf(6, "upstream", "p2") },
      { ...upstreamRep(7), dir: dirOf(7, "upstream", "p2") },
    ]
    await writeCampaignDir(tmp.path, doc([scheduleRun(0, "p1", "fork"), scheduleRun(1, "p1", "fork"), scheduleRun(2, "p1", "upstream"), scheduleRun(3, "p1", "upstream"), scheduleRun(4, "p2", "fork"), scheduleRun(5, "p2", "fork"), scheduleRun(6, "p2", "upstream"), scheduleRun(7, "p2", "upstream")], { spec: spec(["p1", "p2"]) }), entries)
    const result = await campaignReport(tmp.path)
    expect(result.schema).toBe(1)
    expect(result.campaign.spec).toEqual(spec(["p1", "p2"]))
    expect(result.campaign.seed).toBe(7)
    expect(result.campaign.schedule).toHaveLength(8)
    expect(result.campaign.endedAt).toBe("2026-09-15T00:10:00.000Z")
    expect(result.phases.map((phase) => phase.name)).toEqual(["p1", "p2"])
    expect(result.phases.map((phase) => phase.model)).toEqual(["test/model", "test/model"])
    const p1 = result.phases[0]
    const forkLeg = p1.legs[0]
    expect(forkLeg.shape).toBe("fork")
    expect(forkLeg.proxy).toBe(false)
    expect(forkLeg.verdict).toBe("binding")
    expect(forkLeg.verdictMixed).toBe(false)
    expect(forkLeg.runs).toEqual([
      { runDir: "00-p1-fork", verdicts: ["binding"], binary: "fork-1" },
      { runDir: "01-p1-fork", verdicts: ["binding"], binary: "fork-1" },
    ])
    expect(forkLeg.metrics.coldStartInput).toEqual({ median: 20, min: 10, max: 30, values: [10, 30] })
    expect(forkLeg.metrics.sessionInput).toEqual({ median: 26, min: 16, max: 36, values: [16, 36] })
    expect(forkLeg.metrics.payloadChars).toEqual({ median: 93, min: 62, max: 124, values: [62, 124] })
    const turn0 = await turn0Of(path.join(tmp.path, "00-p1-fork"))
    expect(forkLeg.coldStartAttribution).toEqual({
      system: { median: turn0.system, min: turn0.system, max: turn0.system, values: [turn0.system, turn0.system] },
      tools: { median: turn0.tools, min: turn0.tools, max: turn0.tools, values: [turn0.tools, turn0.tools] },
      history: { median: turn0.history, min: turn0.history, max: turn0.history, values: [turn0.history, turn0.history] },
    })
    expect(forkLeg.diagnostics.cacheCollapses).toEqual([])
    const upstreamLeg = p1.legs[1]
    expect(upstreamLeg.shape).toBe("upstream")
    expect(upstreamLeg.verdict).toBeNull()
    expect(upstreamLeg.verdictMixed).toBe(false)
    expect(upstreamLeg.metrics.coldStartInput).toEqual({ median: 13, min: 12, max: 14, values: [12, 14] })
    expect(upstreamLeg.metrics.sessionInput).toEqual({ median: 19, min: 18, max: 20, values: [18, 20] })
    expect(upstreamLeg.metrics.payloadChars).toBeNull()
    expect(p1.comparisons).toEqual([
      {
        kind: "fork-vs-upstream",
        comparable: true,
        verdict: "binding",
        metrics: {
          coldStartInput: { fork: 20, other: 13, delta: 7 },
          sessionInput: { fork: 26, other: 19, delta: 7 },
          payloadChars: null,
        },
      },
    ])
    expect(result.phases[1].comparisons).toEqual(p1.comparisons)
    expect(result.warnings).toEqual([])
  })

  test("verdict gate: fork runs split binding/advisory → comparable false + warning naming both run dirs, medians still emitted", async () => {
    await using tmp = await tmpdir()
    const entries = [forkRep(0, { verdicts: ["binding"] }), forkRep(1, { verdicts: ["advisory"] }), upstreamRep(2), upstreamRep(3)]
    await writeCampaignDir(tmp.path, p1Doc(), entries)
    const result = await campaignReport(tmp.path)
    const forkLeg = result.phases[0].legs[0]
    expect(forkLeg.verdict).toBeNull()
    expect(forkLeg.verdictMixed).toBe(true)
    expect(forkLeg.metrics.sessionInput).toEqual({ median: 26, min: 16, max: 36, values: [16, 36] })
    const comparison = result.phases[0].comparisons[0]
    expect(comparison.comparable).toBe(false)
    expect(comparison.verdict).toBeNull()
    expect(comparison.metrics.sessionInput).toEqual({ fork: 26, other: 19, delta: 7 })
    expect(result.warnings).toHaveLength(1)
    expect(result.warnings[0]).toMatch(/flip/)
    expect(result.warnings[0]).toMatch(/binding in 00-p1-fork/)
    expect(result.warnings[0]).toMatch(/advisory in 01-p1-fork/)
  })

  test("verdict gate: a run with ≥2 verdicts → verdictMixed true + comparable false + warning naming the run dir", async () => {
    await using tmp = await tmpdir()
    const entries = [forkRep(0, { verdicts: ["advisory", "binding"] }), forkRep(1), upstreamRep(2), upstreamRep(3)]
    await writeCampaignDir(tmp.path, p1Doc(), entries)
    const result = await campaignReport(tmp.path)
    const forkLeg = result.phases[0].legs[0]
    expect(forkLeg.verdict).toBeNull()
    expect(forkLeg.verdictMixed).toBe(true)
    expect(result.phases[0].comparisons[0].comparable).toBe(false)
    expect(result.warnings).toHaveLength(1)
    expect(result.warnings[0]).toMatch(/mixed/)
    expect(result.warnings[0]).toMatch(/00-p1-fork/)
  })

  test("old fork run dirs (manifests without verdicts) → fork leg incomparable + warning; metrics still computed", async () => {
    await using tmp = await tmpdir()
    const old = { pin: undefined, verdicts: undefined, binary: undefined }
    const entries = [forkRep(0, old), forkRep(1, old), upstreamRep(2), upstreamRep(3)]
    await writeCampaignDir(tmp.path, p1Doc(), entries)
    const result = await campaignReport(tmp.path)
    const forkLeg = result.phases[0].legs[0]
    expect(forkLeg.verdict).toBeNull()
    expect(forkLeg.verdictMixed).toBe(false)
    expect(forkLeg.metrics.sessionInput).toEqual({ median: 26, min: 16, max: 36, values: [16, 36] })
    expect(result.phases[0].comparisons[0].comparable).toBe(false)
    expect(result.warnings).toHaveLength(1)
    expect(result.warnings[0]).toMatch(/00-p1-fork/)
    expect(result.warnings[0]).toMatch(/no recorded verdict/)
  })

  test("fork-vs-fork-proxy pair: payloadChars delta present; both sides capture-bearing; proxy leg gate-exempt", async () => {
    await using tmp = await tmpdir()
    const proxyRep = (index: number): RunEntry => ({
      dir: dirOf(index, "fork", "p1", true),
      manifest: forkManifest({ verdicts: [] }),
      sessions: ["ses_a"],
      parts: [{ sessionID: "ses_a", data: stepFinish({ input: index % 2 === 0 ? 11 : 31 }) }],
      captures: [{ sessionID: "ses_a", seq: 0, file: capture("ses_a") }],
    })
    const entries = [forkRep(0), forkRep(1), proxyRep(2), proxyRep(3), upstreamRep(4), upstreamRep(5)]
    await writeCampaignDir(
      tmp.path,
      doc([
        scheduleRun(0, "p1", "fork"),
        scheduleRun(1, "p1", "fork"),
        scheduleRun(2, "p1", "fork", true),
        scheduleRun(3, "p1", "fork", true),
        scheduleRun(4, "p1", "upstream"),
        scheduleRun(5, "p1", "upstream"),
      ]),
      entries,
    )
    const result = await campaignReport(tmp.path)
    const legs = result.phases[0].legs
    expect(legs.map((leg) => [leg.shape, leg.proxy])).toEqual([
      ["fork", false],
      ["fork", true],
      ["upstream", false],
    ])
    const proxyLeg = legs[1]
    expect(proxyLeg.verdict).toBeNull()
    expect(proxyLeg.metrics.payloadChars).toEqual({ median: 62, min: 62, max: 62, values: [62, 62] })
    const [upstreamPair, proxyPair] = result.phases[0].comparisons
    expect(upstreamPair.kind).toBe("fork-vs-upstream")
    expect(proxyPair).toEqual({
      kind: "fork-vs-fork-proxy",
      comparable: true,
      verdict: "binding",
      metrics: {
        coldStartInput: { fork: 20, other: 21, delta: -1 },
        sessionInput: { fork: 26, other: 27, delta: -1 },
        payloadChars: { fork: 93, other: 62, delta: 31 },
      },
    })
    expect(result.warnings).toEqual([])
  })

  test("cache-collapse diagnostics: crossing rows listed with runDir + turn + expectedPrefix; absent otherwise", async () => {
    await using tmp = await tmpdir()
    const collapsing: RunEntry = {
      ...forkRep(0),
      parts: [
        { sessionID: "ses_a", data: { type: "step-finish", tokens: { input: 100, output: 2, reasoning: 0, cache: { read: 0, write: 0 } }, cost: 0.5 } },
        { sessionID: "ses_a", data: { type: "step-finish", tokens: { input: 20, output: 2, reasoning: 0, cache: { read: 10, write: 0 } }, cost: 0.5 } },
      ],
    }
    const entries = [collapsing, forkRep(1), upstreamRep(2), upstreamRep(3)]
    await writeCampaignDir(tmp.path, p1Doc(), entries)
    const result = await campaignReport(tmp.path)
    expect(result.phases[0].legs[0].diagnostics.cacheCollapses).toEqual([{ runDir: "00-p1-fork", turn: 1, cacheRead: 10, expectedPrefix: 50 }])
    expect(result.phases[0].legs[1].diagnostics.cacheCollapses).toEqual([])
  })

  test("incomplete campaign (failed/pending run, or null endedAt) refused loudly; missing campaign.json refused loudly", async () => {
    await using tmp = await tmpdir()
    const entries = [forkRep(0), forkRep(1), upstreamRep(2), upstreamRep(3)]
    const runs = [scheduleRun(0, "p1", "fork"), scheduleRun(1, "p1", "fork"), scheduleRun(2, "p1", "upstream"), scheduleRun(3, "p1", "upstream")]
    await writeCampaignDir(tmp.path, p1Doc(), entries)
    await Bun.write(path.join(tmp.path, "campaign.json"), JSON.stringify(doc([{ ...runs[3], status: "failed", error: "boom" }])))
    await expect(campaignReport(tmp.path)).rejects.toThrow(/incomplete.*03-p1-upstream.*failed/)
    await Bun.write(path.join(tmp.path, "campaign.json"), JSON.stringify(doc([{ ...runs[3], status: "pending" }])))
    await expect(campaignReport(tmp.path)).rejects.toThrow(/incomplete/)
    await Bun.write(path.join(tmp.path, "campaign.json"), JSON.stringify(doc(runs, { endedAt: null })))
    await expect(campaignReport(tmp.path)).rejects.toThrow(/incomplete/)
    await using tmp2 = await tmpdir()
    await expect(campaignReport(tmp2.path)).rejects.toThrow(/campaign\.json/)
  })

  test("CLI: campaign <dir> prints the report JSON; -o writes it instead", async () => {
    await using tmp = await tmpdir()
    const entries = [forkRep(0), forkRep(1), upstreamRep(2), upstreamRep(3)]
    await writeCampaignDir(tmp.path, p1Doc(), entries)
    const script = path.join(import.meta.dir, "../script/measure-usage.ts")
    const stdoutProc = Bun.spawn([process.execPath, script, "campaign", tmp.path], { cwd: path.join(import.meta.dir, ".."), stdout: "pipe", stderr: "pipe" })
    const [stdout, code] = await Promise.all([new Response(stdoutProc.stdout).text(), stdoutProc.exited])
    expect(code).toBe(0)
    const parsed = JSON.parse(stdout)
    expect(parsed.schema).toBe(1)
    expect(parsed.phases[0].comparisons[0].comparable).toBe(true)
    expect(parsed.phases[0].comparisons[0].metrics.coldStartInput.delta).toBe(7)
    const outPath = path.join(tmp.path, "report.json")
    const fileProc = Bun.spawn([process.execPath, script, "campaign", tmp.path, "-o", outPath], { cwd: path.join(import.meta.dir, ".."), stdout: "pipe", stderr: "pipe" })
    const [stdout2, code2] = await Promise.all([new Response(fileProc.stdout).text(), fileProc.exited])
    expect(code2).toBe(0)
    expect(stdout2).toBe("")
    expect(JSON.parse(await Bun.file(outPath).text()).schema).toBe(1)
  })

  test("CLI: incomplete campaign exits non-zero with the reason on stderr", async () => {
    await using tmp = await tmpdir()
    const entries = [forkRep(0), forkRep(1), upstreamRep(2)]
    await writeCampaignDir(tmp.path, p1Doc(), entries)
    await Bun.write(path.join(tmp.path, "campaign.json"), JSON.stringify(doc([...p1DocRuns(), { ...scheduleRun(3, "p1", "upstream"), status: "pending" }])))
    const script = path.join(import.meta.dir, "../script/measure-usage.ts")
    const proc = Bun.spawn([process.execPath, script, "campaign", tmp.path], { cwd: path.join(import.meta.dir, ".."), stdout: "pipe", stderr: "pipe" })
    const [stderr, code] = await Promise.all([new Response(proc.stderr).text(), proc.exited])
    expect(code).toBe(1)
    expect(stderr).toMatch(/incomplete/)
  })

  // Estimator-sourced cold-start attribution (R13-006 + SC-5, ticket 31): the
  // leg's `coldStartAttribution` medians are the breakdown core's own turn-0
  // category totals — never a second estimator/parser path, so the drift test
  // runs both entry points on one fixture (breakdown() directly vs
  // campaignReport over the same run dirs).

  const turn0Of = async (runDir: string) => {
    const run = await loadRun(runDir)
    const { breakdown } = await import("../script/breakdown")
    const turn0 = breakdown({ manifest: run.manifest, captures: run.captures, usage: run.usage }).sessions[0]!.turns[0]!
    return {
      system: turn0.system.total,
      tools: turn0.tools.total,
      history: turn0.history.total,
      adapter: turn0.estimator.adapter,
    }
  }

  test("attribution medians match per-run breakdown turn-0 categories (no recomputation drift); adapter engaged", async () => {
    await using tmp = await tmpdir()
    // Differing turn-0 payloads so the leg median aggregates distinct values.
    const differing = (index: number): RunEntry => ({
      ...forkRep(index),
      captures: [
        {
          sessionID: "ses_a",
          seq: 0,
          file: capture("ses_a", "msg_1", { system: index % 2 === 0 ? ["abcd"] : ["abcd", "efghij"] }),
        },
        { sessionID: "ses_a", seq: 1, file: capture("ses_a", "msg_2") },
      ],
    })
    const entries = [differing(0), differing(1), upstreamRep(2), upstreamRep(3)]
    await writeCampaignDir(tmp.path, p1Doc(), entries)
    const result = await campaignReport(tmp.path)
    const forkLeg = result.phases[0].legs[0]
    const run0 = await turn0Of(path.join(tmp.path, "00-p1-fork"))
    const run1 = await turn0Of(path.join(tmp.path, "01-p1-fork"))
    expect(run0.adapter).toBe("o200k")
    expect(run1.adapter).toBe("o200k")
    expect(forkLeg.coldStartAttribution).toEqual({
      system: { median: (run0.system + run1.system) / 2, min: run0.system, max: run1.system, values: [run0.system, run1.system] },
      tools: { median: (run0.tools + run1.tools) / 2, min: run0.tools, max: run1.tools, values: [run0.tools, run1.tools] },
      history: { median: (run0.history + run1.history) / 2, min: run0.history, max: run1.history, values: [run0.history, run1.history] },
    })
  })

  test("attribution: capture-bearing legs carry the full triple, upstream legs null; absent from pair metrics, derivable per leg", async () => {
    await using tmp = await tmpdir()
    const entries = [forkRep(0), forkRep(1), upstreamRep(2), upstreamRep(3)]
    await writeCampaignDir(tmp.path, p1Doc(), entries)
    const result = await campaignReport(tmp.path)
    const [forkLeg, upstreamLeg] = result.phases[0].legs
    const turn0 = await turn0Of(path.join(tmp.path, "00-p1-fork"))
    expect(forkLeg.coldStartAttribution).toEqual({
      system: { median: turn0.system, min: turn0.system, max: turn0.system, values: [turn0.system, turn0.system] },
      tools: { median: turn0.tools, min: turn0.tools, max: turn0.tools, values: [turn0.tools, turn0.tools] },
      history: { median: turn0.history, min: turn0.history, max: turn0.history, values: [turn0.history, turn0.history] },
    })
    expect(upstreamLeg.coldStartAttribution).toBeNull()
    const comparison = result.phases[0].comparisons[0]
    expect(comparison.kind).toBe("fork-vs-upstream")
    expect("coldStartAttribution" in comparison).toBe(false)
    expect(JSON.stringify(forkLeg)).toContain("coldStartAttribution")
  })

  test("attribution: a capture-less run in a capture-bearing leg → dispersion over the remaining runs + warning naming it", async () => {
    await using tmp = await tmpdir()
    const captureless = { ...forkRep(0), manifest: forkManifest({ capturesDir: null }) }
    const entries = [captureless, forkRep(1), upstreamRep(2), upstreamRep(3)]
    await writeCampaignDir(tmp.path, p1Doc(), entries)
    const result = await campaignReport(tmp.path)
    const turn0 = await turn0Of(path.join(tmp.path, "01-p1-fork"))
    const forkLeg = result.phases[0].legs[0]
    expect(forkLeg.coldStartAttribution).toEqual({
      system: { median: turn0.system, min: turn0.system, max: turn0.system, values: [turn0.system] },
      tools: { median: turn0.tools, min: turn0.tools, max: turn0.tools, values: [turn0.tools] },
      history: { median: turn0.history, min: turn0.history, max: turn0.history, values: [turn0.history] },
    })
    expect(result.warnings).toEqual([
      expect.stringMatching(/payloadChars in leg fork unavailable for run dirs unavailable in 00-p1-fork/),
      expect.stringMatching(/coldStartAttribution in leg fork unavailable for run dirs unavailable in 00-p1-fork/),
    ])
  })
})
