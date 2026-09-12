import { describe, expect, spyOn, test } from "bun:test"
import { Database } from "bun:sqlite"
import fs from "fs/promises"
import path from "path"
import { tmpdir } from "./fixture/fixture"
import { diff, loadRun, report, type RunManifest, type StepFinishRecord, type UsageReport } from "../script/measure-usage"
import type { CaptureFile } from "../src/session/llm/prompt-capture"

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

const capture = (sessionID: string, requestID = "msg_1", payload?: Partial<CaptureFile["payload"]>): CaptureFile => ({
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
