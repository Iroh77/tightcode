#!/usr/bin/env bun

import { Database } from "bun:sqlite"
import { Schema } from "effect"
import fs from "fs/promises"
import path from "path"
import type { CaptureFile } from "../src/session/llm/prompt-capture"

// Measurement script (R10-003/R10-004): offline transforms over a reference-workload
// run dir — provider-reported usage from the run's SQLite step-finish parts
// (authoritative), paired in order with capture dumps; component attribution via
// offline chars/4 estimates proportionally reconciled to usage.input.
// Contracts: ARCHITECTURE/detailed/context-observability.md, decision context-observability-02.

// The manifest is the driver's output (ticket 15); the reader validates this
// exact shape so report/diff inputs stay honest. Harness v2 (ticket 28) adds
// the verdict/identity record as additive optional fields — the v2 driver
// always writes them (null/[] when nothing to record), v1 manifests read as
// null/[]. Contracts: ARCHITECTURE/detailed/comparison-testing.md.
export const RunManifestSchema = Schema.Struct({
  bin: Schema.String,
  modelID: Schema.String,
  providerID: Schema.String,
  promptsDigest: Schema.String,
  capture: Schema.Boolean,
  proxyMode: Schema.Boolean,
  dbPath: Schema.String,
  capturesDir: Schema.NullOr(Schema.String),
  startedAt: Schema.String,
  endedAt: Schema.String,
  pin: Schema.optional(Schema.NullOr(Schema.Literals(["binding", "advisory"]))),
  verdicts: Schema.optional(Schema.Array(Schema.Literals(["binding", "advisory"]))),
  binary: Schema.optional(Schema.NullOr(Schema.String)),
})

export type RunManifest = Schema.Schema.Type<typeof RunManifestSchema>

export type StepFinishRecord = {
  sessionID: string
  tokens: { input: number; output: number; reasoning: number; cacheRead: number; cacheWrite: number }
  cost: number
}

export type CaptureRecord = { file: string; capture: CaptureFile }

export type Run = {
  manifest: RunManifest
  sessions: string[]
  captures: CaptureRecord[]
  usage: StepFinishRecord[]
}

export type UsageTurn = {
  index: number
  captureFile?: string
  usage: { input: number; output: number; reasoning: number; cacheRead: number; cacheWrite: number; cost: number } | null
  estimate: Estimate | null
  reconciled: Reconciled | null
}

type Estimate = { systemTokens: number; toolsTokens: number; historyTokens: number; totalTokens: number }
type Reconciled = { systemTokens: number; toolsTokens: number; historyTokens: number }

export type UsageReport = {
  manifest: RunManifest
  sessions: Array<{
    sessionID: string
    turns: UsageTurn[]
    totals: { input: number; output: number; cost: number }
  }>
}

export type DiffReport = {
  promptsDigest: string
  warnings: string[]
  sessions: Array<{
    index: number
    turns: Array<{
      index: number
      delta: { input: number; output: number; cacheRead: number; cacheWrite: number; cost: number } | null
      reconciled: Reconciled | null
    }>
    totals: { input: number; output: number; cost: number }
  }>
  totals: { input: number; output: number; cost: number }
}

// Offline chars/4 estimate (decision context-observability-02): the estimator is attribution, not
// truth — raw estimate AND the reconciled split are both reported.
const messageChars = (message: { role: string; content: unknown }) =>
  JSON.stringify({ role: message.role, content: message.content })?.length ?? 0

// The single source of the char componentization: the v1 estimator ceil()s
// these into tokens, payloadChars sums them raw (R13-001 "exact char
// accounting" — the two can never drift).
const payloadCharComponents = (payload: CaptureFile["payload"]) => {
  let leading = 0
  while (leading < payload.messages.length && payload.messages[leading].role === "system") leading++
  return {
    systemChars:
      payload.system.join("\n").length +
      payload.messages.slice(0, leading).reduce((sum, message) => sum + messageChars(message), 0),
    toolsChars: Object.entries(payload.tools).reduce(
      (sum, [name, tool]) => sum + name.length + (JSON.stringify(tool.inputSchema)?.length ?? 0) + (tool.description?.length ?? 0),
      0,
    ),
    historyChars: payload.messages.slice(leading).reduce((sum, message) => sum + messageChars(message), 0),
  }
}

const estimateTurn = (payload: CaptureFile["payload"]): Estimate => {
  const { systemChars, toolsChars, historyChars } = payloadCharComponents(payload)
  const systemTokens = Math.ceil(systemChars / 4)
  const toolsTokens = Math.ceil(toolsChars / 4)
  const historyTokens = Math.ceil(historyChars / 4)
  return { systemTokens, toolsTokens, historyTokens, totalTokens: systemTokens + toolsTokens + historyTokens }
}

// Each component scaled by k = input / est.total, residual to history, so the
// split sums to the provider's authoritative input. Cache columns stay out.
const reconcileTurn = (estimate: Estimate, input: number): Reconciled => {
  const k = estimate.totalTokens === 0 ? 0 : input / estimate.totalTokens
  const systemTokens = Math.round(estimate.systemTokens * k)
  const toolsTokens = Math.round(estimate.toolsTokens * k)
  return { systemTokens, toolsTokens, historyTokens: input - systemTokens - toolsTokens }
}

const CAPTURE_SEQ = /^(\d+)\.json$/

const seqOf = (file: string) => {
  const match = CAPTURE_SEQ.exec(path.basename(file))
  return match ? Number(match[1]) : Number.MAX_SAFE_INTEGER
}

const sessionReport = (sessionID: string, usageRows: StepFinishRecord[], captureRecords: CaptureRecord[]): UsageReport["sessions"][number] => {
  const sorted = captureRecords.toSorted((a, b) => seqOf(a.file) - seqOf(b.file))
  const turns: UsageTurn[] = []
  for (let i = 0; i < Math.max(sorted.length, usageRows.length); i++) {
    const captureRecord = sorted[i]
    const usageRecord = usageRows[i]
    const estimate = captureRecord ? estimateTurn(captureRecord.capture.payload) : null
    const usage = usageRecord ? { ...usageRecord.tokens, cost: usageRecord.cost } : null
    turns.push({
      index: i,
      ...(captureRecord ? { captureFile: captureRecord.file } : {}),
      usage,
      estimate,
      reconciled: estimate && usage ? reconcileTurn(estimate, usage.input) : null,
    })
  }
  const totals = usageRows.reduce(
    (sum, record) => ({ input: sum.input + record.tokens.input, output: sum.output + record.tokens.output, cost: sum.cost + record.cost }),
    { input: 0, output: 0, cost: 0 },
  )
  return { sessionID, turns, totals }
}

export const report = (run: Run): UsageReport => {
  const usageBySession = new Map<string, StepFinishRecord[]>()
  for (const record of run.usage) {
    const rows = usageBySession.get(record.sessionID)
    if (rows) rows.push(record)
    else usageBySession.set(record.sessionID, [record])
  }
  const capturesBySession = new Map<string, CaptureRecord[]>()
  for (const record of run.captures) {
    const sessionID = record.capture.meta.sessionID
    const rows = capturesBySession.get(sessionID)
    if (rows) rows.push(record)
    else capturesBySession.set(sessionID, [record])
  }
  // Sessions with records but no row in the DB table keep a slot after the
  // table-ordered sessions — never silently dropped.
  const known = new Set(run.sessions)
  const orphanSessions = [...new Set([...usageBySession.keys(), ...capturesBySession.keys()])]
    .filter((sessionID) => !known.has(sessionID))
    .toSorted()
  const sessions = [...run.sessions, ...orphanSessions].map((sessionID) =>
    sessionReport(sessionID, usageBySession.get(sessionID) ?? [], capturesBySession.get(sessionID) ?? []),
  )
  return { manifest: run.manifest, sessions }
}

const deltaOf = (fork: UsageTurn["usage"], upstream: UsageTurn["usage"]) => {
  if (!fork || !upstream) return null
  return {
    input: fork.input - upstream.input,
    output: fork.output - upstream.output,
    cacheRead: fork.cacheRead - upstream.cacheRead,
    cacheWrite: fork.cacheWrite - upstream.cacheWrite,
    cost: fork.cost - upstream.cost,
  }
}

export const diff = (fork: UsageReport, upstream: UsageReport): DiffReport => {
  if (fork.manifest.promptsDigest !== upstream.manifest.promptsDigest)
    throw new Error(
      `diff refused: promptsDigest mismatch (${fork.manifest.promptsDigest} vs ${upstream.manifest.promptsDigest}) — the runs used different workloads`,
    )
  const warnings: string[] = []
  const sessions: DiffReport["sessions"] = []
  const totals = { input: 0, output: 0, cost: 0 }
  for (let i = 0; i < Math.max(fork.sessions.length, upstream.sessions.length); i++) {
    const forkSession = fork.sessions[i]
    const upstreamSession = upstream.sessions[i]
    if (!forkSession || !upstreamSession) {
      warnings.push(`session index ${i} present only in ${forkSession ? "fork" : "upstream"}`)
      sessions.push({ index: i, turns: [], totals: { input: 0, output: 0, cost: 0 } })
      continue
    }
    const turns: DiffReport["sessions"][number]["turns"] = []
    const sessionTotals = { input: 0, output: 0, cost: 0 }
    for (let j = 0; j < Math.max(forkSession.turns.length, upstreamSession.turns.length); j++) {
      const forkTurn = forkSession.turns[j]
      const upstreamTurn = upstreamSession.turns[j]
      if (!forkTurn || !upstreamTurn) {
        warnings.push(`session ${i} (${forkSession.sessionID}): turn ${j} present only in ${forkTurn ? "fork" : "upstream"} — a loop behaving differently is itself a finding`)
        turns.push({ index: j, delta: null, reconciled: forkTurn?.reconciled ?? null })
        continue
      }
      const delta = deltaOf(forkTurn.usage, upstreamTurn.usage)
      turns.push({ index: j, delta, reconciled: forkTurn.reconciled ?? null })
      if (delta) {
        sessionTotals.input += delta.input
        sessionTotals.output += delta.output
        sessionTotals.cost += delta.cost
      }
    }
    sessions.push({ index: i, turns, totals: sessionTotals })
    totals.input += sessionTotals.input
    totals.output += sessionTotals.output
    totals.cost += sessionTotals.cost
  }
  return { promptsDigest: fork.manifest.promptsDigest, warnings, sessions, totals }
}

// --- cache-immune comparison metrics (R13-001, ticket 28) ---
// Pure functions over a loaded run (comparison-testing.md §Module contracts):
// cache-immune by construction, consumed by the v2 campaign aggregation.

// Distinct meta.verdict values across the run's captures, sorted. [] = none
// observable; ≥2 = mixed (a finding, never averaged away).
export const deriveVerdicts = (captureRecords: CaptureRecord[]): Array<"binding" | "advisory"> =>
  [...new Set(captureRecords.map((record) => record.capture.meta.verdict).filter((v) => v !== undefined))].sort()

// First provider turn's input tokens: the run's first step-finish row — always
// the first row of its session (rowid order), so cache-cold by construction.
// Null when the run has no usage rows.
export const coldStartInput = (run: Run): number | null => run.usage[0]?.tokens.input ?? null

// Σ over ALL step-finish rows of (input + cacheRead + cacheWrite). "Cached
// input" = cacheRead + cacheWrite: re-billing (GLM population lag) moves
// tokens between input and cacheWrite, account warmth (DeepSeek) between
// input and cacheRead — never out of the sum.
export const sessionInput = (run: Run): number =>
  run.usage.reduce((sum, record) => sum + record.tokens.input + record.tokens.cacheRead + record.tokens.cacheWrite, 0)

// Σ over captures of the v1 estimator's exact char accounting without
// ceil(/4) (same componentization as estimateTurn). Null when the run has no
// captures.
export const payloadChars = (run: Run): number | null => {
  if (run.captures.length === 0) return null
  return run.captures.reduce((sum, record) => {
    const { systemChars, toolsChars, historyChars } = payloadCharComponents(record.capture.payload)
    return sum + systemChars + toolsChars + historyChars
  }, 0)
}

// Cache-collapse heuristic flag (R13-001): turn i ≥ 1 flagged when its
// cacheRead collapsed below the ratio × previous turn's total input.
// Diagnostic-only by contract — prompt-base append batches legitimately fire
// it, so it is never a headline metric, never in medians. Rows are the
// step-finish records in loadRun order (session-rowid, contiguous per
// session); flags reset per session.
export const CACHE_COLLAPSE_RATIO = 0.5

export const cacheCollapseFlags = (rows: StepFinishRecord[]): Array<{ turn: number; cacheRead: number; expectedPrefix: number }> => {
  const flags: Array<{ turn: number; cacheRead: number; expectedPrefix: number }> = []
  let sessionID: string | undefined
  let turn = -1
  let prevTotal = 0
  for (const row of rows) {
    if (row.sessionID !== sessionID) {
      sessionID = row.sessionID
      turn = -1
      prevTotal = 0
    }
    turn++
    if (turn >= 1) {
      const expectedPrefix = CACHE_COLLAPSE_RATIO * prevTotal
      if (row.tokens.cacheRead < expectedPrefix) flags.push({ turn, cacheRead: row.tokens.cacheRead, expectedPrefix })
    }
    prevTotal = row.tokens.input + row.tokens.cacheRead + row.tokens.cacheWrite
  }
  return flags
}

// --- run-dir readers (offline tool: manifest/DB problems fail loudly, R00-010) ---

const readManifest = async (runDir: string): Promise<RunManifest> => {
  const file = path.join(runDir, "manifest.json")
  const manifest: unknown = await Bun.file(file)
    .json()
    .catch((error: unknown) => {
      throw new Error(`measure-usage: manifest.json unreadable in run dir ${runDir}: ${error instanceof Error ? error.message : String(error)}`)
    })
  if (!Schema.is(RunManifestSchema)(manifest))
    throw new Error(`measure-usage: manifest.json in ${runDir} is not a valid run manifest`)
  return manifest
}

const isCaptureFile = (value: unknown): value is CaptureFile => {
  if (typeof value !== "object" || value === null) return false
  const candidate = value as {
    meta?: { sessionID?: unknown }
    payload?: { system?: unknown; tools?: unknown; messages?: unknown }
  }
  return (
    typeof candidate.meta?.sessionID === "string" &&
    Array.isArray(candidate.payload?.system) &&
    Array.isArray(candidate.payload?.messages) &&
    typeof candidate.payload?.tools === "object" &&
    candidate.payload?.tools !== null
  )
}

export const readCaptures = async (capturesDir: string, runDir: string): Promise<CaptureRecord[]> => {
  const dirEntries = await fs.readdir(capturesDir, { withFileTypes: true }).catch((error: NodeJS.ErrnoException) => {
    if (error.code === "ENOENT") return null
    throw error
  })
  if (!dirEntries) {
    console.error(`measure-usage: captures dir not found, reporting usage without attribution: ${capturesDir}`)
    return []
  }
  const sessionDirs = dirEntries.filter((entry) => entry.isDirectory()).map((entry) => entry.name).toSorted()
  const records: CaptureRecord[] = []
  for (const sessionID of sessionDirs) {
    const dir = path.join(capturesDir, sessionID)
    const files = (await fs.readdir(dir)).filter((name) => CAPTURE_SEQ.test(name)).toSorted((a, b) => seqOf(a) - seqOf(b))
    for (const name of files) {
      const file = path.join(dir, name)
      const parsed = await Bun.file(file).json().catch(() => null)
      if (!isCaptureFile(parsed)) {
        console.error(`measure-usage: skipping unreadable capture file: ${file}`)
        continue
      }
      records.push({ file: path.relative(runDir, file), capture: parsed })
    }
  }
  return records
}

const readUsage = (dbPath: string): { sessions: string[]; usage: StepFinishRecord[] } => {
  const db = new Database(dbPath, { readonly: true })
  try {
    const sessions = db
      .query<{ id: string }, []>("SELECT id FROM session ORDER BY rowid")
      .all()
      .map((row) => row.id)
    const rows = db
      .query<{ rowid: number; sessionID: string; data: string }, []>(
        `SELECT p.rowid AS rowid, p.session_id AS sessionID, p.data AS data
         FROM part p LEFT JOIN session s ON s.id = p.session_id
         WHERE json_extract(p.data, '$.type') = 'step-finish'
         ORDER BY s.rowid, p.rowid`,
      )
      .all()
    const usage = rows.map((row) => {
      const data = parseStepFinish(row)
      return {
        sessionID: row.sessionID,
        tokens: {
          input: data.tokens.input,
          output: data.tokens.output,
          reasoning: data.tokens.reasoning,
          cacheRead: data.tokens.cache.read,
          cacheWrite: data.tokens.cache.write,
        },
        cost: data.cost,
      }
    })
    return { sessions, usage }
  } finally {
    db.close()
  }
}

const StepFinishDataSchema = Schema.Struct({
  tokens: Schema.Struct({
    input: Schema.Number,
    output: Schema.Number,
    reasoning: Schema.Number,
    cache: Schema.Struct({ read: Schema.Number, write: Schema.Number }),
  }),
  cost: Schema.Number,
})

const parseStepFinish = (row: { rowid: number; data: string }) => {
  let parsed: unknown
  try {
    parsed = JSON.parse(row.data)
  } catch {
    throw new Error(`measure-usage: corrupt step-finish part row (rowid ${row.rowid}) in the run DB`)
  }
  if (!Schema.is(StepFinishDataSchema)(parsed))
    throw new Error(`measure-usage: step-finish part row (rowid ${row.rowid}) does not match the expected usage shape`)
  return parsed
}

export const loadRun = async (runDir: string): Promise<Run> => {
  const manifest = await readManifest(runDir)
  const dbPath = path.resolve(runDir, manifest.dbPath)
  if (!(await Bun.file(dbPath).exists())) throw new Error(`measure-usage: usage DB not found at ${dbPath} (manifest.dbPath)`)
  const read = readUsage(dbPath)
  const captures = manifest.capturesDir ? await readCaptures(path.resolve(runDir, manifest.capturesDir), runDir) : []
  return { manifest, sessions: read.sessions, usage: read.usage, captures }
}

// --- CLI ---

const usageText = () =>
  console.error(
    "usage: bun run script/measure-usage.ts report <runDir> [-o out.json]\n       bun run script/measure-usage.ts diff <forkRunDir> <upstreamRunDir> [-o out.json]",
  )

const emit = async (value: unknown, out: string | undefined) => {
  const json = JSON.stringify(value, null, 2) + "\n"
  if (out) await Bun.write(out, json)
  else process.stdout.write(json)
}

const main = async (argv: string[]): Promise<number> => {
  const cmd = argv[0]
  const positional: string[] = []
  let out: string | undefined
  for (let i = 1; i < argv.length; i++) {
    const arg = argv[i]
    if (arg === "-o") {
      const value = argv[i + 1]
      if (value === undefined) {
        usageText()
        return 1
      }
      out = value
      i++
      continue
    }
    positional.push(arg)
  }
  if (cmd === "report" && positional.length === 1) {
    await emit(report(await loadRun(positional[0])), out)
    return 0
  }
  if (cmd === "diff" && positional.length === 2) {
    const fork = report(await loadRun(positional[0]))
    const upstream = report(await loadRun(positional[1]))
    await emit(diff(fork, upstream), out)
    return 0
  }
  usageText()
  return 1
}

if (import.meta.main) {
  try {
    process.exit(await main(process.argv.slice(2)))
  } catch (error) {
    console.error(error instanceof Error ? error.message : error)
    process.exit(1)
  }
}
