#!/usr/bin/env bun

import { Database } from "bun:sqlite"
import { Schema } from "effect"
import fs from "fs/promises"
import path from "path"
import type { CaptureFile } from "../src/session/llm/prompt-capture"
import type { VerdictProvenance } from "../src/session/binding-verdict"
import type { CampaignSchedule, CampaignSpec } from "./reference-workload"

// Measurement script (R10-003/R10-004): offline transforms over a reference-workload
// run dir — provider-reported usage from the run's SQLite step-finish parts
// (authoritative), paired in order with capture dumps; component attribution via
// offline chars/4 estimates proportionally reconciled to usage.input.
// Contracts: ARCHITECTURE/detailed/context-observability.md, decision context-observability-02.

// The manifest is the driver's output (ticket 15); the reader validates this
// exact shape so report/diff inputs stay honest. Harness v2 (ticket 28) adds
// the verdict/identity record as additive optional fields — the v2 driver
// always writes them (null/[] when nothing to record), v1 manifests read as
// null/[]. R12-013 (ticket 36) adds `verdictProvenances` the same way:
// recorded, never gating (R13-003 gates on the verdict alone); old runs
// without the field stay valid. R12-013 Amendment 1 (ticket 41): the union
// shrinks to the collapsed cascade's origins — pin/static-table/default.
// Contracts: ARCHITECTURE/detailed/comparison-testing.md.
const VerdictProvenanceSchema = Schema.Union([
  Schema.Struct({ origin: Schema.Literal("pin") }),
  Schema.Struct({ origin: Schema.Literal("static-table") }),
  Schema.Struct({ origin: Schema.Literal("default") }),
])

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
  verdictProvenances: Schema.optional(Schema.Array(VerdictProvenanceSchema)),
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

// R12-013 mirror of deriveVerdicts over meta.verdictProvenance: distinct
// sorted JSON values. Manifest-only — never consumed by the gate.
export const deriveVerdictProvenances = (captureRecords: CaptureRecord[]): VerdictProvenance[] => {
  const byJson = new Map(
    captureRecords.flatMap((record) =>
      record.capture.meta.verdictProvenance === undefined
        ? []
        : [[JSON.stringify(record.capture.meta.verdictProvenance), record.capture.meta.verdictProvenance] as const],
    ),
  )
  return [...byJson.keys()].sort().flatMap((json) => {
    const provenance = byJson.get(json)
    return provenance === undefined ? [] : [provenance]
  })
}

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

// Estimator-sourced cold-start attribution (R13-006 + SC-5, ticket 31): per
// run, feature 10's breakdown() core over the run — turn 0 of the first
// session's system/tools/history category totals. Never a second
// estimator/parser path: the figures are the breakdown's own (its
// `estimator.adapter` rides through, so they are interpretable per
// tokenizer). Null when the run has no captures (upstream legs) or no
// projected turn 0. Dynamic import because breakdown.ts consumes this
// module's readers at runtime — a static import would create exactly the
// runtime cycle the type-only reference-workload import above avoids.
export type RunAttribution = { system: number; tools: number; history: number }

export const runAttribution = async (run: Run): Promise<RunAttribution | null> => {
  if (run.captures.length === 0) return null
  const { breakdown } = await import("./breakdown")
  const turn0 = breakdown({ manifest: run.manifest, captures: run.captures, usage: run.usage }).sessions[0]?.turns[0]
  return turn0 ? { system: turn0.system.total, tools: turn0.tools.total, history: turn0.history.total } : null
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
    "usage: bun run script/measure-usage.ts report <runDir> [-o out.json]\n       bun run script/measure-usage.ts diff <forkRunDir> <upstreamRunDir> [-o out.json]\n       bun run script/measure-usage.ts campaign <campaignDir> [-o out.json]",
  )

const emit = async (value: unknown, out: string | undefined) => {
  const json = JSON.stringify(value, null, 2) + "\n"
  if (out) await Bun.write(out, json)
  else process.stdout.write(json)
}

// --- campaign aggregation (R13-002/003/006, ticket 30) ---
// ComparisonReport v1 — same versioning discipline as Breakdown JSON (R10-006):
// a schema field from birth, stable key vocabulary, tolerant additive
// evolution. Contracts: ARCHITECTURE/detailed/comparison-testing.md.

// Type-only: the runner imports this module at runtime, so the dependency
// must not circle back (CampaignSpec/CampaignSchedule are erased here).
export type Dispersion = { median: number; min: number; max: number; values: number[] }

export type LegReport = {
  shape: "fork" | "upstream"
  proxy: boolean
  runs: Array<{ runDir: string; verdicts: string[]; binary: string | null }>
  verdict: "binding" | "advisory" | null // the leg's uniform recorded verdict; null = none recorded
  verdictMixed: boolean // runs disagree, or any run carried ≥2 verdicts
  metrics: {
    coldStartInput: Dispersion | null
    sessionInput: Dispersion | null
    payloadChars: Dispersion | null
  }
  coldStartAttribution: { system: Dispersion; tools: Dispersion; history: Dispersion } | null // SC-5 (ticket 31): breakdown-core turn-0 category medians; null on capture-less legs
  diagnostics: { cacheCollapses: Array<{ runDir: string; turn: number; cacheRead: number; expectedPrefix: number }> }
}

export type ComparisonPair = {
  kind: "fork-vs-upstream" | "fork-vs-fork-proxy"
  comparable: boolean // fork side uniform AND recorded (decision comparison-testing-01 §3)
  verdict: "binding" | "advisory" | null // the fork side's verdict column (R13-003/006)
  metrics: {
    coldStartInput: { fork: number; other: number; delta: number } | null
    sessionInput: { fork: number; other: number; delta: number } | null
    payloadChars: { fork: number; other: number; delta: number } | null // fork-vs-fork-proxy only
  }
}

export type ComparisonReport = {
  schema: 1
  campaign: {
    outDir: string
    spec: CampaignSpec
    seed: number
    schedule: CampaignSchedule["runs"]
    startedAt: string
    endedAt: string
  }
  phases: Array<{ name: string; model: string; legs: LegReport[]; comparisons: ComparisonPair[] }>
  warnings: string[]
}

// Tolerant-additive reader for the campaign.json the runner writes (ticket 29):
// the fields the aggregation consumes are validated, the spec echo is carried
// through as parsed.
const CampaignDocSchema = Schema.Struct({
  spec: Schema.Struct({
    phases: Schema.Array(
      Schema.Struct({
        name: Schema.String,
        model: Schema.String,
        mcp: Schema.optional(Schema.Unknown),
      }),
    ),
    legs: Schema.Array(
      Schema.Struct({
        shape: Schema.Literals(["fork", "upstream"]),
        bin: Schema.String,
        proxy: Schema.optional(Schema.Boolean),
      }),
    ),
    runs: Schema.Number,
    seed: Schema.optional(Schema.Number),
    pin: Schema.optional(Schema.Literals(["binding", "advisory"])),
  }),
  seed: Schema.Number,
  startedAt: Schema.String,
  endedAt: Schema.NullOr(Schema.String),
  runs: Schema.Array(
    Schema.Struct({
      index: Schema.Number,
      phase: Schema.String,
      shape: Schema.Literals(["fork", "upstream"]),
      proxy: Schema.Boolean,
      rep: Schema.Number,
      runDir: Schema.String,
      status: Schema.Literals(["done", "failed", "pending"]),
      error: Schema.optional(Schema.String),
    }),
  ),
})

type CampaignDoc = Schema.Schema.Type<typeof CampaignDocSchema>

const readCampaign = async (campaignDir: string): Promise<CampaignDoc> => {
  const doc: unknown = await Bun.file(path.join(campaignDir, "campaign.json"))
    .json()
    .catch((error: unknown) => {
      const cause = error instanceof Error ? error.message : String(error)
      throw new Error(`measure-usage: campaign.json unreadable in ${campaignDir}: ${cause}`)
    })
  if (!Schema.is(CampaignDocSchema)(doc))
    throw new Error(`measure-usage: campaign.json in ${campaignDir} is not a valid campaign document`)
  return doc
}

// Median with dispersion over the leg's run values. Even counts average the
// two middle values; `values` keeps the raw run-order list so the median is
// never the only witness.
const dispersion = (values: number[]): Dispersion => {
  const sorted = values.toSorted((a, b) => a - b)
  const mid = Math.floor(sorted.length / 2)
  const median = sorted.length % 2 === 1 ? sorted[mid] : (sorted[mid - 1] + sorted[mid]) / 2
  return { median, min: sorted[0], max: sorted[sorted.length - 1], values }
}

// Availability rule shared by every leg column: no warning when all runs (or
// none) provide a value; partial availability is a warning naming the runs —
// surfaced, never silently folded (R00-010).
const availableValues = <T>(runDirs: string[], values: Array<T | null>, warnings: string[], label: string): T[] => {
  const present = values.filter((value): value is T => value !== null)
  if (present.length === 0 || present.length === values.length) return present
  const missing = runDirs.filter((_, i) => values[i] === null)
  warnings.push(`measure-usage: ${label} unavailable in ${missing.join(", ")} — dispersion over the remaining runs`)
  return present
}

// One metric column over a leg's runs: null when no run provides it.
const legMetric = (
  runDirs: string[],
  values: Array<number | null>,
  warnings: string[],
  label: string,
): Dispersion | null => {
  const present = availableValues(runDirs, values, warnings, label)
  if (present.length === 0) return null
  return dispersion(present)
}

// One attribution column over a leg's runs (same availability rule as
// legMetric): null when no run provides one.
const legAttribution = (
  runDirs: string[],
  values: Array<RunAttribution | null>,
  warnings: string[],
  label: string,
): LegReport["coldStartAttribution"] => {
  const present = availableValues(runDirs, values, warnings, label)
  if (present.length === 0) return null
  const column = (key: keyof RunAttribution) => dispersion(present.map((value) => value[key]))
  return { system: column("system"), tools: column("tools"), history: column("history") }
}

const legReport = (
  shape: "fork" | "upstream",
  proxy: boolean,
  loaded: Array<{ runDir: string; verdicts: string[]; binary: string | null; attribution: RunAttribution | null; run: Run }>,
  warnings: string[],
): LegReport => {
  const runDirs = loaded.map((entry) => entry.runDir)
  const metrics = {
    coldStartInput: legMetric(runDirs, loaded.map((entry) => coldStartInput(entry.run)), warnings, `coldStartInput in leg ${shape}${proxy ? "-proxy" : ""} unavailable for run dirs`),
    sessionInput: legMetric(runDirs, loaded.map((entry) => sessionInput(entry.run)), warnings, `sessionInput in leg ${shape}${proxy ? "-proxy" : ""} unavailable for run dirs`),
    payloadChars: legMetric(runDirs, loaded.map((entry) => payloadChars(entry.run)), warnings, `payloadChars in leg ${shape}${proxy ? "-proxy" : ""} unavailable for run dirs`),
  }
  const runs = loaded.map((entry) => ({ runDir: entry.runDir, verdicts: entry.verdicts, binary: entry.binary }))
  // R13-003 (comparison-testing-01 §3): the leg's verdict column is its
  // uniform recorded verdict — every run exactly one identical verdict;
  // mixed (≥2 within a run) and flips (disagreement across runs) surface as
  // verdictMixed, unrecordable ([]) as null without mixed.
  const { unrecorded, mixed, byVerdict } = verdictBreakdown(runs)
  const verdict =
    byVerdict.size === 1 && mixed.length === 0 && unrecorded.length === 0
      ? ([...byVerdict.keys()][0] as "binding" | "advisory")
      : null
  const diagnostics = loaded.flatMap((entry) =>
    cacheCollapseFlags(entry.run.usage).map((flag) => ({ runDir: entry.runDir, ...flag })),
  )
  return {
    shape,
    proxy,
    runs,
    verdict,
    verdictMixed: byVerdict.size > 1 || mixed.length > 0,
    metrics,
    coldStartAttribution: legAttribution(
      runDirs,
      loaded.map((entry) => entry.attribution),
      warnings,
      `coldStartAttribution in leg ${shape}${proxy ? "-proxy" : ""} unavailable for run dirs`,
    ),
    diagnostics: { cacheCollapses: diagnostics },
  }
}

// The verdict-shape analysis both the leg's verdict column and the fork-side
// gate read (R13-003): single = exactly one recorded verdict per run,
// unrecorded = [] runs, mixed = ≥2-verdict runs, byVerdict groups the single
// runs' run dirs per distinct verdict.
const verdictBreakdown = (runs: LegReport["runs"]) => {
  const single = runs.filter((entry) => entry.verdicts.length === 1)
  const unrecorded = runs.filter((entry) => entry.verdicts.length === 0)
  const mixed = runs.filter((entry) => entry.verdicts.length >= 2)
  const byVerdict = new Map<string, string[]>()
  for (const entry of single) {
    const dirs = byVerdict.get(entry.verdicts[0]) ?? []
    dirs.push(entry.runDir)
    byVerdict.set(entry.verdicts[0], dirs)
  }
  return { unrecorded, mixed, byVerdict }
}

// Fork-side verdict gate (R13-003): comparable only when uniform AND
// recorded; upstream/proxy sides are exempt (no lazy branch in their
// payloads). Every failure mode gets a warning naming the run dirs —
// surfaced, never silently folded into a delta.
const forkGate = (forkLeg: LegReport): { comparable: boolean; warnings: string[] } => {
  const warnings: string[] = []
  const { unrecorded, mixed, byVerdict } = verdictBreakdown(forkLeg.runs)
  if (mixed.length > 0)
    warnings.push(
      `measure-usage: fork verdicts mixed in ${mixed.map((entry) => entry.runDir).join(", ")} (${mixed
        .flatMap((entry) => entry.verdicts)
        .join(", ")}) — comparison refused (R13-003)`,
    )
  if (unrecorded.length > 0)
    warnings.push(
      `measure-usage: no recorded verdict in ${unrecorded.map((entry) => entry.runDir).join(", ")} — mechanism identity unverifiable, comparison refused (R13-003)`,
    )
  if (byVerdict.size > 1)
    warnings.push(
      `measure-usage: fork verdict flip across runs — ${[...byVerdict]
        .map(([verdict, dirs]) => `${verdict} in ${dirs.join(", ")}`)
        .join(" vs ")} — comparison refused (R13-003)`,
    )
  return { comparable: warnings.length === 0, warnings }
}

const pairMetrics = (
  forkLeg: LegReport,
  otherLeg: LegReport,
  includePayload: boolean,
): ComparisonPair["metrics"] => {
  const delta = (fork: Dispersion | null, other: Dispersion | null) =>
    fork && other ? { fork: fork.median, other: other.median, delta: fork.median - other.median } : null
  return {
    coldStartInput: delta(forkLeg.metrics.coldStartInput, otherLeg.metrics.coldStartInput),
    sessionInput: delta(forkLeg.metrics.sessionInput, otherLeg.metrics.sessionInput),
    payloadChars: includePayload ? delta(forkLeg.metrics.payloadChars, otherLeg.metrics.payloadChars) : null,
  }
}

export const campaignReport = async (campaignDir: string): Promise<ComparisonReport> => {
  const campaign = await readCampaign(campaignDir)
  // Incomplete campaign = operator error (R00-010): refuse loudly, never
  // aggregate a poisoned schedule.
  const notDone = campaign.runs.filter((run) => run.status !== "done")
  if (notDone.length > 0 || campaign.endedAt === null) {
    const detail =
      notDone.length > 0
        ? `${notDone.length} run(s) not done (${notDone.map((run) => `${run.runDir}: ${run.status}${run.error ? ` ${run.error}` : ""}`).join(", ")})`
        : "endedAt is missing"
    throw new Error(`measure-usage: campaign ${campaignDir} is incomplete — ${detail}`)
  }
  const phaseNames = campaign.spec.phases.map((phase) => phase.name)
  for (const run of campaign.runs) {
    if (!phaseNames.includes(run.phase))
      throw new Error(`measure-usage: campaign schedule run ${run.runDir} references unknown phase ${JSON.stringify(run.phase)}`)
  }
  const warnings: string[] = []
  const loaded = await Promise.all(
    campaign.runs.map(async (schedule) => {
      const run = await loadRun(path.resolve(campaignDir, schedule.runDir))
      return {
        schedule,
        runDir: schedule.runDir,
        run,
        verdicts: [...(run.manifest.verdicts ?? [])],
        binary: run.manifest.binary ?? null,
        attribution: await runAttribution(run),
      }
    }),
  )
  const phases = campaign.spec.phases.map((phase) => {
    // Legs grouped by (shape, proxy) in schedule first-appearance order.
    const phaseRuns = loaded.filter((entry) => entry.schedule.phase === phase.name)
    const legOrder: Array<{ shape: "fork" | "upstream"; proxy: boolean }> = []
    const byLeg = new Map<string, typeof phaseRuns>()
    for (const entry of phaseRuns) {
      const key = `${entry.schedule.shape}:${entry.schedule.proxy}`
      const group = byLeg.get(key)
      if (group) group.push(entry)
      else {
        byLeg.set(key, [entry])
        legOrder.push({ shape: entry.schedule.shape, proxy: entry.schedule.proxy })
      }
    }
    const legs = legOrder.map((leg) => legReport(leg.shape, leg.proxy, byLeg.get(`${leg.shape}:${leg.proxy}`) ?? [], warnings))
    const forkLeg = legs.find((leg) => leg.shape === "fork" && !leg.proxy)
    const gate = forkLeg ? forkGate(forkLeg) : null
    if (gate) warnings.push(...gate.warnings)
    const comparisons: ComparisonPair[] = []
    const otherLeg = legs.find((leg) => leg.shape === "upstream" && !leg.proxy)
    if (forkLeg && otherLeg && gate)
      comparisons.push({
        kind: "fork-vs-upstream",
        comparable: gate.comparable,
        verdict: forkLeg.verdict,
        metrics: pairMetrics(forkLeg, otherLeg, false),
      })
    const proxyLeg = legs.find((leg) => leg.shape === "fork" && leg.proxy)
    if (forkLeg && proxyLeg && gate)
      comparisons.push({
        kind: "fork-vs-fork-proxy",
        comparable: gate.comparable,
        verdict: forkLeg.verdict,
        metrics: pairMetrics(forkLeg, proxyLeg, true),
      })
    return { name: phase.name, model: phase.model, legs, comparisons }
  })
  return {
    schema: 1,
    campaign: {
      outDir: campaignDir,
      // Effect Schema's Type is readonly; the report contract is the mutable
      // CampaignSpec — spread into fresh arrays at the echo boundary.
      spec: { ...campaign.spec, phases: [...campaign.spec.phases], legs: [...campaign.spec.legs] },
      seed: campaign.seed,
      schedule: [...campaign.runs],
      startedAt: campaign.startedAt,
      endedAt: campaign.endedAt,
    },
    phases,
    warnings,
  }
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
  if (cmd === "campaign" && positional.length === 1) {
    await emit(await campaignReport(positional[0]), out)
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
