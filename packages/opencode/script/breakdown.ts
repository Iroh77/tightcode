import type { CaptureFile } from "../src/session/llm/prompt-capture"
import { forModel, parseSkillsListing, parseSystemBlocks, type AdapterKind } from "./estimator"
import type { CaptureRecord, RunManifest, StepFinishRecord } from "./measure-usage"

// Breakdown core (R10-006, ticket 24): taxonomy projection over capture dumps
// → Breakdown JSON v1. Pure transform — reuses measure-usage's readers and its
// pairing convention (captures seq-ordered order-zipped with step-finish rows,
// unmatched reported, corrupt capture skipped upstream in readCaptures). The
// MCP cross-cut view is a grouping dimension only: it never enters category
// totals. `inputEstimate` is unreconciled — attribution by real vocabulary;
// the chars/4 reconciliation stays measure-usage's job (context-observability-03 §5).
// Contracts: ARCHITECTURE/detailed/context-observability.md round 2.

export type BlockRow = { key: string; tokens: number; content?: string }
export type ToolRow = {
  name: string
  server: string | null
  tokens: number
  content?: { description?: string; inputSchema?: unknown }
}
export type HistoryRow = { index: number; role: string; tokens: number; content?: string }
export type SkillRow = { name: string; tokens: number; content?: string }

export type TurnBreakdown = {
  index: number
  captureFile: string
  requestID: string
  small: boolean
  estimator: { modelID: string; adapter: AdapterKind }
  system: { blocks: BlockRow[]; mcpInstructionsTags: number | null; total: number }
  skills: { total: number; headers: number; perSkill: SkillRow[] } // sub-view of system.skills
  tools: { entries: ToolRow[]; total: number }
  history: { messages: HistoryRow[]; total: number }
  mcp: {
    servers: Array<{ name: string; instructionsBlock: number; catalogBlock: number | null; tools: Array<{ name: string; tokens: number }>; total: number }>
    unattributed: Array<{ name: string; tokens: number }>
    instructionsTags: number // mirrors system.mcpInstructionsTags
    total: number // Σ servers + instructionsTags
  }
  output: number | null // paired step-finish usage.output (authoritative)
  inputEstimate: number // system.total + tools.total + history.total — unreconciled
  inputReported: number | null
}

// Design leaves SessionTotals open; shape mirrors the per-turn categories the
// test scripts and the future Desktop view aggregate over.
export type SessionTotals = {
  system: number
  tools: number
  history: number
  inputEstimate: number
  output: number | null
  inputReported: number | null
}

export type Breakdown = {
  schema: 1
  run: { modelID: string; providerID: string; modelApi: string; generatedAt: string }
  sessions: Array<{ sessionID: string; turns: TurnBreakdown[]; totals: SessionTotals }>
  totals: SessionTotals
}

// Round-1 char convention, now per component with the real vocabulary.
const toolTokens = (est: { estimate: (text: string) => number }, name: string, entry: CaptureFile["payload"]["tools"][string]) =>
  est.estimate(name) +
  est.estimate(JSON.stringify(entry.inputSchema) ?? "") +
  est.estimate(entry.description ?? "")

const messageTokens = (est: { estimate: (text: string) => number }, message: { role: string; content: unknown }) =>
  est.estimate(JSON.stringify({ role: message.role, content: message.content }) ?? "")

const CATALOG_BULLET = /^- (.+)$/

// Wrapper-session deferred tools appear only as catalog bullets: `- name: desc`
// or `- name` (the producer's two bullet shapes, one line each).
const catalogBullets = (blockText: string): Array<{ name: string; text: string }> => {
  const bullets: Array<{ name: string; text: string }> = []
  for (const line of blockText.split("\n")) {
    const match = CATALOG_BULLET.exec(line)
    if (!match) continue
    const name = match[1].split(":")[0].trim()
    if (name !== "") bullets.push({ name, text: line })
  }
  return bullets
}

const projectTurn = (index: number, record: CaptureRecord, usageRow: StepFinishRecord | null): TurnBreakdown => {
  const capture = record.capture
  const est = forModel(capture.meta.modelID)
  const parsed = parseSystemBlocks(capture.payload.system)

  const blocks: BlockRow[] = []
  const addBlockTokens = (key: string, tokens: number) => {
    const existing = blocks.find((block) => block.key === key)
    if (existing) existing.tokens += tokens
    else blocks.push({ key, tokens })
  }
  let mcpInstructionsTags: number | null = null
  const mcpBlockTokens = new Map<string, number>()
  const catalogBlockTokens = new Map<string | undefined, number>()
  const catalogBlockTexts = new Map<string, string>()
  let skillsText: string | undefined
  let systemTotal = 0

  for (const segment of parsed.segments) {
    const tokens = est.estimate(segment.text)
    systemTotal += tokens
    if (segment.label === "mcpInstructionsTags") {
      mcpInstructionsTags = (mcpInstructionsTags ?? 0) + tokens
    } else if ("server" in segment) {
      mcpBlockTokens.set(segment.server, (mcpBlockTokens.get(segment.server) ?? 0) + tokens)
      addBlockTokens(`mcp:${segment.server}`, tokens)
    } else if (segment.label === "skills") {
      skillsText = segment.text
      addBlockTokens("skills", tokens)
    } else if (segment.label === "catalog" || segment.label.startsWith("catalog:")) {
      const server = segment.label === "catalog" ? undefined : segment.label.slice("catalog:".length)
      catalogBlockTokens.set(server, (catalogBlockTokens.get(server) ?? 0) + tokens)
      addBlockTokens(segment.label, tokens)
      if (server !== undefined) catalogBlockTexts.set(server, segment.text)
    } else {
      addBlockTokens(segment.label, tokens)
    }
  }

  const skills = skillsText
    ? (() => {
        const listing = parseSkillsListing(skillsText)
        return {
          total: est.estimate(skillsText),
          headers: est.estimate(listing.headers),
          perSkill: listing.items.map((item) => ({ name: item.name, tokens: est.estimate(item.text) })),
        }
      })()
    : { total: 0, headers: 0, perSkill: [] as SkillRow[] }

  // Attribution: the capture meta map when present; old captures (no
  // toolServers) fall back to the scoped prefix rule — longest `server_`
  // prefix against the servers visible in the same capture (mcp:* and
  // catalog:<server> blocks); else unattributed. Grouping-only either way.
  const visibleServers = [...mcpBlockTokens.keys(), ...catalogBlockTokens.keys()].filter(
    (server): server is string => server !== undefined,
  )
  const attributeServer = (name: string): string | null => {
    const matches = visibleServers.filter((server) => name.startsWith(`${server}_`))
    return matches.length > 0 ? matches.toSorted((a, b) => b.length - a.length)[0] : null
  }
  const toolServers = capture.meta.toolServers
  const entries: ToolRow[] = Object.entries(capture.payload.tools).map(([name, entry]) => ({
    name,
    server: toolServers ? toolServers[name] ?? null : attributeServer(name),
    tokens: toolTokens(est, name, entry),
  }))
  const toolsTotal = entries.reduce((sum, entry) => sum + entry.tokens, 0)

  // Round-1 convention: leading role:"system" messages count as system and
  // stay outside the history rows.
  let leading = 0
  while (leading < capture.payload.messages.length && capture.payload.messages[leading].role === "system") leading++
  const historyMessages = capture.payload.messages.slice(leading).map((message, offset) => ({
    index: leading + offset,
    role: message.role,
    tokens: messageTokens(est, message),
  }))
  const historyTotal = historyMessages.reduce((sum, message) => sum + message.tokens, 0)

  const serverNames = new Set<string>([
    ...mcpBlockTokens.keys(),
    ...catalogBlockTokens.keys().filter((server): server is string => server !== undefined),
    ...(toolServers ? Object.values(toolServers) : []),
  ])
  const servers = [...serverNames].sort().map((name) => {
    const instructionsBlock = mcpBlockTokens.get(name) ?? 0
    const catalog = catalogBlockTokens.get(name)
    const recorded = entries.filter((entry) => entry.server === name)
    const seen = new Set(recorded.map((entry) => entry.name))
    const bulletRows = (catalogBlockTexts.get(name) ? catalogBullets(catalogBlockTexts.get(name)!) : [])
      .filter((bullet) => !seen.has(bullet.name))
      .map((bullet) => ({ name: bullet.name, tokens: est.estimate(bullet.text) }))
    const serverTools = [
      ...recorded.map((entry) => ({ name: entry.name, tokens: entry.tokens })),
      ...bulletRows,
    ]
    return {
      name,
      instructionsBlock,
      catalogBlock: catalog ?? null,
      tools: serverTools,
      total: instructionsBlock + (catalog ?? 0) + serverTools.reduce((sum, entry) => sum + entry.tokens, 0),
    }
  })
  const unattributed = entries.filter((entry) => entry.server === null).map((entry) => ({ name: entry.name, tokens: entry.tokens }))
  const mcpTotal = servers.reduce((sum, server) => sum + server.total, 0) + (mcpInstructionsTags ?? 0)

  return {
    index,
    captureFile: record.file,
    requestID: capture.meta.requestID,
    small: capture.meta.small,
    estimator: { modelID: est.modelID, adapter: est.adapter },
    system: { blocks, mcpInstructionsTags, total: systemTotal },
    skills,
    tools: { entries, total: toolsTotal },
    history: { messages: historyMessages, total: historyTotal },
    mcp: { servers, unattributed, instructionsTags: mcpInstructionsTags ?? 0, total: mcpTotal },
    output: usageRow ? usageRow.tokens.output : null,
    inputEstimate: systemTotal + toolsTotal + historyTotal,
    inputReported: usageRow ? usageRow.tokens.input : null,
  }
}

const ZERO_TOTALS: SessionTotals = { system: 0, tools: 0, history: 0, inputEstimate: 0, output: null, inputReported: null }

const turnTotals = (turn: TurnBreakdown): SessionTotals => ({
  system: turn.system.total,
  tools: turn.tools.total,
  history: turn.history.total,
  inputEstimate: turn.inputEstimate,
  output: turn.output,
  inputReported: turn.inputReported,
})

// Nulls propagate: output/inputReported stay null while no folded row has one.
const mergeTotals = (a: SessionTotals, b: SessionTotals): SessionTotals => ({
  system: a.system + b.system,
  tools: a.tools + b.tools,
  history: a.history + b.history,
  inputEstimate: a.inputEstimate + b.inputEstimate,
  output: b.output === null ? a.output : (a.output ?? 0) + b.output,
  inputReported: b.inputReported === null ? a.inputReported : (a.inputReported ?? 0) + b.inputReported,
})

const groupBy = <T>(items: T[], key: (item: T) => string): Map<string, T[]> => {
  const map = new Map<string, T[]>()
  for (const item of items) {
    const rows = map.get(key(item))
    if (rows) rows.push(item)
    else map.set(key(item), [item])
  }
  return map
}

const CAPTURE_SEQ = /(\d+)\.json$/

const seqOf = (file: string) => {
  const match = CAPTURE_SEQ.exec(file)
  return match ? Number(match[1]) : Number.MAX_SAFE_INTEGER
}

export const breakdown = (input: { manifest: RunManifest; captures: CaptureRecord[]; usage: StepFinishRecord[] }): Breakdown => {
  const usageBySession = groupBy(input.usage, (record) => record.sessionID)
  const capturesBySession = groupBy(input.captures, (record) => record.capture.meta.sessionID)
  // Usage rows arrive in (session rowid, part rowid) order from loadRun, so
  // first appearance preserves the table order; capture-only sessions follow.
  const sessionIDs = [...usageBySession.keys()]
  for (const sessionID of [...capturesBySession.keys()].sort()) {
    if (!usageBySession.has(sessionID)) sessionIDs.push(sessionID)
  }

  const sessions = sessionIDs.map((sessionID) => {
    const usageRows = usageBySession.get(sessionID) ?? []
    const records = (capturesBySession.get(sessionID) ?? []).toSorted((a, b) => seqOf(a.file) - seqOf(b.file))
    const turns: TurnBreakdown[] = []
    for (let i = 0; i < Math.max(records.length, usageRows.length); i++) {
      const record = records[i]
      const usageRow = usageRows[i]
      if (!record) {
        // A usage row without a capture has no taxonomy to project — reported
        // loudly, never folded into a turn (pairing convention, R00-010).
        console.error(`breakdown: unpaired step-finish usage row (turn ${i}) in session ${sessionID} has no capture — omitted`)
        continue
      }
      turns.push(projectTurn(i, record, usageRow ?? null))
    }
    return { sessionID, turns, totals: turns.map(turnTotals).reduce(mergeTotals, ZERO_TOTALS) }
  })

  return {
    schema: 1,
    run: {
      modelID: input.manifest.modelID,
      providerID: input.manifest.providerID,
      modelApi: input.captures[0]?.capture.meta.modelApi ?? "",
      generatedAt: new Date().toISOString(),
    },
    sessions,
    totals: sessions.map((session) => session.totals).reduce(mergeTotals, ZERO_TOTALS),
  }
}

export * as Breakdown from "./breakdown"
