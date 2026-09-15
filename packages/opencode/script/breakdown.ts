import type { CaptureFile } from "../src/session/llm/prompt-capture"
import { forModel, parseSkillsListing, parseSystemBlocks, type AdapterKind } from "./estimator"
import { loadRun, type CaptureRecord, type RunManifest, type StepFinishRecord } from "./measure-usage"

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

// R10-009 drill-down: selection lives in the core (the projected JSON carries
// no raw texts, so the CLI cannot select alone) — the CLI only parses `--content`
// and passes the keys through. `content` rides the selected nodes only.
type ContentSelection = {
  system: Set<string>
  skills: Set<string>
  tools: Set<string>
  history: Set<number>
}

const parseContentKeys = (keys: readonly string[]): ContentSelection => {
  const selection: ContentSelection = { system: new Set(), skills: new Set(), tools: new Set(), history: new Set() }
  for (const key of keys) {
    const dot = key.indexOf(".")
    const namespace = dot === -1 ? "" : key.slice(0, dot)
    const rest = dot === -1 ? "" : key.slice(dot + 1)
    if (namespace === "system") {
      if (rest.startsWith("skills:")) selection.skills.add(rest.slice("skills:".length))
      else selection.system.add(rest)
    } else if (namespace === "tools") {
      selection.tools.add(rest)
    } else if (namespace === "history") {
      const index = Number(rest)
      if (rest === "" || !Number.isInteger(index) || index < 0)
        throw new Error(`breakdown: invalid --content key "${key}" (history.<index>)`)
      selection.history.add(index)
    } else {
      throw new Error(
        `breakdown: invalid --content key "${key}" (vocabulary: system.<block>, system.skills:<name>, tools.<name>, history.<index>)`,
      )
    }
  }
  return selection
}

const projectTurn = (index: number, record: CaptureRecord, usageRow: StepFinishRecord | null, select: ContentSelection | undefined): TurnBreakdown => {
  const capture = record.capture
  const est = forModel(capture.meta.modelID)
  const parsed = parseSystemBlocks(capture.payload.system)

  const blocks: BlockRow[] = []
  const blockTexts = new Map<string, string[]>()
  const addBlockTokens = (key: string, tokens: number, text: string) => {
    const texts = blockTexts.get(key)
    if (texts) texts.push(text)
    else blockTexts.set(key, [text])
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
      addBlockTokens(`mcp:${segment.server}`, tokens, segment.text)
    } else if (segment.label === "skills") {
      skillsText = segment.text
      addBlockTokens("skills", tokens, segment.text)
    } else if (segment.label === "catalog" || segment.label.startsWith("catalog:")) {
      const server = segment.label === "catalog" ? undefined : segment.label.slice("catalog:".length)
      catalogBlockTokens.set(server, (catalogBlockTokens.get(server) ?? 0) + tokens)
      addBlockTokens(segment.label, tokens, segment.text)
      if (server !== undefined) catalogBlockTexts.set(server, segment.text)
    } else {
      addBlockTokens(segment.label, tokens, segment.text)
    }
  }

  const skills = skillsText
    ? (() => {
        const listing = parseSkillsListing(skillsText)
        return {
          total: est.estimate(skillsText),
          headers: est.estimate(listing.headers),
          perSkill: listing.items.map((item) => ({
            name: item.name,
            tokens: est.estimate(item.text),
            ...(select?.skills.has(item.name) ? { content: item.text } : {}),
          })),
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
    ...(select?.tools.has(name) ? { content: { description: entry.description, inputSchema: entry.inputSchema } } : {}),
  }))
  const toolsTotal = entries.reduce((sum, entry) => sum + entry.tokens, 0)

  // Round-1 convention: leading role:"system" messages count as system and
  // stay outside the history rows.
  let leading = 0
  while (leading < capture.payload.messages.length && capture.payload.messages[leading].role === "system") leading++
  const historyMessages = capture.payload.messages.slice(leading).map((message, offset) => {
    const index = leading + offset
    return {
      index,
      role: message.role,
      tokens: messageTokens(est, message),
      ...(select?.history.has(index)
        ? { content: typeof message.content === "string" ? message.content : (JSON.stringify(message.content) ?? "") }
        : {}),
    }
  })
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

  const systemBlocks: BlockRow[] = blocks.map((block) => {
    const text = blockTexts.get(block.key)
    if (!select?.system.has(block.key) || !text) return block
    return { ...block, content: text.join("\n") }
  })

  return {
    index,
    captureFile: record.file,
    requestID: capture.meta.requestID,
    small: capture.meta.small,
    estimator: { modelID: est.modelID, adapter: est.adapter },
    system: { blocks: systemBlocks, mcpInstructionsTags, total: systemTotal },
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

export const breakdown = (
  input: { manifest: RunManifest; captures: CaptureRecord[]; usage: StepFinishRecord[] },
  options?: { content?: readonly string[] },
): Breakdown => {
  const contentKeys = options?.content
  const select = contentKeys ? parseContentKeys(contentKeys) : undefined
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
      turns.push(projectTurn(i, record, usageRow ?? null, select))
    }
    return { sessionID, turns, totals: turns.map(turnTotals).reduce(mergeTotals, ZERO_TOTALS) }
  })

  if (contentKeys && select) {
    const available = new Set<string>()
    for (const session of sessions)
      for (const turn of session.turns) {
        for (const block of turn.system.blocks) available.add(`system.${block.key}`)
        for (const skill of turn.skills.perSkill) available.add(`system.skills:${skill.name}`)
        for (const entry of turn.tools.entries) available.add(`tools.${entry.name}`)
        for (const message of turn.history.messages) available.add(`history.${message.index}`)
      }
    const unknown = contentKeys.filter((key) => !available.has(key))
    if (unknown.length > 0)
      throw new Error(
        `breakdown: unknown --content key(s) ${unknown.join(", ")} — available keys: ${[...available].sort().join(", ")}`,
      )
  }

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

// --- CLI (R10-006 command surface, R10-009 drill-down, ticket 25) ---
// bun run script/breakdown.ts <runDir> [--session <id>] [--turn <n>] [--content <key>...] [--human] [-o out.json]
// Counts-only JSON v1 by default; `content` fields only on explicit --content
// selection. --human is a stdout projection (counts + % of inputEstimate),
// never a second format of record — mutually exclusive with -o. --session/
// --turn are reader-side selections: totals are recomputed over what remains,
// sessions left without turns by --turn are dropped.

const usageText = () =>
  console.error(
    "usage: bun run script/breakdown.ts <runDir> [--session <id>] [--turn <n>] [--content <key>...] [--human] [-o out.json]",
  )

const emit = async (value: unknown, out: string | undefined) => {
  const json = JSON.stringify(value, null, 2) + "\n"
  if (out) await Bun.write(out, json)
  else process.stdout.write(json)
}

const filterSessions = (result: Breakdown, sessions: readonly string[]): Breakdown => {
  const filtered = result.sessions.filter((session) => sessions.includes(session.sessionID))
  return { ...result, sessions: filtered, totals: filtered.map((session) => session.totals).reduce(mergeTotals, ZERO_TOTALS) }
}

const filterTurns = (result: Breakdown, index: number): Breakdown => {
  const sessions = result.sessions
    .map((session) => {
      const turns = session.turns.filter((turn) => turn.index === index)
      return { ...session, turns, totals: turns.map(turnTotals).reduce(mergeTotals, ZERO_TOTALS) }
    })
    .filter((session) => session.turns.length > 0)
  return { ...result, sessions, totals: sessions.map((session) => session.totals).reduce(mergeTotals, ZERO_TOTALS) }
}

const renderHuman = (result: Breakdown): string => {
  const pct = (part: number, total: number) => (total > 0 ? `${((part / total) * 100).toFixed(1)}%` : "n/a")
  const lines = [
    `run ${result.run.modelID} (${result.run.providerID}) — inputEstimate ${result.totals.inputEstimate} (system ${result.totals.system}, tools ${result.totals.tools}, history ${result.totals.history}), output ${result.totals.output}, inputReported ${result.totals.inputReported}`,
  ]
  for (const session of result.sessions) {
    lines.push(`session ${session.sessionID} — inputEstimate ${session.totals.inputEstimate}`)
    for (const turn of session.turns) {
      lines.push(`  turn ${turn.index} ${turn.requestID}${turn.small ? " [small]" : ""} — inputEstimate ${turn.inputEstimate}`)
      lines.push(`    system ${turn.system.total} (${pct(turn.system.total, turn.inputEstimate)})`)
      for (const block of turn.system.blocks) lines.push(`      ${block.key} ${block.tokens} (${pct(block.tokens, turn.inputEstimate)})`)
      if (turn.system.mcpInstructionsTags !== null)
        lines.push(`      mcp instructions tags ${turn.system.mcpInstructionsTags} (${pct(turn.system.mcpInstructionsTags, turn.inputEstimate)})`)
      lines.push(`    tools ${turn.tools.total} (${pct(turn.tools.total, turn.inputEstimate)})`)
      for (const entry of turn.tools.entries) lines.push(`      ${entry.name} ${entry.tokens} (${pct(entry.tokens, turn.inputEstimate)})`)
      lines.push(`    history ${turn.history.total} (${pct(turn.history.total, turn.inputEstimate)})`)
    }
  }
  return lines.join("\n")
}

export const main = async (argv: string[]): Promise<number> => {
  try {
    const positional: string[] = []
    const sessions: string[] = []
    const content: string[] = []
    let out: string | undefined
    let turn: number | undefined
    let human = false
    for (let i = 0; i < argv.length; i++) {
      const arg = argv[i]
      const value = argv[i + 1]
      if (arg === "-o" || arg === "--session" || arg === "--content" || arg === "--turn") {
        if (value === undefined) {
          usageText()
          return 1
        }
        if (arg === "-o") out = value
        else if (arg === "--session") sessions.push(value)
        else if (arg === "--content") content.push(value)
        else {
          const index = Number(value)
          if (!Number.isInteger(index) || index < 0) {
            console.error(`breakdown: --turn expects a non-negative integer, got "${value}"`)
            return 1
          }
          turn = index
        }
        i++
        continue
      }
      if (arg === "--human") {
        human = true
        continue
      }
      if (arg.startsWith("-")) {
        usageText()
        return 1
      }
      positional.push(arg)
    }
    if (positional.length !== 1) {
      usageText()
      return 1
    }
    if (human && out) {
      console.error("breakdown: --human prints to stdout; -o writes the JSON of record — use one, not both")
      return 1
    }
    if (human && content.length > 0) {
      console.error("breakdown: --human renders counts only; --content drill-down needs the JSON output")
      return 1
    }
    const run = await loadRun(positional[0])
    let result = breakdown(
      { manifest: run.manifest, captures: run.captures, usage: run.usage },
      content.length > 0 ? { content } : undefined,
    )
    if (sessions.length > 0) result = filterSessions(result, sessions)
    if (turn !== undefined) result = filterTurns(result, turn)
    if (human) process.stdout.write(renderHuman(result) + "\n")
    else await emit(result, out)
    return 0
  } catch (error) {
    console.error(error instanceof Error ? error.message : error)
    return 1
  }
}

if (import.meta.main) process.exit(await main(process.argv.slice(2)))

export * as Breakdown from "./breakdown"
