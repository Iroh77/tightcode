#!/usr/bin/env bun

import { Database } from "bun:sqlite"
import fs from "fs/promises"
import path from "path"
import type { CaptureFile } from "../../src/session/llm/prompt-capture"

// Echo-stub bin for the reference-workload driver (--stub, R10-004): mimics an
// opencode-shaped CLI without a provider so the driver mechanics can be
// verified offline. Per run invocation it appends one deterministic step-finish
// usage row at OPENCODE_DB, writes a capture dump per turn when
// OPENCODE_ENABLE_PROMPT_CAPTURE is set (mirroring the fork's dump layout), and
// records the observed OPENCODE_* flags to <runDir>/stub-env.json so the
// driver's env propagation is verifiable.

const SESSION_ID = "stub-session"
const CAPTURE_SEQ = /^(\d{4})\.json$/

const args = Bun.argv.slice(2)
const body = args[0] === "run" ? args.slice(1) : args
let model: string | undefined
const words: string[] = []
for (let i = 0; i < body.length; i++) {
  const arg = body[i]
  if (arg === "--model") {
    model = body[i + 1]
    i++
  } else if (arg === "--agent") {
    i++
  } else if (arg !== "--continue") {
    words.push(arg)
  }
}
const prompt = words.join(" ")
if (!prompt) {
  console.error("stub-bin: missing prompt")
  process.exit(1)
}

const xdg = process.env.XDG_DATA_HOME
const dbPath = process.env.OPENCODE_DB
if (!xdg || !dbPath) {
  console.error("stub-bin: XDG_DATA_HOME and OPENCODE_DB must be set (driver-provided)")
  process.exit(1)
}

const db = new Database(dbPath)
db.exec("CREATE TABLE IF NOT EXISTS session (id TEXT PRIMARY KEY)")
db.exec("CREATE TABLE IF NOT EXISTS part (session_id TEXT, data TEXT)")
db.run("INSERT OR IGNORE INTO session (id) VALUES (?)", [SESSION_ID])
const prior = db.query<{ n: number }, [string]>("SELECT COUNT(*) AS n FROM part WHERE session_id = ?").get(SESSION_ID)
const seq = prior?.n ?? 0
const stepFinish = {
  type: "step-finish",
  tokens: { input: 1000 + Math.ceil(prompt.length / 4), output: 10 + 5 * seq, reasoning: 0, cache: { read: 7, write: 3 } },
  cost: 0.01,
}
db.run("INSERT INTO part (session_id, data) VALUES (?, ?)", [SESSION_ID, JSON.stringify(stepFinish)])
db.close()

if (process.env.OPENCODE_ENABLE_PROMPT_CAPTURE === "1") {
  const capturesDir = path.join(xdg, "opencode", "prompt-captures", SESSION_ID)
  await fs.mkdir(capturesDir, { recursive: true })
  const names = await fs.readdir(capturesDir).catch((error: NodeJS.ErrnoException) => {
    if (error.code === "ENOENT") return []
    throw error
  })
  const next = names
    .map((name) => CAPTURE_SEQ.exec(name)?.[1])
    .filter((match) => match !== undefined)
    .reduce((max, match) => Math.max(max, Number(match)), -1) + 1
  const [providerID = "stub", ...modelRest] = (model ?? "stub/stub").split("/")
  const capture: CaptureFile = {
    meta: {
      version: 1,
      sessionID: SESSION_ID,
      providerID,
      modelID: modelRest.join("/"),
      modelApi: "stub",
      agent: "build",
      small: false,
      requestID: `stub-${next}`,
      createdAt: new Date().toISOString(),
      optimized: {
        lazyTools: process.env.OPENCODE_DISABLE_LAZY_TOOLS !== "1",
        staticSlimming: process.env.OPENCODE_DISABLE_STATIC_SLIMMING !== "1",
      },
    },
    payload: {
      system: ["You are a stub assistant for reference-workload mechanics verification."],
      tools: {},
      messages: [{ role: "user", content: prompt }],
    },
  }
  await Bun.write(path.join(capturesDir, `${String(next).padStart(4, "0")}.json`), JSON.stringify(capture, null, 2))
}

await Bun.write(
  path.join(path.dirname(xdg), "stub-env.json"),
  JSON.stringify(
    {
      OPENCODE_ENABLE_PROMPT_CAPTURE: process.env.OPENCODE_ENABLE_PROMPT_CAPTURE,
      OPENCODE_DISABLE_LAZY_TOOLS: process.env.OPENCODE_DISABLE_LAZY_TOOLS,
      OPENCODE_DISABLE_STATIC_SLIMMING: process.env.OPENCODE_DISABLE_STATIC_SLIMMING,
    },
    null,
    2,
  ),
)

console.log(`[stub] ${SESSION_ID} turn ${seq}: ${prompt}`)