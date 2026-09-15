import { DateTime, Effect } from "effect"
import { type ModelMessage, type Tool } from "ai"
import path from "path"
import fs from "fs/promises"
import type { Prepared } from "./request"

// The dump envelope is a whitelist, not a redaction (R10-002): `params`,
// `messageTransformOptions` and `headers` are structurally absent, so no
// credential can enter the file — auth attaches at transport, outside
// `Prepared`, and these are the only plausible carriers.
export type CaptureMeta = {
  readonly version: 1
  readonly sessionID: string
  readonly parentSessionID?: string
  readonly providerID: string
  readonly modelID: string
  readonly modelApi: string
  readonly agent: string
  readonly small: boolean
  readonly requestID: string
  readonly createdAt: string
  readonly optimized: { lazyTools: boolean; staticSlimming: boolean }
  // MCP attribution for the breakdown's cross-cut view (R10-006): tool name →
  // sanitized MCP server, assembled from the frozen listing entries. Absent on
  // bypass/small turns (no frozen base) — additive optional, tolerant readers.
  readonly toolServers?: Record<string, string>
  // Mechanism + binary self-identification for the comparison harness
  // (R13-003/005, SC-4): the resolved schema-binding verdict that shaped the
  // payload (absent when no lazy branch ran — bypass/small/proxy/kill-switch
  // turns) and the producing binary's version constant. Additive optional —
  // old captures parse and report.
  readonly verdict?: "binding" | "advisory"
  readonly binary?: string
}

export type ToolCapture = {
  readonly description?: string
  readonly inputSchema: unknown
  readonly strict?: boolean
}

export type CaptureFile = {
  readonly meta: CaptureMeta
  readonly payload: {
    readonly system: string[]
    readonly tools: Record<string, ToolCapture>
    readonly messages: ModelMessage[]
  }
}

// Everything the caller supplies; `version` and `createdAt` are dump-time facts.
export type CaptureSource = Omit<CaptureMeta, "version" | "createdAt">

export type DumpInput = {
  readonly data: string
  readonly prepared: Prepared
  readonly meta: CaptureSource
}

// The AI SDK marks jsonSchema() wrappers with Symbol.for("vercel.ai.schema");
// a wrapper's `.jsonSchema` getter yields the wire schema directly. Anything
// else (zod, plain objects, PromiseLike schemas) is a serializer contract
// violation — the turn's dump is skipped rather than written misleading.
const AI_SCHEMA = Symbol.for("vercel.ai.schema")

type RawSchema = { ok: true; schema: unknown } | { ok: false }

const rawJsonSchema = (inputSchema: unknown): RawSchema => {
  if (typeof inputSchema !== "object" || inputSchema === null) return { ok: false }
  const candidate = inputSchema as { [AI_SCHEMA]?: unknown; jsonSchema?: unknown }
  if (candidate[AI_SCHEMA] !== true) return { ok: false }
  const raw = candidate.jsonSchema
  if (typeof raw !== "object" || raw === null) return { ok: false }
  // A thenable jsonSchema cannot be serialized synchronously; JSON schema
  // objects with a `then` keyword are fine — `then` there is not a function.
  if (typeof (raw as { then?: unknown }).then === "function") return { ok: false }
  return { ok: true, schema: raw }
}

type Projection = { ok: true; tools: Record<string, ToolCapture> } | { ok: false; name: string }

const serializeTools = (tools: Record<string, Tool>): Projection => {
  const projected: Record<string, ToolCapture> = {}
  for (const [name, tool] of Object.entries(tools)) {
    const extracted = rawJsonSchema(tool.inputSchema)
    if (!extracted.ok) return { ok: false, name }
    projected[name] = {
      ...(tool.description === undefined ? {} : { description: tool.description }),
      inputSchema: extracted.schema,
      ...(tool.strict === undefined ? {} : { strict: tool.strict }),
    }
  }
  return { ok: true, tools: projected }
}

// Stateless turn sequence: re-derived per dump from the session's capture dir,
// so resume across processes needs no counter and produces no collisions.
// Same-session concurrent turns may collide; last write wins (benign — a dump
// is reproducible from a fresh turn).
const SEQ_PATTERN = /^(\d{4})\.json$/

const nextSeq = (entries: string[]): number => {
  let max = -1
  for (const entry of entries) {
    const match = SEQ_PATTERN.exec(entry)
    if (match) max = Math.max(max, Number(match[1]))
  }
  return max + 1
}

// Never fails (R00-010 at this seam): any error is logged and swallowed — a
// dump failure must not break the provider turn. The caller gates on
// OPENCODE_ENABLE_PROMPT_CAPTURE, so the flag-off path reaches no IO at all.
export const dump = Effect.fn("PromptCapture.dump")(function* (input: DumpInput) {
  const projected = serializeTools(input.prepared.tools)
  if (!projected.ok) {
    yield* Effect.logError(
      "prompt capture skipped: tool listing contains a tool whose inputSchema is not a jsonSchema() wrapper",
      { sessionID: input.meta.sessionID, tool: projected.name },
    )
    return
  }
  const now = yield* DateTime.nowAsDate
  const file: CaptureFile = {
    meta: { ...input.meta, version: 1, createdAt: now.toISOString() },
    payload: {
      system: input.prepared.system,
      tools: projected.tools,
      messages: input.prepared.messages,
    },
  }
  const dir = path.join(input.data, "prompt-captures", input.meta.sessionID)
  yield* Effect.tryPromise(() => fs.mkdir(dir, { recursive: true }))
  const entries = yield* Effect.tryPromise(() => fs.readdir(dir))
  const seq = String(nextSeq(entries)).padStart(4, "0")
  yield* Effect.tryPromise(() => Bun.write(path.join(dir, `${seq}.json`), JSON.stringify(file, null, 2)))
}, Effect.catchCause((cause) => Effect.logError("prompt capture dump failed, turn continues", { cause })))

export * as PromptCapture from "./prompt-capture"
