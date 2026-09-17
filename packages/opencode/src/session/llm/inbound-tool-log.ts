import type { LLMEvent } from "@opencode-ai/llm"
import { Global } from "@opencode-ai/core/global"
import { RuntimeFlags } from "@/effect/runtime-flags"
import { LayerNode } from "@opencode-ai/core/effect/layer-node"
import { Context, DateTime, Effect, Layer } from "effect"
import fs from "fs/promises"
import path from "path"

// R10-010: the forensic seam between the provider stream and OpenCode's
// transforms — the assistant message's tool calls exactly as received,
// pre-parse, pre-`isRecord` wrap, pre-storage. A debugging artifact, not a
// measurement input: separate file and flag from the R10-001 capture axis.

// The per-turn context the log line needs. One fresh object per provider
// turn, also the WeakMap key that scopes the delta buffers to a single
// stream attempt: a retried turn starts with empty buffers.
export type CaptureContext = {
  readonly sessionID: string
  readonly messageID: string
  readonly providerID: string
  readonly modelID: string
}

export type InboundToolCallLine = {
  time: number
  sessionID: string
  messageID: string
  providerID: string
  modelID: string
  callID: string
  tool: string
  providerExecuted?: boolean
  raw: string | null
  input: unknown
}

export interface Interface {
  readonly capture: (event: LLMEvent, context: CaptureContext) => Effect.Effect<void>
}

export class Service extends Context.Service<Service, Interface>()("@opencode/InboundToolLog") {}

export const layer = Layer.effect(
  Service,
  Effect.gen(function* () {
    const flags = yield* RuntimeFlags.Service
    const global = yield* Global.Service
    const file = path.join(global.data, "inbound-tool-calls.jsonl")
    // Flag read once at layer init (capture precedent): off by default, and
    // off means a pure no-op — zero IO, zero buffering (R10-010).
    if (!flags.enableInboundToolLog) return Service.of({ capture: () => Effect.void })

    const buffers = new WeakMap<CaptureContext, Map<string, { name: string; chunks: string[] }>>()
    const entry = (context: CaptureContext) => {
      let existing = buffers.get(context)
      if (existing === undefined) {
        existing = new Map()
        buffers.set(context, existing)
      }
      return existing
    }

    // Total: filters to tool-input-*/tool-call internally, never fails (R00-010
    // at this seam — a write failure is logged loudly, the turn continues, and
    // a later call retries the append independently).
    const capture = Effect.fn("InboundToolLog.capture")(function* (event: LLMEvent, context: CaptureContext) {
      if (event.type === "tool-input-start") {
        entry(context).set(event.id, { name: event.name, chunks: [] })
        return
      }
      if (event.type === "tool-input-delta") {
        // A delta without a start is a provider quirk; buffering from the
        // delta keeps `raw` complete anyway.
        const buffered = entry(context).get(event.id)
        if (buffered === undefined) entry(context).set(event.id, { name: event.name, chunks: [event.text] })
        else buffered.chunks.push(event.text)
        return
      }
      if (event.type !== "tool-call") return
      const buffered = entry(context).get(event.id)
      entry(context).delete(event.id)
      const now = yield* DateTime.nowAsDate
      const line: InboundToolCallLine = {
        time: now.getTime(),
        sessionID: context.sessionID,
        messageID: context.messageID,
        providerID: context.providerID,
        modelID: context.modelID,
        callID: event.id,
        tool: event.name,
        ...(event.providerExecuted === true ? { providerExecuted: true } : {}),
        // Verbatim pre-parse argument text, never truncated or reformatted;
        // null when the provider streamed no arguments (R10-010).
        raw: buffered === undefined ? null : buffered.chunks.join(""),
        input: event.input,
      }
      yield* Effect.tryPromise(() => fs.appendFile(file, `${JSON.stringify(line)}\n`)).pipe(
        Effect.catchCause((cause) => Effect.logError("inbound tool log write failed, turn continues", { file, cause })),
      )
    })

    return Service.of({ capture })
  }),
)

export * as InboundToolLog from "./inbound-tool-log"

export const node = LayerNode.make({
  service: Service,
  layer: layer,
  deps: [RuntimeFlags.node, Global.node],
})
