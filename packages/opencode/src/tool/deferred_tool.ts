import type { JSONSchema7 } from "@ai-sdk/provider"
import type { Tool, ToolExecutionOptions } from "ai"
import { Effect } from "effect"
import type { EffectBridge } from "../effect/bridge"
import type { SessionProcessor } from "../session/processor"
import { schemaBlock, type ToolSeed } from "../session/tool-listing"
import { errorMessage } from "../util/error"
import { isRecord } from "../util/record"
import DESCRIPTION from "./deferred_tool.txt"

// R12-012: the wrapper's single eager meta-tool, defined once with a closed,
// deployment-uniform schema (R12-009 — identical shape across provider
// families). The seed enters the listing through SessionTools.resolve on
// binding+wrapper sessions only; the AITool instance is constructed at that
// confluence closing over the completed shaped record — never a registry
// builtin (decision tool-lazy-loading-04 §1). This file owns the frozen
// definition facts plus the dispatch executor.
export const DEFERRED_TOOL_SCHEMA: JSONSchema7 = {
  type: "object",
  properties: {
    name: { type: "string", description: "Deferred tool name exactly as listed in the <deferred_tools> catalog" },
    args: { type: "string", description: "JSON string holding the tool's arguments object" },
  },
  required: ["name", "args"],
  additionalProperties: false,
}

export const DEFERRED_TOOL_DESCRIPTION = DESCRIPTION

// The unwrap metadata key (decision tool-lazy-loading-04 §3): rides the
// ToolPart's state.metadata, mirroring the load_tool marker convention.
// Written at dispatch start on the running part — failToolCall preserves
// running metadata, so the error state keeps it — and merged into the
// returned output's metadata, because completeToolCall writes output.metadata
// over the part. History stays verbatim: metadata is not model-visible.
type UpdateToolCall = SessionProcessor.Handle["updateToolCall"]

const writeMetadata = (input: { run: EffectBridge.Shape; updateToolCall: UpdateToolCall }, callID: string, patch: Record<string, unknown>) =>
  input.run.promise(
    input.updateToolCall(callID, (part) => {
      if (part.state.status !== "running") return part
      return { ...part, state: { ...part.state, metadata: { ...part.state.metadata, ...patch } } }
    }).pipe(
      Effect.catchCause((cause) =>
        Effect.logWarning("deferred_tool could not update the tool part metadata", { callID, cause }).pipe(Effect.asVoid),
      ),
    ),
  )

// The ai-sdk may begin executing the tool before the processor has consumed
// the stream's "tool-call" part, so the part can still be pending at dispatch
// start and a single write races the pending→running transition (the running
// transition rebuilds the state, dropping any earlier write). The stream order
// guarantees "tool-call" (running) is consumed strictly before "tool-result"
// (failToolCall), so waiting for the running status lands the write before the
// error state is written. Bounded per R00-010; a part that is already settled
// or an exhausted wait logs — the marker would otherwise be lost silently.
const METADATA_ATTEMPTS = 100
const METADATA_DELAY_MS = 5

const writeMetadataWhenRunning = async (
  input: { run: EffectBridge.Shape; updateToolCall: UpdateToolCall },
  callID: string,
  patch: Record<string, unknown>,
) => {
  for (let attempt = 0; attempt < METADATA_ATTEMPTS; attempt++) {
    const part = await writeMetadata(input, callID, patch)
    if (!part) break
    if (part.state.status === "running") return
    if (part.state.status !== "pending") break
    await Bun.sleep(METADATA_DELAY_MS)
  }
  await input.run.promise(
    Effect.logWarning("deferred_tool metadata write did not reach the running part", {
      callID,
      keys: Object.keys(patch),
    }),
  )
}

// Dispatch (decision tool-lazy-loading-04 §4): the map is the per-turn shaped
// deferred seeds only (permissible — shape already filtered denials). Unknown
// or eager names error without any schema (R12-001 dominance; eager tools are
// reachable directly); unparseable args on a known name deliver the inner
// tool's full schema with the load_tool marker (schema-in-error, the
// binding-session stand-in for R12-006). Otherwise the withFallback-wrapped
// inner closure runs untouched: permission ruleset, plugin triggers and
// fallback semantics unwrap naturally — no wrapper-level triggers, no
// double-append.
export const dispatch = (input: {
  seeds: ToolSeed[]
  shaped: Record<string, Tool>
  run: EffectBridge.Shape
  updateToolCall: UpdateToolCall
}) =>
  async (args: unknown, options: ToolExecutionOptions) => {
    const call = isRecord(args) ? args : {}
    const name = typeof call.name === "string" ? call.name : ""
    await writeMetadataWhenRunning(input, options.toolCallId, { deferred_tool: { tool: name } })
    const seed = input.seeds.find((item) => item.name === name)
    if (!seed)
      throw new Error(`Unknown deferred tool: ${name}. Deferred tools are listed in the <deferred_tools> catalog blocks.`)
    if (seed.kind === "eager")
      throw new Error(`${name} is not a deferred tool — it is listed in the tool listing, call it directly.`)
    let parsedArgs: unknown
    try {
      parsedArgs = JSON.parse(typeof call.args === "string" ? call.args : "")
    } catch (error) {
      await writeMetadataWhenRunning(input, options.toolCallId, { load_tool: { tools: [name] } })
      throw new Error(`${errorMessage(error)}\n\n${schemaBlock(seed)}`)
    }
    if (!isRecord(parsedArgs)) {
      await writeMetadataWhenRunning(input, options.toolCallId, { load_tool: { tools: [name] } })
      throw new Error(`Arguments must be a JSON object holding the ${name} tool's arguments.\n\n${schemaBlock(seed)}`)
    }
    const inner = input.shaped[name]
    if (!inner?.execute) throw new Error(`${name} has no executable closure in this turn.`)
    const output = await inner.execute(parsedArgs, options)
    if (!isRecord(output)) return output
    const metadata = isRecord(output.metadata) ? output.metadata : {}
    return { ...output, metadata: { ...metadata, deferred_tool: { tool: name } } }
  }
