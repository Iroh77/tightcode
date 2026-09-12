import type { JSONSchema7 } from "@ai-sdk/provider"
import { PermissionV1 } from "@opencode-ai/core/v1/permission"
import type { SessionV1 } from "@opencode-ai/core/v1/session"
import type { ToolExecutionOptions } from "ai"
import { Effect, Schema } from "effect"
import type { EffectBridge } from "@/effect/bridge"
import type { SessionProcessor } from "@/session/processor"
import { delivered } from "@/tool/load_tool"
import { Tool } from "@/tool/tool"
import { errorMessage } from "@/util/error"
import { Permission } from "@/permission"

export type Verdict = "binding" | "advisory"

const SOURCES = ["builtin", "resource", "mcp", "plugin", "custom"] as const

export type UniverseTool = {
  name: string
  description: string
  jsonSchema: JSONSchema7
  source: (typeof SOURCES)[number]
  server?: string
}

// Produced per turn by shape: the pre-freeze fact. Carries the untruncated
// description and the full (model-sanitized) schema so that frozen entries
// stay mode-independent — the advisory placeholder must never overwrite the
// only copy the frozen base holds (load_tool re-serves it, R12-004/R12-005).
export type ToolSeed = {
  name: string
  kind: "eager" | "deferred"
  fullDescription: string
  jsonSchema: JSONSchema7
  server?: string
}

// Frozen at first write; never mutated (R12-007).
export type FrozenToolEntry = ToolSeed

export type ListingEntry = { name: string; description: string; jsonSchema: JSONSchema7 }

export class MalformedToolEntryError extends Schema.TaggedErrorClass<MalformedToolEntryError>()(
  "MalformedToolEntryError",
  { name: Schema.String, reason: Schema.String },
) {}

// Several provider families require the field, so a truly absent schema is not
// sendable; one family-agnostic constant keeps R12-009 intact (R12-003 amendment).
export const PLACEHOLDER: JSONSchema7 = { type: "object", properties: {} }

// R12-002 eager set, keyed by production registry ids. The shell tool's
// exposed id is "bash" (ShellID.ToolID — kept for plugin/permission
// compatibility, rename planned upstream), so the requirement's "shell"
// resolves to "bash" here.
const EAGER = new Set(["bash", "read", "load_tool"])

const TRUNCATE_BOUND = 100

// Decision tool-lazy-loading-03 §1: first ≤100 chars, cut at the last word
// boundary within the bound, "..." appended when truncated; empty stays empty.
const truncate100 = (description: string) => {
  if (description.length <= TRUNCATE_BOUND) return description
  const bound = description.slice(0, TRUNCATE_BOUND)
  const cut = bound.lastIndexOf(" ")
  return (cut > 0 ? bound.slice(0, cut) : bound) + "..."
}

export const shape = (input: { universe: UniverseTool[]; ruleset: PermissionV1.Ruleset }): ToolSeed[] => {
  validate(input.universe)
  const denied = Permission.disabled(
    input.universe.map((tool) => tool.name),
    input.ruleset,
  )
  return input.universe
    .filter((tool) => !denied.has(tool.name))
    .map((tool) => ({
      name: tool.name,
      kind: EAGER.has(tool.name) ? ("eager" as const) : ("deferred" as const),
      fullDescription: tool.description,
      jsonSchema: tool.jsonSchema,
      ...(tool.server !== undefined ? { server: tool.server } : {}),
    }))
}

export const render = (entries: FrozenToolEntry[], mode: Verdict): ListingEntry[] => {
  // R12-003 prefix-once, owned by the group's first-written entry (decision
  // tool-lazy-loading-03 §2 as amended at ticket 02 close): recomputing an
  // alphabetical winner per render would move the prefix onto later appends
  // and mutate frozen entries, which R12-008 forbids.
  const owner = new Map<string, string>()
  for (const entry of entries) {
    if (entry.kind !== "deferred" || entry.server === undefined) continue
    if (!owner.has(entry.server)) owner.set(entry.server, entry.name)
  }
  return entries.map((entry) => {
    if (entry.kind === "eager") {
      return { name: entry.name, description: entry.fullDescription, jsonSchema: entry.jsonSchema }
    }
    const prefix = entry.server !== undefined && owner.get(entry.server) === entry.name ? `${entry.server}: ` : ""
    return {
      name: entry.name,
      description: prefix + truncate100(entry.fullDescription),
      jsonSchema: mode === "binding" ? entry.jsonSchema : PLACEHOLDER,
    }
  })
}

function validate(universe: UniverseTool[]) {
  for (const [index, tool] of universe.entries()) {
    const at = `universe[${index}]`
    if (typeof tool.name !== "string" || tool.name === "")
      throw new MalformedToolEntryError({ name: "", reason: `${at}.name must be a non-empty string` })
    if (typeof tool.description !== "string")
      throw new MalformedToolEntryError({ name: tool.name, reason: `${at}.description must be a string` })
    if (typeof tool.jsonSchema !== "object" || tool.jsonSchema === null || Array.isArray(tool.jsonSchema))
      throw new MalformedToolEntryError({ name: tool.name, reason: `${at}.jsonSchema must be an object` })
    if (!SOURCES.includes(tool.source))
      throw new MalformedToolEntryError({ name: tool.name, reason: `${at}.source must be one of ${SOURCES.join(", ")}` })
    if (tool.server !== undefined && (typeof tool.server !== "string" || tool.server === ""))
      throw new MalformedToolEntryError({ name: tool.name, reason: `${at}.server must be a non-empty string when present` })
  }
}

// Type-only reference (erased at runtime): keeps the fallback's expectations
// in sync with the real processor handle without importing the processor into
// this module.
type UpdateToolCall = SessionProcessor.Handle["updateToolCall"]

const schemaBlock = (seed: ToolSeed) =>
  [`The ${seed.name} tool has not been loaded. Its full input schema is:`, "", JSON.stringify(seed.jsonSchema, null, 2)].join("\n")

// R12-006 direct-call fallback (decision tool-lazy-loading-02 §3/§4): a
// deferred tool executes when called without a prior load; on failure its full
// schema rides the error output and the part is marked loaded, so recovery is
// protocol-level rather than willingness-level. Permission denials and aborts
// are not arg-shape failures and stay upstream (R12-001 enforcement unchanged).
// A schema-validation failure additionally feeds BindingVerdict.observe:
// grammar-enforced serving cannot produce one, so the signal is proof of
// advisory — future sessions only, independent of the load state.
export const withFallback = (
  input: {
    seed: ToolSeed
    messages: SessionV1.WithParts[]
    run: EffectBridge.Shape
    updateToolCall: UpdateToolCall
    observe: Effect.Effect<void>
  },
  execute: (args: unknown, options: ToolExecutionOptions) => Promise<unknown>,
): ((args: unknown, options: ToolExecutionOptions) => Promise<unknown>) =>
  async (args, options) => {
    try {
      return await execute(args, options)
    } catch (error) {
      if (options.abortSignal?.aborted) throw error
      if (
        error instanceof PermissionV1.RejectedError ||
        error instanceof PermissionV1.DeniedError ||
        error instanceof PermissionV1.CorrectedError
      )
        throw error
      if (error instanceof Tool.InvalidArgumentsError) await input.run.promise(input.observe)
      // Repeat failures stay quiet: the marker re-opens only when the model no
      // longer sees the part that delivered the full content.
      if (delivered(input.seed.name, input.messages)) throw error
      // The marker must land while the part is still running — failToolCall
      // preserves running metadata into the error state.
      await input.run.promise(
        input
          .updateToolCall(options.toolCallId, (part) => {
            if (part.state.status !== "running") return part
            return {
              ...part,
              state: {
                ...part.state,
                metadata: { ...part.state.metadata, load_tool: { tools: [input.seed.name] } },
              },
            }
          })
          .pipe(
            Effect.catchCause((cause) =>
              Effect.logWarning("tool fallback could not set the delivery marker", {
                tool: input.seed.name,
                callID: options.toolCallId,
                cause,
              }),
            ),
            Effect.asVoid,
          ),
      )
      throw new Error(`${errorMessage(error)}\n\n${schemaBlock(input.seed)}`)
    }
  }

export * as ToolListing from "./tool-listing"
