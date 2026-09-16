import type { JSONSchema7 } from "@ai-sdk/provider"
import { PermissionV1 } from "@opencode-ai/core/v1/permission"
import type { SessionV1 } from "@opencode-ai/core/v1/session"
import type { ToolExecutionOptions } from "ai"
import { Effect, Schema } from "effect"
import type { EffectBridge } from "@/effect/bridge"
import type { SystemBlock } from "@/session/llm/prompt-base"
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
// `source` rides additively (frozen with the entry): the catalog producer's
// grouping key (R12-012).
export type ToolSeed = {
  name: string
  kind: "eager" | "deferred"
  fullDescription: string
  jsonSchema: JSONSchema7
  server?: string
  source: (typeof SOURCES)[number]
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
// resolves to "bash" here. "deferred_tool" is only ever IN the universe on
// binding+wrapper sessions (gated at the SessionTools.resolve confluence),
// where it must be eager (R12-012).
const EAGER = new Set(["bash", "read", "load_tool", "deferred_tool"])

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
      source: tool.source,
      ...(tool.server !== undefined ? { server: tool.server } : {}),
    }))
}

// The (verdict, wrapperAxis) listing policy (R12-012): on binding sessions
// with the wrapper enabled, deferred entries produce NO listing entry at all —
// the payload reaches them through deferred_tool and the catalog instead.
// Advisory wins defensively: the pair advisory+wrapper cannot occur
// (wrapperActive requires binding). wrapper=false restores the round-1 bytes
// for both modes (the kill-switch path).
export const render = (entries: FrozenToolEntry[], mode: Verdict, wrapper: boolean): ListingEntry[] => {
  // R12-003 prefix-once, owned by the group's first-written entry (decision
  // tool-lazy-loading-03 §2 as amended at ticket 02 close): recomputing an
  // alphabetical winner per render would move the prefix onto later appends
  // and mutate frozen entries, which R12-008 forbids. Ownership is computed
  // over the full frozen list before any omission — wrapper sessions never
  // move an owner (owners only matter where deferred entries render).
  const owner = new Map<string, string>()
  for (const entry of entries) {
    if (entry.kind !== "deferred" || entry.server === undefined) continue
    if (!owner.has(entry.server)) owner.set(entry.server, entry.name)
  }
  const views: ListingEntry[] = []
  for (const entry of entries) {
    if (entry.kind === "eager") {
      views.push({ name: entry.name, description: entry.fullDescription, jsonSchema: entry.jsonSchema })
      continue
    }
    if (mode === "binding" && wrapper) continue
    const prefix = entry.server !== undefined && owner.get(entry.server) === entry.name ? `${entry.server}: ` : ""
    views.push({
      name: entry.name,
      description: prefix + truncate100(entry.fullDescription),
      jsonSchema: mode === "binding" ? entry.jsonSchema : PLACEHOLDER,
    })
  }
  return views
}

// R12-012 discovery: the deferred-tool catalog blocks, produced purely per
// turn from the seeds (only wrapper sessions assemble them — prompt.ts gates
// on wrapperActive; advisory/kill-switch sessions never produce catalog keys).
// Grouping by the seed's source: mcp → `catalog:<server>` (one block per
// server), resource → "catalog:resources", everything else → "catalog".
// Each block carries its own <deferred_tools> delimiters; bullets are
// `- name: truncate100(description)` in seed (frozen-append) order, the name
// alone when the description is empty. No server description prefix inside a
// block — the block is the grouping (R12-003's prefix-once governs listing
// entries, which do not exist for deferred tools in this mode). reconcileSystem's
// first-write-wins turns re-emission into a no-op and a late MCP connect into
// a new-key append (one batch per event, R12-008). The how-to-call instruction
// lives in deferred_tool's description, never in a block — blocks stay pure
// listings and never mutate.
export const catalogBlocks = (input: { seeds: ToolSeed[] }): SystemBlock[] => {
  const groups = new Map<string, { key: SystemBlock["key"]; server?: string; bullets: string[] }>()
  for (const seed of input.seeds) {
    if (seed.kind !== "deferred") continue
    if (seed.source === "mcp" && seed.server === undefined)
      throw new MalformedToolEntryError({ name: seed.name, reason: "an mcp seed must carry its server for catalog grouping" })
    const server = seed.source === "mcp" ? seed.server : undefined
    const key: SystemBlock["key"] = server !== undefined ? `catalog:${server}` : seed.source === "resource" ? "catalog:resources" : "catalog"
    const group = groups.get(key) ?? { key, server, bullets: [] }
    const description = truncate100(seed.fullDescription)
    group.bullets.push(description === "" ? `- ${seed.name}` : `- ${seed.name}: ${description}`)
    groups.set(key, group)
  }
  return [...groups.values()].map((group) => ({
    key: group.key,
    content: [`<deferred_tools${group.server ? ` server="${group.server}"` : ""}>`, ...group.bullets, "</deferred_tools>"].join("\n"),
  }))
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

// The ai-sdk may begin executing the tool before the processor has consumed
// the stream's "tool-call" part, so the part can still be pending when the
// fallback marker writes — and the pending→running transition rebuilds the
// state, dropping the write (same race deferred_tool.ts documents). The
// observe hop used to mask this by accident; the Amendment 4 gate removed it,
// so the write now waits for the running part explicitly. The marker must
// land while the part is running — failToolCall preserves running metadata
// into the error state. Bounded; a settled part or an exhausted wait logs.
const METADATA_ATTEMPTS = 100
const METADATA_DELAY_MS = 5
// Runtime-agnostic bounded wait (R00-016): the desktop embeds the server under
// Electron's Node, where the Bun global does not exist.
const sleep = (ms: number) => new Promise<void>((resolve) => setTimeout(resolve, ms))

const writeMarkerWhenRunning = async (
  input: { run: EffectBridge.Shape; updateToolCall: UpdateToolCall; seed: ToolSeed },
  callID: string,
) => {
  for (let attempt = 0; attempt < METADATA_ATTEMPTS; attempt++) {
    const part = await input.run.promise(
      input.updateToolCall(callID, (part) => {
        if (part.state.status !== "running") return part
        return {
          ...part,
          state: {
            ...part.state,
            metadata: { ...part.state.metadata, load_tool: { tools: [input.seed.name] } },
          },
        }
      }).pipe(
        Effect.catchCause((cause) =>
          Effect.logWarning("tool fallback could not set the delivery marker", {
            tool: input.seed.name,
            callID,
            cause,
          }).pipe(Effect.as(undefined)),
        ),
      ),
    )
    if (!part) break
    if (part.state.status === "running") return
    if (part.state.status !== "pending") break
    await sleep(METADATA_DELAY_MS)
  }
  await input.run.promise(
    Effect.logWarning("tool fallback marker did not reach the running part", {
      tool: input.seed.name,
      callID,
    }),
  )
}

// Shared schema-in-error block: the direct-call fallback (R12-006) and the
// deferred_tool dispatch (R12-012) append the same round-1 format.
export const schemaBlock = (seed: ToolSeed) =>
  [`The ${seed.name} tool has not been loaded. Its full input schema is:`, "", JSON.stringify(seed.jsonSchema, null, 2)].join("\n")

// R12-006 direct-call fallback (decision tool-lazy-loading-02 §3/§4): a
// deferred tool executes when called without a prior load; on failure its full
// schema rides the error output and the part is marked loaded, so recovery is
// protocol-level rather than willingness-level. Permission denials and aborts
// are not arg-shape failures and stay upstream (R12-001 enforcement unchanged).
// A schema-validation failure feeds BindingVerdict.observe when the caller
// passed one (R12-010 Amendment 4: only informative sessions — wrapper-active
// binding — wire it; the factory receives the failing args for the evidence
// digest). Future sessions only, independent of the load state.
export const withFallback = (
  input: {
    seed: ToolSeed
    messages: SessionV1.WithParts[]
    run: EffectBridge.Shape
    updateToolCall: UpdateToolCall
    observe?: (args: unknown) => Effect.Effect<void>
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
      if (error instanceof Tool.InvalidArgumentsError && input.observe !== undefined)
        await input.run.promise(input.observe(args))
      // Repeat failures stay quiet: the marker re-opens only when the model no
      // longer sees the part that delivered the full content.
      if (delivered(input.seed.name, input.messages)) throw error
      await writeMarkerWhenRunning(input, options.toolCallId)
      throw new Error(`${errorMessage(error)}\n\n${schemaBlock(input.seed)}`)
    }
  }

export * as ToolListing from "./tool-listing"
