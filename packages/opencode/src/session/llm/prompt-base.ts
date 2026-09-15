import { LayerNode } from "@opencode-ai/core/effect/layer-node"
import { InstanceState } from "@/effect/instance-state"
import type { Provider } from "@/provider/provider"
import type { FrozenToolEntry, ToolSeed, Verdict } from "@/session/tool-listing"
import { Context, Effect, Layer, Schema } from "effect"

export type SystemBlockKey =
  | "environment"
  | "instructions"
  | `mcp:${string}`
  | "skills"
  // Catalog family (R12-012, fork-invented — no upstream bytes): the
  // deferred-tool discovery blocks written only on binding+wrapper sessions.
  | "catalog"
  | `catalog:${string}`
  | "structured_output"

export type SystemBlock = { key: SystemBlockKey; content: string }

export class DuplicateSystemBlockError extends Schema.TaggedErrorClass<DuplicateSystemBlockError>()(
  "DuplicateSystemBlockError",
  { key: Schema.String },
) {}

export class DuplicateToolEntryError extends Schema.TaggedErrorClass<DuplicateToolEntryError>()(
  "DuplicateToolEntryError",
  { name: Schema.String },
) {}

const isMcp = (key: SystemBlockKey): key is `mcp:${string}` => key.startsWith("mcp:")

const isCatalog = (key: SystemBlockKey): key is "catalog" | `catalog:${string}` =>
  key === "catalog" || key.startsWith("catalog:")

// Per-message conditionals are outside the frozen prompt base (decision
// tool-lazy-loading-01): they are rendered per turn, never frozen.
const PER_TURN = new Set<SystemBlockKey>(["structured_output"])

const endpoint = (input: { model: Provider.Model; provider: Provider.Info }) => {
  const baseURL = input.provider.options.baseURL
  // Upstream resolves an empty-string baseURL to the model URL at request
  // time (provider.ts), so the state key follows the effective endpoint.
  return typeof baseURL === "string" && baseURL !== "" ? baseURL : input.model.api.url
}

const stateKey = (input: { sessionID: string; model: Provider.Model; provider: Provider.Info }) =>
  `${input.sessionID}:${input.model.providerID}/${input.model.id}/${endpoint(input)}`

// Canonical ordering: environment, instructions, mcp group, skills, catalog
// blocks, structured_output (SC-2). Catalog blocks are the R12-012 discovery
// family (fork-invented): they project after skills and before per-turn keys,
// in frozen append order; sessions without catalog keys render round-1 bytes
// exactly. Projecting by key class keeps the rendered bytes independent of
// the frozen array's append order for the fixed slots.
export const render = (blocks: SystemBlock[]): string[] => {
  const environment = blocks.find((block) => block.key === "environment")
  const instructions = blocks.find((block) => block.key === "instructions")
  const mcp = blocks.filter((block) => isMcp(block.key))
  const skills = blocks.find((block) => block.key === "skills")
  const catalog = blocks.filter((block) => isCatalog(block.key))
  const perTurn = blocks.filter((block) => PER_TURN.has(block.key))
  return [
    environment?.content,
    instructions?.content,
    mcp.length > 0
      ? ["<mcp_instructions>", ...mcp.map((block) => block.content), "</mcp_instructions>"].join("\n")
      : undefined,
    skills?.content,
    ...catalog.map((block) => block.content),
    ...perTurn.map((block) => block.content),
  ].filter((entry): entry is string => entry !== undefined)
}

export interface Interface {
  readonly reconcileSystem: (input: {
    sessionID: string
    model: Provider.Model
    provider: Provider.Info
    blocks: SystemBlock[]
  }) => Effect.Effect<{ blocks: SystemBlock[]; appended: string[] }, DuplicateSystemBlockError>
  readonly reconcileTools: (input: {
    sessionID: string
    model: Provider.Model
    provider: Provider.Info
    seeds: ToolSeed[]
    mode: Verdict
    wrapper: boolean
  }) => Effect.Effect<
    { entries: FrozenToolEntry[]; appended: string[]; mode: Verdict; wrapper: boolean },
    DuplicateToolEntryError
  >
  // Read view over the frozen base (load_tool path): the entries plus the
  // frozen (mode, wrapper) policy pair. An unknown key yields an empty view —
  // callers render their own error lines from it.
  readonly entries: (input: {
    sessionID: string
    model: Provider.Model
    provider: Provider.Info
  }) => Effect.Effect<{ entries: FrozenToolEntry[]; mode: Verdict | undefined; wrapper: boolean | undefined }>
}

export class Service extends Context.Service<Service, Interface>()("@opencode/PromptBase") {}

type PromptBaseState = {
  systemBlocks: SystemBlock[]
  toolEntries: Map<string, FrozenToolEntry>
  // The R12-010 verdict frozen with the first tool write: a mid-session
  // BindingVerdict.observe flip must never re-render frozen entries (R12-007).
  mode?: Verdict
  // The R12-012 wrapper axis frozen next to the mode: the pair (mode, wrapper)
  // is exactly what decided the written listing shapes.
  wrapper?: boolean
}

const emptyState = (): PromptBaseState => ({ systemBlocks: [], toolEntries: new Map() })

const layer = Layer.effect(
  Service,
  Effect.gen(function* () {
    const state = yield* InstanceState.make(
      Effect.fn("PromptBase.state")(function* () {
        return new Map<string, PromptBaseState>()
      }),
    )

    const reconcileSystem = Effect.fn("PromptBase.reconcileSystem")(function* (input: {
      sessionID: string
      model: Provider.Model
      provider: Provider.Info
      blocks: SystemBlock[]
    }) {
      const map = yield* InstanceState.get(state)
      const key = stateKey(input)
      const current = map.get(key) ?? emptyState()
      const seen = new Set(current.systemBlocks.map((block) => block.key))
      const appended: SystemBlock[] = []
      const perTurn: SystemBlock[] = []
      for (const block of input.blocks) {
        if (PER_TURN.has(block.key)) {
          perTurn.push(block)
          continue
        }
        if (appended.some((b) => b.key === block.key)) yield* new DuplicateSystemBlockError({ key: block.key })
        if (seen.has(block.key)) continue
        appended.push(block)
        seen.add(block.key)
      }
      if (appended.length > 0)
        map.set(key, { ...current, systemBlocks: [...current.systemBlocks, ...appended] })
      return { blocks: [...current.systemBlocks, ...appended, ...perTurn], appended: appended.map((block) => block.key) }
    })

    const reconcileTools = Effect.fn("PromptBase.reconcileTools")(function* (input: {
      sessionID: string
      model: Provider.Model
      provider: Provider.Info
      seeds: ToolSeed[]
      mode: Verdict
      wrapper: boolean
    }) {
      const map = yield* InstanceState.get(state)
      const key = stateKey(input)
      const current = map.get(key) ?? emptyState()
      const mode = current.mode ?? input.mode
      // Same first-write rule as the mode: the frozen pair (mode, wrapper) is
      // what decided the written shapes; a mid-session change never re-reads.
      const wrapper = current.wrapper ?? input.wrapper
      const appended: ToolSeed[] = []
      for (const seedItem of input.seeds) {
        if (current.toolEntries.has(seedItem.name)) continue
        if (appended.some((existing) => existing.name === seedItem.name))
          yield* new DuplicateToolEntryError({ name: seedItem.name })
        appended.push(seedItem)
      }
      const toolEntries = appended.reduce((entries, seedItem) => {
        entries.set(seedItem.name, seedItem)
        return entries
      }, new Map(current.toolEntries))
      if (appended.length > 0) map.set(key, { ...current, toolEntries, mode, wrapper })
      return { entries: [...toolEntries.values()], appended: appended.map((seedItem) => seedItem.name), mode, wrapper }
    })

    const entries = Effect.fn("PromptBase.entries")(function* (input: {
      sessionID: string
      model: Provider.Model
      provider: Provider.Info
    }) {
      const map = yield* InstanceState.get(state)
      const current = map.get(stateKey(input))
      return current
        ? { entries: [...current.toolEntries.values()], mode: current.mode, wrapper: current.wrapper }
        : { entries: [] as FrozenToolEntry[], mode: undefined, wrapper: undefined }
    })

    return Service.of({ reconcileSystem, reconcileTools, entries })
  }),
)

export const node = LayerNode.make({ service: Service, layer: layer, deps: [] })

export * as PromptBase from "./prompt-base"
