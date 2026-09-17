import { Effect, Schema } from "effect"
import { SessionV1 } from "@opencode-ai/core/v1/session"
import { PromptBase } from "../session/llm/prompt-base"
import { Provider } from "../provider/provider"
import type { FrozenToolEntry, Verdict } from "../session/tool-listing"
import { Tool } from "./tool"

// load_tool's listing description discloses what a loaded tool can do in the
// resolved regime — one pinned variant per (verdict, wrapper) case, selected
// at SessionTools.resolve before shape (decision tool-lazy-loading-06). The
// old static load_tool.txt could not be accurate across the three regimes
// and hard-coded the eager set.
export const loadToolDescriptions = {
  advisory: "Load a deferred tool's full description and schema (also supports multiple names to load several tools at once). The loaded tool can then be used normally.",
  wrapper: "Load a deferred tool's full description and schema (also supports multiple names to load several tools at once). A loaded tool cannot be called directly. Instead, it must be executed through the deferred_tool tool, passing the arguments as deferred_tool requires.",
  schemaEager: "Load a deferred tool's full description (also supports multiple names to load several tools at once).",
} as const

export const loadToolDescription = (verdict: Verdict, wrapper: boolean): string =>
  verdict === "advisory" ? loadToolDescriptions.advisory : wrapper ? loadToolDescriptions.wrapper : loadToolDescriptions.schemaEager

export const Parameters = Schema.Struct({
  tools: Schema.mutable(Schema.Array(Schema.String)).annotate({ description: "Deferred tool names to load" }),
})

type Metadata = {
  load_tool?: { tools: string[] }
  truncated?: boolean
}

// R12-004/R12-005: `name` counts as delivered when the model-visible history
// (the filterCompacted view handed to every execute as ctx.messages) holds a
// tool part whose marker says that part's output delivered the full content.
// Completed parts qualify only when not pruned by compaction (the prune stamp
// makes the model see "[Old tool result content cleared]" instead); error
// parts qualify by marker alone (the direct-call fallback marks deferred tools
// that failed without a prior load). The marker rides part.state.metadata —
// the record completeToolCall writes and failToolCall preserves.
export const delivered = (name: string, messages: SessionV1.WithParts[]): boolean =>
  messages.some((message) =>
    message.parts.some((part) => {
      if (part.type !== "tool") return false
      if (part.state.status !== "completed" && part.state.status !== "error") return false
      const tools = part.state.metadata?.load_tool?.tools
      if (!Array.isArray(tools) || !tools.includes(name)) return false
      return part.state.status === "error" || part.state.time.compacted === undefined
    }),
  )

// Three-way serve off the frozen (mode, wrapper) pair (R12-012): advisory and
// binding+wrapper serve the full schema (on wrapper sessions no listing entry
// carries the schema — this is its only channel, R12-005); binding without the
// wrapper serves the description only, the listing entry already carries the
// full schema (R12-010).
const fullServe = (entry: FrozenToolEntry, mode: Verdict, wrapper: boolean) => {
  const head = [`### ${entry.name}`, "", entry.fullDescription]
  return mode === "advisory" || wrapper
    ? [...head, "", "Input schema:", JSON.stringify(entry.jsonSchema, null, 2)].join("\n")
    : [...head, "", "The full input schema is already registered in the tool listing."].join("\n")
}

export const LoadTool = Tool.define<typeof Parameters, Metadata, PromptBase.Service | Provider.Service>(
  "load_tool",
  Effect.gen(function* () {
    const promptBase = yield* PromptBase.Service
    const providerService = yield* Provider.Service

    return {
      // The listing/payload description is overridden per session at
      // SessionTools.resolve; the advisory variant is the static default.
      description: loadToolDescriptions.advisory,
      parameters: Parameters,
      execute: (params: Schema.Schema.Type<typeof Parameters>, ctx: Tool.Context<Metadata>) =>
        Effect.gen(function* () {
          // session/tools.ts attaches the active model to builtin tool context;
          // absent means load_tool ran outside the session wiring.
          const model = ctx.extra?.model as Provider.Model | undefined
          if (!model) return yield* Effect.fail(new Error("load_tool was executed without the session model context"))
          const provider = yield* providerService.getProvider(model.providerID)
          const base = yield* promptBase.entries({ sessionID: ctx.sessionID, model, provider })
          // A missing mode with entries present cannot happen (the mode is
          // written with the first tool batch); advisory is the conservative
          // fallback — the schema must reach the model when it is not already
          // in the listing (R12-005). A missing wrapper defaults false —
          // round-1 semantics (the flag is written with the first batch).
          const mode = base.mode ?? "advisory"
          const wrapper = base.wrapper ?? false
          const loadable = base.entries.filter((entry) => entry.kind === "deferred").map((entry) => entry.name)
          const sections: string[] = []
          const served: string[] = []
          // Per requested name, deduplicated, in order; per-name errors are
          // output lines — never fatal to the turn (R00-010).
          for (const name of [...new Set(params.tools)]) {
            const entry = base.entries.find((item) => item.name === name)
            if (!entry) {
              sections.push(
                loadable.length > 0
                  ? `${name}: unknown tool. Loadable tools: ${loadable.join(", ")}`
                  : `${name}: unknown tool. No deferred tools are available to load.`,
              )
              continue
            }
            if (entry.kind === "eager") {
              sections.push(`${name}: already fully listed in the tool listing.`)
              continue
            }
            if (delivered(name, ctx.messages)) {
              sections.push(`${name}: already loaded.`)
              continue
            }
            sections.push(fullServe(entry, mode, wrapper))
            served.push(name)
          }
          return {
            title: `Loaded ${served.length} tool${served.length === 1 ? "" : "s"}`,
            output: sections.length > 0 ? sections.join("\n\n") : "No tool names provided.",
            metadata: {
              // Confirmation-only calls mark nothing: only parts whose output
              // actually delivered full content re-open delivered() later.
              ...(served.length > 0 ? { load_tool: { tools: served } } : {}),
              // The load output is the schema-delivery channel (R12-005);
              // opting out of the generic truncation pass keeps it intact.
              truncated: false,
            },
          }
        }).pipe(Effect.orDie),
    }
  }),
)