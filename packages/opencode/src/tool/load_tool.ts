import { Effect, Schema } from "effect"
import { SessionV1 } from "@opencode-ai/core/v1/session"
import { PromptBase } from "../session/llm/prompt-base"
import { Provider } from "../provider/provider"
import type { FrozenToolEntry, Verdict } from "../session/tool-listing"
import DESCRIPTION from "./load_tool.txt"
import { Tool } from "./tool"

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

const fullServe = (entry: FrozenToolEntry, mode: Verdict) => {
  const head = [`### ${entry.name}`, "", entry.fullDescription]
  // Advisory: the listing entry carries only the placeholder schema, so the
  // full schema reaches the model here (R12-005). Binding: the entry already
  // carries the full schema (R12-010), only the description was deferred.
  return mode === "advisory"
    ? [...head, "", "Input schema:", JSON.stringify(entry.jsonSchema, null, 2)].join("\n")
    : [...head, "", "The full input schema is already registered in the tool listing."].join("\n")
}

export const LoadTool = Tool.define<typeof Parameters, Metadata, PromptBase.Service | Provider.Service>(
  "load_tool",
  Effect.gen(function* () {
    const promptBase = yield* PromptBase.Service
    const providerService = yield* Provider.Service

    return {
      description: DESCRIPTION,
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
          // in the listing (R12-005).
          const mode = base.mode ?? "advisory"
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
            sections.push(fullServe(entry, mode))
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