import { PermissionV1 } from "@opencode-ai/core/v1/permission"
import type { Auth } from "@/auth"
import { SessionV1 } from "@opencode-ai/core/v1/session"
import type { RuntimeFlags } from "@/effect/runtime-flags"
import { InstanceState } from "@/effect/instance-state"
import { Permission } from "@/permission"
import type { Agent } from "@/agent/agent"
import type { MessageV2 } from "../message-v2"
import type { Provider } from "@/provider/provider"
import { ProviderTransform } from "@/provider/transform"
import { SystemPrompt } from "../system"
import { ToolListing, type FrozenToolEntry, type ToolSeed, type Verdict } from "@/session/tool-listing"
import { PromptBase } from "./prompt-base"
import type { SystemBlock } from "./prompt-base"
import { PromptCapture } from "./prompt-capture"
import { InstallationVersion } from "@opencode-ai/core/installation/version"
import { Effect, Record } from "effect"
import { jsonSchema, tool as aiTool, type ModelMessage, type Tool } from "ai"
import type { Plugin } from "@/plugin"
import { mergeDeep } from "remeda"

const USER_AGENT = `opencode/${InstallationVersion}`

type PrepareInput = {
  readonly user: SessionV1.User
  readonly sessionID: string
  readonly parentSessionID?: string
  readonly model: Provider.Model
  readonly agent: Agent.Info
  readonly permission?: PermissionV1.Ruleset
  readonly system: SystemBlock[]
  readonly messages: ModelMessage[]
  readonly small?: boolean
  readonly tools: Record<string, Tool>
  readonly toolSeeds?: ToolSeed[]
  readonly toolVerdict?: Verdict
  readonly provider: Provider.Info
  readonly auth: Auth.Info | undefined
  readonly plugin: Plugin.Interface
  readonly promptBase: PromptBase.Interface
  readonly flags: RuntimeFlags.Info
  // App data dir, captured at the LLM layer build (Global is a build-time
  // dependency there, not a runtime service — the capture sink must not add
  // one to the per-turn context).
  readonly data: string
  readonly isWorkflow: boolean
}

export type Prepared = {
  readonly system: string[]
  readonly messages: ModelMessage[]
  readonly tools: Record<string, Tool>
  readonly params: {
    readonly temperature?: number
    readonly topP?: number
    readonly topK?: number
    readonly maxOutputTokens?: number
    readonly options: Record<string, any>
  }
  readonly messageTransformOptions: Record<string, any>
  readonly headers: Record<string, string>
}

const mergeOptions = (target: Record<string, any>, source: Record<string, any> | undefined): Record<string, any> =>
  mergeDeep(target, source ?? {}) as Record<string, any>

// A frozen listing entry whose execute closure the per-turn record no longer
// produces (e.g. its MCP server died mid-session) stays listed (R12-007:
// nothing is ever removed) with an execute that fails with a clear error —
// the AI SDK silently drops execute-less tools, leaving a dangling tool-call
// that only surfaces as a provider 400 on the next turn.
// The returned promise always rejects; the shape matches the AI SDK's tool
// result so the failing execute typechecks as a real tool.
const unavailableExecute = (name: string) => async (): Promise<{ output: string; title: string; metadata: object }> => {
  throw new Error(
    `The ${name} tool is no longer available in this session: the component that produced it stopped doing so mid-session (for example its MCP server disconnected).`,
  )
}

// Impose (detailed design [c]): the frozen tool base is the authoritative
// listing shape at the payload. Every frozen name is projected with its frozen
// description/inputSchema, pairing execute closures from the per-turn record
// while they exist; names outside the base (StructuredOutput, _noop) pass
// through untouched. The permission/user.tools filter downstream stays
// permissive-relevant for the current agent (R12-001 dominates after impose).
const impose = (tools: Record<string, Tool>, frozen: { entries: FrozenToolEntry[]; mode: Verdict }): Record<string, Tool> => {
  const result: Record<string, Tool> = {}
  for (const entry of ToolListing.render(frozen.entries, frozen.mode)) {
    const live = tools[entry.name]
    result[entry.name] = live
      ? { ...live, description: entry.description, inputSchema: jsonSchema(entry.jsonSchema) }
      : aiTool({ description: entry.description, inputSchema: jsonSchema(entry.jsonSchema), execute: unavailableExecute(entry.name) })
  }
  for (const [name, tool] of Object.entries(tools)) {
    if (name in result) continue
    result[name] = tool
  }
  return result
}

export const prepare = Effect.fn("LLMRequestPrep.prepare")(function* (input: PrepareInput) {
  const isOpenaiOauth = input.provider.id === "openai" && input.auth?.type === "oauth"
  // Kill-switch (SC-3) and small turns (summary/compaction) stay per-turn
  // upstream behavior — no reconcile, no impose, no frozen state.
  const bypass = input.small || input.flags.disableLazyTools
  const systemBlocks = bypass
    ? input.system
    : (
        yield* input.promptBase.reconcileSystem({
          sessionID: input.sessionID,
          model: input.model,
          provider: input.provider,
          blocks: input.system,
        })
      ).blocks
  const frozenTools = bypass
    ? undefined
    : yield* input.promptBase.reconcileTools({
        sessionID: input.sessionID,
        model: input.model,
        provider: input.provider,
        seeds: input.toolSeeds ?? [],
        // The resolved R12-010 verdict rides the stream input from
        // SessionTools.resolve; the base freezes it at the first write
        // (reconcileTools returns the frozen mode, so a mid-session observe
        // flip never re-renders written entries). A missing verdict is an
        // unresolvable one — conservative binding (R12-010).
        mode: input.toolVerdict ?? "binding",
      })
  const system = [
    [
      ...(input.agent.prompt ? [input.agent.prompt] : SystemPrompt.provider(input.model)),
      ...PromptBase.render(systemBlocks),
      ...(input.user.system ? [input.user.system] : []),
    ]
      .filter((x) => x)
      .join("\n"),
  ]

  const header = system[0]
  yield* input.plugin.trigger(
    "experimental.chat.system.transform",
    { sessionID: input.sessionID, model: input.model },
    { system },
  )
  if (system.length > 2 && system[0] === header) {
    const rest = system.slice(1)
    system.length = 0
    system.push(header, rest.join("\n"))
  }

  const variant =
    !input.small && input.model.variants && input.user.model.variant
      ? input.model.variants[input.user.model.variant]
      : {}
  const base = input.small
    ? ProviderTransform.smallOptions(input.model)
    : ProviderTransform.options({
        model: input.model,
        sessionID: input.sessionID,
        providerOptions: input.provider.options,
      })
  const options = mergeOptions(mergeOptions(mergeOptions(base, input.model.options), input.agent.options), variant)
  if (
    input.model.api.npm === "@ai-sdk/azure" &&
    (input.provider.options.useCompletionUrls || input.model.options.useCompletionUrls || options.useCompletionUrls)
  ) {
    delete options.reasoningSummary
    delete options.include
  }
  if (isOpenaiOauth) options.instructions = system.join("\n")

  const messages =
    isOpenaiOauth || input.isWorkflow
      ? input.messages
      : [
          ...system.map(
            (x): ModelMessage => ({
              role: "system",
              content: x,
            }),
          ),
          ...input.messages,
        ]

  const params = yield* input.plugin.trigger(
    "chat.params",
    {
      sessionID: input.sessionID,
      agent: input.agent.name,
      model: input.model,
      provider: input.provider,
      message: input.user,
    },
    {
      temperature: input.model.capabilities.temperature
        ? (input.agent.temperature ?? ProviderTransform.temperature(input.model))
        : undefined,
      topP: input.agent.topP ?? ProviderTransform.topP(input.model),
      topK: ProviderTransform.topK(input.model),
      maxOutputTokens: ProviderTransform.maxOutputTokens(input.model, input.flags.outputTokenMax),
      options,
    },
  )

  const { headers } = yield* input.plugin.trigger(
    "chat.headers",
    {
      sessionID: input.sessionID,
      agent: input.agent.name,
      model: input.model,
      provider: input.provider,
      message: input.user,
    },
    {
      headers: {},
    },
  )

  const tools = resolveTools({ ...input, tools: frozenTools ? impose(input.tools, frozenTools) : input.tools })
  // Codex parity: OpenAI Responses-family providers hardcode `strict: false`
  // on every function tool so MCP-sourced and dynamic schemas that don't
  // satisfy OpenAI's structured-outputs constraints still register.
  if (
    input.model.api.npm === "@ai-sdk/openai" ||
    input.model.api.npm === "@ai-sdk/azure" ||
    input.model.api.npm === "@ai-sdk/amazon-bedrock/mantle"
  ) {
    for (const key of Object.keys(tools)) tools[key] = { ...tools[key], strict: false }
  }
  if (
    input.model.providerID.includes("github-copilot") &&
    Object.keys(tools).length === 0 &&
    hasToolCalls(input.messages)
  ) {
    // Copilot needs a tools field when replaying prior tool calls, even if no tools are currently enabled.
    tools["_noop"] = aiTool({
      description: "Do not call this tool. It exists only for API compatibility and must never be invoked.",
      inputSchema: jsonSchema({
        type: "object",
        properties: {
          reason: { type: "string", description: "Unused" },
        },
      }),
      execute: async () => ({ output: "", title: "", metadata: {} }),
    })
  }

  const opencodeProjectID = input.model.providerID.startsWith("opencode")
    ? (yield* InstanceState.context).project.id
    : undefined

  const prepared = {
    system,
    messages,
    tools: Object.fromEntries(Object.entries(tools).toSorted(([a], [b]) => a.localeCompare(b))),
    params,
    messageTransformOptions: options,
    headers: {
      ...(input.model.providerID.startsWith("opencode")
        ? {
            ...(opencodeProjectID ? { "x-opencode-project": opencodeProjectID } : {}),
            "x-opencode-session": input.sessionID,
            "x-opencode-request": input.user.id,
            "x-opencode-client": input.flags.client,
            "User-Agent": USER_AGENT,
          }
        : {
            "x-session-affinity": input.sessionID,
            "X-Session-Id": input.sessionID,
            "User-Agent": USER_AGENT,
          }),
      ...(input.parentSessionID ? { "x-parent-session-id": input.parentSessionID } : {}),
      ...input.model.headers,
      ...headers,
    },
  }
  // Gated at the call site (decision context-observability-01): flag off = no
  // IO. The dump sees the exact returned object (SC-1).
  if (input.flags.enablePromptCapture)
    yield* PromptCapture.dump({
      data: input.data,
      prepared,
      meta: {
        sessionID: input.sessionID,
        parentSessionID: input.parentSessionID,
        providerID: input.provider.id,
        modelID: input.model.id,
        modelApi: input.model.api.npm,
        agent: input.agent.name,
        small: input.small ?? false,
        requestID: input.user.id,
        optimized: {
          lazyTools: !input.flags.disableLazyTools,
          staticSlimming: !input.flags.disableStaticSlimming,
        },
      },
    })
  return prepared
})

function resolveTools(input: Pick<PrepareInput, "tools" | "agent" | "permission" | "user">) {
  const disabled = Permission.disabled(
    Object.keys(input.tools),
    Permission.merge(input.agent.permission, input.permission ?? []),
  )
  return Record.filter(input.tools, (_, k) => input.user.tools?.[k] !== false && !disabled.has(k))
}

export function hasToolCalls(messages: ModelMessage[]): boolean {
  for (const msg of messages) {
    if (!Array.isArray(msg.content)) continue
    for (const part of msg.content) {
      if (part.type === "tool-call" || part.type === "tool-result") return true
    }
  }
  return false
}

export * as LLMRequestPrep from "./request"
