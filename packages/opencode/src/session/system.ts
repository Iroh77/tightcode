import { LayerNode } from "@opencode-ai/core/effect/layer-node"
import { Context, Effect, Layer } from "effect"

import { InstanceState } from "@/effect/instance-state"

import PROMPT_ANTHROPIC from "./prompt/anthropic.txt"
import PROMPT_DEFAULT from "./prompt/default.txt"
import PROMPT_BEAST from "./prompt/beast.txt"
import PROMPT_GEMINI from "./prompt/gemini.txt"
import PROMPT_GPT from "./prompt/gpt.txt"
import PROMPT_ASTRA from "./prompt/gpt-astra.txt"
import PROMPT_KIMI from "./prompt/kimi.txt"
import PROMPT_META from "./prompt/meta.txt"

import PROMPT_CODEX from "./prompt/codex.txt"
import PROMPT_TRINITY from "./prompt/trinity.txt"
import type { Provider } from "@/provider/provider"
import type { Agent } from "@/agent/agent"
import { Permission } from "@/permission"
import { Skill } from "@/skill"
import { AbsolutePath } from "@opencode-ai/core/schema"
import { Location } from "@opencode-ai/core/location"
import { LocationServiceMap, locationServiceMapLayer } from "@opencode-ai/core/location-services"
import { Reference } from "@opencode-ai/core/reference"
import { MCP } from "@/mcp"
import { PermissionV1 } from "@opencode-ai/core/v1/permission"
import { RuntimeFlags } from "@/effect/runtime-flags"
import { ContextSlimmer } from "./context-slimmer"
import type { SystemBlock } from "./llm/prompt-base"

// Template selection exposed for the R11-007 override cascade: the override
// config key vocabulary is the built-in template name, so the upstream
// substring matching must be visible to the position-0 seam (request.ts).
// provider stays byte-identical — it composes base + render.
export type TemplateName = "anthropic" | "beast" | "codex" | "default" | "gemini" | "gpt" | "kimi" | "meta" | "trinity"

export function base(model: Provider.Model): { template: TemplateName; raw: string; name: string | undefined } {
  if (model.api.id.includes("muse")) {
    const name = model.api.id.includes("muse-glimmer") ? "Muse Glimmer" : "Muse Spark"
    return { template: "meta", raw: PROMPT_META, name }
  }
  if (model.api.id.includes("gpt-4") || model.api.id.includes("o1") || model.api.id.includes("o3"))
    return { template: "beast", raw: PROMPT_BEAST, name: undefined }
  if (model.api.id.includes("gpt")) {
    if (model.api.id.includes("gpt-6")) return { template: "gpt", raw: PROMPT_ASTRA, name: undefined }
    if (model.api.id.includes("codex")) {
      return { template: "codex", raw: PROMPT_CODEX, name: undefined }
    }
    return { template: "gpt", raw: PROMPT_GPT, name: undefined }
  }
  if (model.api.id.includes("gemini-")) return { template: "gemini", raw: PROMPT_GEMINI, name: undefined }
  if (model.api.id.includes("claude")) return { template: "anthropic", raw: PROMPT_ANTHROPIC, name: undefined }
  if (model.api.id.toLowerCase().includes("trinity")) return { template: "trinity", raw: PROMPT_TRINITY, name: undefined }
  if (
    model.api.id.toLowerCase().includes("kimi") ||
    ["kimi-for-coding", "moonshotai", "moonshotai-cn"].includes(model.providerID)
  )
    return { template: "kimi", raw: PROMPT_KIMI, name: undefined }
  return { template: "default", raw: PROMPT_DEFAULT, name: undefined }
}

// Shared rendering step for the built-in base and override values alike
// (Amendment 1 §3): the muse display name is substituted iff the selected
// template is meta — meta was the only template carrying the placeholder.
export function render(text: string, base: { name: string | undefined }): string {
  return base.name === undefined ? text : text.replaceAll("{{MODEL_NAME}}", base.name)
}

export function provider(model: Provider.Model) {
  const selected = base(model)
  return [render(selected.raw, selected)]
}

export interface Interface {
  readonly environment: (model: Provider.Model) => Effect.Effect<string[]>
  readonly skills: (agent: Agent.Info) => Effect.Effect<string | undefined>
  readonly mcp: (agent: Agent.Info, permission?: PermissionV1.Ruleset) => Effect.Effect<string | undefined>
  readonly mcpBlocks: (agent: Agent.Info, permission?: PermissionV1.Ruleset) => Effect.Effect<SystemBlock[]>
}

export class Service extends Context.Service<Service, Interface>()("@opencode/SystemPrompt") {}

const layer = Layer.effect(
  Service,
  Effect.gen(function* () {
    const skill = yield* Skill.Service
    const mcp = yield* MCP.Service
    const locations = yield* LocationServiceMap.Service
    const flags = yield* RuntimeFlags.Service

    const visibleServers = Effect.fnUntraced(function* (ruleset: PermissionV1.Ruleset) {
      return (yield* mcp.instructions())
        .filter((item) => item.tools.length === 0 || Permission.disabled(item.tools, ruleset).size < item.tools.length)
        .map((item) =>
          flags.disableStaticSlimming
            ? item
            : { ...item, instructions: ContextSlimmer.mcpInstructions(item.instructions) },
        )
    })

    return Service.of({
      environment: Effect.fn("SystemPrompt.environment")(function* (model: Provider.Model) {
        const ctx = yield* InstanceState.context
        const references = yield* Effect.gen(function* () {
          return (yield* (yield* Reference.Service).list()).filter((reference) => reference.description !== undefined)
        }).pipe(Effect.provide(locations.get(Location.Ref.make({ directory: AbsolutePath.make(ctx.directory) }))))
        return [
          [
            `You are powered by the model named ${model.api.id}. The exact model ID is ${model.providerID}/${model.api.id}`,
            `Here is some useful information about the environment you are running in:`,
            `<env>`,
            `  Working directory: ${ctx.directory}`,
            `  Workspace root folder: ${ctx.worktree}`,
            `  Is directory a git repo: ${ctx.project.vcs === "git" ? "yes" : "no"}`,
            `  Platform: ${process.platform}`,
            `  Today's date: ${new Date().toDateString()}`,
            `</env>`,
          ].join("\n"),
          references.length === 0
            ? undefined
            : [
                "Project references provide additional directories that can be accessed when relevant.",
                "<available_references>",
                ...references
                  .toSorted((a, b) => a.name.localeCompare(b.name))
                  .flatMap((reference) => [
                    "  <reference>",
                    `    <name>${reference.name}</name>`,
                    `    <path>${reference.path}</path>`,
                    ...(reference.description === undefined
                      ? []
                      : [`    <description>${reference.description}</description>`]),
                    "  </reference>",
                  ]),
                "</available_references>",
              ].join("\n"),
        ].filter((part): part is string => part !== undefined)
      }),

      skills: Effect.fn("SystemPrompt.skills")(function* (agent: Agent.Info) {
        if (Permission.disabled(["skill"], agent.permission).has("skill")) return

        const list = yield* skill.available(agent)

        return [
          "Skills provide specialized instructions and workflows for specific tasks.",
          "Use the skill tool to load a skill when a task matches its description.",
          Skill.fmt(list, { verbose: flags.disableStaticSlimming }),
        ].join("\n")
      }),

      mcp: Effect.fn("SystemPrompt.mcp")(function* (agent: Agent.Info, permission?: PermissionV1.Ruleset) {
        const instructions = yield* visibleServers(Permission.merge(agent.permission, permission ?? []))
        if (instructions.length === 0) return

        return [
          "<mcp_instructions>",
          ...instructions.flatMap((item) => serverSection(item)),
          "</mcp_instructions>",
        ].join("\n")
      }),

      mcpBlocks: Effect.fn("SystemPrompt.mcpBlocks")(function* (agent: Agent.Info, permission?: PermissionV1.Ruleset) {
        const instructions = yield* visibleServers(Permission.merge(agent.permission, permission ?? []))
        return instructions.map(
          (item): SystemBlock => ({ key: `mcp:${item.name}`, content: serverSection(item).join("\n") }),
        )
      }),
    })
  }),
)

function serverSection(item: { name: string; instructions: string }) {
  return [
    `  <server name="${item.name}">`,
    ...item.instructions.split("\n").map((line) => `    ${line}`),
    "  </server>",
  ]
}

const locationServiceMapNode = LayerNode.make({
  service: LocationServiceMap.Service,
  layer: locationServiceMapLayer,
  deps: [],
})

export const node = LayerNode.make({
  service: Service,
  layer: layer,
  deps: [Skill.node, MCP.node, locationServiceMapNode, RuntimeFlags.node],
})

export * as SystemPrompt from "./system"
