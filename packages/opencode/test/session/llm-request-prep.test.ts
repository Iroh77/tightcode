import { describe, expect } from "bun:test"
import { LayerNode } from "@opencode-ai/core/effect/layer-node"
import { ModelV2 } from "@opencode-ai/core/model"
import { ProviderV2 } from "@opencode-ai/core/provider"
import { Effect } from "effect"
import type { Agent } from "../../src/agent/agent"
import { RuntimeFlags } from "../../src/effect/runtime-flags"
import { LLMRequestPrep } from "../../src/session/llm/request"
import { PromptBase, type SystemBlock } from "../../src/session/llm/prompt-base"
import type { Plugin } from "../../src/plugin"
import type { Provider } from "../../src/provider/provider"
import { MessageID, SessionID } from "../../src/session/schema"
import { SystemPrompt } from "../../src/session/system"
import { testEffect } from "../lib/effect"

const it = testEffect(LayerNode.compile(LayerNode.group([PromptBase.node, RuntimeFlags.node])))

const model: Provider.Model = {
  id: ModelV2.ID.make("test-model"),
  providerID: ProviderV2.ID.make("test"),
  api: { id: "test-model", url: "https://api.test", npm: "@ai-sdk/test" },
  name: "Test Model",
  capabilities: {
    temperature: true,
    reasoning: false,
    attachment: false,
    toolcall: true,
    input: { text: true, audio: false, image: false, video: false, pdf: false },
    output: { text: true, audio: false, image: false, video: false, pdf: false },
    interleaved: false,
  },
  cost: { input: 0, output: 0, cache: { read: 0, write: 0 } },
  limit: { context: 128000, output: 8192 },
  status: "active",
  options: {},
  headers: {},
  release_date: "",
}

const provider: Provider.Info = {
  id: ProviderV2.ID.make("test"),
  name: "Test",
  source: "custom",
  env: [],
  options: {},
  models: {},
}

const agent: Agent.Info = {
  name: "build",
  mode: "primary",
  permission: [],
  options: {},
}

const plugin: Plugin.Interface = {
  trigger: (_name, _input, output) => Effect.succeed(output),
  list: () => Effect.succeed([]),
  init: () => Effect.void,
}

const blocks = (instructions: string): SystemBlock[] => [
  { key: "environment", content: "env block" },
  { key: "instructions", content: instructions },
  { key: "skills", content: "skills block" },
]

const expectedSystem = (turnBlocks: SystemBlock[]) =>
  [[...SystemPrompt.provider(model), ...PromptBase.render(turnBlocks)].filter((x) => x).join("\n")]

describe("session.llm-request-prep.prompt-freeze", () => {
  const prepareWith = (input: {
    promptBase: PromptBase.Interface
    flags: RuntimeFlags.Info
    system?: SystemBlock[]
    small?: boolean
  }) =>
    Effect.gen(function* () {
      return yield* LLMRequestPrep.prepare({
        user: {
          id: MessageID.ascending(),
          sessionID: SessionID.make("ses_prep"),
          role: "user",
          time: { created: 0 },
          agent: "build",
          model: { providerID: ProviderV2.ID.make("test"), modelID: ModelV2.ID.make("test-model") },
        },
        sessionID: "ses_prep",
        model,
        agent,
        permission: [],
        system: input.system ?? blocks("instructions v1"),
        messages: [],
        small: input.small,
        tools: {},
        provider,
        auth: undefined,
        plugin,
        promptBase: input.promptBase,
        flags: input.flags,
        isWorkflow: false,
      })
    })

  it.instance("freezes system blocks across turns; later turns re-serve the first write", () =>
    Effect.gen(function* () {
      const promptBase = yield* PromptBase.Service
      const base = yield* RuntimeFlags.Service
      const first = yield* prepareWith({ promptBase, flags: base })
      expect(first.system).toEqual(expectedSystem(blocks("instructions v1")))

      const second = yield* prepareWith({
        promptBase,
        flags: base,
        system: blocks("instructions v2 CHANGED"),
      })
      expect(second.system).toEqual(first.system)
    }),
  )

  it.instance("kill-switch restores upstream per-turn bytes", () =>
    Effect.gen(function* () {
      const promptBase = yield* PromptBase.Service
      const base = yield* RuntimeFlags.Service
      const frozen = yield* prepareWith({ promptBase, flags: base })
      const bypass = yield* prepareWith({
        promptBase,
        flags: { ...base, disableLazyTools: true },
        system: blocks("instructions v2 CHANGED"),
      })
      expect(bypass.system).toEqual(expectedSystem(blocks("instructions v2 CHANGED")))
      expect(bypass.system).not.toEqual(frozen.system)
    }),
  )

  it.instance("small turns bypass the freeze entirely", () =>
    Effect.gen(function* () {
      const promptBase = yield* PromptBase.Service
      const base = yield* RuntimeFlags.Service
      const small = yield* prepareWith({
        promptBase,
        flags: base,
        small: true,
        system: blocks("instructions small-turn"),
      })
      expect(small.system).toEqual(expectedSystem(blocks("instructions small-turn")))

      const normal = yield* prepareWith({ promptBase, flags: base, system: blocks("instructions v1") })
      expect(normal.system).toEqual(expectedSystem(blocks("instructions v1")))

      const stable = yield* prepareWith({ promptBase, flags: base, system: blocks("instructions v2 CHANGED") })
      expect(stable.system).toEqual(normal.system)
    }),
  )
})
