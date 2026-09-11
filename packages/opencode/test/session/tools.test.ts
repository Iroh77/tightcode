import { describe, expect } from "bun:test"
import type { JSONSchema7 } from "@ai-sdk/provider"
import { ModelV2 } from "@opencode-ai/core/model"
import { ProviderV2 } from "@opencode-ai/core/provider"
import { SessionV1 } from "@opencode-ai/core/v1/session"
import { Agent } from "@/agent/agent"
import { BindingVerdict } from "@/session/binding-verdict"
import { MCP } from "@/mcp"
import { Permission } from "@/permission"
import { Provider } from "@/provider/provider"
import { Session } from "@/session/session"
import { MessageID, PartID, SessionID } from "@/session/schema"
import { SessionProcessor } from "@/session/processor"
import { SessionTools } from "@/session/tools"
import { Tool } from "@/tool/tool"
import { ToolRegistry } from "@/tool/registry"
import { Truncate } from "@/tool/truncate"
import { Plugin } from "@/plugin"
import { RuntimeFlags } from "@/effect/runtime-flags"
import { Effect, Layer, Schema } from "effect"
import { jsonSchema, type ToolExecutionOptions } from "ai"
import type { Verdict } from "@/session/tool-listing"
import { testEffect } from "../lib/effect"

const callID = "call-test"
const sessionID = SessionID.make("ses_test")
const messageID = MessageID.ascending()
const partID = PartID.ascending()

const agent: Agent.Info = {
  name: "build",
  mode: "primary",
  options: {},
  permission: [{ permission: "*", pattern: "*", action: "allow" }],
}

const model = {
  providerID: ProviderV2.ID.make("test"),
  api: { id: "test-model" },
} as Provider.Model

function fakeMcp() {
  return MCP.Service.of({
    tools: () => Effect.succeed({}),
    clients: () => Effect.succeed({}),
  } as Partial<MCP.Interface> as MCP.Interface)
}

const fakePlugin = Plugin.Service.of({
  init: () => Effect.void,
  list: () => Effect.succeed([]),
  trigger: (_name, _input, output) => Effect.succeed(output),
} satisfies Plugin.Interface)

const fakePermission = Permission.Service.of({
  ask: () => Effect.void,
  reply: () => Effect.void,
  list: () => Effect.succeed([]),
} satisfies Permission.Interface)

const fakeTruncate = Truncate.Service.of({
  cleanup: () => Effect.void,
  write: () => Effect.succeed("output.txt"),
  output: (text: string) => Effect.succeed({ content: text, truncated: false }),
  limits: () => Effect.succeed({ maxLines: 2000, maxBytes: 50 * 1024 }),
} satisfies Truncate.Interface)

const sessionStub = { id: sessionID, permission: [] } as unknown as Session.Info
const promptOpsStub = {} as never

const timingRegistry = Layer.succeed(
  ToolRegistry.Service,
  ToolRegistry.Service.of({
    ids: () => Effect.succeed(["timing"]),
    all: () => Effect.succeed([]),
    named: () => Effect.die("unused"),
    tools: () =>
      Effect.succeed([
        {
          id: "timing",
          description: "updates metadata more than once",
          parameters: Schema.Struct({}),
          jsonSchema: { type: "object", properties: {} },
          execute: (_args, ctx) =>
            Effect.gen(function* () {
              yield* ctx.metadata({ metadata: { output: "first" } })
              yield* ctx.metadata({ metadata: { output: "second" } })
              return { title: "timing", metadata: {}, output: "done" }
            }),
        } satisfies Tool.Def,
      ]),
  }),
)

const baseLayer = (
  options: { flags?: Partial<RuntimeFlags.Info>; registry?: Layer.Layer<ToolRegistry.Service> } = {},
) =>
  Layer.mergeAll(
    Layer.succeed(Plugin.Service, fakePlugin),
    Layer.succeed(Permission.Service, fakePermission),
    Layer.succeed(MCP.Service, fakeMcp()),
    Layer.succeed(Truncate.Service, fakeTruncate),
    RuntimeFlags.layer(options.flags),
    options.registry ?? timingRegistry,
  )

const providerInfo: Provider.Info = {
  id: ProviderV2.ID.make("test"),
  name: "Test",
  source: "custom",
  env: [],
  options: {},
  models: {},
}

const calls: BindingVerdict.Target[] = []

const verdictStub = (verdict: Verdict | undefined) =>
  Layer.succeed(
    BindingVerdict.Service,
    BindingVerdict.Service.of({
      resolve: (target) =>
        verdict === undefined
          ? Effect.die("BindingVerdict.resolve must not be called")
          : Effect.sync(() => {
              calls.push(target)
              return verdict
            }),
      observe: () => Effect.void,
    }),
  )

const providerStub = Layer.mock(Provider.Service, {
  getProvider: () => Effect.succeed(providerInfo),
})

const layer = baseLayer()

const it = testEffect(Layer.mergeAll(layer, verdictStub("advisory"), providerStub))

it.effect("preserves running tool start time across metadata updates", () =>
  Effect.gen(function* () {
    const state: SessionV1.ToolPart = {
      id: partID,
      sessionID,
      messageID,
      type: "tool",
      tool: "timing",
      callID,
      state: {
        status: "running",
        input: {},
        time: { start: 100 },
      },
    }
    const updates: number[] = []
    const processor = {
      message: {
        id: messageID,
        sessionID,
        role: "assistant",
        parentID: MessageID.ascending(),
        agent: "build",
        mode: "build",
        path: { cwd: "/tmp", root: "/tmp" },
        cost: 0,
        tokens: { input: 0, output: 0, reasoning: 0, cache: { read: 0, write: 0 } },
        modelID: ModelV2.ID.make("test-model"),
        providerID: ProviderV2.ID.make("test"),
        time: { created: 1 },
      } satisfies SessionV1.Assistant,
      updateToolCall: (_toolCallID, update) =>
        Effect.sync(() => {
          const next = update(state)
          state.state = next.state
          if (state.state.status === "running") updates.push(state.state.time.start)
          return state
        }),
      completeToolCall: () => Effect.void,
    } satisfies Pick<SessionProcessor.Handle, "message" | "updateToolCall" | "completeToolCall">

    const resolved = yield* SessionTools.resolve({
      agent,
      model,
      session: sessionStub,
      processor,
      bypassAgentCheck: false,
      messages: [],
      promptOps: promptOpsStub,
    })
    const execute = resolved.tools.timing.execute
    if (!execute) throw new Error("timing tool is missing execute")

    yield* Effect.promise(() =>
      execute(
        {},
        {
          toolCallId: callID,
          abortSignal: new AbortController().signal,
          messages: [],
        },
      ),
    )

    expect(updates).toEqual([100, 100])
    expect(state.state.status).toBe("running")
    if (state.state.status === "running") {
      expect(state.state.time.start).toBe(100)
    }
  }),
)

describe("session.tools lazy listing (ticket 05)", () => {
  const long = "find files by glob patterns " + "y".repeat(80)
  const globSchema: JSONSchema7 = {
    type: "object",
    properties: { pattern: { type: "string" } },
    required: ["pattern"],
  }

  const globRegistry = Layer.succeed(
    ToolRegistry.Service,
    ToolRegistry.Service.of({
      ids: () => Effect.succeed(["glob"]),
      all: () => Effect.succeed([]),
      named: () => Effect.die("unused"),
      tools: () =>
        Effect.succeed([
          {
            id: "glob",
            description: long,
            parameters: Schema.Struct({}),
            jsonSchema: globSchema,
            execute: () => Effect.succeed({ title: "glob", metadata: {}, output: "" }),
          } satisfies Tool.Def,
        ]),
    }),
  )

  const resolveGlob = Effect.gen(function* () {
    return yield* SessionTools.resolve({
      agent,
      model,
      session: sessionStub,
      processor: {
        message: {
          id: messageID,
          sessionID,
          role: "assistant",
          parentID: MessageID.ascending(),
          agent: "build",
          mode: "build",
          path: { cwd: "/tmp", root: "/tmp" },
          cost: 0,
          tokens: { input: 0, output: 0, reasoning: 0, cache: { read: 0, write: 0 } },
          modelID: ModelV2.ID.make("test-model"),
          providerID: ProviderV2.ID.make("test"),
          time: { created: 1 },
        } satisfies SessionV1.Assistant,
        updateToolCall: () => Effect.succeed(undefined),
        completeToolCall: () => Effect.void,
      } satisfies Pick<SessionProcessor.Handle, "message" | "updateToolCall" | "completeToolCall">,
      bypassAgentCheck: false,
      messages: [],
      promptOps: promptOpsStub,
    })
  })

  const itBinding = testEffect(
    Layer.mergeAll(baseLayer({ registry: globRegistry }), verdictStub("binding"), providerStub),
  )
  const itAdvisory = testEffect(
    Layer.mergeAll(baseLayer({ registry: globRegistry }), verdictStub("advisory"), providerStub),
  )
  const itKillSwitch = testEffect(
    Layer.mergeAll(
      baseLayer({ flags: { disableLazyTools: true }, registry: globRegistry }),
      verdictStub(undefined),
      providerStub,
    ),
  )

  itBinding.effect("resolves the verdict at resolve time and shapes the per-turn listing view with it (binding)", () =>
    Effect.gen(function* () {
      calls.length = 0
      const resolved = yield* resolveGlob
      // resolved before the session's first provider request, scoped to the (provider, model) tuple
      expect(calls).toEqual([{ model, provider: providerInfo }])
      expect(resolved.verdict).toBe("binding")
      // binding: the per-turn AITool record carries the full schema in place of the placeholder
      expect(resolved.tools.glob.inputSchema).toEqual(jsonSchema(globSchema))
      expect(resolved.tools.glob.description).toBe(long.slice(0, long.lastIndexOf(" ")) + "...")
      // seeds stay full facts (mode-independent first-write source)
      expect(resolved.seeds[0].fullDescription).toBe(long)
      expect(resolved.seeds[0].jsonSchema).toEqual(globSchema)
    }),
  )

  itAdvisory.effect("advisory verdict keeps the constant placeholder in the per-turn listing view", () =>
    Effect.gen(function* () {
      calls.length = 0
      const resolved = yield* resolveGlob
      expect(resolved.verdict).toBe("advisory")
      expect(resolved.tools.glob.inputSchema).toEqual(jsonSchema({ type: "object", properties: {} }))
    }),
  )

  itKillSwitch.effect("kill-switch: the verdict is never resolved, upstream per-turn shapes pass through", () =>
    Effect.gen(function* () {
      const resolved = yield* resolveGlob
      expect(resolved.verdict).toBeUndefined()
      expect(resolved.seeds).toEqual([])
      expect(resolved.tools.glob.description).toBe(long)
      expect(resolved.tools.glob.inputSchema).toEqual(jsonSchema(globSchema))
    }),
  )
})

describe("session.tools direct-call fallback (ticket 07)", () => {
  const globSchema: JSONSchema7 = {
    type: "object",
    properties: { pattern: { type: "string" } },
    required: ["pattern"],
  }
  const messageID = MessageID.ascending()

  const failingGlobRegistry = Layer.succeed(
    ToolRegistry.Service,
    ToolRegistry.Service.of({
      ids: () => Effect.succeed(["glob"]),
      all: () => Effect.succeed([]),
      named: () => Effect.die("unused"),
      tools: () =>
        Effect.succeed([
          {
            id: "glob",
            description: "finds files",
            parameters: Schema.Struct({}),
            jsonSchema: globSchema,
            // Production shape: Tool.wrap raises the decode failure as a
            // defect (Effect.orDie), which is what the fallback catches.
            execute: () => Effect.die(new Tool.InvalidArgumentsError({ tool: "glob", detail: "missing pattern" })),
          } satisfies Tool.Def,
        ]),
    }),
  )

  const observed: string[] = []
  const verdictLearn = Layer.succeed(
    BindingVerdict.Service,
    BindingVerdict.Service.of({
      resolve: () => Effect.succeed("advisory" as Verdict),
      observe: () =>
        Effect.sync(() => {
          observed.push("schema-violation")
        }),
    }),
  )

  const itFallback = testEffect(
    Layer.mergeAll(baseLayer({ registry: failingGlobRegistry }), verdictLearn, providerStub),
  )
  const itOff = testEffect(
    Layer.mergeAll(
      baseLayer({ flags: { disableLazyTools: true }, registry: failingGlobRegistry }),
      verdictStub(undefined),
      providerStub,
    ),
  )

  const runningGlobPart = (): SessionV1.ToolPart => ({
    id: PartID.ascending(),
    sessionID,
    messageID,
    type: "tool",
    tool: "glob",
    callID,
    state: { status: "running", input: {}, time: { start: 0 } },
  })

  const failingResolveInput = (state: SessionV1.ToolPart) => ({
    agent,
    model,
    session: sessionStub,
    processor: {
      message: {
        id: messageID,
        sessionID,
        role: "assistant",
        parentID: MessageID.ascending(),
        agent: "build",
        mode: "build",
        path: { cwd: "/tmp", root: "/tmp" },
        cost: 0,
        tokens: { input: 0, output: 0, reasoning: 0, cache: { read: 0, write: 0 } },
        modelID: ModelV2.ID.make("test-model"),
        providerID: ProviderV2.ID.make("test"),
        time: { created: 1 },
      } satisfies SessionV1.Assistant,
      updateToolCall: (_toolCallID: string, update: (part: SessionV1.ToolPart) => SessionV1.ToolPart) =>
        Effect.sync(() => {
          state.state = update(state).state
          return state
        }),
      completeToolCall: () => Effect.void,
    } satisfies Pick<SessionProcessor.Handle, "message" | "updateToolCall" | "completeToolCall">,
    bypassAgentCheck: false,
    messages: [] as SessionV1.WithParts[],
    promptOps: promptOpsStub,
  })

  const rethrown = (execute: (args: unknown, options: ToolExecutionOptions) => Promise<unknown>) =>
    Effect.promise(() =>
      execute({}, { toolCallId: callID, messages: [], abortSignal: new AbortController().signal }).then(
        () => new Error("expected the call to fail"),
        (error: unknown) => error,
      ),
    )

  itFallback.effect("a failed deferred call appends the schema, marks the part and observes the verdict", () =>
    Effect.gen(function* () {
      observed.length = 0
      const state = runningGlobPart()
      const resolved = yield* SessionTools.resolve(failingResolveInput(state))

      const execute = resolved.tools.glob.execute
      if (!execute) throw new Error("glob is missing execute")
      const rejection = yield* rethrown(execute)
      expect(rejection).toBeInstanceOf(Error)
      expect((rejection as Error).message).toContain("missing pattern")
      expect((rejection as Error).message).toContain(JSON.stringify(globSchema, null, 2))
      if (state.state.status !== "running") throw new Error("part should still be running")
      expect(state.state.metadata?.load_tool).toEqual({ tools: ["glob"] })
      expect(observed).toEqual(["schema-violation"])
    }),
  )

  itOff.effect("kill-switch: a failed deferred call stays upstream — no schema append, no marker, no observe", () =>
    Effect.gen(function* () {
      observed.length = 0
      const state = runningGlobPart()
      const resolved = yield* SessionTools.resolve(failingResolveInput(state))

      const execute = resolved.tools.glob.execute
      if (!execute) throw new Error("glob is missing execute")
      const rejection = yield* rethrown(execute)
      expect(rejection).toBeInstanceOf(Tool.InvalidArgumentsError)
      expect((rejection as Error).message).not.toContain(JSON.stringify(globSchema))
      if (state.state.status !== "running") throw new Error("part should still be running")
      expect(state.state.metadata?.load_tool).toBeUndefined()
      expect(observed).toEqual([])
    }),
  )
})
