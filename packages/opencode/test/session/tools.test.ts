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
import { DEFERRED_TOOL_DESCRIPTION, DEFERRED_TOOL_SCHEMA } from "@/tool/deferred_tool"
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
              return { verdict, provenance: { origin: "static-table" as const } }
            }),
    }),
  )


const layer = baseLayer()


const processorStub = (state: SessionV1.ToolPart, onUpdate?: (part: SessionV1.ToolPart) => void) =>
  ({
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
        const next = update(state)
        state.state = next.state
        onUpdate?.(state)
        return state
      }),
    completeToolCall: () => Effect.void,
  }) satisfies Pick<SessionProcessor.Handle, "message" | "updateToolCall" | "completeToolCall">

const it = testEffect(Layer.mergeAll(layer, verdictStub("advisory")))

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
    const processor = processorStub(state, (part) => {
      if (part.state.status === "running") updates.push(part.state.time.start)
    })


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

const envelopeRegistry = Layer.succeed(
  ToolRegistry.Service,
  ToolRegistry.Service.of({
    ids: () => Effect.succeed(["inner"]),
    all: () => Effect.succeed([]),
    named: () => Effect.die("unused"),
    tools: () =>
      Effect.succeed([
        {
          id: "inner",
          description: "streams progress under the wrapper",
          parameters: Schema.Struct({}),
          jsonSchema: { type: "object", properties: {} },
          execute: (_args, ctx) =>
            Effect.gen(function* () {
              yield* ctx.metadata({ title: "progress", metadata: { step: 1 } })
              return { title: "inner", metadata: {}, output: "done" }
            }),
        } satisfies Tool.Def,
      ]),
  }),
)

const envelopeIt = testEffect(
  Layer.mergeAll(baseLayer({ registry: envelopeRegistry }), verdictStub("advisory")),
)

envelopeIt.effect("metadata writes never rewrite the running part's input (R12-012 verbatim)", () =>
  Effect.gen(function* () {
    // Wrapper-shaped part: the stored input is the emitted deferred_tool
    // envelope while the inner execute receives the parsed inner args.
    const envelope = { name: "edit", args: '{"filePath":"/f.txt"}' }
    const state: SessionV1.ToolPart = {
      id: partID,
      sessionID,
      messageID,
      type: "tool",
      tool: "deferred_tool",
      callID,
      state: {
        status: "running",
        input: envelope,
        time: { start: 100 },
      },
    }
    const processor = processorStub(state)
    const resolved = yield* SessionTools.resolve({
      agent,
      model,
      session: sessionStub,
      processor,
      bypassAgentCheck: false,
      messages: [],
      promptOps: promptOpsStub,
    })
    const execute = resolved.tools.inner.execute
    if (!execute) throw new Error("inner tool is missing execute")

    yield* Effect.promise(() =>
      execute(
        { filePath: "/f.txt" },
        {
          toolCallId: callID,
          abortSignal: new AbortController().signal,
          messages: [],
        },
      ),
    )

    if (state.state.status !== "running") throw new Error("part should still be running")
    expect(state.state.input).toEqual(envelope)
    expect(state.state.metadata).toEqual({ step: 1 })
    expect(state.state.title).toBe("progress")
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
    Layer.mergeAll(baseLayer({ registry: globRegistry }), verdictStub("binding")),
  )
  const itBindingNoWrapper = testEffect(
    Layer.mergeAll(
      baseLayer({ flags: { disableToolWrapper: true }, registry: globRegistry }),
      verdictStub("binding"),
    ),
  )
  const itAdvisory = testEffect(
    Layer.mergeAll(baseLayer({ registry: globRegistry }), verdictStub("advisory")),
  )
  const itKillSwitch = testEffect(
    Layer.mergeAll(
      baseLayer({ flags: { disableLazyTools: true }, registry: globRegistry }),
      verdictStub(undefined),
    ),
  )

  itBinding.effect("binding + wrapper: the meta-tool seed joins eager, deferred entries keep full-fact shapes", () =>
    Effect.gen(function* () {
      calls.length = 0
      const resolved = yield* resolveGlob
      // resolved before the session's first provider request, scoped to the model
      expect(calls).toEqual([{ model }])
      expect(resolved.verdict).toBe("binding")
      expect(resolved.verdictProvenance).toEqual({ origin: "static-table" })
      expect(resolved.wrapper).toBe(true)
      // the meta-tool seed is pushed before shape and classified eager; the
      // closed definition rides the seed verbatim
      const meta = resolved.seeds.find((seed) => seed.name === "deferred_tool")
      expect(meta?.kind).toBe("eager")
      expect(meta?.jsonSchema).toEqual(DEFERRED_TOOL_SCHEMA)
      expect(meta?.fullDescription).toBe(DEFERRED_TOOL_DESCRIPTION)
      // glob's listing view is omitted on wrapper sessions, so the per-turn
      // AITool keeps its full-fact shape — the payload omission happens at
      // impose, never here (the closures are the dispatch targets)
      expect(resolved.tools.glob.inputSchema).toEqual(jsonSchema(globSchema))
      expect(resolved.tools.glob.description).toBe(long)
      // seeds stay full facts (mode-independent first-write source)
      expect(resolved.seeds.find((seed) => seed.name === "glob")?.fullDescription).toBe(long)
      expect(resolved.seeds.find((seed) => seed.name === "glob")?.jsonSchema).toEqual(globSchema)
    }),
  )

  itBindingNoWrapper.effect("binding + wrapper kill-switch: round-1 schema-eager listing, no meta seed", () =>
    Effect.gen(function* () {
      const resolved = yield* resolveGlob
      expect(resolved.verdict).toBe("binding")
      expect(resolved.wrapper).toBe(false)
      expect(resolved.seeds.some((seed) => seed.name === "deferred_tool")).toBe(false)
      // round-1 binding: the per-turn AITool carries the full schema with the
      // truncated description (R12-010 amendment 1)
      expect(resolved.tools.glob.inputSchema).toEqual(jsonSchema(globSchema))
      expect(resolved.tools.glob.description).toBe(long.slice(0, long.lastIndexOf(" ")) + "...")
    }),
  )

  itAdvisory.effect("advisory verdict keeps the constant placeholder in the per-turn listing view", () =>
    Effect.gen(function* () {
      calls.length = 0
      const resolved = yield* resolveGlob
      expect(resolved.verdict).toBe("advisory")
      expect(resolved.wrapper).toBe(false)
      expect(resolved.seeds.some((seed) => seed.name === "deferred_tool")).toBe(false)
      expect(resolved.tools.glob.inputSchema).toEqual(jsonSchema({ type: "object", properties: {} }))
    }),
  )

  itKillSwitch.effect("kill-switch: the verdict is never resolved, upstream per-turn shapes pass through", () =>
    Effect.gen(function* () {
      const resolved = yield* resolveGlob
      expect(resolved.verdict).toBeUndefined()
      expect(resolved.verdictProvenance).toBeUndefined()
      expect(resolved.wrapper).toBeUndefined()
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

  const itFallback = testEffect(
    Layer.mergeAll(baseLayer({ registry: failingGlobRegistry }), verdictStub("advisory")),
  )
  const itOff = testEffect(
    Layer.mergeAll(
      baseLayer({ flags: { disableLazyTools: true }, registry: failingGlobRegistry }),
      verdictStub(undefined),
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

  itFallback.effect("a failed deferred call appends the schema and marks the part", () =>
      Effect.gen(function* () {
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
      }),
  )

  itOff.effect("kill-switch: a failed deferred call stays upstream — no schema append, no marker", () =>
    Effect.gen(function* () {
      const state = runningGlobPart()
      const resolved = yield* SessionTools.resolve(failingResolveInput(state))

      const execute = resolved.tools.glob.execute
      if (!execute) throw new Error("glob is missing execute")
      const rejection = yield* rethrown(execute)
      expect(rejection).toBeInstanceOf(Tool.InvalidArgumentsError)
      expect((rejection as Error).message).not.toContain(JSON.stringify(globSchema))
      if (state.state.status !== "running") throw new Error("part should still be running")
      expect(state.state.metadata?.load_tool).toBeUndefined()
    }),
  )
})

describe("session.tools deferred_tool meta-tool (ticket 19)", () => {
  const globSchema: JSONSchema7 = {
    type: "object",
    properties: { pattern: { type: "string" }, path: { type: "string" } },
    required: ["pattern"],
  }
  const messageID = MessageID.ascending()

  const globRegistry = (execute: Tool.Def["execute"]) =>
    Layer.succeed(
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
              execute,
            } satisfies Tool.Def,
          ]),
      }),
    )

  const okGlob = globRegistry(() =>
    Effect.succeed({ title: "glob", metadata: { files: 3 }, output: "a.ts\nb.ts" }),
  )
  const failingGlob = globRegistry(() =>
    // Production shape: Tool.wrap raises the decode failure as a defect
    // (Effect.orDie), which the fallback catches.
    Effect.die(new Tool.InvalidArgumentsError({ tool: "glob", detail: "missing pattern" })),
  )

  const itWrapper = testEffect(Layer.mergeAll(baseLayer({ registry: okGlob }), verdictStub("binding")))
  const itWrapperFailing = testEffect(
    Layer.mergeAll(baseLayer({ registry: failingGlob }), verdictStub("binding")),
  )
  const itDenied = testEffect(Layer.mergeAll(baseLayer({ registry: okGlob }), verdictStub("binding")))
  const itAdvisory = testEffect(
    Layer.mergeAll(baseLayer({ registry: okGlob }), verdictStub("advisory")),
  )
  const itBindingNoWrapper = testEffect(
    Layer.mergeAll(
      baseLayer({ flags: { disableToolWrapper: true }, registry: okGlob }),
      verdictStub("binding"),
    ),
  )
  const itSchemaEagerFailing = testEffect(
    Layer.mergeAll(
      baseLayer({ flags: { disableToolWrapper: true }, registry: failingGlob }),
      verdictStub("binding"),
    ),
  )
  const itKillSwitch = testEffect(
    Layer.mergeAll(
      baseLayer({ flags: { disableLazyTools: true }, registry: okGlob }),
      verdictStub(undefined),
    ),
  )

  // A running deferred_tool part plus the resolve input whose updateToolCall
  // writes through to it — mirroring what the processor holds while execute()
  // is in flight (same shape as failingResolveInput above).
  const resolveInput = (state: SessionV1.ToolPart, agentOverride?: Agent.Info) => ({
    agent: agentOverride ?? agent,
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
    },
    bypassAgentCheck: false,
    messages: [] as SessionV1.WithParts[],
    promptOps: promptOpsStub,
  })

  const metaPart = (): SessionV1.ToolPart => ({
    id: PartID.ascending(),
    sessionID,
    messageID,
    type: "tool",
    tool: "deferred_tool",
    callID,
    state: { status: "running", input: {}, time: { start: 0 } },
  })

  const opts = () => ({ toolCallId: callID, abortSignal: new AbortController().signal, messages: [] })

  const rejectionOf = (execute: unknown, args: unknown) =>
    Effect.promise(() =>
      (execute as (args: unknown, options: ToolExecutionOptions) => Promise<unknown>)(args, opts()).then(
        () => new Error("expected the dispatch to fail"),
        (error: unknown) => error,
      ),
    )

  itWrapper.effect(
    "binding+wrapper constructs the meta-tool; dispatch executes the inner tool through its wrapped closure",
    () =>
      Effect.gen(function* () {
        const state = metaPart()
        const resolved = yield* SessionTools.resolve(resolveInput(state))
        expect(resolved.wrapper).toBe(true)
        const meta = resolved.tools.deferred_tool
        const execute = meta?.execute
        if (!execute) throw new Error("deferred_tool must be constructed on wrapper sessions")
        expect(meta?.description).toBe(DEFERRED_TOOL_DESCRIPTION)
        expect(meta?.inputSchema).toEqual(jsonSchema(DEFERRED_TOOL_SCHEMA))

        const output = yield* Effect.promise(() => execute({ name: "glob", args: '{"pattern":"*.ts"}' }, opts()))
        // the inner output with the unwrap metadata merged — what
        // completeToolCall writes over the part (the completed part carries it)
        expect(output).toEqual({
          title: "glob",
          metadata: { files: 3, deferred_tool: { tool: "glob" } },
          output: "a.ts\nb.ts",
        })
        // unwrap metadata landed on the running part pre-execute
        if (state.state.status !== "running") throw new Error("part should still be running")
        expect(state.state.metadata?.deferred_tool).toEqual({ tool: "glob" })
      }),
  )

  itWrapper.effect("unknown name through resolve: clear error, no schema, no marker", () =>
    Effect.gen(function* () {
      const state = metaPart()
      const resolved = yield* SessionTools.resolve(resolveInput(state))
      const meta = resolved.tools.deferred_tool
      if (!meta?.execute) throw new Error("deferred_tool must be constructed on wrapper sessions")
      const error = yield* rejectionOf(meta.execute, { name: "bogus", args: "{}" })
      expect(error).toBeInstanceOf(Error)
      expect((error as Error).message).toContain(
        "Unknown deferred tool: bogus. Deferred tools are listed in the <deferred_tools> catalog blocks.",
      )
      expect((error as Error).message).not.toContain('"pattern"')
      if (state.state.status !== "running") throw new Error("part should still be running")
      expect(state.state.metadata?.load_tool).toBeUndefined()
      expect(state.state.metadata?.deferred_tool).toEqual({ tool: "bogus" })
    }),
  )

  itDenied.effect("non-permissible name: the dispatch map holds only permissible seeds — no schema leak (R12-001)", () =>
    Effect.gen(function* () {
      const state = metaPart()
      const denyingAgent: Agent.Info = {
        ...agent,
        permission: [{ permission: "glob", pattern: "*", action: "deny" }],
      }
      const resolved = yield* SessionTools.resolve(resolveInput(state, denyingAgent))
      expect(resolved.seeds.some((seed) => seed.name === "glob")).toBe(false)
      const meta = resolved.tools.deferred_tool
      if (!meta?.execute) throw new Error("deferred_tool must be constructed on wrapper sessions")
      const error = yield* rejectionOf(meta.execute, { name: "glob", args: '{"pattern":"*"}' })
      expect((error as Error).message).toContain("Unknown deferred tool: glob")
      expect((error as Error).message).not.toContain('"pattern"')
      if (state.state.status !== "running") throw new Error("part should still be running")
      expect(state.state.metadata?.load_tool).toBeUndefined()
    }),
  )

  itWrapper.effect("unparseable args on a known name: schema-in-error + delivery marker on the part", () =>
    Effect.gen(function* () {
      const state = metaPart()
      const resolved = yield* SessionTools.resolve(resolveInput(state))
      const meta = resolved.tools.deferred_tool
      if (!meta?.execute) throw new Error("deferred_tool must be constructed on wrapper sessions")
      const error = yield* rejectionOf(meta.execute, { name: "glob", args: "{oops" })
      expect((error as Error).message).toContain(JSON.stringify(globSchema, null, 2))
      if (state.state.status !== "running") throw new Error("part should still be running")
      expect(state.state.metadata?.load_tool).toEqual({ tools: ["glob"] })
      expect(state.state.metadata?.deferred_tool).toEqual({ tool: "glob" })
    }),
  )

  itWrapperFailing.effect("inner InvalidArgumentsError rides the fallback: schema once, marker once", () =>
    Effect.gen(function* () {
      const state = metaPart()
      const resolved = yield* SessionTools.resolve(resolveInput(state))
      const meta = resolved.tools.deferred_tool
      if (!meta?.execute) throw new Error("deferred_tool must be constructed on wrapper sessions")
      const error = yield* rejectionOf(meta.execute, { name: "glob", args: '{"pattern":"*"}' })
      const message = (error as Error).message
      expect(message).toContain("missing pattern")
      expect(message.split(JSON.stringify(globSchema, null, 2)).length - 1).toBe(1)
      if (state.state.status !== "running") throw new Error("part should still be running")
      expect(state.state.metadata?.load_tool).toEqual({ tools: ["glob"] })
      expect(state.state.metadata?.deferred_tool).toEqual({ tool: "glob" })
    }),
  )

  itSchemaEagerFailing.effect("wrapper kill-switch (schema-eager binding): a failed deferred call still appends the schema", () =>
    Effect.gen(function* () {
      const state = metaPart()
      const resolved = yield* SessionTools.resolve(resolveInput(state))
      const meta = resolved.tools.deferred_tool
      expect(meta).toBeUndefined()
      const execute = resolved.tools.glob.execute
      if (!execute) throw new Error("glob is missing execute")
      const rejection = yield* rejectionOf(execute, { pattern: "*" })
      expect((rejection as Error).message).toContain(JSON.stringify(globSchema, null, 2))
    }),
  )

  itAdvisory.effect("advisory: no meta-tool constructed, universe and record stay clean", () =>
    Effect.gen(function* () {
      const resolved = yield* SessionTools.resolve(resolveInput(metaPart()))
      expect(resolved.tools.deferred_tool).toBeUndefined()
      expect(resolved.seeds.some((seed) => seed.name === "deferred_tool")).toBe(false)
    }),
  )

  itBindingNoWrapper.effect("wrapper kill-switch: no meta-tool, schema-eager binding stands", () =>
    Effect.gen(function* () {
      const resolved = yield* SessionTools.resolve(resolveInput(metaPart()))
      expect(resolved.wrapper).toBe(false)
      expect(resolved.tools.deferred_tool).toBeUndefined()
      expect(resolved.seeds.some((seed) => seed.name === "deferred_tool")).toBe(false)
    }),
  )

  itKillSwitch.effect("lazy-tools kill-switch: no meta-tool, no verdict, upstream shapes", () =>
    Effect.gen(function* () {
      const resolved = yield* SessionTools.resolve(resolveInput(metaPart()))
      expect(resolved.verdict).toBeUndefined()
      expect(resolved.tools.deferred_tool).toBeUndefined()
      expect(resolved.seeds).toEqual([])
    }),
  )
})
