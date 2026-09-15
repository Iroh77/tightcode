import { describe, expect, test } from "bun:test"
import type { JSONSchema7 } from "@ai-sdk/provider"
import { SessionV1 } from "@opencode-ai/core/v1/session"
import { Effect } from "effect"
import type { EffectBridge } from "../../src/effect/bridge"
import type { SessionProcessor } from "../../src/session/processor"
import { MessageID, PartID, SessionID } from "../../src/session/schema"
import { schemaBlock, type ToolSeed } from "../../src/session/tool-listing"
import { dispatch } from "../../src/tool/deferred_tool"
import { delivered } from "../../src/tool/load_tool"
import type { Tool as AITool, ToolExecutionOptions } from "ai"

const sessionID = SessionID.make("ses_deferred-tool")
const messageID = MessageID.ascending()
const callID = "call_deferred"

const globSchema: JSONSchema7 = {
  type: "object",
  properties: { pattern: { type: "string" }, path: { type: "string" } },
  required: ["pattern"],
}

const seeds: ToolSeed[] = [
  {
    name: "read",
    kind: "eager",
    fullDescription: "read a file",
    jsonSchema: { type: "object", properties: {} },
    source: "builtin",
  },
  {
    name: "glob",
    kind: "deferred",
    fullDescription: "find files",
    jsonSchema: globSchema,
    source: "builtin",
  },
  {
    name: "grep",
    kind: "deferred",
    fullDescription: "search files",
    jsonSchema: { type: "object", properties: { pattern: { type: "string" } } },
    source: "builtin",
  },
]

const run = {
  promise: (effect: Effect.Effect<unknown, unknown, never>) => Effect.runPromise(effect),
} as EffectBridge.Shape

const options = (): ToolExecutionOptions => ({
  toolCallId: callID,
  messages: [],
  abortSignal: new AbortController().signal,
})

// A running deferred_tool part whose state survives updateToolCall, mirroring
// what the processor holds while execute() is in flight.
const stateStub = () => {
  const part: SessionV1.ToolPart = {
    id: PartID.ascending(),
    sessionID,
    messageID,
    type: "tool",
    tool: "deferred_tool",
    callID,
    state: { status: "running", input: {}, time: { start: 0 } },
  }
  const updateToolCall: SessionProcessor.Handle["updateToolCall"] = (_toolCallID, update) =>
    Effect.sync(() => {
      const next = update(part)
      part.state = next.state
      return next
    })
  return { part, updateToolCall }
}

const innerCalls: unknown[] = []
const shaped = {
  read: {
    execute: async () => {
      throw new Error("read must never be dispatched")
    },
  },
  glob: {
    execute: async (args: unknown) => {
      innerCalls.push(args)
      return { title: "glob", metadata: { files: 3 }, output: `ran ${JSON.stringify(args)}` }
    },
  },
} as unknown as Record<string, AITool>

const rejection = async (promise: Promise<unknown>): Promise<Error> =>
  promise.then(
    () => new Error("expected the dispatch to fail") as unknown as Error,
    (error: unknown) => error as Error,
  )

const messageWith = (part: SessionV1.ToolPart): SessionV1.WithParts[] => [
  { info: {} as SessionV1.Assistant, parts: [part] },
]

const errored = (part: SessionV1.ToolPart): SessionV1.ToolPart => ({
  ...part,
  state: {
    ...part.state,
    status: "error",
    error: "boom",
    time: { start: 0, end: 1 },
    metadata: part.state.status === "running" ? part.state.metadata : undefined,
  },
})

describe("tool.deferred_tool dispatch (ticket 19)", () => {
  test("happy path: parsed args reach the inner closure, unwrap metadata rides running part and output", async () => {
    const { part, updateToolCall } = stateStub()
    innerCalls.length = 0
    const execute = dispatch({ seeds, shaped, run, updateToolCall })

    const output = (await execute({ name: "glob", args: '{"pattern":"*.ts"}' }, options())) as {
      title: string
      metadata: Record<string, unknown>
      output: string
    }

    // inner tool received the parsed args object, not the JSON string
    expect(innerCalls).toEqual([{ pattern: "*.ts" }])
    // success output = inner output with the unwrap metadata merged
    expect(output.title).toBe("glob")
    expect(output.metadata).toEqual({ files: 3, deferred_tool: { tool: "glob" } })
    expect(output.output).toBe('ran {"pattern":"*.ts"}')
    // unwrap metadata landed on the running part (pre-execute — written before
    // the inner call, asserted by the inner having already seen it)
    if (part.state.status !== "running") throw new Error("part should still be running")
    expect(part.state.metadata?.deferred_tool).toEqual({ tool: "glob" })
  })

  test("unknown name: pinned error, no schema leak, no marker", async () => {
    const { part, updateToolCall } = stateStub()
    innerCalls.length = 0
    const execute = dispatch({ seeds, shaped, run, updateToolCall })

    const error = await rejection(execute({ name: "bogus", args: "{}" }, options()))
    expect(error.message).toBe(
      "Unknown deferred tool: bogus. Deferred tools are listed in the <deferred_tools> catalog blocks.",
    )
    expect(error.message).not.toContain("pattern")
    innerCalls.length = 0
    expect(innerCalls).toEqual([])
    if (part.state.status !== "running") throw new Error("part should still be running")
    expect(part.state.metadata?.deferred_tool).toEqual({ tool: "bogus" })
    expect(part.state.metadata?.load_tool).toBeUndefined()
  })

  test("missing name behaves as unknown", async () => {
    const { updateToolCall } = stateStub()
    const execute = dispatch({ seeds, shaped, run, updateToolCall })
    const error = await rejection(execute({ args: "{}" }, options()))
    expect(error.message).toContain("Unknown deferred tool")
    expect(error.message).not.toContain("pattern")
  })

  test("eager target: call-it-directly error, no schema, no marker", async () => {
    const { part, updateToolCall } = stateStub()
    innerCalls.length = 0
    const execute = dispatch({ seeds, shaped, run, updateToolCall })

    const error = await rejection(execute({ name: "read", args: "{}" }, options()))
    expect(error.message).toContain("not a deferred tool")
    expect(error.message).toContain("call it directly")
    expect(error.message).not.toContain("pattern")
    expect(innerCalls).toEqual([])
    if (part.state.status !== "running") throw new Error("part should still be running")
    expect(part.state.metadata?.load_tool).toBeUndefined()
  })

  test("unparseable args on a known name: schema-in-error + delivery marker", async () => {
    const { part, updateToolCall } = stateStub()
    innerCalls.length = 0
    const execute = dispatch({ seeds, shaped, run, updateToolCall })

    const error = await rejection(execute({ name: "glob", args: "{oops" }, options()))
    // parse error first, schema block appended (round-1 withFallback format)
    expect(error.message.endsWith(schemaBlock(seeds[1]))).toBe(true)
    expect(error.message.length).toBeGreaterThan(schemaBlock(seeds[1]).length)
    expect(innerCalls).toEqual([])
    if (part.state.status !== "running") throw new Error("part should still be running")
    expect(part.state.metadata?.load_tool).toEqual({ tools: ["glob"] })
    // the marker on the error part re-opens the short confirmation: a
    // subsequent load_tool(["glob"]) answers "already loaded." (load-tool tests)
    expect(delivered("glob", messageWith(errored(part)))).toBe(true)
  })

  test("JSON but not an object: same schema-in-error path", async () => {
    const { part, updateToolCall } = stateStub()
    const execute = dispatch({ seeds, shaped, run, updateToolCall })

    const error = await rejection(execute({ name: "glob", args: '["nope"]' }, options()))
    expect(error.message).toContain(schemaBlock(seeds[1]))
    if (part.state.status !== "running") throw new Error("part should still be running")
    expect(part.state.metadata?.load_tool).toEqual({ tools: ["glob"] })
  })

  test("unparseable args on an unknown name: unknown error only, no schema, no marker", async () => {
    const { part, updateToolCall } = stateStub()
    const execute = dispatch({ seeds, shaped, run, updateToolCall })

    const error = await rejection(execute({ name: "bogus", args: "{oops" }, options()))
    expect(error.message).toContain("Unknown deferred tool: bogus")
    expect(error.message).not.toContain("pattern")
    if (part.state.status !== "running") throw new Error("part should still be running")
    expect(part.state.metadata?.load_tool).toBeUndefined()
  })
})
