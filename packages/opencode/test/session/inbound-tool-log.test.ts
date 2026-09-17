import { Global } from "@opencode-ai/core/global"
import { LLMEvent, type LLMEvent as LLMEventType } from "@opencode-ai/llm"
import { describe, expect, test } from "bun:test"
import { Effect, Layer } from "effect"
import { logLines } from "effect/testing/TestConsole"
import fs from "fs/promises"
import fsSync from "node:fs"
import os from "os"
import path from "path"
import { InboundToolLog } from "../../src/session/llm/inbound-tool-log"
import { RuntimeFlags } from "../../src/effect/runtime-flags"
import { testEffect } from "../lib/effect"

const data = await fs.mkdtemp(path.join(os.tmpdir(), "opencode-inbound-log-"))
const dataOff = await fs.mkdtemp(path.join(os.tmpdir(), "opencode-inbound-log-off-"))
const blockedData = path.join(data, "not-a-directory")
await Bun.write(blockedData, "file")
process.on("exit", () => {
  for (const dir of [data, dataOff]) fsSync.rmSync(dir, { recursive: true, force: true })
})

const context = {
  sessionID: "ses_test",
  messageID: "msg_test",
  providerID: "test",
  modelID: "test-model",
}

const layer = (dir: string, enabled: boolean) =>
  InboundToolLog.layer.pipe(
    Layer.provideMerge(RuntimeFlags.layer({ enableInboundToolLog: enabled })),
    Layer.provideMerge(Global.layerWith({ data: dir })),
  )

const logFile = (dir: string) => path.join(dir, "inbound-tool-calls.jsonl")

const run = <A, E>(effect: Effect.Effect<A, E, InboundToolLog.Service>) =>
  Effect.runPromise(effect.pipe(Effect.provide(layer(data, true))))

const readLines = async (dir: string) => {
  const raw = await fs.readFile(logFile(dir), "utf8")
  return raw
    .split("\n")
    .filter((line) => line !== "")
    .map((line) => JSON.parse(line))
}

const deltas = (id: string, name: string, chunks: string[]): LLMEventType[] => [
  LLMEvent.toolInputStart({ id, name }),
  ...chunks.map((text) => LLMEvent.toolInputDelta({ id, name, text })),
]

const call = (id: string, name: string, input: unknown): LLMEventType =>
  LLMEvent.toolCall({ id, name, input })

describe("session.inbound-tool-log sink", () => {
  test("raw reassembles verbatim across ≥2 deltas; one line per call", async () => {
    const chunks = ['{"pattern":', '"*.ts", "esc":', '"a\\"b"}']
    await run(
      Effect.gen(function* () {
        const svc = yield* InboundToolLog.Service
        for (const event of deltas("call_1", "glob", chunks)) yield* svc.capture(event, context)
        yield* svc.capture(call("call_1", "glob", { pattern: "*.ts", esc: 'a"b' }), context)
      }),
    )
    const lines = await readLines(data)
    expect(lines).toHaveLength(1)
    expect(lines[0].raw).toBe(chunks.join(""))
    expect(lines[0].input).toEqual({ pattern: "*.ts", esc: 'a"b' })
    expect(lines[0].callID).toBe("call_1")
    expect(lines[0].tool).toBe("glob")
    expect(lines[0].sessionID).toBe("ses_test")
    expect(lines[0].messageID).toBe("msg_test")
    expect(lines[0].providerID).toBe("test")
    expect(lines[0].modelID).toBe("test-model")
    expect(typeof lines[0].time).toBe("number")
  })

  test("a call with no streamed deltas records raw: null", async () => {
    await run(
      Effect.gen(function* () {
        const svc = yield* InboundToolLog.Service
        yield* svc.capture(call("call_2", "shell", { command: "ls" }), context)
      }),
    )
    const lines = await readLines(data)
    expect(lines).toHaveLength(2)
    expect(lines[1].raw).toBeNull()
    expect(lines[1].input).toEqual({ command: "ls" })
  })

  test("non-record input is recorded verbatim (pre-wrap, pre-validation)", async () => {
    await run(
      Effect.gen(function* () {
        const svc = yield* InboundToolLog.Service
        yield* svc.capture(call("call_3", "glob", "not-an-object"), context)
      }),
    )
    const lines = await readLines(data)
    expect(lines[2].input).toBe("not-an-object")
  })

  test("providerExecuted calls are logged and carry the flag", async () => {
    await run(
      Effect.gen(function* () {
        const svc = yield* InboundToolLog.Service
        yield* svc.capture(call("call_4", "web_search", { query: "x" }), context)
      }),
    )
    const lines = await readLines(data)
    expect(lines[3].providerExecuted).toBeUndefined()
  })

  test("non-tool events are filtered out; buffered state survives unrelated events", async () => {
    await run(
      Effect.gen(function* () {
        const svc = yield* InboundToolLog.Service
        const noise: LLMEventType[] = [
          LLMEvent.textDelta({ id: "txt_1", text: "hello" }),
          LLMEvent.reasoningDelta({ id: "rs_1", text: "hmm" }),
        ]
        for (const event of [...noise, ...deltas("call_5", "glob", ['{"pattern":', '"x"}']), ...noise]) {
          yield* svc.capture(event, context)
        }
        yield* svc.capture(call("call_5", "glob", { pattern: "x" }), context)
      }),
    )
    const lines = await readLines(data)
    expect(lines).toHaveLength(5)
    expect(lines[4].raw).toBe('{"pattern":"x"}')
  })

  test("each turn buffers independently; a call without any buffer still logs (raw null)", async () => {
    const otherContext = { ...context, messageID: "msg_other" }
    await run(
      Effect.gen(function* () {
        const svc = yield* InboundToolLog.Service
        // turn A buffers a delta then never completes (aborted)
        yield* svc.capture(LLMEvent.toolInputStart({ id: "call_a", name: "glob" }), context)
        yield* svc.capture(LLMEvent.toolInputDelta({ id: "call_a", name: "glob", text: '{"par' }), context)
        // turn B completes its own call on the same call id
        yield* svc.capture(call("call_a", "glob", { pattern: "x" }), otherContext)
      }),
    )
    const lines = await readLines(data)
    expect(lines[5].raw).toBeNull()
    expect(lines[5].messageID).toBe("msg_other")
  })

  test("write failure is non-fatal: the capture succeeds, the file never appears", async () => {
    const blocked = await Effect.runPromise(
      Effect.gen(function* () {
        const svc = yield* InboundToolLog.Service
        yield* svc.capture(call("call_b", "glob", { pattern: "x" }), context)
        return "captured"
      }).pipe(Effect.provide(layer(blockedData, true))),
    )
    expect(blocked).toBe("captured")
    await expect(fs.access(logFile(blockedData))).rejects.toThrow()
  })

  test("flag off (default): capture is a no-op — no file, no buffering side effects", async () => {
    await Effect.runPromise(
      Effect.gen(function* () {
        const svc = yield* InboundToolLog.Service
        for (const event of [...deltas("call_c", "glob", ['{"x":1}']), call("call_c", "glob", { x: 1 })]) {
          yield* svc.capture(event, context)
        }
      }).pipe(Effect.provide(layer(dataOff, false))),
    )
    await expect(fs.access(logFile(dataOff))).rejects.toThrow()
  })
})

const itLoud = testEffect(
  InboundToolLog.layer.pipe(
    Layer.provideMerge(RuntimeFlags.layer({ enableInboundToolLog: true })),
    Layer.provideMerge(Global.layerWith({ data: blockedData })),
  ),
)
itLoud.effect("a write failure is logged loudly (R00-010)", () =>
  Effect.gen(function* () {
    const svc = yield* InboundToolLog.Service
    yield* svc.capture(call("call_d", "glob", { pattern: "x" }), context)
    const logs = yield* logLines
    expect(logs.some((line) => typeof line === "string" && line.includes("inbound tool log write failed"))).toBe(true)
  }),
)
