import { Global } from "@opencode-ai/core/global"
import { AppNodeBuilder } from "@opencode-ai/core/effect/app-node-builder"
import { describe, expect, test } from "bun:test"
import { ConfigProvider, Effect, Layer } from "effect"
import { tool as aiTool, jsonSchema, type ModelMessage, type Tool } from "ai"
import fs from "fs/promises"
import path from "path"
import { tmpdir } from "../fixture/fixture"
import { PromptCapture, type CaptureSource } from "../../src/session/llm/prompt-capture"
import type { Prepared } from "../../src/session/llm/request"
import { RuntimeFlags } from "../../src/effect/runtime-flags"

const meta: CaptureSource = {
  sessionID: "ses_capture",
  providerID: "test",
  modelID: "test-model",
  modelApi: "@ai-sdk/openai-compatible",
  agent: "build",
  small: false,
  requestID: "msg_user_1",
  optimized: { lazyTools: true, staticSlimming: true },
}

const prepared = (input?: { tools?: Record<string, Tool>; messages?: ModelMessage[] }): Prepared => ({
  system: ["You are a test agent.", "<env>test</env>"],
  messages: input?.messages ?? [{ role: "user", content: "hello" }],
  tools: input?.tools ?? {
    read: aiTool({
      description: "Read a file",
      inputSchema: jsonSchema({ type: "object", properties: { filePath: { type: "string" } } }),
    }),
  },
  params: { temperature: 0, options: { apiKey: "sk-secret-do-not-serialize" } },
  messageTransformOptions: { apiKey: "sk-transform-secret" },
  headers: { Authorization: "Bearer sk-header-secret", "User-Agent": "opencode/test" },
})

// Sources the dump target the way production does — through Global.Service.data
// (Global.layerWith override), mirroring the request.ts wire contract.
const dumpTo = (data: string, input: { prepared: Prepared; meta: typeof meta }) =>
  Effect.gen(function* () {
    const global = yield* Global.Service
    yield* PromptCapture.dump({ data: global.data, prepared: input.prepared, meta: input.meta })
  }).pipe(Effect.provide(Global.layerWith({ data })), Effect.runPromise)

const readFile = (file: string) => Bun.file(file).json()

const walk = (value: unknown, visit: (key: string, value: unknown) => void) => {
  if (Array.isArray(value)) {
    for (const entry of value) walk(entry, visit)
    return
  }
  if (value && typeof value === "object") {
    for (const [key, entry] of Object.entries(value)) {
      visit(key, entry)
      walk(entry, visit)
    }
  }
}

describe("session.prompt-capture.serializer", () => {
  test("whitelist projection: raw jsonSchema extraction, strict passthrough", async () => {
    await using tmp = await tmpdir()
    await dumpTo(tmp.path, {
      prepared: prepared({
        tools: {
          read: aiTool({
            description: "Read a file",
            inputSchema: jsonSchema({ type: "object", properties: { filePath: { type: "string" } } }),
          }),
          strictTool: {
            ...aiTool({ description: "strict", inputSchema: jsonSchema({ type: "object", properties: {} }) }),
            strict: false,
          },
          bare: aiTool({ inputSchema: jsonSchema({ type: "object", properties: {} }) }),
        },
      }),
      meta,
    })
    const file = await readFile(path.join(tmp.path, "prompt-captures", meta.sessionID, "0000.json")) as {
      meta: unknown
      payload: { system: string[]; tools: Record<string, { description?: string; inputSchema: unknown; strict?: boolean }>; messages: unknown[] }
    }
    // the raw JSON schema reaches the file, not the jsonSchema() wrapper
    expect(file.payload.tools.read).toEqual({
      description: "Read a file",
      inputSchema: { type: "object", properties: { filePath: { type: "string" } } },
    })
    // strict only when the tool object carries it
    expect(file.payload.tools.strictTool).toEqual({
      description: "strict",
      inputSchema: { type: "object", properties: {} },
      strict: false,
    })
    expect(file.payload.tools.bare).toEqual({ inputSchema: { type: "object", properties: {} } })
  })

  test("params, messageTransformOptions and headers never enter the dump (R10-002)", async () => {
    await using tmp = await tmpdir()
    await dumpTo(tmp.path, { prepared: prepared(), meta })
    const raw = await Bun.file(path.join(tmp.path, "prompt-captures", meta.sessionID, "0000.json")).text()
    expect(raw).not.toContain("sk-secret-do-not-serialize")
    expect(raw).not.toContain("sk-transform-secret")
    expect(raw).not.toContain("sk-header-secret")
    expect(raw).not.toContain("Authorization")
    const file = JSON.parse(raw)
    expect(Object.keys(file).toSorted()).toEqual(["meta", "payload"])
    expect(Object.keys(file.payload).toSorted()).toEqual(["messages", "system", "tools"])
    expect(Object.keys(file.meta).toSorted()).toEqual([
      "agent",
      "createdAt",
      "modelApi",
      "modelID",
      "optimized",
      "providerID",
      "requestID",
      "sessionID",
      "small",
      "version",
    ])
  })
})

describe("session.prompt-capture.dump", () => {
  test("writes one JSON file per turn under data/prompt-captures/<sessionID>", async () => {
    await using tmp = await tmpdir()
    await dumpTo(tmp.path, { prepared: prepared(), meta })
    const file = await readFile(path.join(tmp.path, "prompt-captures", meta.sessionID, "0000.json")) as {
      payload: { system: string[]; messages: unknown[] }
    }
    expect(file.payload.system).toEqual(["You are a test agent.", "<env>test</env>"])
    expect(file.payload.messages).toEqual([{ role: "user", content: "hello" }])
  })

  test("meta.toolServers rides the file; the payload is untouched (R12-009 / R10-001)", async () => {
    await using tmp = await tmpdir()
    await dumpTo(tmp.path, { prepared: prepared(), meta: { ...meta, toolServers: { glob: "firecrawl" } } })
    await dumpTo(tmp.path, { prepared: prepared(), meta })
    const withMap = await readFile(path.join(tmp.path, "prompt-captures", meta.sessionID, "0000.json")) as {
      meta: Record<string, unknown>
      payload: { tools: Record<string, unknown> }
    }
    const without = await readFile(path.join(tmp.path, "prompt-captures", meta.sessionID, "0001.json")) as {
      meta: Record<string, unknown>
      payload: { tools: Record<string, unknown> }
    }
    expect(withMap.meta.toolServers).toEqual({ glob: "firecrawl" })
    expect("toolServers" in without.meta).toBe(false)
    expect(JSON.stringify(withMap.payload.tools)).toBe(JSON.stringify(without.payload.tools))
  })

  test("meta.verdict, its provenance and meta.binary ride the file; undefined fields stay absent (SC-4, R13-003/005, R12-013)", async () => {
    await using tmp = await tmpdir()
    await dumpTo(tmp.path, {
      prepared: prepared(),
      meta: {
        ...meta,
        verdict: "binding",
        verdictProvenance: { origin: "static-table" },
        binary: "opencode/1.2.3",
      },
    })
    await dumpTo(tmp.path, { prepared: prepared(), meta })
    const withFields = await readFile(path.join(tmp.path, "prompt-captures", meta.sessionID, "0000.json")) as {
      meta: Record<string, unknown>
    }
    const without = await readFile(path.join(tmp.path, "prompt-captures", meta.sessionID, "0001.json")) as {
      meta: Record<string, unknown>
    }
    expect(withFields.meta.verdict).toBe("binding")
    expect(withFields.meta.verdictProvenance).toEqual({ origin: "static-table" })
    expect(withFields.meta.binary).toBe("opencode/1.2.3")
    expect("verdict" in without.meta).toBe(false)
    expect("verdictProvenance" in without.meta).toBe(false)
    expect("binary" in without.meta).toBe(false)
  })

  test("a non-jsonSchema tool skips the turn's dump entirely (contract violation)", async () => {
    await using tmp = await tmpdir()
    const violating = {
      broken: {
        description: "not a jsonSchema wrapper",
        // zod-style schema object: no jsonSchema() wrapper — none exist in the repo today
        inputSchema: { type: "object", properties: {} },
      },
    } as unknown as Record<string, Tool>
    await dumpTo(tmp.path, { prepared: prepared({ tools: violating }), meta })
    const dir = path.join(tmp.path, "prompt-captures", meta.sessionID)
    const entries = await fs.readdir(dir).catch((error: NodeJS.ErrnoException) =>
      error.code === "ENOENT" ? [] : Promise.reject(error),
    )
    expect(entries.filter((name) => name.endsWith(".json"))).toEqual([])
  })

  test("seq resumes from a directory scan: pre-existing files never collide", async () => {
    await using tmp = await tmpdir()
    const dir = path.join(tmp.path, "prompt-captures", meta.sessionID)
    await fs.mkdir(dir, { recursive: true })
    await Bun.write(path.join(dir, "0000.json"), "{}")
    await Bun.write(path.join(dir, "0001.json"), "{}")
    await Bun.write(path.join(dir, "junk.json"), "{}")
    await Bun.write(path.join(dir, "0002.txt"), "not a capture")
    await dumpTo(tmp.path, { prepared: prepared(), meta })
    const names = await Array.fromAsync(new Bun.Glob("*.json").scan({ cwd: dir }))
    expect(names.toSorted()).toEqual(["0000.json", "0001.json", "0002.json", "junk.json"])
    const file = await readFile(path.join(dir, "0002.json")) as { meta: { requestID: string } }
    expect(file.meta.requestID).toBe("msg_user_1")
  })
})

describe("session.prompt-capture.flag", () => {
  // RuntimeFlags.layer hardwires an empty config; the node reads the ambient
  // ConfigProvider, so precedence is proven against fromUnknown.
  const fromConfig = (input: Record<string, unknown>) =>
    AppNodeBuilder.build(RuntimeFlags.node).pipe(Layer.provide(ConfigProvider.layer(ConfigProvider.fromUnknown(input))))

  const readCaptureFlag = RuntimeFlags.Service.useSync((flags) => flags.enablePromptCapture)

  test("capture is opt-in: unset flag defaults to false (upstream-identical)", async () => {
    const enabled = await Effect.runPromise(readCaptureFlag.pipe(Effect.provide(fromConfig({}))))
    expect(enabled).toBe(false)
  })

  test("OPENCODE_ENABLE_PROMPT_CAPTURE set turns capture on", async () => {
    const enabled = await Effect.runPromise(
      readCaptureFlag.pipe(Effect.provide(fromConfig({ OPENCODE_ENABLE_PROMPT_CAPTURE: "true" }))),
    )
    expect(enabled).toBe(true)
  })
})
