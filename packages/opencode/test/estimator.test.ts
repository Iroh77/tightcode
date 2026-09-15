import { describe, expect, spyOn, test } from "bun:test"
import fs from "fs/promises"
import path from "path"
import type { CaptureFile } from "../src/session/llm/prompt-capture"
import { forModel, verifyAssets } from "../script/estimator"
import { tmpdir } from "./fixture/fixture"

// Known-count fixtures: expected counts computed independently with the HF
// `tokenizers` / python `tiktoken` oracles (throwaway tooling, not committed).
const GLM_TEXT = "大型语言模型使用字节对编码将文本转换为标记。中文字符通常比英文单词消耗更多标记，因为词汇表对中文的覆盖有限。这一点在成本估算中很重要。"
const DEEPSEEK_TEXT = "Math: ∑∏∫ ≈ ≠ ≤ ≥ ∞ ∈ ∅ αβγδε Ω≈π"
const O200K_TEXT = "Accents: éàèùçêîôû ÀÉÈÙ Ça va très bien!"

const committedAssets = path.join(import.meta.dir, "..", "script", "estimator", "assets")

const assetsInit = (mode: "tampered" | "corrupt" | "missing") => async (dir: string) => {
  for (const name of ["glm", "deepseek"]) {
    await fs.mkdir(path.join(dir, name), { recursive: true })
    await fs.copyFile(path.join(committedAssets, name, "tokenizer.json"), path.join(dir, name, "tokenizer.json"))
  }
  await fs.copyFile(path.join(committedAssets, "pins.json"), path.join(dir, "pins.json"))
  if (mode === "tampered") await fs.appendFile(path.join(dir, "glm", "tokenizer.json"), "tampered")
  if (mode === "corrupt") await Bun.write(path.join(dir, "glm", "tokenizer.json"), "not json at all")
  if (mode === "missing") await fs.rm(path.join(dir, "glm", "tokenizer.json"))
  return dir
}

describe("estimator forModel", () => {
  test("selects the tokenizer by model-id prefix", () => {
    expect(forModel("glm-4.6").adapter).toBe("glm")
    expect(forModel("z-ai/glm-5.3-flash").adapter).toBe("glm")
    expect(forModel("deepseek-chat").adapter).toBe("deepseek")
    expect(forModel("openrouter/deepseek/deepseek-v4-flash-0731").adapter).toBe("deepseek")
    expect(forModel("claude-sonnet-4-5").adapter).toBe("o200k")
    expect(forModel("anything").adapter).toBe("o200k")
  })

  test("is memoized per process", () => {
    expect(forModel("glm-4.6")).toBe(forModel("glm-4.6"))
    expect(forModel("claude-sonnet-4-5")).toBe(forModel("claude-sonnet-4-5"))
  })

  test("real-asset estimate paths return the known counts", () => {
    const glm = forModel("glm-5.3-flash")
    expect(glm.estimate(GLM_TEXT)).toBe(39)
    expect(glm.adapter).toBe("glm")

    const deepseek = forModel("deepseek-v4-flash")
    expect(deepseek.estimate(DEEPSEEK_TEXT)).toBe(20)
    expect(deepseek.adapter).toBe("deepseek")

    const o200k = forModel("claude-sonnet-4-5")
    expect(o200k.estimate(O200K_TEXT)).toBe(21)
    expect(o200k.adapter).toBe("o200k")
  })

  test("special tokens are never added; non-special added tokens stay real vocabulary", () => {
    expect(forModel("glm-4.6").estimate("<|system|>")).toBeGreaterThan(1)
    expect(forModel("glm-4.6").estimate("<think>")).toBeGreaterThan(1)
    expect(forModel("deepseek-v4").estimate("<｜begin▁of▁sentence｜>")).toBeGreaterThan(1)
    expect(forModel("deepseek-v4").estimate("<|EOT|>")).toBe(1)
  })
})

describe("estimator verify", () => {
  test("passes on the committed assets", () => {
    verifyAssets()
  })

  test("fails loudly on a tampered copy", async () => {
    await using tmp = await tmpdir({ init: assetsInit("tampered") })
    expect(() => verifyAssets(tmp.path)).toThrow(/mismatch/)
  })
})

describe("estimator fallback", () => {
  test.each(["corrupt", "missing"] as const)("falls back to o200k on %s asset, logged, never thrown", async (mode) => {
    await using tmp = await tmpdir({ init: assetsInit(mode) })
    const log = spyOn(console, "error")
    try {
      const estimator = forModel("glm-4.6", { assetsDir: tmp.path })
      expect(estimator.adapter).toBe("glm")
      expect(estimator.estimate("hello world")).toBeGreaterThan(0)
      expect(estimator.adapter).toBe("o200k-fallback")
      expect(log).toHaveBeenCalled()
    } finally {
      log.mockRestore()
    }
  })

  test("hash mismatch also falls back", async () => {
    await using tmp = await tmpdir({ init: assetsInit("tampered") })
    const estimator = forModel("glm-4.6", { assetsDir: tmp.path })
    expect(estimator.estimate("hello world")).toBeGreaterThan(0)
    expect(estimator.adapter).toBe("o200k-fallback")
  })
})

describe("R10-008 bound", () => {
  const payloadText = (capture: CaptureFile): string => {
    let start = 0
    while (start < capture.payload.messages.length && capture.payload.messages[start].role === "system") start++
    return [
      capture.payload.system.join("\n"),
      ...Object.entries(capture.payload.tools).map(([name, tool]) =>
        JSON.stringify({ type: "function", function: { name, description: tool.description, parameters: tool.inputSchema } }),
      ),
      ...capture.payload.messages.slice(start).map((message) => JSON.stringify(message)),
    ].join("\n")
  }

  test.each(["glm", "deepseek"] as const)("cold-start fixture within ±10%% with the real adapter engaged (%s)", async (name) => {
    const capture: CaptureFile = JSON.parse(await Bun.file(path.join(import.meta.dir, "fixture", "estimator", name, "capture.json")).text())
    const usage: { tokens: { input: number; cache: { read: number; write: number } } } = JSON.parse(
      await Bun.file(path.join(import.meta.dir, "fixture", "estimator", name, "usage.json")).text(),
    )
    const estimator = forModel(capture.meta.modelID)
    const total = estimator.estimate(payloadText(capture))
    const reported = usage.tokens.input + usage.tokens.cache.read + usage.tokens.cache.write
    expect(estimator.adapter).toBe(name)
    expect(Math.abs(total - reported) / reported).toBeLessThanOrEqual(0.1)
  })
})
