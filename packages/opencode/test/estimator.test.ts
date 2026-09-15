import { describe, expect, spyOn, test } from "bun:test"
import fs from "fs/promises"
import path from "path"
import { ToolListing } from "../src/session/tool-listing"
import type { CaptureFile } from "../src/session/llm/prompt-capture"
import { Skill } from "../src/skill"
import { forModel, parseSkillsListing, parseSystemBlocks, verifyAssets } from "../script/estimator"
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

// --- parseSystemBlocks / parseSkillsListing (SC-5 parser ownership, R10-006) ---
// Committed fixture strings cut from real renders. The environment block,
// instruction entries, mcp server sections and the structured-output constant
// are byte-faithful copies of the producer sites (src/session/system.ts,
// src/session/instruction.ts, src/session/prompt.ts); the skills listing and
// the catalog blocks come from the real Skill.fmt / ToolListing.catalogBlocks
// producers, so drift on those fails here directly.

const ENV_BLOCK = [
  "You are powered by the model named glm-5.3-flash. The exact model ID is z-ai/glm-5.3-flash",
  "Here is some useful information about the environment you are running in:",
  "<env>",
  "  Working directory: /tmp/fixture-repo",
  "  Workspace root folder: /tmp/fixture-repo",
  "  Is directory a git repo: yes",
  "  Platform: linux",
  "  Today's date: Tue Sep 15 2026",
  "</env>",
].join("\n")

const INSTRUCTIONS_BLOCK = [
  "Instructions from: /tmp/fixture-repo/AGENTS.md",
  "Root instructions, line one.",
  "Root instructions, line two.",
  "Instructions from: /home/u/.config/opencode/AGENTS.md",
  "Global instructions.",
].join("\n")

const MCP_GROUP = [
  "<mcp_instructions>",
  '  <server name="github">',
  "    GitHub instructions.",
  "    More GitHub detail.",
  "  </server>",
  '  <server name="linear">',
  "    Linear instructions.",
  "  </server>",
  "</mcp_instructions>",
].join("\n")

const STRUCTURED_OUTPUT_BLOCK =
  "IMPORTANT: The user has requested structured output. You MUST use the StructuredOutput tool to provide your final response. Do NOT respond with plain text - you MUST call the StructuredOutput tool with your answer formatted according to the schema."

const BASE_BLOCK = ["You are a helpful coding agent.", "Follow the user's instructions."].join("\n")

const SKILLS_INTRO = [
  "Skills provide specialized instructions and workflows for specific tasks.",
  "Use the skill tool to load a skill when a task matches its description.",
]

const skillsBlock = (verbose: boolean) =>
  [
    ...SKILLS_INTRO,
    Skill.fmt(
      [
        { name: "tdd", description: "Test-driven development guidance", location: "/tmp/skills/tdd/SKILL.md", content: "" },
        { name: "grill-me", description: "A relentless interview", location: "/tmp/skills/grill-me/SKILL.md", content: "" },
      ],
      { verbose },
    ),
  ].join("\n")

const catalogBlocks = ToolListing.catalogBlocks({
  seeds: [
    { name: "mcp_github_create_issue", kind: "deferred", fullDescription: "Create an issue", jsonSchema: { type: "object" }, server: "github", source: "mcp" },
    { name: "mcp_github_list_prs", kind: "deferred", fullDescription: "", jsonSchema: { type: "object" }, server: "github", source: "mcp" },
    { name: "shellrun", kind: "deferred", fullDescription: "Run a shell command", jsonSchema: { type: "object" }, source: "builtin" },
  ],
})

const CANONICAL_SYSTEM = [
  BASE_BLOCK,
  ENV_BLOCK,
  INSTRUCTIONS_BLOCK,
  MCP_GROUP,
  skillsBlock(false),
  ...catalogBlocks.map((block) => block.content),
  STRUCTURED_OUTPUT_BLOCK,
  "Closing guidance supplied by the user.",
].join("\n")

const o200k = forModel("claude-parser-test")

const bytePartition = (segments: Array<{ text: string }>, text: string) => segments.map((segment) => segment.text).join("") === text

describe("parseSystemBlocks", () => {
  test("segments the canonical render in SC-2 order, every block type", () => {
    const { segments, text } = parseSystemBlocks(CANONICAL_SYSTEM.split("\n"))
    expect(text).toBe(CANONICAL_SYSTEM)
    expect(segments.map((segment) => segment.label)).toEqual([
      "base",
      "environment",
      // Contiguous anchor runs stay separate segments; the breakdown folds
      // them into ONE instructions row (per-file detail rides content).
      "instructions",
      "instructions",
      "mcpInstructionsTags",
      "mcp:github",
      "mcp:linear",
      "mcpInstructionsTags",
      "skills",
      "catalog:github",
      "catalog",
      "structured_output",
      "user_system",
    ])
  })

  test("canonical fixture: other empty, segments tile the joined bytes and tokens exactly", () => {
    const { segments, text } = parseSystemBlocks(CANONICAL_SYSTEM.split("\n"))
    expect(segments.filter((segment) => segment.label === "other")).toEqual([])
    expect(bytePartition(segments, text)).toBe(true)
    expect(segments.reduce((sum, segment) => sum + o200k.estimate(segment.text), 0)).toBe(o200k.estimate(text))
  })

  test("server names ride mcp segments; catalog server attr splits keys", () => {
    const { segments } = parseSystemBlocks(CANONICAL_SYSTEM.split("\n"))
    const mcpSegments = segments.filter((segment): segment is Extract<typeof segment, { server: string }> => "server" in segment)
    expect(mcpSegments.map((segment) => `${segment.label}|${segment.server}`)).toEqual(["mcp:github|github", "mcp:linear|linear"])
    const tags = segments.filter((segment) => segment.label === "mcpInstructionsTags")
    expect(tags.map((segment) => segment.text.trim())).toEqual(["<mcp_instructions>", "</mcp_instructions>"])
  })

  test("mid-text unknown spills to other; leading text is base", () => {
    const text = [ENV_BLOCK, "Some free text in the middle.", INSTRUCTIONS_BLOCK].join("\n")
    const { segments } = parseSystemBlocks(text.split("\n"))
    expect(segments.map((segment) => segment.label)).toEqual(["environment", "other", "instructions", "instructions"])
    expect(segments.find((segment) => segment.label === "other")?.text).toBe("Some free text in the middle.\n")
    expect(segments.find((segment) => segment.label === "base")).toBeUndefined()
  })

  test("references section after </env> stays environment (SC-2)", () => {
    const references = [
      "Project references provide additional directories that can be accessed when relevant.",
      "<available_references>",
      "  <reference>",
      "    <name>effect</name>",
      "  </reference>",
      "</available_references>",
    ].join("\n")
    const text = [ENV_BLOCK, references, INSTRUCTIONS_BLOCK].join("\n")
    const { segments } = parseSystemBlocks(text.split("\n"))
    expect(segments.map((segment) => segment.label)).toEqual(["environment", "environment", "instructions", "instructions"])
    expect(segments.filter((segment) => segment.label === "other")).toEqual([])
    expect(bytePartition(segments, text)).toBe(true)
  })

  test("trailing unknown is user_system", () => {
    const text = [ENV_BLOCK, "User suffix text."].join("\n")
    const { segments } = parseSystemBlocks(text.split("\n"))
    expect(segments.map((segment) => segment.label)).toEqual(["environment", "user_system"])
    expect(segments.at(-1)?.text).toBe("User suffix text.")
  })

  test("text with no anchors is entirely base", () => {
    const { segments } = parseSystemBlocks([BASE_BLOCK])
    expect(segments.map((segment) => segment.label)).toEqual(["base"])
    expect(bytePartition(segments, BASE_BLOCK)).toBe(true)
  })

  test("plugin split restores byte-exact on join", () => {
    const joined = parseSystemBlocks([CANONICAL_SYSTEM])
    const cut = CANONICAL_SYSTEM.indexOf("\n")
    const split = parseSystemBlocks([CANONICAL_SYSTEM.slice(0, cut), CANONICAL_SYSTEM.slice(cut + 1)])
    expect(split.text).toBe(joined.text)
    expect(split.segments).toEqual(joined.segments)
  })

  test("parses the committed real-render capture (glm baseline turn)", async () => {
    const capture: CaptureFile = JSON.parse(
      await Bun.file(path.join(import.meta.dir, "fixture", "estimator", "glm", "capture.json")).text(),
    )
    const real = forModel(capture.meta.modelID)
    const { segments, text } = parseSystemBlocks(capture.payload.system)
    expect(segments.map((segment) => segment.label)).toEqual(["base", "environment", "skills"])
    expect(segments.filter((segment) => segment.label === "other")).toEqual([])
    expect(bytePartition(segments, text)).toBe(true)
    expect(segments.reduce((sum, segment) => sum + real.estimate(segment.text), 0)).toBe(real.estimate(text))
  })
})

describe("parseSkillsListing", () => {
  test("non-verbose: headers carry intro + heading, items carry bullets", () => {
    const text = skillsBlock(false)
    const { headers, items } = parseSkillsListing(text)
    expect(headers).toContain(SKILLS_INTRO[0])
    expect(headers).toContain("## Available Skills")
    expect(items.map((item) => item.name).sort()).toEqual(["grill-me", "tdd"])
    expect(headers + items.map((item) => item.text).join("")).toBe(text)
  })

  test("verbose: items from XML, wrapper tags land in headers", () => {
    const text = skillsBlock(true)
    const { headers, items } = parseSkillsListing(text)
    expect(items.map((item) => item.name).sort()).toEqual(["grill-me", "tdd"])
    expect(headers).toContain("<available_skills>")
    expect(headers).toContain("</available_skills>")
    // headers = text before the first item + text after the last item, so the
    // byte order is head + items + tail; assert the token tiling instead.
    expect(o200k.estimate(headers) + items.reduce((sum, item) => sum + o200k.estimate(item.text), 0)).toBe(o200k.estimate(text))
    expect(items.every((item) => item.text.includes("<location>"))).toBe(true)
  })

  test("no skills: whole text is headers", () => {
    const text = [...SKILLS_INTRO, Skill.fmt([], { verbose: false })].join("\n")
    const { headers, items } = parseSkillsListing(text)
    expect(items).toEqual([])
    expect(headers).toBe(text)
  })

  test("items + headers tile the skills text in tokens", () => {
    for (const verbose of [false, true]) {
      const text = skillsBlock(verbose)
      const { headers, items } = parseSkillsListing(text)
      expect(o200k.estimate(headers) + items.reduce((sum, item) => sum + o200k.estimate(item.text), 0)).toBe(o200k.estimate(text))
    }
  })
})
