import { describe, expect, test } from "bun:test"
import { deferredToolName, toolDisplayMetadata, webSearchProviderLabel } from "../../src/util/tool-display"

describe("webSearchProviderLabel", () => {
  test("labels known providers", () => {
    expect(webSearchProviderLabel("parallel")).toBe("Parallel Web Search")
    expect(webSearchProviderLabel("exa")).toBe("Exa Web Search")
  })

  for (const [name, provider] of [
    ["undefined", undefined],
    ["null", null],
    ["an object", {}],
    ["an array", []],
    ["a number", 1],
    ["an unexpected string", "other"],
  ] as const) {
    test(`uses the generic label for ${name}`, () => {
      expect(webSearchProviderLabel(provider)).toBe("Web Search")
    })
  }
})

describe("toolDisplayMetadata", () => {
  test("returns structured metadata for non-pending states", () => {
    const structured = { provider: "parallel", numResults: 3 }

    expect(toolDisplayMetadata({ status: "running", structured })).toBe(structured)
    expect(toolDisplayMetadata({ status: "completed", structured })).toBe(structured)
    expect(toolDisplayMetadata({ status: "error", structured })).toBe(structured)
  })

  test("does not expose pending or malformed metadata", () => {
    expect(toolDisplayMetadata({ status: "pending", structured: { provider: "exa" } })).toEqual({})
    expect(toolDisplayMetadata({ status: "completed" })).toEqual({})
    expect(toolDisplayMetadata({ status: "completed", structured: null })).toEqual({})
    expect(toolDisplayMetadata({ status: "completed", structured: [] })).toEqual({})
    expect(toolDisplayMetadata(undefined)).toEqual({})
  })
})

describe("deferredToolName", () => {
  test("unwraps deferred_tool parts via the metadata key", () => {
    expect(deferredToolName("deferred_tool", { deferred_tool: { tool: "glob" } })).toBe("glob")
  })

  test("keeps the wrapper name when the key is absent or malformed", () => {
    expect(deferredToolName("deferred_tool", {})).toBe("deferred_tool")
    expect(deferredToolName("deferred_tool", { deferred_tool: {} })).toBe("deferred_tool")
    expect(deferredToolName("deferred_tool", { deferred_tool: { tool: "" } })).toBe("deferred_tool")
    expect(deferredToolName("deferred_tool", { deferred_tool: "nope" })).toBe("deferred_tool")
    expect(deferredToolName("deferred_tool", undefined as unknown as Record<string, unknown>)).toBe("deferred_tool")
  })

  test("passes other tools through untouched", () => {
    expect(deferredToolName("skill", { name: "legit-skill" })).toBe("skill")
    expect(deferredToolName("bash", { output: "hi" })).toBe("bash")
  })
})
