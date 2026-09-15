import { describe, expect, test } from "bun:test"
import { deferredToolName } from "./deferred-tool-display"

describe("deferredToolName", () => {
  test("unwraps deferred_tool parts via the metadata key", () => {
    expect(deferredToolName("deferred_tool", { deferred_tool: { tool: "glob" } })).toBe("glob")
  })

  test("keeps the wrapper name when the key is absent or malformed", () => {
    expect(deferredToolName("deferred_tool", {})).toBe("deferred_tool")
    expect(deferredToolName("deferred_tool", undefined)).toBe("deferred_tool")
    expect(deferredToolName("deferred_tool", { deferred_tool: {} })).toBe("deferred_tool")
    expect(deferredToolName("deferred_tool", { deferred_tool: { tool: "" } })).toBe("deferred_tool")
    expect(deferredToolName("deferred_tool", { deferred_tool: "nope" })).toBe("deferred_tool")
  })

  test("passes other tools through untouched", () => {
    expect(deferredToolName("skill", { name: "legit-skill" })).toBe("skill")
    expect(deferredToolName("bash", { output: "hi" })).toBe("bash")
  })
})
