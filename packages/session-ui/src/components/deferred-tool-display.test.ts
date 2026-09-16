import { describe, expect, test } from "bun:test"
import { deferredToolInput, deferredToolName } from "./deferred-tool-display"

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

describe("deferredToolInput", () => {
  test("parses the wrapper envelope's args JSON into the inner arguments object", () => {
    const input = { name: "edit", args: JSON.stringify({ filePath: "/a/b.txt", oldString: "a", newString: "b" }) }
    expect(deferredToolInput(input)).toEqual({ filePath: "/a/b.txt", oldString: "a", newString: "b" })
  })

  test("returns undefined for unparseable or non-object args", () => {
    const args = (args: unknown) => ({ name: "edit", args })
    expect(deferredToolInput(args(""))).toBeUndefined()
    expect(deferredToolInput(args("not json"))).toBeUndefined()
    expect(deferredToolInput(args("null"))).toBeUndefined()
    expect(deferredToolInput(args("[1,2]"))).toBeUndefined()
    expect(deferredToolInput(args("\"text\""))).toBeUndefined()
    expect(deferredToolInput(args(42))).toBeUndefined()
    expect(deferredToolInput(args(undefined))).toBeUndefined()
  })

  test("returns undefined for non-envelope input shapes", () => {
    expect(deferredToolInput(undefined)).toBeUndefined()
    expect(deferredToolInput({})).toBeUndefined()
    expect(deferredToolInput({ args: "{}" })).toBeUndefined()
    expect(deferredToolInput({ name: 42, args: "{}" })).toBeUndefined()
  })
})
