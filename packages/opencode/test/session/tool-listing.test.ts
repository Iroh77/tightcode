import { describe, expect, test } from "bun:test"
import type { JSONSchema7 } from "@ai-sdk/provider"
import { PermissionV1 } from "@opencode-ai/core/v1/permission"
import type { SessionV1 } from "@opencode-ai/core/v1/session"
import type { ToolExecutionOptions } from "ai"
import { Effect } from "effect"
import type { EffectBridge } from "../../src/effect/bridge"
import { MessageID, PartID, SessionID } from "../../src/session/schema"
import { Tool } from "../../src/tool/tool"
import { MalformedToolEntryError, PLACEHOLDER, ToolListing, type ToolSeed, type UniverseTool } from "../../src/session/tool-listing"

const schema = (label: string): JSONSchema7 => ({
  type: "object",
  properties: { label: { type: "string", const: label } },
  required: ["label"],
  additionalProperties: false,
})

const def = (name: string, description: string, source: UniverseTool["source"] = "builtin", server?: string) => ({
  name,
  description,
  jsonSchema: schema(name),
  source,
  ...(server ? { server } : {}),
})

describe("session.tool-listing", () => {
  describe("shape", () => {
    test("keeps the eager set full and defers everything else with full facts", () => {
      const long = "L".repeat(150)
      const seeds = ToolListing.shape({
        universe: [
          def("bash", "Runs a shell command"),
          def("read", "Reads a file"),
          def("load_tool", "Loads a deferred tool"),
          def("deferred_tool", "Execute a deferred tool"),
          def("glob", long),
          def("firecrawl_scrape", "Scrapes a page", "mcp", "firecrawl"),
          def("list_mcp_resources", "Lists resources", "resource"),
          def("custom_tool", "A custom tool", "custom"),
        ],
        ruleset: [],
      })
      const byName = new Map(seeds.map((seed) => [seed.name, seed]))
      expect(byName.get("bash")?.kind).toBe("eager")
      expect(byName.get("read")?.kind).toBe("eager")
      expect(byName.get("load_tool")?.kind).toBe("eager")
      // the wrapper meta-tool is eager — it only ever enters the universe on
      // binding+wrapper sessions, gated at the resolve confluence
      expect(byName.get("deferred_tool")?.kind).toBe("eager")
      expect(byName.get("glob")?.kind).toBe("deferred")
      expect(byName.get("firecrawl_scrape")?.kind).toBe("deferred")
      expect(byName.get("list_mcp_resources")?.kind).toBe("deferred")
      expect(byName.get("custom_tool")?.kind).toBe("deferred")
      // full facts: shape never truncates and never replaces the schema
      expect(byName.get("glob")?.fullDescription).toEqual(long)
      expect(byName.get("glob")?.jsonSchema).toEqual(schema("glob"))
      expect(byName.get("bash")?.fullDescription).toEqual("Runs a shell command")
      expect(byName.get("firecrawl_scrape")?.server).toBe("firecrawl")
      expect(byName.get("bash")?.server).toBeUndefined()
      // input order preserved (becomes frozen insertion order downstream)
      expect(seeds.map((seed) => seed.name)).toEqual([
        "bash",
        "read",
        "load_tool",
        "deferred_tool",
        "glob",
        "firecrawl_scrape",
        "list_mcp_resources",
        "custom_tool",
      ])
    })

    test("drops blanket-denied tools in any form", () => {
      const seeds = ToolListing.shape({
        universe: [
          def("bash", "Runs commands"),
          def("read", "Reads files"),
          def("glob", "Finds files"),
          def("edit", "Edits files"),
          def("write", "Writes files"),
          def("list_mcp_resources", "Lists resources", "resource"),
          def("firecrawl_search", "Searches the web", "mcp", "firecrawl"),
          def("firecrawl_scrape", "Scrapes", "mcp", "firecrawl"),
        ],
        ruleset: [
          { permission: "edit", pattern: "*", action: "deny" },
          { permission: "read", pattern: "*", action: "deny" },
          { permission: "glob", pattern: "*", action: "deny" },
          { permission: "firecrawl_search", pattern: "*", action: "deny" },
        ],
      })
      // R12-001 dominates the eager set: a denied eager name is not listed either
      expect(seeds.map((seed) => seed.name)).toEqual(["bash", "firecrawl_scrape"])
    })

    test("does not mutate its input", () => {
      const universe = [def("bash", "Runs commands"), def("glob", "Finds files")]
      const snapshot = structuredClone(universe)
      ToolListing.shape({ universe, ruleset: [{ permission: "bash", pattern: "*", action: "deny" }] })
      expect(universe).toEqual(snapshot)
    })

    test("raises a typed error on malformed universe entries instead of swallowing", () => {
      const base = def("ok", "fine")
      expect(() => ToolListing.shape({ universe: [{ ...base, name: "" }], ruleset: [] })).toThrow(
        MalformedToolEntryError,
      )
      expect(() => ToolListing.shape({ universe: [{ ...base, description: undefined as unknown as string }], ruleset: [] })).toThrow(
        MalformedToolEntryError,
      )
      expect(() => ToolListing.shape({ universe: [{ ...base, jsonSchema: null as unknown as JSONSchema7 }], ruleset: [] })).toThrow(
        MalformedToolEntryError,
      )
      expect(() => ToolListing.shape({ universe: [{ ...base, source: "unknown" as UniverseTool["source"] }], ruleset: [] })).toThrow(
        MalformedToolEntryError,
      )
    })
  })

  describe("render", () => {
    test("advisory mode: eager full, deferred truncated with placeholder schema", () => {
      const view = ToolListing.render(
        ToolListing.shape({
          universe: [def("bash", "Runs a shell command"), def("glob", "find files by glob patterns " + "y".repeat(80))],
          ruleset: [],
        }),
        "advisory",
        false,
      )
      expect(view).toEqual([
        { name: "bash", description: "Runs a shell command", jsonSchema: schema("bash") },
        {
          name: "glob",
          description: "find files by glob patterns...",
          jsonSchema: { type: "object", properties: {} },
        },
      ])
    })

    test("truncation bound: exactly 100 chars unchanged, longer cut at the last word boundary", () => {
      const bound = "z".repeat(100)
      const view = ToolListing.render(
        ToolListing.shape({
          universe: [
            def("tool_a", bound),
            def("tool_b", "short description"),
            def("tool_c", bound + " tail"),
            def("tool_d", ""),
          ],
          ruleset: [],
        }),
        "advisory",
        false,
      )
      expect(view[0].description).toEqual(bound)
      expect(view[1].description).toEqual("short description")
      // no word boundary within the first 100 chars: the bound is kept whole
      expect(view[2].description).toEqual(bound + "...")
      expect(view[3].description).toEqual("")
    })

    test("writes the server prefix once, on the group's first-written entry", () => {
      const view = ToolListing.render(
        ToolListing.shape({
          universe: [
            def("firecrawl_scrape", "Scrapes", "mcp", "firecrawl"),
            def("firecrawl_search", "Searches", "mcp", "firecrawl"),
            def("notion_search", "Searches notion", "mcp", "notion"),
            def("bash", "Runs commands"),
          ],
          ruleset: [],
        }),
        "advisory",
        false,
      )
      expect(view.map((entry) => entry.name)).toEqual(["firecrawl_scrape", "firecrawl_search", "notion_search", "bash"])
      expect(view[0].description).toBe("firecrawl: Scrapes")
      expect(view[1].description).toBe("Searches")
      expect(view[2].description).toBe("notion: Searches notion")
      expect(view[3].description).toBe("Runs commands")
    })

    test("a later same-server entry never moves the prefix (append stability)", () => {
      const frozen = ToolListing.shape({ universe: [def("firecrawl_search", "Searches", "mcp", "firecrawl")], ruleset: [] })
      const appended = ToolListing.shape({
        universe: [def("firecrawl_api_call", "Calls the API", "mcp", "firecrawl")],
        ruleset: [],
      })
      // frozen insertion order: firecrawl_search written first, api_call appended later
      const view = ToolListing.render([...frozen, ...appended], "advisory", false)
      expect(view[0].description).toBe("firecrawl: Searches")
      expect(view[1].description).toBe("Calls the API")
    })

    test("keeps empty descriptions empty; the first-written entry carries the prefix alone", () => {
      const view = ToolListing.render(
        ToolListing.shape({
          universe: [
            def("srv_one", "", "mcp", "srv"),
            def("srv_two", "", "mcp", "srv"),
            def("bare", "", "builtin"),
          ],
          ruleset: [],
        }),
        "advisory",
        false,
      )
      expect(view[0].description).toBe("srv: ")
      expect(view[1].description).toBe("")
      expect(view[2].description).toBe("")
    })

    test("one uniform listing shape for every tool source", () => {
      const universe = [
        def("tool_a", "Builtin", "builtin"),
        def("tool_b", "Resource", "resource"),
        def("tool_c", "Mcp", "mcp", "srv"),
        def("tool_d", "Plugin", "plugin"),
        def("tool_e", "Custom", "custom"),
      ]
      const seeds = ToolListing.shape({ universe, ruleset: [] })
      expect(Object.keys(seeds[0]).toSorted()).toEqual(["fullDescription", "jsonSchema", "kind", "name", "source"])
      // the source rides the seed (frozen with the entry) so the catalog
      // producer can group without a universe re-read
      expect(seeds.map((seed) => seed.source)).toEqual(["builtin", "resource", "mcp", "plugin", "custom"])
      const view = ToolListing.render(seeds, "advisory", false)
      for (const entry of view) {
        expect(Object.keys(entry).toSorted()).toEqual(["description", "jsonSchema", "name"])
        expect(entry.jsonSchema).toEqual({ type: "object", properties: {} })
      }
    })
  })

  describe("render binding mode (R12-010)", () => {
    const long = "search the workspace for " + "x".repeat(90)

    const seeds = ToolListing.shape({
      universe: [
        def("bash", "Runs a shell command"),
        def("glob", long),
        def("firecrawl_scrape", "Scrapes a page", "mcp", "firecrawl"),
        def("firecrawl_search", "Searches the web", "mcp", "firecrawl"),
      ],
      ruleset: [],
    })

    test("binding: every deferred entry carries its full schema; the description stays truncated", () => {
      const view = ToolListing.render(seeds, "binding", false)
      expect(view[0]).toEqual({ name: "bash", description: "Runs a shell command", jsonSchema: schema("bash") })
      expect(view[1].name).toBe("glob")
      // schema-eager, description-deferred (R12-010 amendment 1)
      expect(view[1].jsonSchema).toEqual(schema("glob"))
      expect(view[1].description).toBe(long.slice(0, long.lastIndexOf(" ")) + "...")
    })

    test("advisory: the same entries carry the constant placeholder instead", () => {
      const view = ToolListing.render(seeds, "advisory", false)
      expect(view[1].jsonSchema).toEqual(PLACEHOLDER)
      expect(view[2].jsonSchema).toEqual(PLACEHOLDER)
      // the mode changes which schema value is written, never the entry shape
      expect(Object.keys(view[1]).toSorted()).toEqual(["description", "jsonSchema", "name"])
    })

    test("prefix-once and eager fullness are mode-independent", () => {
      const binding = ToolListing.render(seeds, "binding", false)
      const advisory = ToolListing.render(seeds, "advisory", false)
      expect(binding.map((entry) => entry.description)).toEqual(advisory.map((entry) => entry.description))
      expect(binding[2].description).toBe("firecrawl: Scrapes a page")
      expect(binding[0].jsonSchema).toEqual(advisory[0].jsonSchema)
    })
  })

  describe("render wrapper policy (R12-012, ticket 17)", () => {
    const long = "search the workspace for " + "x".repeat(90)

    const seeds = ToolListing.shape({
      universe: [
        def("bash", "Runs a shell command"),
        def("glob", long),
        def("firecrawl_scrape", "Scrapes a page", "mcp", "firecrawl"),
        def("firecrawl_search", "Searches the web", "mcp", "firecrawl"),
      ],
      ruleset: [],
    })

    test("binding + wrapper: deferred entries produce no listing entry; eager entries stay", () => {
      const view = ToolListing.render(seeds, "binding", true)
      expect(view.map((entry) => entry.name)).toEqual(["bash"])
      expect(view[0]).toEqual({ name: "bash", description: "Runs a shell command", jsonSchema: schema("bash") })
    })

    test("the eager meta-tool seed rides the wrapper listing with full facts", () => {
      const withMeta = ToolListing.shape({
        universe: [def("glob", long), def("deferred_tool", "Execute a deferred tool")],
        ruleset: [],
      })
      const view = ToolListing.render(withMeta, "binding", true)
      expect(view.map((entry) => entry.name)).toEqual(["deferred_tool"])
      expect(view[0].description).toBe("Execute a deferred tool")
      expect(view[0].jsonSchema).toEqual(schema("deferred_tool"))
    })

    test("kill-switch (wrapper=false): round-1 binding bytes", () => {
      const view = ToolListing.render(seeds, "binding", false)
      expect(view.map((entry) => entry.name)).toEqual(["bash", "glob", "firecrawl_scrape", "firecrawl_search"])
      expect(view[1].jsonSchema).toEqual(schema("glob"))
      expect(view[1].description).toBe(long.slice(0, long.lastIndexOf(" ")) + "...")
      expect(view[2].description).toBe("firecrawl: Scrapes a page")
    })

    test("advisory golden: byte-identical to the round-1 render regardless of the wrapper axis", () => {
      // the pair advisory+wrapper cannot occur (wrapperActive requires
      // binding); defensively the advisory shape wins either way
      expect(ToolListing.render(seeds, "advisory", true)).toEqual(ToolListing.render(seeds, "advisory", false))
      expect(ToolListing.render(seeds, "advisory", false)).toEqual([
        { name: "bash", description: "Runs a shell command", jsonSchema: schema("bash") },
        { name: "glob", description: long.slice(0, long.lastIndexOf(" ")) + "...", jsonSchema: PLACEHOLDER },
        { name: "firecrawl_scrape", description: "firecrawl: Scrapes a page", jsonSchema: PLACEHOLDER },
        { name: "firecrawl_search", description: "Searches the web", jsonSchema: PLACEHOLDER },
      ])
    })

    test("prefix-once ownership is computed over the full frozen list, unaffected by omission", () => {
      // the omission render never moves an owner: the same frozen list rendered
      // without the wrapper still carries the prefix on the first-written entry
      expect(ToolListing.render(seeds, "binding", true).map((entry) => entry.name)).toEqual(["bash"])
      const restored = ToolListing.render(seeds, "binding", false)
      expect(restored[2].description).toBe("firecrawl: Scrapes a page")
      expect(restored[3].description).toBe("Searches the web")
    })
  })

  describe("catalogBlocks (R12-012 discovery, ticket 18)", () => {
    // Grouping fixtures in seed order: two builtin-deferred, one custom, one
    // resource, two firecrawl, one notion — the same seed order the frozen
    // append produces.
    const catalogSeeds = ToolListing.shape({
      universe: [
        def("glob", "Finds files " + "x".repeat(100)),
        def("task", "", "custom"),
        def("list_mcp_resources", "Lists resources", "resource"),
        def("firecrawl_scrape", "Scrapes a page", "mcp", "firecrawl"),
        def("firecrawl_search", "Searches the web", "mcp", "firecrawl"),
        def("notion_search", "Searches notion", "mcp", "notion"),
      ],
      ruleset: [],
    })

    test("groups deferred seeds by source into the three-key family in first-appearance order", () => {
      const blocks = ToolListing.catalogBlocks({ seeds: catalogSeeds })
      expect(blocks.map((block) => block.key)).toEqual([
        "catalog",
        "catalog:resources",
        "catalog:firecrawl",
        "catalog:notion",
      ])
    })

    test("block content is byte-exact: own delimiters, name + truncated description bullets, seed order", () => {
      const blocks = ToolListing.catalogBlocks({ seeds: catalogSeeds })
      const long = "Finds files " + "x".repeat(100)
      // no server description prefix inside blocks — the block is the grouping
      // (R12-003's prefix-once governs listing entries, absent in this mode)
      expect(blocks[0]).toEqual({
        key: "catalog",
        content: `<deferred_tools>\n- glob: ${long.slice(0, long.lastIndexOf(" "))}...\n- task\n</deferred_tools>`,
      })
      expect(blocks[1]).toEqual({
        key: "catalog:resources",
        content: "<deferred_tools>\n- list_mcp_resources: Lists resources\n</deferred_tools>",
      })
      // one block per server; only mcp blocks carry the server attribute
      expect(blocks[2]).toEqual({
        key: "catalog:firecrawl",
        content:
          '<deferred_tools server="firecrawl">\n- firecrawl_scrape: Scrapes a page\n- firecrawl_search: Searches the web\n</deferred_tools>',
      })
      expect(blocks[3]).toEqual({
        key: "catalog:notion",
        content: '<deferred_tools server="notion">\n- notion_search: Searches notion\n</deferred_tools>',
      })
    })

    test("empty description renders the name alone", () => {
      const blocks = ToolListing.catalogBlocks({
        seeds: ToolListing.shape({ universe: [def("bare", "")], ruleset: [] }),
      })
      expect(blocks).toEqual([{ key: "catalog", content: "<deferred_tools>\n- bare\n</deferred_tools>" }])
    })

    test("empty groups emit nothing; zero deferred seeds produce no blocks", () => {
      // no resource seeds → no catalog:resources key; no mcp seeds → no per-server keys
      const builtinOnly = ToolListing.catalogBlocks({
        seeds: ToolListing.shape({ universe: [def("glob", "Finds files")], ruleset: [] }),
      })
      expect(builtinOnly.map((block) => block.key)).toEqual(["catalog"])
      // eager-only seed list (e.g. a wrapper session with no deferred tools at
      // first write) emits nothing at all
      const eagerOnly = ToolListing.catalogBlocks({
        seeds: ToolListing.shape({ universe: [def("bash", "Runs commands"), def("deferred_tool", "Execute a deferred tool")], ruleset: [] }),
      })
      expect(eagerOnly).toEqual([])
    })

    test("eager seeds never enter a catalog block (the meta-tool is not its own discovery)", () => {
      const blocks = ToolListing.catalogBlocks({
        seeds: ToolListing.shape({ universe: [def("deferred_tool", "Execute a deferred tool"), def("glob", "Finds files")], ruleset: [] }),
      })
      expect(blocks).toEqual([{ key: "catalog", content: "<deferred_tools>\n- glob: Finds files\n</deferred_tools>" }])
    })

    test("pure and stateless: same input → identical blocks, input seeds unmutated", () => {
      const snapshot = structuredClone(catalogSeeds)
      expect(ToolListing.catalogBlocks({ seeds: catalogSeeds })).toEqual(ToolListing.catalogBlocks({ seeds: catalogSeeds }))
      expect(catalogSeeds).toEqual(snapshot)
    })

    test("an mcp seed without a server raises instead of silently regrouping", () => {
      const orphan = { ...ToolListing.shape({ universe: [def("srv_tool", "Does things", "mcp", "srv")], ruleset: [] })[0], server: undefined }
      expect(() => ToolListing.catalogBlocks({ seeds: [orphan] })).toThrow(MalformedToolEntryError)
    })
  })
})

describe("session.tool-listing withFallback (R12-006)", () => {
  const globSeed: ToolSeed = {
    name: "glob",
    kind: "deferred",
    fullDescription: "glob full description",
    jsonSchema: schema("glob"),
    source: "builtin",
  }

  const sessionID = SessionID.make("ses_fallback")

  const part = (state: SessionV1.ToolState): SessionV1.ToolPart => ({
    id: PartID.ascending(),
    sessionID,
    messageID: MessageID.ascending(),
    type: "tool",
    callID: "call_fallback",
    tool: "glob",
    state,
  })

  const runningPart = () =>
    part({ status: "running", input: { pattern: "*" }, time: { start: 0 }, metadata: { progress: "half" } })

  const completedPart = () =>
    part({
      status: "completed",
      input: {},
      output: "done",
      title: "done",
      metadata: { preset: true },
      time: { start: 0, end: 1 },
    })

  // A delivered marker on an error-shaped part (what a first fallback failure
  // leaves behind) makes delivered() true for the next failure.
  const fallbackHistory = (): SessionV1.WithParts[] => [
    {
      info: {} as SessionV1.Assistant,
      parts: [
        part({
          status: "error",
          input: { pattern: "*" },
          error: "boom\n\nschema",
          metadata: { load_tool: { tools: ["glob"] } },
          time: { start: 0, end: 1 },
        }),
      ],
    },
  ]

  const options = (): ToolExecutionOptions => ({
    toolCallId: "call_fallback",
    messages: [],
    abortSignal: new AbortController().signal,
  })
  const abortedOptions = (): ToolExecutionOptions => {
    const controller = new AbortController()
    controller.abort()
    return { toolCallId: "call_fallback", messages: [], abortSignal: controller.signal }
  }

  const harness = (input: {
    messages?: SessionV1.WithParts[]
    part?: SessionV1.ToolPart
    execute: (args: unknown, options: ToolExecutionOptions) => Promise<unknown>
  }) => {
    const observed: string[] = []
    const marked: SessionV1.ToolPart[] = []
    let current = input.part
    const run = {
      promise: (effect: Effect.Effect<unknown, unknown, never>) => Effect.runPromise(effect),
    } as EffectBridge.Shape
    const wrapped = ToolListing.withFallback(
      {
        seed: globSeed,
        messages: input.messages ?? [],
        run,
        updateToolCall: (_toolCallID, update) =>
          Effect.sync(() => {
            if (!current) return undefined
            current = update(current)
            marked.push(current)
            return current
          }),
        observe: Effect.sync(() => observed.push("schema-violation")),
      },
      input.execute,
    )
    return { wrapped, observed, marked, current: () => current }
  }

  const rejection = async (promise: Promise<unknown>): Promise<unknown> =>
    promise.then(
      () => {
        throw new Error("expected the call to fail")
      },
      (error) => error,
    )

  test("failed call appends the full schema to the error text and marks the part loaded", async () => {
    const h = harness({ part: runningPart(), execute: async () => { throw new Error("boom") } })

    const error = await rejection(h.wrapped({}, options()))

    expect(error).toBeInstanceOf(Error)
    expect((error as Error).message).toContain("boom")
    expect((error as Error).message).toContain("glob")
    expect((error as Error).message).toContain(JSON.stringify(schema("glob"), null, 2))
    const state = h.current()?.state
    if (state?.status !== "running") throw new Error("part should still be running")
    expect(state.metadata?.load_tool).toEqual({ tools: ["glob"] })
    // progress metadata written while running survives for failToolCall to preserve
    expect(state.metadata?.progress).toBe("half")
    expect(h.observed).toEqual([])
  })

  test("repeat failure stays quiet: the delivery marker already opened the tool", async () => {
    const h = harness({ messages: fallbackHistory(), part: runningPart(), execute: async () => { throw new Error("boom again") } })
    const before = structuredClone(h.current())

    const error = await rejection(h.wrapped({}, options()))

    expect((error as Error).message).toBe("boom again")
    expect(h.current()).toEqual(before)
    expect(h.marked).toEqual([])
  })

  test("abort is excluded: the original error passes through untouched", async () => {
    const h = harness({ part: runningPart(), execute: async () => { throw new Error("aborted mid-execution") } })
    const before = structuredClone(h.current())

    const error = await rejection(h.wrapped({}, abortedOptions()))

    expect((error as Error).message).toBe("aborted mid-execution")
    expect(h.current()).toEqual(before)
    expect(h.marked).toEqual([])
  })

  test("permission rejections are excluded in every shape", async () => {
    for (const denial of [
      new PermissionV1.RejectedError(),
      new PermissionV1.DeniedError({ ruleset: [] }),
      new PermissionV1.CorrectedError({ feedback: "no" }),
    ]) {
      const h = harness({ part: runningPart(), execute: async () => { throw denial } })
      const before = structuredClone(h.current())

      const error = await rejection(h.wrapped({}, options()))

      expect(error).toBe(denial)
      expect(h.current()).toEqual(before)
      expect(h.marked).toEqual([])
    }
  })

  test("invalid arguments observe the verdict even when already delivered, and stay quiet", async () => {
    const h = harness({
      messages: fallbackHistory(),
      part: runningPart(),
      execute: async () => { throw new Tool.InvalidArgumentsError({ tool: "glob", detail: "missing pattern" }) },
    })
    const before = structuredClone(h.current())

    const error = await rejection(h.wrapped({}, options()))

    expect(h.observed).toEqual(["schema-violation"])
    expect(error).toBeInstanceOf(Tool.InvalidArgumentsError)
    expect(h.current()).toEqual(before)
    expect(h.marked).toEqual([])
  })

  test("invalid arguments when not delivered: observe, append the schema, mark", async () => {
    const h = harness({
      part: runningPart(),
      execute: async () => { throw new Tool.InvalidArgumentsError({ tool: "glob", detail: "missing pattern" }) },
    })

    const error = await rejection(h.wrapped({}, options()))

    expect(h.observed).toEqual(["schema-violation"])
    expect((error as Error).message).toContain("missing pattern")
    expect((error as Error).message).toContain(JSON.stringify(schema("glob"), null, 2))
    const marked = h.current()?.state
    if (marked?.status !== "running") throw new Error("part should still be running")
    expect(marked.metadata?.load_tool).toEqual({ tools: ["glob"] })
  })

  test("the marker only lands on a running part", async () => {
    const h = harness({ part: completedPart(), execute: async () => { throw new Error("boom") } })

    await rejection(h.wrapped({}, options()))

    expect((h.current()?.state as { metadata?: unknown }).metadata).toEqual({ preset: true })
  })
})
