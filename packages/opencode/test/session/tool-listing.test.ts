import { describe, expect, test } from "bun:test"
import type { JSONSchema7 } from "@ai-sdk/provider"
import { MalformedToolEntryError, PLACEHOLDER, ToolListing, type UniverseTool } from "../../src/session/tool-listing"

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
          def("shell", "Runs a shell command"),
          def("read", "Reads a file"),
          def("load_tool", "Loads a deferred tool"),
          def("glob", long),
          def("firecrawl_scrape", "Scrapes a page", "mcp", "firecrawl"),
          def("list_mcp_resources", "Lists resources", "resource"),
          def("custom_tool", "A custom tool", "custom"),
        ],
        ruleset: [],
      })
      const byName = new Map(seeds.map((seed) => [seed.name, seed]))
      expect(byName.get("shell")?.kind).toBe("eager")
      expect(byName.get("read")?.kind).toBe("eager")
      expect(byName.get("load_tool")?.kind).toBe("eager")
      expect(byName.get("glob")?.kind).toBe("deferred")
      expect(byName.get("firecrawl_scrape")?.kind).toBe("deferred")
      expect(byName.get("list_mcp_resources")?.kind).toBe("deferred")
      expect(byName.get("custom_tool")?.kind).toBe("deferred")
      // full facts: shape never truncates and never replaces the schema
      expect(byName.get("glob")?.fullDescription).toEqual(long)
      expect(byName.get("glob")?.jsonSchema).toEqual(schema("glob"))
      expect(byName.get("shell")?.fullDescription).toEqual("Runs a shell command")
      expect(byName.get("firecrawl_scrape")?.server).toBe("firecrawl")
      expect(byName.get("shell")?.server).toBeUndefined()
      // input order preserved (becomes frozen insertion order downstream)
      expect(seeds.map((seed) => seed.name)).toEqual([
        "shell",
        "read",
        "load_tool",
        "glob",
        "firecrawl_scrape",
        "list_mcp_resources",
        "custom_tool",
      ])
    })

    test("drops blanket-denied tools in any form", () => {
      const seeds = ToolListing.shape({
        universe: [
          def("shell", "Runs commands"),
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
      expect(seeds.map((seed) => seed.name)).toEqual(["shell", "firecrawl_scrape"])
    })

    test("does not mutate its input", () => {
      const universe = [def("shell", "Runs commands"), def("glob", "Finds files")]
      const snapshot = structuredClone(universe)
      ToolListing.shape({ universe, ruleset: [{ permission: "shell", pattern: "*", action: "deny" }] })
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
          universe: [def("shell", "Runs a shell command"), def("glob", "find files by glob patterns " + "y".repeat(80))],
          ruleset: [],
        }),
        "advisory",
      )
      expect(view).toEqual([
        { name: "shell", description: "Runs a shell command", jsonSchema: schema("shell") },
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
            def("shell", "Runs commands"),
          ],
          ruleset: [],
        }),
        "advisory",
      )
      expect(view.map((entry) => entry.name)).toEqual(["firecrawl_scrape", "firecrawl_search", "notion_search", "shell"])
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
      const view = ToolListing.render([...frozen, ...appended], "advisory")
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
      expect(Object.keys(seeds[0]).toSorted()).toEqual(["fullDescription", "jsonSchema", "kind", "name"])
      const view = ToolListing.render(seeds, "advisory")
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
        def("shell", "Runs a shell command"),
        def("glob", long),
        def("firecrawl_scrape", "Scrapes a page", "mcp", "firecrawl"),
        def("firecrawl_search", "Searches the web", "mcp", "firecrawl"),
      ],
      ruleset: [],
    })

    test("binding: every deferred entry carries its full schema; the description stays truncated", () => {
      const view = ToolListing.render(seeds, "binding")
      expect(view[0]).toEqual({ name: "shell", description: "Runs a shell command", jsonSchema: schema("shell") })
      expect(view[1].name).toBe("glob")
      // schema-eager, description-deferred (R12-010 amendment 1)
      expect(view[1].jsonSchema).toEqual(schema("glob"))
      expect(view[1].description).toBe(long.slice(0, long.lastIndexOf(" ")) + "...")
    })

    test("advisory: the same entries carry the constant placeholder instead", () => {
      const view = ToolListing.render(seeds, "advisory")
      expect(view[1].jsonSchema).toEqual(PLACEHOLDER)
      expect(view[2].jsonSchema).toEqual(PLACEHOLDER)
      // the mode changes which schema value is written, never the entry shape
      expect(Object.keys(view[1]).toSorted()).toEqual(["description", "jsonSchema", "name"])
    })

    test("prefix-once and eager fullness are mode-independent", () => {
      const binding = ToolListing.render(seeds, "binding")
      const advisory = ToolListing.render(seeds, "advisory")
      expect(binding.map((entry) => entry.description)).toEqual(advisory.map((entry) => entry.description))
      expect(binding[2].description).toBe("firecrawl: Scrapes a page")
      expect(binding[0].jsonSchema).toEqual(advisory[0].jsonSchema)
    })
  })
})
