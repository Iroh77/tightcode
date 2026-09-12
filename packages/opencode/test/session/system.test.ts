import { describe, expect, test } from "bun:test"
import { LayerNode } from "@opencode-ai/core/effect/layer-node"
import { Effect, Layer } from "effect"
import type { Agent } from "../../src/agent/agent"
import { NamedError } from "@opencode-ai/core/util/error"
import { Skill } from "../../src/skill"
import { Permission } from "../../src/permission"
import type { Provider } from "../../src/provider/provider"
import { SystemPrompt } from "../../src/session/system"
import { MCP } from "../../src/mcp"
import { RuntimeFlags } from "../../src/effect/runtime-flags"
import { testEffect } from "../lib/effect"

const skills: Skill.Info[] = [
  {
    name: "zeta-skill",
    description: "Zeta skill.",
    location: "/tmp/zeta-skill/SKILL.md",
    content: "# zeta-skill",
  },
  {
    name: "alpha-skill",
    description: "Alpha skill.",
    location: "/tmp/alpha-skill/SKILL.md",
    content: "# alpha-skill",
  },
  {
    name: "middle-skill",
    description: "Middle skill.",
    location: "/tmp/middle-skill/SKILL.md",
    content: "# middle-skill",
  },
  {
    name: "manual-skill",
    location: "/tmp/manual-skill/SKILL.md",
    content: "# manual-skill",
  },
]

const build: Agent.Info = {
  name: "build",
  mode: "primary",
  permission: Permission.fromConfig({ "*": "allow" }),
  options: {},
}

const it = testEffect(
  LayerNode.compile(SystemPrompt.node, [
    [
      MCP.node,
      Layer.mock(MCP.Service, {
        instructions: () =>
          Effect.succeed([
            {
              name: "guide-server",
              instructions: "Use lookup before mutate.",
              tools: [],
            },
            {
              name: "tool-server",
              instructions: "Prefer search before update.",
              tools: ["tool-server_search", "tool-server_update"],
            },
          ]),
      }),
    ],
    [
      Skill.node,
      Layer.succeed(
        Skill.Service,
        Skill.Service.of({
          get: (name) => Effect.succeed(skills.find((skill) => skill.name === name)),
          require: (name) => {
            const info = skills.find((skill) => skill.name === name)
            if (info) return Effect.succeed(info)
            return Effect.fail(new Skill.NotFoundError({ name, available: skills.map((skill) => skill.name) }))
          },
          all: () => Effect.succeed(skills),
          dirs: () => Effect.succeed([]),
          available: () => Effect.succeed(skills),
        }),
      ),
    ],
  ]),
)

describe("session.system", () => {
  test("selects the Meta prompt for Muse Spark model IDs", () => {
    for (const id of ["meta/muse-spark-preview", "muse-spark-1.1", "muse-spark-1.2"]) {
      const prompt = SystemPrompt.provider({ api: { id } } as Provider.Model)[0]
      expect(prompt).toContain("powered by Muse Spark,")
      expect(prompt).toContain("using Meta Muse Spark.")
      expect(prompt).not.toContain("{{MODEL_NAME}}")
    }
  })

  test("selects the Meta prompt for Muse Glimmer model IDs", () => {
    for (const id of ["meta/muse-glimmer", "meta/muse-glimmer-30b", "muse-glimmer-30b"]) {
      const prompt = SystemPrompt.provider({ api: { id } } as Provider.Model)[0]
      expect(prompt).toContain("powered by Muse Glimmer,")
      expect(prompt).toContain("using Meta Muse Glimmer.")
      expect(prompt).not.toContain("{{MODEL_NAME}}")
    }
  })

  test("selects the Kimi prompt for official provider model IDs", () => {
    for (const providerID of ["kimi-for-coding", "moonshotai", "moonshotai-cn"]) {
      const prompt = SystemPrompt.provider({ providerID, api: { id: "k3" } } as Provider.Model)[0]
      expect(prompt).toContain("# Prompt and Tool Use")
    }
  })

  it.effect("skills output is sorted by name and stable across calls", () =>
    Effect.gen(function* () {
      const prompt = yield* SystemPrompt.Service
      const first = yield* prompt.skills(build)
      const second = yield* prompt.skills(build)
      const output = first ?? (yield* Effect.fail(new NamedError.Unknown({ message: "missing skills output" })))

      expect(first).toBe(second)

      const alpha = output.indexOf("<name>alpha-skill</name>")
      const middle = output.indexOf("<name>middle-skill</name>")
      const zeta = output.indexOf("<name>zeta-skill</name>")

      expect(alpha).toBeGreaterThan(-1)
      expect(middle).toBeGreaterThan(alpha)
      expect(zeta).toBeGreaterThan(middle)
      expect(output).not.toContain("manual-skill")
    }),
  )

  it.effect("MCP output includes connected server instructions", () =>
    Effect.gen(function* () {
      const prompt = yield* SystemPrompt.Service
      const output = yield* prompt.mcp(build)

      expect(output).toBe(
        [
          "<mcp_instructions>",
          '  <server name="guide-server">',
          "    Use lookup before mutate.",
          "  </server>",
          '  <server name="tool-server">',
          "    Prefer search before update.",
          "  </server>",
          "</mcp_instructions>",
        ].join("\n"),
      )
    }),
  )

  it.effect("MCP output omits servers when all advertised tools are denied", () =>
    Effect.gen(function* () {
      const prompt = yield* SystemPrompt.Service
      const output = yield* prompt.mcp(build, Permission.fromConfig({ "tool-server_*": "deny" }))

      expect(output).toBe(
        [
          "<mcp_instructions>",
          '  <server name="guide-server">',
          "    Use lookup before mutate.",
          "  </server>",
          "</mcp_instructions>",
        ].join("\n"),
      )
    }),
  )
})

describe("SystemPrompt.mcpInstructions truncation (R11-002)", () => {
  const fixture = { instructions: "" }

  const mcpLayer = Layer.mock(MCP.Service, {
    instructions: () => Effect.succeed([{ name: "guide-server", instructions: fixture.instructions, tools: [] }]),
  })

  const skillLayer = Layer.mock(Skill.Service, {
    get: (name) => Effect.succeed(skills.find((skill) => skill.name === name)),
    require: (name) => {
      const info = skills.find((skill) => skill.name === name)
      if (info) return Effect.succeed(info)
      return Effect.fail(new Skill.NotFoundError({ name, available: skills.map((skill) => skill.name) }))
    },
    all: () => Effect.succeed(skills),
    dirs: () => Effect.succeed([]),
    available: () => Effect.succeed(skills),
  })

  const compile = (flags: Partial<RuntimeFlags.Info>) =>
    LayerNode.compile(SystemPrompt.node, [
      [MCP.node, mcpLayer],
      [Skill.node, skillLayer],
      [RuntimeFlags.node, RuntimeFlags.layer(flags)],
    ])

  const itSlimming = testEffect(compile({}))
  const itUpstream = testEffect(compile({ disableStaticSlimming: true }))

  // Expected values build on upstream's server-section rendering (split +
  // 4-space indent), which this ticket leaves untouched.
  const section = (instructions: string) =>
    [`  <server name="guide-server">`, ...instructions.split("\n").map((line) => `    ${line}`), "  </server>"].join(
      "\n",
    )

  const render = (instructions: string) =>
    ["<mcp_instructions>", section(instructions), "</mcp_instructions>"].join("\n")

  const assertBothRenderers = (instructions: string) =>
    Effect.gen(function* () {
      const prompt = yield* SystemPrompt.Service
      const output = yield* prompt.mcp(build)
      const blocks = yield* prompt.mcpBlocks(build)

      expect(output).toBe(render(instructions))
      expect(blocks).toEqual([{ key: "mcp:guide-server", content: section(instructions) }])
    })

  itSlimming.effect("instructions exactly 250 characters render unchanged", () =>
    Effect.gen(function* () {
      fixture.instructions = "x".repeat(250)
      yield* assertBothRenderers(fixture.instructions)
    }),
  )

  itSlimming.effect("instructions over 250 characters truncate at the last word boundary within the budget", () =>
    Effect.gen(function* () {
      fixture.instructions = "a".repeat(240) + " " + "b".repeat(10)
      yield* assertBothRenderers("a".repeat(240) + "...")
    }),
  )

  itSlimming.effect("word boundaries at the budget edge keep the result within 250 units", () =>
    Effect.gen(function* () {
      // space at the head's last index (247): the word cut still fits the ellipsis
      fixture.instructions = "a".repeat(247) + " bcd"
      yield* assertBothRenderers("a".repeat(247) + "...")
      // space past the head: falls back to the hard cut, still within budget
      fixture.instructions = "a".repeat(249) + " b"
      yield* assertBothRenderers("a".repeat(247) + "...")
    }),
  )

  itSlimming.effect("instructions over 250 characters without a space stay budget-safe", () =>
    Effect.gen(function* () {
      fixture.instructions = "a".repeat(251)
      yield* assertBothRenderers("a".repeat(247) + "...")
    }),
  )

  itSlimming.effect("multi-line instructions truncate before indentation", () =>
    Effect.gen(function* () {
      fixture.instructions = "Intro\n" + "a".repeat(300)
      yield* assertBothRenderers("Intro\n" + "a".repeat(241) + "...")
    }),
  )

  itSlimming.effect("empty instructions render unchanged", () =>
    Effect.gen(function* () {
      fixture.instructions = ""
      yield* assertBothRenderers(fixture.instructions)
    }),
  )

  itUpstream.effect("disableStaticSlimming flag set renders upstream-identical full instructions", () =>
    Effect.gen(function* () {
      fixture.instructions = "a".repeat(300)
      yield* assertBothRenderers(fixture.instructions)
    }),
  )
})
