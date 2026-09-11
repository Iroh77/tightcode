import { describe, expect, test } from "bun:test"
import { SessionV1 } from "@opencode-ai/core/v1/session"
import path from "path"
import { Effect, FileSystem, Layer } from "effect"
import { CrossSpawnSpawner } from "@opencode-ai/core/cross-spawn-spawner"

import { Instruction } from "../../src/session/instruction"
import type { MessageV2 } from "../../src/session/message-v2"
import { MessageID, PartID, SessionID } from "../../src/session/schema"
import { Global } from "@opencode-ai/core/global"
import { RuntimeFlags } from "../../src/effect/runtime-flags"
import { provideInstance, provideTmpdirInstance, testInstanceStoreLayer, tmpdirScoped } from "../fixture/fixture"
import { testEffect } from "../lib/effect"
import { TestConfig } from "../fixture/config"
import { ProviderV2 } from "@opencode-ai/core/provider"
import { ModelV2 } from "@opencode-ai/core/model"
import { AppNodeBuilder } from "@opencode-ai/core/effect/app-node-builder"
import { LayerNode } from "@opencode-ai/core/effect/layer-node"
import { LayerNodePlatform } from "@opencode-ai/core/effect/app-node-platform"
import { InstanceStore } from "@/project/instance-store"
import { InstanceBootstrap } from "@/project/bootstrap"
import { Config } from "@/config/config"

const it = testEffect(
  AppNodeBuilder.build(LayerNode.group([CrossSpawnSpawner.node, LayerNodePlatform.filesystem, InstanceStore.node]), [
    [
      InstanceBootstrap.node,
      Layer.succeed(InstanceBootstrap.Service, InstanceBootstrap.Service.of({ run: Effect.void })),
    ],
  ]),
)

const configLayer = Layer.succeed(Config.Service, TestConfig.make())

const instructionLayer = (global: Partial<Global.Interface>, flags: Partial<RuntimeFlags.Info> = {}) =>
  AppNodeBuilder.build(Instruction.node, [
    [Config.node, configLayer],
    [Global.node, Global.layerWith(global)],
    [RuntimeFlags.node, RuntimeFlags.layer(flags)],
  ])

const provideInstruction =
  (global: Partial<Global.Interface>, flags?: Partial<RuntimeFlags.Info>) =>
  <A, E, R>(self: Effect.Effect<A, E, R>) =>
    self.pipe(Effect.provide(instructionLayer(global, flags)))

const write = (filepath: string, content: string) =>
  Effect.gen(function* () {
    const fs = yield* FileSystem.FileSystem
    yield* fs.makeDirectory(path.dirname(filepath), { recursive: true })
    yield* fs.writeFileString(filepath, content)
  })

const writeFiles = (dir: string, files: Record<string, string>) =>
  Effect.all(
    Object.entries(files).map(([file, content]) => write(path.join(dir, file), content)),
    { discard: true },
  )

const withFiles = <A, E, R>(files: Record<string, string>, self: (dir: string) => Effect.Effect<A, E, R>) =>
  provideTmpdirInstance((dir) =>
    Effect.gen(function* () {
      yield* writeFiles(dir, files)
      return yield* self(dir).pipe(provideInstruction({ home: dir, config: dir }))
    }),
  )

const tmpWithFiles = (files: Record<string, string>) =>
  Effect.gen(function* () {
    const dir = yield* tmpdirScoped()
    yield* writeFiles(dir, files)
    return dir
  })

// Git-rooted worktree with the session cwd at a subdirectory, so instruction
// discovery has ancestors strictly between cwd and worktree. Global files live
// in a separate empty dir unless the test opts in via `global`.
const withWorktree = <A, E, R>(
  files: Record<string, string>,
  self: (worktree: string, globalDir: string) => Effect.Effect<A, E, R>,
  options: { global?: Record<string, string>; cwd?: "sub-deep" | "worktree"; flags?: Partial<RuntimeFlags.Info> } = {},
) =>
  Effect.gen(function* () {
    const globalDir = yield* tmpdirScoped()
    yield* writeFiles(globalDir, options.global ?? {})
    const worktree = yield* tmpdirScoped({ git: true })
    yield* writeFiles(worktree, files)
    const directory = options.cwd === "worktree" ? worktree : path.join(worktree, "sub", "deep")
    return yield* self(worktree, globalDir).pipe(
      provideInstance(directory),
      provideInstruction({ home: globalDir, config: globalDir }, options.flags),
    )
  }).pipe(Effect.provide(testInstanceStoreLayer))

function loaded(filepath: string): SessionV1.WithParts[] {
  const sessionID = SessionID.make("session-loaded-1")
  const messageID = MessageID.make("msg_message-loaded-1")

  return [
    {
      info: {
        id: messageID,
        sessionID,
        role: "user",
        time: { created: 0 },
        agent: "build",
        model: {
          providerID: ProviderV2.ID.make("anthropic"),
          modelID: ModelV2.ID.make("claude-sonnet-4-20250514"),
        },
      },
      parts: [
        {
          id: PartID.make("prt_part-loaded-1"),
          messageID,
          sessionID,
          type: "tool",
          callID: "call-loaded-1",
          tool: "read",
          state: {
            status: "completed",
            input: {},
            output: "done",
            title: "Read",
            metadata: { loaded: [filepath] },
            time: { start: 0, end: 1 },
          },
        },
      ],
    },
  ]
}

describe("Instruction.resolve", () => {
  it.live("returns empty when AGENTS.md is at project root (already in systemPaths)", () =>
    withFiles({ "AGENTS.md": "# Root Instructions", "src/file.ts": "const x = 1" }, (dir) =>
      Effect.gen(function* () {
        const svc = yield* Instruction.Service
        const system = yield* svc.systemPaths()
        expect(system.has(path.join(dir, "AGENTS.md"))).toBe(true)

        const results = yield* svc.resolve([], path.join(dir, "src", "file.ts"), MessageID.make("msg_message-test-1"))
        expect(results).toEqual([])
      }),
    ),
  )

  it.live("returns AGENTS.md from subdirectory (not in systemPaths)", () =>
    withFiles({ "subdir/AGENTS.md": "# Subdir Instructions", "subdir/nested/file.ts": "const x = 1" }, (dir) =>
      Effect.gen(function* () {
        const svc = yield* Instruction.Service
        const system = yield* svc.systemPaths()
        expect(system.has(path.join(dir, "subdir", "AGENTS.md"))).toBe(false)

        const results = yield* svc.resolve(
          [],
          path.join(dir, "subdir", "nested", "file.ts"),
          MessageID.make("msg_message-test-2"),
        )
        expect(results.length).toBe(1)
        expect(results[0].filepath).toBe(path.join(dir, "subdir", "AGENTS.md"))
      }),
    ),
  )

  it.live("doesn't reload AGENTS.md when reading it directly", () =>
    withFiles({ "subdir/AGENTS.md": "# Subdir Instructions", "subdir/nested/file.ts": "const x = 1" }, (dir) =>
      Effect.gen(function* () {
        const svc = yield* Instruction.Service
        const filepath = path.join(dir, "subdir", "AGENTS.md")
        const system = yield* svc.systemPaths()
        expect(system.has(filepath)).toBe(false)

        const results = yield* svc.resolve([], filepath, MessageID.make("msg_message-test-3"))
        expect(results).toEqual([])
      }),
    ),
  )

  it.live("does not reattach the same nearby instructions twice for one message", () =>
    withFiles({ "subdir/AGENTS.md": "# Subdir Instructions", "subdir/nested/file.ts": "const x = 1" }, (dir) =>
      Effect.gen(function* () {
        const svc = yield* Instruction.Service
        const filepath = path.join(dir, "subdir", "nested", "file.ts")
        const id = MessageID.make("msg_message-claim-1")

        const first = yield* svc.resolve([], filepath, id)
        const second = yield* svc.resolve([], filepath, id)

        expect(first).toHaveLength(1)
        expect(first[0].filepath).toBe(path.join(dir, "subdir", "AGENTS.md"))
        expect(second).toEqual([])
      }),
    ),
  )

  it.live("clear allows nearby instructions to be attached again for the same message", () =>
    withFiles({ "subdir/AGENTS.md": "# Subdir Instructions", "subdir/nested/file.ts": "const x = 1" }, (dir) =>
      Effect.gen(function* () {
        const svc = yield* Instruction.Service
        const filepath = path.join(dir, "subdir", "nested", "file.ts")
        const id = MessageID.make("msg_message-claim-2")

        const first = yield* svc.resolve([], filepath, id)
        yield* svc.clear(id)
        const second = yield* svc.resolve([], filepath, id)

        expect(first).toHaveLength(1)
        expect(second).toHaveLength(1)
        expect(second[0].filepath).toBe(path.join(dir, "subdir", "AGENTS.md"))
      }),
    ),
  )

  it.live("skips instructions already reported by prior read metadata", () =>
    withFiles({ "subdir/AGENTS.md": "# Subdir Instructions", "subdir/nested/file.ts": "const x = 1" }, (dir) =>
      Effect.gen(function* () {
        const svc = yield* Instruction.Service
        const agents = path.join(dir, "subdir", "AGENTS.md")
        const filepath = path.join(dir, "subdir", "nested", "file.ts")
        const id = MessageID.make("msg_message-claim-3")

        const results = yield* svc.resolve(loaded(agents), filepath, id)
        expect(results).toEqual([])
      }),
    ),
  )

  test.todo("fetches remote instructions from config URLs via HttpClient", () => {})
})

describe("Instruction.system", () => {
  it.live("loads both project and global AGENTS.md when both exist", () =>
    Effect.gen(function* () {
      const globalTmp = yield* tmpWithFiles({ "AGENTS.md": "# Global Instructions" })
      const projectTmp = yield* tmpWithFiles({ "AGENTS.md": "# Project Instructions" })

      yield* Effect.gen(function* () {
        const svc = yield* Instruction.Service
        const paths = yield* svc.systemPaths()
        expect(paths.has(path.join(projectTmp, "AGENTS.md"))).toBe(true)
        expect(paths.has(path.join(globalTmp, "AGENTS.md"))).toBe(true)

        const rules = yield* svc.system()
        expect(rules).toHaveLength(2)
        expect(rules[0]).toBe(`Instructions from: ${path.join(globalTmp, "AGENTS.md")}\n# Global Instructions`)
        expect(rules[1]).toBe(`Instructions from: ${path.join(projectTmp, "AGENTS.md")}\n# Project Instructions`)
      }).pipe(provideInstance(projectTmp), provideInstruction({ home: globalTmp, config: globalTmp }))
    }),
  )

  it.live("skips project and global CLAUDE.md when Claude Code prompt is disabled", () =>
    Effect.gen(function* () {
      const globalTmp = yield* tmpWithFiles({ ".claude/CLAUDE.md": "# Global Claude" })
      const projectTmp = yield* tmpWithFiles({ "CLAUDE.md": "# Project Claude" })

      yield* Effect.gen(function* () {
        const svc = yield* Instruction.Service
        const paths = yield* svc.systemPaths()
        expect(paths.has(path.join(globalTmp, ".claude", "CLAUDE.md"))).toBe(false)
        expect(paths.has(path.join(projectTmp, "CLAUDE.md"))).toBe(false)
        expect(yield* svc.system()).toEqual([])
      }).pipe(
        provideInstance(projectTmp),
        provideInstruction({ home: globalTmp, config: globalTmp }, { disableClaudeCodePrompt: true }),
      )
    }),
  )
})

describe("Instruction.systemPaths global config", () => {
  it.live("uses Global.Service config AGENTS.md", () =>
    Effect.gen(function* () {
      const globalTmp = yield* tmpWithFiles({ "AGENTS.md": "# Global Instructions" })
      const projectTmp = yield* tmpdirScoped()

      yield* Effect.gen(function* () {
        const svc = yield* Instruction.Service
        const paths = yield* svc.systemPaths()
        expect(paths.has(path.join(globalTmp, "AGENTS.md"))).toBe(true)
      }).pipe(provideInstance(projectTmp), provideInstruction({ home: globalTmp, config: globalTmp }))
    }),
  )
})

describe("Instruction.systemPaths root-wins (R11-001)", () => {
  it.live("root AGENTS.md present: keeps cwd-level and worktree-root, drops intermediate ancestors", () =>
    withWorktree({ "AGENTS.md": "# Root", "sub/AGENTS.md": "# Mid", "sub/deep/AGENTS.md": "# Deep" }, (worktree) =>
      Effect.gen(function* () {
        const svc = yield* Instruction.Service
        const paths = yield* svc.systemPaths()
        expect(paths.size).toBe(2)
        expect(paths.has(path.join(worktree, "sub", "deep", "AGENTS.md"))).toBe(true)
        expect(paths.has(path.join(worktree, "AGENTS.md"))).toBe(true)
        expect(paths.has(path.join(worktree, "sub", "AGENTS.md"))).toBe(false)
      }),
    ),
  )

  it.live("root AGENTS.md present without a cwd-level match: worktree-root kept alone", () =>
    withWorktree({ "AGENTS.md": "# Root", "sub/AGENTS.md": "# Mid" }, (worktree) =>
      Effect.gen(function* () {
        const svc = yield* Instruction.Service
        const paths = yield* svc.systemPaths()
        expect(paths.size).toBe(1)
        expect(paths.has(path.join(worktree, "AGENTS.md"))).toBe(true)
        expect(paths.has(path.join(worktree, "sub", "AGENTS.md"))).toBe(false)
      }),
    ),
  )

  it.live("root AGENTS.md absent: upstream stacking unchanged", () =>
    withWorktree({ "sub/AGENTS.md": "# Mid", "sub/deep/AGENTS.md": "# Deep" }, (worktree) =>
      Effect.gen(function* () {
        const svc = yield* Instruction.Service
        const paths = yield* svc.systemPaths()
        expect(paths.size).toBe(2)
        expect(paths.has(path.join(worktree, "sub", "deep", "AGENTS.md"))).toBe(true)
        expect(paths.has(path.join(worktree, "sub", "AGENTS.md"))).toBe(true)
      }),
    ),
  )

  it.live("CLAUDE.md-only walk: root-wins never applies", () =>
    withWorktree({ "CLAUDE.md": "# Root", "sub/CLAUDE.md": "# Mid", "sub/deep/CLAUDE.md": "# Deep" }, (worktree) =>
      Effect.gen(function* () {
        const svc = yield* Instruction.Service
        const paths = yield* svc.systemPaths()
        expect(paths.size).toBe(3)
        expect(paths.has(path.join(worktree, "sub", "deep", "CLAUDE.md"))).toBe(true)
        expect(paths.has(path.join(worktree, "sub", "CLAUDE.md"))).toBe(true)
        expect(paths.has(path.join(worktree, "CLAUDE.md"))).toBe(true)
      }),
    ),
  )

  it.live("global instructions file unaffected by root-wins", () =>
    withWorktree(
      { "AGENTS.md": "# Root", "sub/AGENTS.md": "# Mid", "sub/deep/AGENTS.md": "# Deep" },
      (worktree, globalDir) =>
        Effect.gen(function* () {
          const svc = yield* Instruction.Service
          const paths = yield* svc.systemPaths()
          expect(paths.size).toBe(3)
          expect(paths.has(path.join(globalDir, "AGENTS.md"))).toBe(true)
          expect(paths.has(path.join(worktree, "sub", "deep", "AGENTS.md"))).toBe(true)
          expect(paths.has(path.join(worktree, "AGENTS.md"))).toBe(true)
          expect(paths.has(path.join(worktree, "sub", "AGENTS.md"))).toBe(false)
        }),
      { global: { "AGENTS.md": "# Global" } },
    ),
  )

  it.live("cwd equals worktree: single match, no behavior change", () =>
    withWorktree(
      { "AGENTS.md": "# Root" },
      (worktree) =>
        Effect.gen(function* () {
          const svc = yield* Instruction.Service
          const paths = yield* svc.systemPaths()
          expect(paths.size).toBe(1)
          expect(paths.has(path.join(worktree, "AGENTS.md"))).toBe(true)
        }),
      { cwd: "worktree" },
    ),
  )

  it.live("disableStaticSlimming flag set: upstream stacking passthrough", () =>
    withWorktree(
      { "AGENTS.md": "# Root", "sub/AGENTS.md": "# Mid", "sub/deep/AGENTS.md": "# Deep" },
      (worktree) =>
        Effect.gen(function* () {
          const svc = yield* Instruction.Service
          const paths = yield* svc.systemPaths()
          expect(paths.size).toBe(3)
          expect(paths.has(path.join(worktree, "sub", "deep", "AGENTS.md"))).toBe(true)
          expect(paths.has(path.join(worktree, "sub", "AGENTS.md"))).toBe(true)
          expect(paths.has(path.join(worktree, "AGENTS.md"))).toBe(true)
        }),
      { flags: { disableStaticSlimming: true } },
    ),
  )
})
