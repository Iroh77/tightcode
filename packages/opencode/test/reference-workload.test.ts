import { describe, expect, test } from "bun:test"
import { Database } from "bun:sqlite"
import fs from "fs/promises"
import path from "path"
import { tmpdir } from "./fixture/fixture"
import { diff, loadRun, report } from "../script/measure-usage"
import { digestWorkload, workloadDigest } from "../script/reference-workload"
import { prompts } from "../script/reference-workload/prompts"

// Seam tests for the reference-workload driver (R10-004): the driver's output
// contract is the run dir consumed by measure-usage.loadRun (design
// context-observability.md, decision context-observability-02).
// Mechanics run against the in-repo echo stub — no provider, no network.

const DRIVER = path.join(import.meta.dir, "../script/reference-workload.ts")

const spawnDriver = async (args: string[], env?: Record<string, string>) => {
  const proc = Bun.spawn([process.execPath, DRIVER, ...args], {
    stdout: "pipe",
    stderr: "pipe",
    ...(env ? { env: { ...process.env, ...env } } : {}),
  })
  const [stdout, stderr, code] = await Promise.all([new Response(proc.stdout).text(), new Response(proc.stderr).text(), proc.exited])
  return { stdout, stderr, code }
}

const runArgs = (runDir: string, extra: string[] = []) => ["--bin", "stub", "--run-dir", runDir, "--model", "stub/test-model", "--stub", ...extra]

const readDb = async (dbPath: string) => {
  const db = new Database(dbPath, { readonly: true })
  try {
    const sessions = db.query<{ id: string }, []>("SELECT id FROM session ORDER BY rowid").all().map((row) => row.id)
    const parts = db
      .query<{ data: string }, []>("SELECT data FROM part ORDER BY rowid")
      .all()
      .map((row) => JSON.parse(row.data) as { type: string; tokens: { input: number; output: number }; cost: number })
    return { sessions, parts }
  } finally {
    db.close()
  }
}

describe("reference-workload prompts", () => {
  test("workload digest covers the prompt list and is content-dependent", async () => {
    const digest = await digestWorkload()
    expect(digest).toMatch(/^[0-9a-f]{64}$/)
    expect(workloadDigest({ prompts: [...prompts], files: {} })).toBe(workloadDigest({ prompts: [...prompts], files: {} }))
    expect(workloadDigest({ prompts: [...prompts, "extra"], files: {} })).not.toBe(workloadDigest({ prompts, files: {} }))
    expect(workloadDigest({ prompts, files: { "README.md": "a" } })).not.toBe(workloadDigest({ prompts, files: { "README.md": "b" } }))
  })
})

describe("reference-workload driver", () => {
  test("--stub produces a complete run dir that measure-usage loads", async () => {
    await using tmp = await tmpdir()
    const runDir = path.join(tmp.path, "run")
    const { code, stderr } = await spawnDriver(runArgs(runDir))
    expect(stderr).toBe("")
    expect(code).toBe(0)

    const manifest = JSON.parse(await Bun.file(path.join(runDir, "manifest.json")).text())
    expect(manifest.bin).toBe("stub")
    expect(manifest.providerID).toBe("stub")
    expect(manifest.modelID).toBe("test-model")
    expect(manifest.promptsDigest).toBe(await digestWorkload())
    expect(manifest.capture).toBe(true)
    expect(manifest.proxyMode).toBe(false)
    expect(manifest.dbPath).toBe("data/opencode.db")
    expect(manifest.capturesDir).toBe("data/opencode/prompt-captures")
    expect(typeof manifest.startedAt).toBe("string")
    expect(typeof manifest.endedAt).toBe("string")

    const db = await readDb(path.join(runDir, "data/opencode.db"))
    expect(db.sessions).toEqual(["stub-session"])
    expect(db.parts).toHaveLength(prompts.length)
    for (const part of db.parts) {
      expect(part.type).toBe("step-finish")
      expect(part.tokens.input).toBeGreaterThan(0)
      expect(part.cost).toBeGreaterThan(0)
    }

    const capturesDir = path.join(runDir, "data/opencode/prompt-captures/stub-session")
    const captureFiles = (await fs.readdir(capturesDir)).sort()
    expect(captureFiles).toHaveLength(prompts.length)
    const first = JSON.parse(await Bun.file(path.join(capturesDir, captureFiles[0])).text())
    expect(first.meta.sessionID).toBe("stub-session")
    expect(first.payload.messages[0].content).toBe(prompts[0])

    const cwd = path.join(runDir, "cwd")
    await fs.access(path.join(cwd, "README.md"))
    await fs.access(path.join(cwd, "src/math.ts"))
    const config = JSON.parse(await Bun.file(path.join(cwd, "opencode.json")).text())
    expect(config.model).toBe("stub/test-model")
    expect(config.agent.build.temperature).toBe(0)

    const auth = JSON.parse(await Bun.file(path.join(runDir, "data/opencode/auth.json")).text())
    expect(Object.keys(auth)).toEqual(["stub"])

    const run = await loadRun(runDir)
    const usage = report(run)
    expect(usage.sessions).toHaveLength(1)
    expect(usage.sessions[0].turns).toHaveLength(prompts.length)
    for (const turn of usage.sessions[0].turns) {
      expect(turn.usage).not.toBeNull()
      expect(turn.reconciled).not.toBeNull()
    }
  })

  test("two stub runs align by index and diff to zero deltas; --proxy sets kill-switches", async () => {
    await using tmp = await tmpdir()
    const forkDir = path.join(tmp.path, "fork")
    const proxyDir = path.join(tmp.path, "proxy")
    expect((await spawnDriver(runArgs(forkDir))).code).toBe(0)
    expect((await spawnDriver(runArgs(proxyDir, ["--proxy"]))).code).toBe(0)

    const stubEnv = async (runDir: string) =>
      JSON.parse(await Bun.file(path.join(runDir, "stub-env.json")).text()) as Record<string, string | undefined>
    expect(await stubEnv(forkDir)).toEqual({
      OPENCODE_ENABLE_PROMPT_CAPTURE: "1",
      OPENCODE_DISABLE_LAZY_TOOLS: undefined,
      OPENCODE_DISABLE_STATIC_SLIMMING: undefined,
    })
    expect(await stubEnv(proxyDir)).toEqual({
      OPENCODE_ENABLE_PROMPT_CAPTURE: "1",
      OPENCODE_DISABLE_LAZY_TOOLS: "1",
      OPENCODE_DISABLE_STATIC_SLIMMING: "1",
    })

    const proxyManifest = JSON.parse(await Bun.file(path.join(proxyDir, "manifest.json")).text())
    expect(proxyManifest.proxyMode).toBe(true)

    const forkReport = report(await loadRun(forkDir))
    const proxyReport = report(await loadRun(proxyDir))
    const d = diff(forkReport, proxyReport)
    expect(d.warnings).toEqual([])
    expect(d.sessions).toHaveLength(1)
    expect(d.sessions[0].turns).toHaveLength(prompts.length)
    for (const turn of d.sessions[0].turns) {
      expect(turn.delta).toEqual({ input: 0, output: 0, cacheRead: 0, cacheWrite: 0, cost: 0 })
    }
    expect(d.totals).toEqual({ input: 0, output: 0, cost: 0 })
  })

  test("refuses a non-empty run dir", async () => {
    await using tmp = await tmpdir()
    const runDir = path.join(tmp.path, "run")
    expect((await spawnDriver(runArgs(runDir))).code).toBe(0)
    const second = await spawnDriver(runArgs(runDir))
    expect(second.code).toBe(1)
    expect(second.stderr).toContain("not empty")
  })

  test("fails loudly when the provider has no auth in real mode", async () => {
    await using tmp = await tmpdir()
    await using home = await tmpdir()
    const runDir = path.join(tmp.path, "run")
    const { code, stderr } = await spawnDriver(["--bin", "definitely-not-a-real-bin", "--run-dir", runDir, "--model", "test-provider/test-model"], {
      XDG_DATA_HOME: home.path,
    })
    expect(code).toBe(1)
    expect(stderr).toContain("test-provider")
  })
})