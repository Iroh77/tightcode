import { describe, expect, test } from "bun:test"
import { Database } from "bun:sqlite"
import fs from "fs/promises"
import os from "os"
import path from "path"
import { tmpdir } from "./fixture/fixture"
import { diff, loadRun, report } from "../script/measure-usage"
import { digestWorkload, scheduleRuns, validateSpec, workloadDigest } from "../script/reference-workload"
import type { CampaignSpec } from "../script/reference-workload"
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
      OPENCODE_PIN_BINDING_VERDICT: undefined,
    })
    expect(await stubEnv(proxyDir)).toEqual({
      OPENCODE_ENABLE_PROMPT_CAPTURE: "1",
      OPENCODE_DISABLE_LAZY_TOOLS: "1",
      OPENCODE_DISABLE_STATIC_SLIMMING: "1",
      OPENCODE_PIN_BINDING_VERDICT: undefined,
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

  // The stub-mirror seam (ticket 27): stub-bin.ts directly, with the driver's
  // minimal child env — the driver's pin wiring arrives with ticket 28's
  // --pin flag (childEnv stays a fresh allowlist until then).
  const spawnStub = async (runDir: string, env?: Record<string, string>) => {
    await fs.mkdir(path.join(runDir, "data"), { recursive: true })
    const proc = Bun.spawn(
      [process.execPath, path.join(import.meta.dir, "../script/reference-workload/stub-bin.ts"), "hello world"],
      {
        stdout: "pipe",
        stderr: "pipe",
        env: {
          PATH: process.env.PATH ?? "",
          HOME: process.env.HOME ?? os.homedir(),
          XDG_DATA_HOME: path.join(runDir, "data"),
          OPENCODE_DB: path.join(runDir, "data", "opencode.db"),
          ...(env ?? {}),
        },
      },
    )
    const [stdout, stderr, code] = await Promise.all([
      new Response(proc.stdout).text(),
      new Response(proc.stderr).text(),
      proc.exited,
    ])
    if (code !== 0) throw new Error(`stub-bin exited ${code}: ${stderr || stdout}`)
    return code
  }

  test("captures self-identify: verdict from the pin env, binary stub (R13-003/005, ticket 27)", async () => {
    await using tmp = await tmpdir()
    const binding = path.join(tmp.path, "binding")
    const advisory = path.join(tmp.path, "advisory")
    const unpinned = path.join(tmp.path, "unpinned")
    expect(await spawnStub(binding, { OPENCODE_ENABLE_PROMPT_CAPTURE: "1", OPENCODE_PIN_BINDING_VERDICT: "binding" })).toBe(0)
    expect(await spawnStub(advisory, { OPENCODE_ENABLE_PROMPT_CAPTURE: "1", OPENCODE_PIN_BINDING_VERDICT: "advisory" })).toBe(0)
    expect(await spawnStub(unpinned, { OPENCODE_ENABLE_PROMPT_CAPTURE: "1" })).toBe(0)

    const firstCapture = async (runDir: string) => {
      const capturesDir = path.join(runDir, "data", "opencode", "prompt-captures", "stub-session")
      const files = (await fs.readdir(capturesDir)).sort()
      return JSON.parse(await Bun.file(path.join(capturesDir, files[0])).text()) as { meta: Record<string, unknown> }
    }
    const stubEnv = async (runDir: string) =>
      JSON.parse(await Bun.file(path.join(runDir, "stub-env.json")).text()) as Record<string, string | undefined>

    const bindingCapture = await firstCapture(binding)
    expect(bindingCapture.meta.verdict).toBe("binding")
    expect(bindingCapture.meta.binary).toBe("stub")
    expect((await stubEnv(binding)).OPENCODE_PIN_BINDING_VERDICT).toBe("binding")

    const advisoryCapture = await firstCapture(advisory)
    expect(advisoryCapture.meta.verdict).toBe("advisory")
    expect(advisoryCapture.meta.binary).toBe("stub")

    const unpinnedCapture = await firstCapture(unpinned)
    expect("verdict" in unpinnedCapture.meta).toBe(false)
    expect(unpinnedCapture.meta.binary).toBe("stub")
    expect((await stubEnv(unpinned)).OPENCODE_PIN_BINDING_VERDICT).toBeUndefined()
  })

  test("captures without verdict/binary still parse and report (tolerant read, SC-4)", async () => {
    await using tmp = await tmpdir()
    const runDir = path.join(tmp.path, "run")
    expect((await spawnDriver(runArgs(runDir))).code).toBe(0)
    const capturesDir = path.join(runDir, "data/opencode/prompt-captures/stub-session")
    const files = (await fs.readdir(capturesDir)).sort()
    const first = path.join(capturesDir, files[0])
    const capture = JSON.parse(await Bun.file(first).text()) as { meta: Record<string, unknown> }
    delete capture.meta.verdict
    delete capture.meta.binary
    await Bun.write(first, JSON.stringify(capture, null, 2))
    const usage = report(await loadRun(runDir))
    expect(usage.sessions).toHaveLength(1)
    expect(usage.sessions[0].turns).toHaveLength(prompts.length)
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

  // Harness v2 seams (R13-003/005, ticket 28): --pin wiring, post-run verdict
  // derivation, binary identity resolution.
  test("--pin propagates to the child env and lands in the manifest with derived verdicts", async () => {
    await using tmp = await tmpdir()
    const runDir = path.join(tmp.path, "run")
    const { code, stderr } = await spawnDriver(runArgs(runDir, ["--pin", "binding"]))
    expect(stderr).toBe("")
    expect(code).toBe(0)

    const stubEnv = JSON.parse(await Bun.file(path.join(runDir, "stub-env.json")).text()) as Record<string, string | undefined>
    expect(stubEnv.OPENCODE_PIN_BINDING_VERDICT).toBe("binding")

    const manifest = JSON.parse(await Bun.file(path.join(runDir, "manifest.json")).text())
    expect(manifest.pin).toBe("binding")
    expect(manifest.verdicts).toEqual(["binding"])
    expect(manifest.binary).toBe("stub")
  })

  test("unpinned run: pin null, verdicts [] (stub emits no meta.verdict), binary stub", async () => {
    await using tmp = await tmpdir()
    const runDir = path.join(tmp.path, "run")
    expect((await spawnDriver(runArgs(runDir))).code).toBe(0)
    const manifest = JSON.parse(await Bun.file(path.join(runDir, "manifest.json")).text())
    expect(manifest.pin).toBeNull()
    expect(manifest.verdicts).toEqual([])
    expect(manifest.binary).toBe("stub")
  })

  test("invalid --pin value fails loudly", async () => {
    await using tmp = await tmpdir()
    const { code, stderr } = await spawnDriver(runArgs(path.join(tmp.path, "run"), ["--pin", "bogus"]))
    expect(code).toBe(1)
    expect(stderr).toContain("--pin")
  })

  // Real-mode binary identity: a fake bin whose workload invocations succeed
  // (auth seeded in the driver home) but whose --version fails or prints.
  test("binary identity: first non-empty --version line; failure → null + warning, manifest still written", async () => {
    await using fake = await tmpdir()
    const authDir = path.join(fake.path, "opencode")
    await fs.mkdir(authDir, { recursive: true })
    await Bun.write(
      path.join(authDir, "auth.json"),
      JSON.stringify({ "test-provider": { type: "api", key: "test-key" } }),
    )
    await using tmp = await tmpdir()

    const writeFakeBin = async (versionBehavior: "fail" | "print") => {
      const file = path.join(tmp.path, versionBehavior === "fail" ? "fail-version.ts" : "print-version.ts")
      await Bun.write(
        file,
        versionBehavior === "fail"
          ? "if (Bun.argv.includes('--version')) process.exit(1)\nprocess.exit(0)\n"
          : "if (Bun.argv.includes('--version')) { process.stdout.write('\\n\\n9.9.9\\n'); process.exit(0) }\nprocess.exit(0)\n",
      )
      return `bun ${file}`
    }

    const printBin = await writeFakeBin("print")
    const printDir = path.join(tmp.path, "print")
    const ok = await spawnDriver(["--bin", printBin, "--run-dir", printDir, "--model", "test-provider/test-model"], {
      XDG_DATA_HOME: fake.path,
    })
    expect(ok.code).toBe(0)
    const printManifest = JSON.parse(await Bun.file(path.join(printDir, "manifest.json")).text())
    expect(printManifest.binary).toBe("9.9.9")
    expect(printManifest.bin).toBe(printBin)

    const failBin = await writeFakeBin("fail")
    const failDir = path.join(tmp.path, "fail")
    const failed = await spawnDriver(["--bin", failBin, "--run-dir", failDir, "--model", "test-provider/test-model"], {
      XDG_DATA_HOME: fake.path,
    })
    expect(failed.code).toBe(0)
    expect(failed.stderr).toMatch(/--version/)
    const failManifest = JSON.parse(await Bun.file(path.join(failDir, "manifest.json")).text())
    expect(failManifest.binary).toBeNull()
    expect(failManifest.bin).toBe(failBin)
  })
})

// Campaign runner seams (R13-002/003, R00-010, ticket 29): spec validation and
// the seeded schedule are pure exports; execution runs against the echo stub —
// no provider, no network. The output contract is <out>/campaign.json plus v1
// run dirs (design comparison-testing.md, decision comparison-testing-01 §4).
describe("campaign runner", () => {
  const baseSpec: CampaignSpec = {
    phases: [
      { name: "small", model: "stub/test-model" },
      { name: "mcp", model: "stub2/other-model", mcp: { firecrawl: { type: "remote", url: "https://mcp.firecrawl.dev" } } },
    ],
    legs: [
      { shape: "fork", bin: "fork-bin" },
      { shape: "upstream", bin: "upstream-bin" },
      { shape: "fork", bin: "proxy-bin", proxy: true },
    ],
    runs: 2,
  }

  const writeSpec = async (dir: string, spec: unknown) => {
    const file = path.join(dir, "spec.json")
    await Bun.write(file, JSON.stringify(spec))
    return file
  }

  const campaignArgs = (specFile: string, out: string, extra: string[] = []) => ["--campaign", specFile, "--out", out, ...extra]

  const readCampaign = async (out: string) =>
    JSON.parse(await Bun.file(path.join(out, "campaign.json")).text()) as {
      spec: unknown
      seed: number
      startedAt: string
      endedAt: string | null
      runs: Array<{ index: number; phase: string; shape: string; proxy: boolean; rep: number; runDir: string; status: string; error?: string }>
    }

  test("spec validation fails loudly: duplicate names, zero phases/legs, runs: 1, bad shapes", () => {
    expect(() =>
      validateSpec({ ...baseSpec, phases: [{ name: "a", model: "x/y" }, { name: "a", model: "x/y" }] }),
    ).toThrow(/duplicate phase name/)
    expect(() => validateSpec({ ...baseSpec, phases: [] })).toThrow(/phases/)
    expect(() => validateSpec({ ...baseSpec, legs: [] })).toThrow(/legs/)
    expect(() => validateSpec({ ...baseSpec, runs: 1 })).toThrow(/never single runs/)
    expect(() => validateSpec({ ...baseSpec, runs: 0 })).toThrow(/never single runs/)
    expect(() => validateSpec({ ...baseSpec, legs: [{ shape: "nope", bin: "b" }] })).toThrow(/shape/)
    expect(() => validateSpec({ ...baseSpec, phases: [{ name: "a b", model: "x/y" }] })).toThrow(/name/)
    expect(() => validateSpec({ ...baseSpec, phases: [{ name: "a", model: "noprovider" }] })).toThrow(/model/)
  })

  test("spec validation echoes a normalized spec; runs defaults to 3", () => {
    expect(validateSpec({ phases: [{ name: "p", model: "a/b" }], legs: [{ shape: "fork", bin: "b" }] })).toEqual({
      phases: [{ name: "p", model: "a/b" }],
      legs: [{ shape: "fork", bin: "b" }],
      runs: 3,
    })
    expect(validateSpec({ ...baseSpec, pin: "binding" })).toEqual({ ...baseSpec, runs: 2, pin: "binding" })
  })

  test("schedule determinism: same spec + seed → identical order; both shapes N times per phase", () => {
    const spec = validateSpec(baseSpec)
    const a = scheduleRuns(spec, 1234, "/tmp/out")
    const b = scheduleRuns(spec, 1234, "/tmp/out")
    expect(a).toEqual(b)
    expect(a).toHaveLength(12)
    expect(a.map((run) => run.index)).toEqual([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11])
    let offset = 0
    for (const phase of spec.phases) {
      const block = a.slice(offset, offset + spec.runs * spec.legs.length)
      for (const run of block) {
        expect(run.phase).toBe(phase.name)
        expect(run.runDir).toBe(
          path.join("/tmp/out", `${String(run.index).padStart(2, "0")}-${phase.name}-${run.shape}${run.proxy ? "-proxy" : ""}`),
        )
      }
      for (const leg of spec.legs) {
        const proxy = leg.proxy ?? false
        expect(block.filter((run) => run.shape === leg.shape && run.proxy === proxy)).toHaveLength(spec.runs)
      }
      for (const run of block) expect(run.rep).toBeLessThan(spec.runs)
      offset += spec.runs * spec.legs.length
    }
  })

  test("stub campaign end-to-end: spec echo + schedule + all done, v1 layout, per-leg capture asymmetry, mcp + auth per phase", async () => {
    await using tmp = await tmpdir()
    const out = path.join(tmp.path, "campaign")
    const specFile = await writeSpec(tmp.path, { ...baseSpec, pin: "binding" })
    const { code, stderr } = await spawnDriver(campaignArgs(specFile, out, ["--stub"]))
    expect(stderr).toBe("")
    expect(code).toBe(0)

    const campaign = await readCampaign(out)
    expect(campaign.spec).toEqual({ ...baseSpec, pin: "binding" })
    expect(Number.isInteger(campaign.seed)).toBe(true)
    expect(campaign.seed).toBeGreaterThanOrEqual(0)
    expect(campaign.seed).toBeLessThanOrEqual(0xffffffff)
    expect(typeof campaign.startedAt).toBe("string")
    expect(typeof campaign.endedAt).toBe("string")
    expect(campaign.runs).toHaveLength(12)
    for (const run of campaign.runs) expect(run.status).toBe("done")

    // The recorded seed reproduces the recorded schedule.
    expect(scheduleRuns(validateSpec({ ...baseSpec, pin: "binding" }), campaign.seed, out).map((run) => run.runDir)).toEqual(
      campaign.runs.map((run) => run.runDir),
    )

    for (const run of campaign.runs) {
      const manifest = JSON.parse(await Bun.file(path.join(run.runDir, "manifest.json")).text())
      expect(typeof manifest.startedAt).toBe("string")
      expect(typeof manifest.endedAt).toBe("string")
      expect(manifest.pin).toBe("binding")
      expect(manifest.binary).toBe("stub")
      const stubEnv = JSON.parse(await Bun.file(path.join(run.runDir, "stub-env.json")).text()) as Record<string, string | undefined>
      expect(stubEnv.OPENCODE_PIN_BINDING_VERDICT).toBe("binding")
      // The [-proxy] run-dir suffix matches the leg's proxy flag, and proxy
      // legs ride with the upstream-shaped kill-switches.
      expect(path.basename(run.runDir).endsWith("-proxy")).toBe(run.proxy)
      expect(stubEnv.OPENCODE_DISABLE_LAZY_TOOLS).toBe(run.proxy ? "1" : undefined)
      expect(stubEnv.OPENCODE_DISABLE_STATIC_SLIMMING).toBe(run.proxy ? "1" : undefined)
      const provider = run.phase === "mcp" ? "stub2" : "stub"
      const config = JSON.parse(await Bun.file(path.join(run.runDir, "cwd/opencode.json")).text())
      expect(config.model).toBe(run.phase === "mcp" ? "stub2/other-model" : "stub/test-model")
      const auth = JSON.parse(await Bun.file(path.join(run.runDir, "data/opencode/auth.json")).text())
      expect(Object.keys(auth)).toEqual([provider])
      if (run.shape === "fork") {
        expect(stubEnv.OPENCODE_ENABLE_PROMPT_CAPTURE).toBe("1")
        expect(manifest.capture).toBe(true)
        expect(manifest.verdicts).toEqual(["binding"])
        const captures = await fs.readdir(path.join(run.runDir, "data/opencode/prompt-captures/stub-session"))
        expect(captures).toHaveLength(prompts.length)
      } else {
        expect(stubEnv.OPENCODE_ENABLE_PROMPT_CAPTURE).toBeUndefined()
        expect(manifest.capture).toBe(false)
        expect(manifest.verdicts).toEqual([])
        await expect(fs.access(path.join(run.runDir, "data/opencode/prompt-captures"))).rejects.toThrow()
      }
      if (run.phase === "mcp") expect(config.mcp).toEqual(baseSpec.phases[1].mcp)
      else expect(config.mcp).toBeUndefined()
    }
  })

  test("omitted seed records a random seed that reproduces the schedule; non-empty out refused", async () => {
    await using tmp = await tmpdir()
    const out = path.join(tmp.path, "campaign")
    const specFile = await writeSpec(tmp.path, { phases: [{ name: "p", model: "stub/m" }], legs: baseSpec.legs, runs: 2 })
    expect((await spawnDriver(campaignArgs(specFile, out, ["--stub"]))).code).toBe(0)
    const campaign = await readCampaign(out)
    expect(scheduleRuns(validateSpec({ phases: [{ name: "p", model: "stub/m" }], legs: baseSpec.legs, runs: 2 }), campaign.seed, out).map((run) => run.runDir)).toEqual(
      campaign.runs.map((run) => run.runDir),
    )
    const rerun = await spawnDriver(campaignArgs(specFile, out, ["--stub"]))
    expect(rerun.code).toBe(1)
    expect(rerun.stderr).toContain("not empty")
  })

  test("campaign CLI: --out required, single-run flags rejected in campaign mode", async () => {
    await using tmp = await tmpdir()
    const specFile = await writeSpec(tmp.path, baseSpec)
    const missingOut = await spawnDriver(["--campaign", specFile, "--stub"])
    expect(missingOut.code).toBe(1)
    expect(missingOut.stderr).toContain("--out")
    const withModel = await spawnDriver([...campaignArgs(specFile, path.join(tmp.path, "out")), "--stub", "--model", "a/b"])
    expect(withModel.code).toBe(1)
    expect(withModel.stderr).toContain("campaign mode")
  })

  test("campaign.json is written before any run completes (progressive from the start)", async () => {
    await using tmp = await tmpdir()
    const out = path.join(tmp.path, "campaign")
    const spec = {
      phases: [{ name: "p", model: "test-provider/test-model" }],
      legs: [{ shape: "fork", bin: "definitely-missing-bin-xyz" }],
      runs: 2,
    }
    const specFile = await writeSpec(tmp.path, spec)
    const { code } = await spawnDriver(campaignArgs(specFile, out))
    expect(code).toBe(1)
    const campaign = await readCampaign(out)
    expect(campaign.runs).toHaveLength(2)
    expect(campaign.runs[0].status).toBe("failed")
    expect(campaign.runs[1].status).toBe("pending")
  })

  test("mid-campaign failure aborts: failed + error + endedAt, later runs pending, exit non-zero", async () => {
    await using fake = await tmpdir()
    const authDir = path.join(fake.path, "opencode")
    await fs.mkdir(authDir, { recursive: true })
    await Bun.write(path.join(authDir, "auth.json"), JSON.stringify({ "test-provider": { type: "api", key: "test-key" } }))
    await using tmp = await tmpdir()
    const goodBin = path.join(tmp.path, "good.ts")
    await Bun.write(
      goodBin,
      `if (Bun.argv.includes("--version")) { process.stdout.write("1.0.0\\n"); process.exit(0) }\nprocess.exit(0)\n`,
    )
    const spec = {
      phases: [{ name: "p", model: "test-provider/test-model" }],
      legs: [
        { shape: "fork", bin: `bun ${goodBin}` },
        { shape: "upstream", bin: "definitely-missing-bin-xyz" },
      ],
      runs: 2,
      seed: 7,
    }
    const specFile = await writeSpec(tmp.path, spec)
    const out = path.join(tmp.path, "campaign")
    const { code, stderr } = await spawnDriver(campaignArgs(specFile, out), { XDG_DATA_HOME: fake.path })
    expect(code).toBe(1)
    expect(stderr).toContain("aborted")

    const campaign = await readCampaign(out)
    expect(typeof campaign.endedAt).toBe("string")
    const failed = campaign.runs.filter((run) => run.status === "failed")
    expect(failed).toHaveLength(1)
    expect(failed[0].error).toBeTruthy()
    const failedIndex = failed[0].index
    for (const run of campaign.runs) {
      if (run.index < failedIndex) expect(run.status).toBe("done")
      if (run.index > failedIndex) {
        expect(run.status).toBe("pending")
        await expect(fs.access(run.runDir)).rejects.toThrow()
      }
    }
  })
})
