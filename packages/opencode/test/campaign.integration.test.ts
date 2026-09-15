import { describe, expect, test } from "bun:test"
import fs from "fs/promises"
import path from "path"
import { tmpdir } from "./fixture/fixture"
import type { ComparisonReport } from "../script/measure-usage"
import { scheduleRuns, validateSpec } from "../script/reference-workload"
import type { CampaignSpec } from "../script/reference-workload"
import { prompts } from "../script/reference-workload/prompts"

// Feature-13 integration (R13-004/006, ticket 32): stub-driver campaign → v2
// report with verdict columns, end to end and offline. Driver and measure-usage
// CLI are spawned as processes (as in test/reference-workload.test.ts); the
// output contracts are <out>/campaign.json + the run dirs and the
// ComparisonReport JSON (design comparison-testing.md §Integration test,
// decision comparison-testing-01).

const DRIVER = path.join(import.meta.dir, "../script/reference-workload.ts")
const MEASURE = path.join(import.meta.dir, "../script/measure-usage.ts")

const spawn = async (script: string, args: string[], env?: Record<string, string>) => {
  const proc = Bun.spawn([process.execPath, script, ...args], {
    stdout: "pipe",
    stderr: "pipe",
    ...(env ? { env: { ...process.env, ...env } } : {}),
  })
  const [stdout, stderr, code] = await Promise.all([
    new Response(proc.stdout).text(),
    new Response(proc.stderr).text(),
    proc.exited,
  ])
  return { stdout, stderr, code }
}

// 2 phases × {fork, upstream} × N=2, fixed seed (R13-002 minimum, pinned
// verdict — the stub maps the pin env to meta.verdict).
const spec: CampaignSpec = {
  phases: [
    { name: "alpha", model: "stub/test-model" },
    { name: "beta-mcp", model: "stub2/other-model", mcp: { firecrawl: { type: "remote", url: "https://mcp.firecrawl.dev" } } },
  ],
  legs: [
    { shape: "fork", bin: "fork-bin" },
    { shape: "upstream", bin: "upstream-bin" },
  ],
  runs: 2,
  seed: 42,
  pin: "binding",
}

const runCampaign = async (dir: string, campaignSpec: CampaignSpec, options?: { stub?: boolean; env?: Record<string, string> }) => {
  const specFile = path.join(dir, "spec.json")
  await Bun.write(specFile, JSON.stringify(campaignSpec))
  const out = path.join(dir, "campaign")
  const args = ["--campaign", specFile, "--out", out]
  if (options?.stub ?? true) args.push("--stub")
  const result = await spawn(DRIVER, args, options?.env)
  return { ...result, out }
}

const reportCampaign = async (out: string): Promise<ComparisonReport> => {
  // Exit code only: aggregating capture-less upstream legs logs the benign
  // tolerant-read line ("captures dir not found") to stderr.
  const { stdout, code } = await spawn(MEASURE, ["campaign", out])
  expect(code).toBe(0)
  return JSON.parse(stdout) as ComparisonReport
}

// The stub's deterministic usage row: input 1000 + ceil(prompt.length/4),
// cache read 7 + write 3 (stub-bin.ts) — the metrics' exact expected values.
const stubColdStart = 1000 + Math.ceil(prompts[0].length / 4)
const stubSessionInput = prompts.reduce((sum, prompt) => sum + 1000 + Math.ceil(prompt.length / 4) + 7 + 3, 0)

const degenerate = (d: ComparisonReport["phases"][number]["legs"][number]["metrics"]["coldStartInput"] | undefined, expected: number) => {
  expect(d).not.toBeNull()
  expect(d?.values).toEqual([expected, expected])
  expect(d?.median).toBe(expected)
  expect(d?.min).toBe(expected)
  expect(d?.max).toBe(expected)
}

describe("campaign integration", () => {
  test("pinned stub campaign → v2 report with verdict columns (scenarios 1-3)", async () => {
    await using tmp = await tmpdir()
    const { code, stderr, out } = await runCampaign(tmp.path, spec)
    expect(stderr).toBe("")
    expect(code).toBe(0)

    // Scenario 1: campaign.json carries the 8 runs in the seeded-shuffle
    // order, all done; each run dir is the v1 layout + v2 manifest.
    const campaign = JSON.parse(await Bun.file(path.join(out, "campaign.json")).text()) as {
      runs: Array<{ runDir: string; status: string }>
    }
    expect(campaign.runs).toHaveLength(8)
    expect(campaign.runs.map((run) => run.runDir)).toEqual(scheduleRuns(validateSpec(spec), spec.seed ?? 0, out).map((run) => run.runDir))
    for (const run of campaign.runs) {
      expect(run.status).toBe("done")
      const manifest = JSON.parse(await Bun.file(path.join(run.runDir, "manifest.json")).text()) as Record<string, unknown>
      expect(manifest.pin).toBe("binding")
      expect(manifest.binary).toBe("stub")
      expect(manifest.promptsDigest).toBeTypeOf("string")
      await fs.access(path.join(run.runDir, "cwd", "opencode.json"))
      await fs.access(path.join(run.runDir, "data", "opencode.db"))
    }

    // Scenario 2: fork legs captured, upstream legs did not — the offline
    // capture asymmetry (capture env on fork-shape legs only).
    const schedule = scheduleRuns(validateSpec(spec), spec.seed ?? 0, out)
    for (const run of campaign.runs) {
      const capturesDir = path.join(run.runDir, "data", "opencode", "prompt-captures", "stub-session")
      if (schedule.find((plan) => plan.runDir === run.runDir)?.shape === "fork") {
        expect(await fs.readdir(capturesDir)).toHaveLength(prompts.length)
      } else {
        await expect(fs.access(path.join(run.runDir, "data", "opencode", "prompt-captures"))).rejects.toThrow()
      }
    }

    // Scenario 3: measure-usage campaign → ComparisonReport schema 1 with
    // degenerate dispersions (N=2 deterministic stub runs), the verdict
    // columns, comparable pairs and the payloadChars presence pattern.
    const report = await reportCampaign(out)
    expect(report.schema).toBe(1)
    expect(report.warnings).toEqual([])
    expect(report.phases.map((phase) => phase.name)).toEqual(["alpha", "beta-mcp"])
    for (const phase of report.phases) {
      const forkLeg = phase.legs.find((leg) => leg.shape === "fork")
      const upstreamLeg = phase.legs.find((leg) => leg.shape === "upstream")
      expect(forkLeg).toBeDefined()
      expect(upstreamLeg).toBeDefined()
      expect(forkLeg?.verdict).toBe("binding")
      expect(forkLeg?.verdictMixed).toBe(false)
      expect(forkLeg?.runs.map((entry) => entry.verdicts)).toEqual([["binding"], ["binding"]])
      // Gate-exempt by absence of the mechanism: no captures, no verdict.
      expect(upstreamLeg?.verdict).toBeNull()
      expect(upstreamLeg?.verdictMixed).toBe(false)
      expect(upstreamLeg?.runs.map((entry) => entry.verdicts)).toEqual([[], []])
      degenerate(forkLeg?.metrics.coldStartInput, stubColdStart)
      degenerate(forkLeg?.metrics.sessionInput, stubSessionInput)
      degenerate(upstreamLeg?.metrics.coldStartInput, stubColdStart)
      degenerate(upstreamLeg?.metrics.sessionInput, stubSessionInput)
      const payloadChars = forkLeg?.metrics.payloadChars
      expect(payloadChars).not.toBeNull()
      const payloadValue = payloadChars?.values[0] ?? -1
      expect(payloadChars?.values).toEqual([payloadValue, payloadValue])
      expect(payloadValue).toBeGreaterThan(0)
      expect(upstreamLeg?.metrics.payloadChars).toBeNull()
      // fork-vs-upstream pair (no proxy leg in this spec): comparable, fork
      // verdict column "binding", zero deltas over the identical stub legs,
      // no payloadChars in the totals-currency pair.
      expect(phase.comparisons).toHaveLength(1)
      const pair = phase.comparisons[0]
      expect(pair.kind).toBe("fork-vs-upstream")
      expect(pair.comparable).toBe(true)
      expect(pair.verdict).toBe("binding")
      expect(pair.metrics.coldStartInput).toEqual({ fork: stubColdStart, other: stubColdStart, delta: 0 })
      expect(pair.metrics.sessionInput).toEqual({ fork: stubSessionInput, other: stubSessionInput, delta: 0 })
      expect(pair.metrics.payloadChars).toBeNull()
    }
  })

  test("unpinned campaign: fork verdict unrecordable → comparisons refused with a warning (scenario 4)", async () => {
    await using tmp = await tmpdir()
    const unpinned: CampaignSpec = { ...spec, pin: undefined }
    const { code, out } = await runCampaign(tmp.path, unpinned)
    expect(code).toBe(0)

    const report = await reportCampaign(out)
    expect(report.schema).toBe(1)
    const forkRunDirs = report.phases.flatMap((phase) =>
      phase.legs.filter((leg) => leg.shape === "fork").flatMap((leg) => leg.runs.map((entry) => entry.runDir)),
    )
    for (const phase of report.phases) {
      const forkLeg = phase.legs.find((leg) => leg.shape === "fork")
      expect(forkLeg?.verdict).toBeNull()
      expect(forkLeg?.verdictMixed).toBe(false)
      // Medians stay emitted, labeled incomparable — never silently folded.
      expect(forkLeg?.metrics.coldStartInput).not.toBeNull()
      const pair = phase.comparisons[0]
      expect(pair.comparable).toBe(false)
      expect(pair.verdict).toBeNull()
      expect(pair.metrics.coldStartInput).not.toBeNull()
    }
    expect(report.warnings.length).toBeGreaterThan(0)
    expect(report.warnings.some((warning) => warning.includes("no recorded verdict"))).toBe(true)
    for (const runDir of forkRunDirs) {
      expect(report.warnings.some((warning) => warning.includes(runDir))).toBe(true)
    }
  })

  test("runs: 1 spec fails loudly (scenario 5a)", async () => {
    await using tmp = await tmpdir()
    const { code, stderr } = await runCampaign(tmp.path, { ...spec, runs: 1, pin: undefined })
    expect(code).toBe(1)
    expect(stderr).toContain("never single runs")
  })

  test("interrupted campaign: report refuses loudly (scenario 5b)", async () => {
    await using fake = await tmpdir()
    const authDir = path.join(fake.path, "opencode")
    await fs.mkdir(authDir, { recursive: true })
    await Bun.write(path.join(authDir, "auth.json"), JSON.stringify({ "test-provider": { type: "api", key: "test-key" } }))
    await using tmp = await tmpdir()
    // A fork bin that succeeds and an upstream bin that does not exist: the
    // campaign aborts mid-flight (runner contract, ticket 29).
    const goodBin = path.join(tmp.path, "good.ts")
    await Bun.write(goodBin, `if (Bun.argv.includes("--version")) { process.stdout.write("1.0.0\\n"); process.exit(0) }\nprocess.exit(0)\n`)
    const aborted = await runCampaign(
      tmp.path,
      {
        phases: [{ name: "p", model: "test-provider/test-model" }],
        legs: [
          { shape: "fork", bin: `bun ${goodBin}` },
          { shape: "upstream", bin: "definitely-missing-bin-xyz" },
        ],
        runs: 2,
        seed: 7,
      },
      { stub: false, env: { XDG_DATA_HOME: fake.path } },
    )
    expect(aborted.code).toBe(1)
    expect(aborted.stderr).toContain("aborted")

    const { code, stderr } = await spawn(MEASURE, ["campaign", aborted.out])
    expect(code).toBe(1)
    expect(stderr).toContain("incomplete")
  })
})

// R13-004 structural assert: the reference prompts are owner-authored
// committed literals — a non-empty list of non-empty string literals. The
// owner pass (re-authoring/self-containment review) is Antoine's; the digest
// change it causes is the invalidation mechanism working by design.
describe("reference-workload prompts (R13-004)", () => {
  test("prompts is a non-empty array of non-empty string literals", () => {
    expect(Array.isArray(prompts)).toBe(true)
    expect(prompts.length).toBeGreaterThan(0)
    for (const prompt of prompts) {
      expect(typeof prompt).toBe("string")
      expect(prompt.trim().length).toBeGreaterThan(0)
    }
  })
})
