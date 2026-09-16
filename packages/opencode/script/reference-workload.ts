#!/usr/bin/env bun

import fs from "fs/promises"
import os from "os"
import path from "path"
import { Schema } from "effect"
import { deriveVerdicts, deriveVerdictProvenances, readCaptures, RunManifestSchema, type RunManifest } from "./measure-usage"
import { prompts } from "./reference-workload/prompts"

// Reference-workload driver (R10-004): runs the fixed prompt list (prompts.ts)
// sequentially against one session on any opencode-shaped binary — upstream CLI
// or fork — inside a fully isolated run dir: fixture copy as cwd, local
// opencode.json (model pin, temperature 0), seeded auth.json, XDG_DATA_HOME +
// OPENCODE_DB inside the run dir (xdg-basedir reads env at import time). The
// run dir is the input contract of script/measure-usage.ts (report/diff).
// Contracts: ARCHITECTURE/detailed/context-observability.md + decision
// context-observability-02. Capture env is always set: upstream binaries ignore
// it (no captures → null attribution in the report), fork runs capture.
// --proxy adds the optimization kill-switches for upstream-shaped attribution.
// Campaign mode (--campaign, R13-002/003, decision comparison-testing-01 §4):
// phases × legs × N runs Fisher–Yates-shuffled with a recorded seed, executed
// strictly sequentially, each run reusing the single-run path; progressive
// campaign.json, abort on first failure.

const USAGE =
  "usage: bun run script/reference-workload.ts (--bin <cmd> --run-dir <dir> --model <provider/model> [--pin <binding|advisory>] [--proxy] | --campaign <spec.json> --out <dir>) [--stub]"

const FIXTURE_DIR = path.join(import.meta.dir, "reference-workload", "fixture-repo")

// Workload identity (R10-004): the fixed prompts AND the fixture repo content —
// the digest in RunManifest is what measure-usage diff guards on, so editing
// either input produces an incomparable run.
export const workloadDigest = (input: { prompts: string[]; files: Record<string, string> }): string =>
  new Bun.CryptoHasher("sha256")
    .update(JSON.stringify({ prompts: input.prompts, files: Object.entries(input.files).sort() }))
    .digest("hex")

export const digestWorkload = async (): Promise<string> => {
  const names = (await Array.fromAsync(new Bun.Glob("**/*").scan({ cwd: FIXTURE_DIR, dot: true }))).sort()
  const files: Record<string, string> = {}
  for (const name of names) files[name] = await Bun.file(path.join(FIXTURE_DIR, name)).text()
  return workloadDigest({ prompts, files })
}

type DriverArgs = {
  campaign: string | null
  out: string | null
  bin: string
  runDir: string
  model: string
  proxy: boolean
  stub: boolean
  pin: "binding" | "advisory" | null
}

const parseArgs = (argv: string[]): DriverArgs => {
  const flags = { campaign: "", out: "", bin: "", runDir: "", model: "", proxy: false, stub: false, pin: null as string | null }
  const valueFlag = (arg: string): "bin" | "runDir" | "model" | "pin" | "campaign" | "out" | undefined =>
    arg === "--bin"
      ? "bin"
      : arg === "--run-dir"
        ? "runDir"
        : arg === "--model"
          ? "model"
          : arg === "--pin"
            ? "pin"
            : arg === "--campaign"
              ? "campaign"
              : arg === "--out"
                ? "out"
                : undefined
  for (let i = 0; i < argv.length; i++) {
    const flag = valueFlag(argv[i])
    if (flag) {
      const value = argv[i + 1]
      if (value === undefined) throw new Error(`reference-workload: ${argv[i]} requires a value\n${USAGE}`)
      flags[flag] = value
      i++
    } else if (argv[i] === "--proxy") flags.proxy = true
    else if (argv[i] === "--stub") flags.stub = true
    else throw new Error(`reference-workload: unknown argument "${argv[i]}"\n${USAGE}`)
  }
  // R13-003: pin the verdict per run (cascade step 0 in the child). Anything
  // but the two verdict literals — including an empty value — fails loudly: a
  // typo'd pin would silently measure the probe instead.
  if (flags.pin !== null && flags.pin !== "binding" && flags.pin !== "advisory")
    throw new Error(`reference-workload: --pin must be "binding" or "advisory", got "${flags.pin}"\n${USAGE}`)
  if (flags.campaign !== "") {
    if (!flags.out) throw new Error(`reference-workload: --out is required with --campaign\n${USAGE}`)
    // Campaign-mode flags live in the spec — silently ignoring a --bin/--model
    // would measure the wrong thing (R00-010).
    if (flags.bin) throw new Error(`reference-workload: --bin is not valid in campaign mode (legs carry the bin)\n${USAGE}`)
    if (flags.model) throw new Error(`reference-workload: --model is not valid in campaign mode (phases carry the model)\n${USAGE}`)
    if (flags.runDir) throw new Error(`reference-workload: --run-dir is not valid in campaign mode (use --out)\n${USAGE}`)
    if (flags.pin !== null) throw new Error(`reference-workload: --pin is not valid in campaign mode (pin lives in the spec)\n${USAGE}`)
    if (flags.proxy) throw new Error(`reference-workload: --proxy is not valid in campaign mode (proxy lives on the legs)\n${USAGE}`)
    return { campaign: flags.campaign, out: flags.out, bin: "", runDir: "", model: "", proxy: false, stub: flags.stub, pin: null }
  }
  if (!flags.bin || !flags.runDir || !flags.model)
    throw new Error(`reference-workload: --bin, --run-dir and --model are required\n${USAGE}`)
  return {
    campaign: null,
    out: null,
    bin: flags.bin,
    runDir: flags.runDir,
    model: flags.model,
    proxy: flags.proxy,
    stub: flags.stub,
    pin: flags.pin === "binding" || flags.pin === "advisory" ? flags.pin : null,
  }
}

const parseModel = (model: string) => {
  const slash = model.indexOf("/")
  if (slash <= 0 || slash === model.length - 1)
    throw new Error(`reference-workload: --model must be <provider/model>, got "${model}"\n${USAGE}`)
  return { providerID: model.slice(0, slash), modelID: model.slice(slash + 1) }
}

const seedAuth = async (input: { providerID: string; stub: boolean }) => {
  if (input.stub) return { [input.providerID]: { type: "api", key: "stub-key-not-a-secret" } }
  const xdg = process.env.XDG_DATA_HOME ?? path.join(os.homedir(), ".local", "share")
  const file = path.join(xdg, "opencode", "auth.json")
  const auth = await Bun.file(file)
    .json()
    .catch((error: unknown) => {
      const cause = error instanceof Error ? error.message : String(error)
      throw new Error(
        `reference-workload: no auth for provider "${input.providerID}" at ${file} (${cause}) — authenticate the provider or use --stub`,
      )
    })
  const entry = (auth as Record<string, unknown>)[input.providerID]
  if (!entry) throw new Error(`reference-workload: provider "${input.providerID}" is not authenticated in ${file}`)
  return { [input.providerID]: entry }
}

const prepareRunDir = async (input: { runDir: string; model: string; providerID: string; stub: boolean; mcp?: unknown }) => {
  const entries = await fs.readdir(input.runDir).catch((error: NodeJS.ErrnoException) => {
    if (error.code === "ENOENT") return null
    throw error
  })
  if (entries && entries.length > 0)
    throw new Error(`reference-workload: run dir ${input.runDir} is not empty — refusing to overwrite an existing run`)
  const cwd = path.join(input.runDir, "cwd")
  await fs.cp(path.join(import.meta.dir, "reference-workload", "fixture-repo"), cwd, { recursive: true })
  const config: Record<string, unknown> = {
    $schema: "https://opencode.ai/config.json",
    model: input.model,
    agent: { build: { temperature: 0 } },
  }
  if (input.mcp !== undefined) config.mcp = input.mcp
  await Bun.write(path.join(cwd, "opencode.json"), JSON.stringify(config, null, 2) + "\n")
  const dataDir = path.join(input.runDir, "data", "opencode")
  await fs.mkdir(dataDir, { recursive: true })
  await Bun.write(path.join(dataDir, "auth.json"), JSON.stringify(await seedAuth(input), null, 2) + "\n")
}

const childEnv = (input: { runDir: string; proxy: boolean; pin: DriverArgs["pin"]; capture: boolean }): Record<string, string> => {
  if (!process.env.PATH) throw new Error("reference-workload: PATH is not set — cannot spawn the binary")
  // Fresh allowlisted env (xdg-basedir reads env at import time): operator-shell
  // XDG_*/OPENCODE_* must not leak into the measurement. Proxy reachability vars
  // pass through so both legs share identical network conditions.
  const env: Record<string, string> = {
    PATH: process.env.PATH,
    HOME: process.env.HOME ?? os.homedir(),
    XDG_DATA_HOME: path.join(input.runDir, "data"),
    OPENCODE_DB: path.join(input.runDir, "data", "opencode.db"),
  }
  // Campaign capture asymmetry (comparison-testing-01 §4): fork-shape legs only.
  // Real upstream binaries ignore the flag either way; stub legs obey it, so the
  // upstream capture-lessness is simulated faithfully offline.
  if (input.capture) env.OPENCODE_ENABLE_PROMPT_CAPTURE = "1"
  for (const key of ["HTTP_PROXY", "HTTPS_PROXY", "NO_PROXY", "http_proxy", "https_proxy", "no_proxy"]) {
    const value = process.env[key]
    if (value) env[key] = value
  }
  // Credential passthrough: provider + MCP keys from the operator shell, when
  // set. Identical for both legs (fair comparison); never written to the run
  // config, the manifest, or the repo. Needed so MCP legs interpolate
  // `{env:...}` keys in the child like the operator's real setup does.
  for (const key of ["OPENROUTER_API_KEY", "FIRECRAWL_API_KEY", "FAL_AI_API_KEY", "N8N_MCP_TOKEN"]) {
    const value = process.env[key]
    if (value) env[key] = value
  }
  if (input.proxy) {
    env.OPENCODE_DISABLE_LAZY_TOOLS = "1"
    env.OPENCODE_DISABLE_STATIC_SLIMMING = "1"
  }
  // R13-003: the pin rides to the child (cascade step 0). Upstream binaries
  // ignore it — pinning a proxy/upstream leg is a no-op by design.
  if (input.pin) env.OPENCODE_PIN_BINDING_VERDICT = input.pin
  return env
}

const runWorkload = async (input: { cmd: string[]; model: string; cwd: string; env: Record<string, string> }) => {
  for (const [index, prompt] of prompts.entries()) {
    const args = ["run", "--model", input.model, "--agent", "build"]
    if (index > 0) args.push("--continue")
    args.push(prompt)
    const proc = Bun.spawn([...input.cmd, ...args], {
      cwd: input.cwd,
      env: input.env,
      stdin: "ignore",
      stdout: "inherit",
      stderr: "inherit",
    })
    const code = await proc.exited
    if (code !== 0)
      throw new Error(`reference-workload: prompt ${index} exited with code ${code} (${input.cmd.join(" ")} run)`)
    console.log(`[${index + 1}/${prompts.length}] ${prompt}`)
  }
}

// Binary identity (R13-005): first non-empty stdout line of `<bin> --version`,
// trimmed. "stub" in stub mode; failure → null + warning — the recorded `bin`
// remains the fallback identity, the manifest is still written.
const resolveBinary = async (input: { cmd: string[]; stub: boolean }): Promise<string | null> => {
  if (input.stub) return "stub"
  let proc: Bun.Subprocess<"ignore", "pipe", "pipe">
  try {
    proc = Bun.spawn([...input.cmd, "--version"], { stdout: "pipe", stderr: "pipe" })
  } catch (error) {
    console.error(`reference-workload: <bin> --version failed to spawn, binary identity unresolved: ${error instanceof Error ? error.message : String(error)}`)
    return null
  }
  const [stdout, code] = await Promise.all([new Response(proc.stdout).text(), proc.exited])
  const line = stdout.split("\n").find((candidate) => candidate.trim() !== "")
  if (code !== 0 || line === undefined) {
    console.error(`reference-workload: <bin> --version failed (exit ${code}), binary identity unresolved: ${input.cmd.join(" ")}`)
    return null
  }
  return line.trim()
}

// One isolated run: fixture copy + config + auth, sequential prompts, v2
// manifest. Shared by single-run mode and every campaign run.
const executeRun = async (input: {
  cmd: string[]
  bin: string
  stub: boolean
  runDir: string
  model: string
  mcp?: unknown
  proxy: boolean
  pin: DriverArgs["pin"]
  capture: boolean
  promptsDigest: string
}): Promise<RunManifest> => {
  const { providerID, modelID } = parseModel(input.model)
  const startedAt = new Date().toISOString()
  await prepareRunDir({ runDir: input.runDir, model: input.model, providerID, stub: input.stub, mcp: input.mcp })
  await runWorkload({
    cmd: input.cmd,
    model: input.model,
    cwd: path.join(input.runDir, "cwd"),
    env: childEnv({ runDir: input.runDir, proxy: input.proxy, pin: input.pin, capture: input.capture }),
  })
  // Harness v2 manifest (R13-003/005): the run's distinct recorded verdicts are
  // derived from the captures post-run (per-turn truth, never the verdict
  // cache's end state); binary identity from `<bin> --version`. A missing
  // captures dir is the upstream-leg norm (capture env not set) — any other
  // stat error stays loud (R00-010).
  const capturesDir = path.join(input.runDir, "data", "opencode", "prompt-captures")
  const capturesDirExists = await fs.stat(capturesDir).then(
    () => true,
    (error: NodeJS.ErrnoException) => {
      if (error.code === "ENOENT") return false
      throw error
    },
  )
  const captureRecords = capturesDirExists ? await readCaptures(capturesDir, input.runDir) : []
  const manifest: RunManifest = {
    bin: input.stub ? "stub" : input.bin,
    modelID,
    providerID,
    promptsDigest: input.promptsDigest,
    capture: input.capture,
    proxyMode: input.proxy,
    dbPath: "data/opencode.db",
    capturesDir: "data/opencode/prompt-captures",
    startedAt,
    endedAt: new Date().toISOString(),
    pin: input.pin,
    verdicts: deriveVerdicts(captureRecords),
    verdictProvenances: deriveVerdictProvenances(captureRecords),
    binary: await resolveBinary({ cmd: input.cmd, stub: input.stub }),
  }
  if (!Schema.is(RunManifestSchema)(manifest))
    throw new Error("reference-workload: assembled manifest does not match the RunManifest schema")
  await Bun.write(path.join(input.runDir, "manifest.json"), JSON.stringify(manifest, null, 2) + "\n")
  return manifest
}

// Campaign spec (driver input, JSON) + schedule record — the aggregation input
// of measure-usage campaign mode (R13-002/003, decision comparison-testing-01 §4).
export type CampaignSpec = {
  phases: Array<{ name: string; model: string; mcp?: unknown }>
  legs: Array<{ shape: "fork" | "upstream"; bin: string; proxy?: boolean }>
  runs: number // N per (phase × leg), minimum 2 — never single runs
  seed?: number
  pin?: "binding" | "advisory"
}

export type CampaignSchedule = {
  spec: CampaignSpec
  seed: number
  startedAt: string
  endedAt: string | null
  runs: Array<{
    index: number
    phase: string
    shape: "fork" | "upstream"
    proxy: boolean
    rep: number
    runDir: string
    status: "done" | "failed" | "pending"
    error?: string
  }>
}

type RunPlan = CampaignSchedule["runs"][number] & { legIndex: number; phaseIndex: number }

export const validateSpec = (raw: unknown): CampaignSpec => {
  // A function declaration, not an arrow const — never-return control-flow
  // narrowing (used below after every guard) only applies to declarations.
  function fail(detail: string): never {
    throw new Error(`reference-workload: invalid campaign spec — ${detail}`)
  }
  if (typeof raw !== "object" || raw === null || Array.isArray(raw)) fail("expected a JSON object")
  const doc = raw as Record<string, unknown>
  if (!Array.isArray(doc.phases) || doc.phases.length === 0) fail("phases must be a non-empty array")
  if (!Array.isArray(doc.legs) || doc.legs.length === 0) fail("legs must be a non-empty array")
  const names = new Set<string>()
  const phases = doc.phases.map((entry, i): { name: string; model: string; mcp?: unknown } => {
    if (typeof entry !== "object" || entry === null || Array.isArray(entry)) fail(`phases[${i}] must be an object`)
    const phase = entry as Record<string, unknown>
    // The run-dir name embeds the phase, so it must be filesystem-safe.
    if (typeof phase.name !== "string" || !/^[A-Za-z0-9][A-Za-z0-9._-]*$/.test(phase.name))
      fail(
        `phases[${i}].name must be a filesystem-safe name ([A-Za-z0-9._-], leading alphanumeric), got ${JSON.stringify(phase.name)}`,
      )
    if (names.has(phase.name)) fail(`duplicate phase name "${phase.name}"`)
    names.add(phase.name)
    if (typeof phase.model !== "string" || !/^[^/]+\/.+/.test(phase.model))
      fail(`phases[${i}].model must be <provider/model>, got ${JSON.stringify(phase.model)}`)
    return "mcp" in phase ? { name: phase.name, model: phase.model, mcp: phase.mcp } : { name: phase.name, model: phase.model }
  })
  const legs = doc.legs.map((entry, i): CampaignSpec["legs"][number] => {
    if (typeof entry !== "object" || entry === null || Array.isArray(entry)) fail(`legs[${i}] must be an object`)
    const leg = entry as Record<string, unknown>
    if (leg.shape !== "fork" && leg.shape !== "upstream")
      fail(`legs[${i}].shape must be "fork" or "upstream", got ${JSON.stringify(leg.shape)}`)
    const shape = leg.shape as "fork" | "upstream"
    if (typeof leg.bin !== "string" || leg.bin.trim() === "") fail(`legs[${i}].bin must be a non-empty string`)
    if (leg.proxy !== undefined && typeof leg.proxy !== "boolean") fail(`legs[${i}].proxy must be a boolean`)
    const bin = leg.bin as string
    if (leg.proxy === undefined) return { shape, bin }
    return { shape, bin, proxy: leg.proxy as boolean }
  })
  let runs = 3
  if (doc.runs !== undefined) {
    if (!Number.isInteger(doc.runs) || (doc.runs as number) < 2)
      fail(`runs must be an integer >= 2 (never single runs), got ${JSON.stringify(doc.runs)}`)
    runs = doc.runs as number
  }
  if (doc.seed !== undefined && (!Number.isInteger(doc.seed) || (doc.seed as number) < 0 || (doc.seed as number) > 0xffffffff))
    fail(`seed must be a 32-bit unsigned integer, got ${JSON.stringify(doc.seed)}`)
  if (doc.pin !== undefined && doc.pin !== "binding" && doc.pin !== "advisory")
    fail(`pin must be "binding" or "advisory", got ${JSON.stringify(doc.pin)}`)
  return {
    phases,
    legs,
    runs,
    ...(doc.seed !== undefined ? { seed: doc.seed as number } : {}),
    ...(doc.pin !== undefined ? { pin: doc.pin as "binding" | "advisory" } : {}),
  }
}

// Classic 32-bit LCG (Numerical Recipes constants): deterministic stream from
// the recorded seed, continued across phases so the whole schedule reproduces.
const lcg = (seed: number) => (): number => {
  seed = (Math.imul(seed, 1664525) + 1013904223) | 0
  return (seed >>> 0) / 2 ** 32
}

export const scheduleRuns = (spec: CampaignSpec, seed: number, outDir: string): RunPlan[] => {
  const rand = lcg(seed)
  const plans: RunPlan[] = []
  let index = 0
  for (const [phaseIndex, phase] of spec.phases.entries()) {
    const entries = spec.legs.flatMap((leg, legIndex) =>
      Array.from({ length: spec.runs }, (_, rep) => ({ legIndex, rep, shape: leg.shape, proxy: leg.proxy ?? false })),
    )
    for (let i = entries.length - 1; i > 0; i--) {
      const j = Math.floor(rand() * (i + 1))
      ;[entries[i], entries[j]] = [entries[j], entries[i]]
    }
    for (const entry of entries) {
      plans.push({
        index,
        phase: phase.name,
        shape: entry.shape,
        proxy: entry.proxy,
        rep: entry.rep,
        runDir: path.join(outDir, `${String(index).padStart(2, "0")}-${phase.name}-${entry.shape}${entry.proxy ? "-proxy" : ""}`),
        status: "pending",
        legIndex: entry.legIndex,
        phaseIndex,
      })
      index++
    }
  }
  return plans
}

const runCampaign = async (input: { specPath: string; out: string; stub: boolean }): Promise<number> => {
  const raw = await Bun.file(input.specPath).json().catch((error: unknown) => {
    const cause = error instanceof Error ? error.message : String(error)
    throw new Error(`reference-workload: cannot read campaign spec ${input.specPath}: ${cause}`)
  })
  const spec = validateSpec(raw)
  const outDir = path.resolve(input.out)
  const existing = await fs.readdir(outDir).catch((error: NodeJS.ErrnoException) => {
    if (error.code === "ENOENT") return null
    throw error
  })
  if (existing && existing.length > 0)
    throw new Error(`reference-workload: campaign out dir ${outDir} is not empty — refusing to overwrite an existing campaign`)
  await fs.mkdir(outDir, { recursive: true })
  const seed = spec.seed ?? crypto.getRandomValues(new Uint32Array(1))[0]
  const plans = scheduleRuns(spec, seed, outDir)
  const startedAt = new Date().toISOString()
  let endedAt: string | null = null
  const writeCampaign = async () => {
    const doc: CampaignSchedule = {
      spec,
      seed,
      startedAt,
      endedAt,
      runs: plans.map(({ legIndex, phaseIndex, ...entry }) => entry),
    }
    await Bun.write(path.join(outDir, "campaign.json"), JSON.stringify(doc, null, 2) + "\n")
  }
  // Progressive: written up front (all pending), then after every run — a
  // killed operator session still leaves the schedule and per-run states.
  await writeCampaign()
  const promptsDigest = await digestWorkload()
  const stubCmd = [process.execPath, path.join(import.meta.dir, "reference-workload", "stub-bin.ts")]
  for (const plan of plans) {
    const leg = spec.legs[plan.legIndex]
    const phase = spec.phases[plan.phaseIndex]
    try {
      await executeRun({
        cmd: input.stub ? stubCmd : leg.bin.trim().split(/\s+/),
        bin: leg.bin,
        stub: input.stub,
        runDir: plan.runDir,
        model: phase.model,
        mcp: phase.mcp,
        proxy: leg.proxy ?? false,
        pin: spec.pin ?? null,
        capture: leg.shape === "fork",
        promptsDigest,
      })
      plan.status = "done"
    } catch (error) {
      // First failure aborts: a half-run leg would poison its medians.
      plan.status = "failed"
      plan.error = error instanceof Error ? error.message : String(error)
      endedAt = new Date().toISOString()
      await writeCampaign()
      console.error(`reference-workload: campaign aborted at run ${plan.index} (${path.basename(plan.runDir)}): ${plan.error}`)
      return 1
    }
    await writeCampaign()
  }
  endedAt = new Date().toISOString()
  await writeCampaign()
  return 0
}

const main = async (argv: string[]): Promise<number> => {
  const args = parseArgs(argv)
  if (args.campaign) return runCampaign({ specPath: args.campaign, out: args.out ?? "", stub: args.stub })
  await executeRun({
    cmd: args.stub ? [process.execPath, path.join(import.meta.dir, "reference-workload", "stub-bin.ts")] : args.bin.trim().split(/\s+/),
    bin: args.bin,
    stub: args.stub,
    runDir: path.resolve(args.runDir),
    model: args.model,
    proxy: args.proxy,
    pin: args.pin,
    capture: true,
    promptsDigest: await digestWorkload(),
  })
  return 0
}

if (import.meta.main) {
  try {
    process.exit(await main(process.argv.slice(2)))
  } catch (error) {
    console.error(error instanceof Error ? error.message : error)
    process.exit(1)
  }
}