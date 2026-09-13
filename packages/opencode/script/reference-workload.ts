#!/usr/bin/env bun

import fs from "fs/promises"
import os from "os"
import path from "path"
import { Schema } from "effect"
import { RunManifestSchema, type RunManifest } from "./measure-usage"
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

const USAGE =
  "usage: bun run script/reference-workload.ts --bin <cmd> --run-dir <dir> --model <provider/model> [--proxy] [--stub]"

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

type DriverArgs = { bin: string; runDir: string; model: string; proxy: boolean; stub: boolean }

const parseArgs = (argv: string[]): DriverArgs => {
  const flags = { bin: "", runDir: "", model: "", proxy: false, stub: false }
  const valueFlag = (arg: string): "bin" | "runDir" | "model" | undefined =>
    arg === "--bin" ? "bin" : arg === "--run-dir" ? "runDir" : arg === "--model" ? "model" : undefined
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
  if (!flags.bin || !flags.runDir || !flags.model)
    throw new Error(`reference-workload: --bin, --run-dir and --model are required\n${USAGE}`)
  return { bin: flags.bin, runDir: flags.runDir, model: flags.model, proxy: flags.proxy, stub: flags.stub }
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

const prepareRunDir = async (input: { runDir: string; model: string; providerID: string; stub: boolean }) => {
  const entries = await fs.readdir(input.runDir).catch((error: NodeJS.ErrnoException) => {
    if (error.code === "ENOENT") return null
    throw error
  })
  if (entries && entries.length > 0)
    throw new Error(`reference-workload: run dir ${input.runDir} is not empty — refusing to overwrite an existing run`)
  const cwd = path.join(input.runDir, "cwd")
  await fs.cp(path.join(import.meta.dir, "reference-workload", "fixture-repo"), cwd, { recursive: true })
  const config = { $schema: "https://opencode.ai/config.json", model: input.model, agent: { build: { temperature: 0 } } }
  await Bun.write(path.join(cwd, "opencode.json"), JSON.stringify(config, null, 2) + "\n")
  const dataDir = path.join(input.runDir, "data", "opencode")
  await fs.mkdir(dataDir, { recursive: true })
  await Bun.write(path.join(dataDir, "auth.json"), JSON.stringify(await seedAuth(input), null, 2) + "\n")
}

const childEnv = (input: { runDir: string; proxy: boolean }): Record<string, string> => {
  if (!process.env.PATH) throw new Error("reference-workload: PATH is not set — cannot spawn the binary")
  // Fresh allowlisted env (xdg-basedir reads env at import time): operator-shell
  // XDG_*/OPENCODE_* must not leak into the measurement. Proxy reachability vars
  // pass through so both legs share identical network conditions.
  const env: Record<string, string> = {
    PATH: process.env.PATH,
    HOME: process.env.HOME ?? os.homedir(),
    XDG_DATA_HOME: path.join(input.runDir, "data"),
    OPENCODE_DB: path.join(input.runDir, "data", "opencode.db"),
    OPENCODE_ENABLE_PROMPT_CAPTURE: "1",
  }
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

const main = async (argv: string[]): Promise<number> => {
  const args = parseArgs(argv)
  const { providerID, modelID } = parseModel(args.model)
  const cmd = args.stub
    ? [process.execPath, path.join(import.meta.dir, "reference-workload", "stub-bin.ts")]
    : args.bin.trim().split(/\s+/)
  const runDir = path.resolve(args.runDir)
  const startedAt = new Date().toISOString()
  const promptsDigest = await digestWorkload()
  await prepareRunDir({ runDir, model: args.model, providerID, stub: args.stub })
  await runWorkload({ cmd, model: args.model, cwd: path.join(runDir, "cwd"), env: childEnv({ runDir, proxy: args.proxy }) })
  const manifest: RunManifest = {
    bin: args.stub ? "stub" : args.bin,
    modelID,
    providerID,
    promptsDigest,
    capture: true,
    proxyMode: args.proxy,
    dbPath: "data/opencode.db",
    capturesDir: "data/opencode/prompt-captures",
    startedAt,
    endedAt: new Date().toISOString(),
  }
  if (!Schema.is(RunManifestSchema)(manifest))
    throw new Error("reference-workload: assembled manifest does not match the RunManifest schema")
  await Bun.write(path.join(runDir, "manifest.json"), JSON.stringify(manifest, null, 2) + "\n")
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