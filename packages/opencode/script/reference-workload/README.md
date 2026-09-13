# Reference workload (R10-004)

Scripted fork-vs-upstream comparison: the fixed prompt list (`prompts.ts`) runs
identically on the upstream CLI and on the fork over the committed fixture repo
(`fixture-repo/`), once per run dir; `script/measure-usage.ts` then reports and
diffs the two runs' usage (provider-reported totals are the comparison
currency — decision `context-observability-02` in the vault).

## Usage

```
bun run script/reference-workload.ts --bin <cmd> --run-dir <dir> --model <provider/model> [--proxy] [--stub]
```

- `--bin <cmd>` — an opencode-shaped CLI, whitespace-split into argv (no quoted
  arguments). Examples: `opencode` (installed upstream CLI), a compiled fork
  binary, or `bun /abs/path/packages/opencode/src/index.ts` (fork dev entry).
- `--run-dir <dir>` — must not exist or be empty; the run is fully contained
  inside: `cwd/` (fixture copy + generated `opencode.json`), `data/`
  (`XDG_DATA_HOME`), `manifest.json`, and — in `--stub` mode — `stub-env.json`.
- `--model <provider/model>` — pinned in the generated `opencode.json`
  (`agent.build.temperature: 0`) and passed to `opencode run` as
  `--model <model> --agent build`, so the pinned temperature applies to the
  default `build` agent; its provider entry from the operator's `auth.json` is
  copied into the run dir.
- `--proxy` — fork attribution run: adds `OPENCODE_DISABLE_LAZY_TOOLS=1` +
  `OPENCODE_DISABLE_STATIC_SLIMMING=1` (upstream-shaped payload per axis).
  Serves attribution only; totals comparisons always use the real upstream run.
- `--stub` — replaces the spawn target with the in-repo echo stub
  (`stub-bin.ts`): mechanics verification without a provider; no network.

`manifest.json` carries the workload identity digest (fixed prompts + fixture
repo content) — `measure-usage.ts diff` refuses runs whose digests differ.

Run dir layout:

```
<run-dir>/
├── manifest.json                                  # RunManifest (schema in script/measure-usage.ts)
├── stub-env.json                                  # --stub only: OPENCODE_* flags the stub observed
├── cwd/                                           # spawn cwd: fixture copy + opencode.json (model, temperature 0)
└── data/                                          # XDG_DATA_HOME
    ├── opencode.db                                # OPENCODE_DB — usage source (step-finish parts)
    └── opencode/
        ├── auth.json                              # seeded: {provider: entry}
        └── prompt-captures/<sessionID>/NNNN.json  # fork runs only (upstream ignores the env)
```

Isolation notes: the subprocess env is an allowlist (`PATH`, `HOME`, the
driver-set `XDG_*`/`OPENCODE_*` values, proxy vars) so operator-shell
`XDG_*`/`OPENCODE_*` cannot leak into a leg; prompts chain into one session via
`opencode run --continue` in the fresh data dir. The operator's global config
at `$HOME/.config/opencode` is still read by both legs — keep it neutral, and
run both legs from the same shell.

## Baseline procedure (first R10-005 measurement)

1. Pick the pinned model `<provider/model>` and make sure the operator
   `auth.json` has an entry for `<provider>`.
2. Upstream leg (installed upstream CLI):
   ```
   bun run script/reference-workload.ts --bin opencode --run-dir runs/upstream --model <provider/model>
   ```
3. Fork leg (same shell; either the compiled binary or the dev entry):
   ```
   bun run script/reference-workload.ts --bin "<fork bin>" --run-dir runs/fork --model <provider/model>
   ```
4. Diff (totals currency; `promptsDigest` must match):
   ```
   bun run script/measure-usage.ts diff runs/fork runs/upstream -o diff.json
   ```
5. Optional attribution run on the fork with upstream-shaped payloads:
   ```
   bun run script/reference-workload.ts --bin "<fork bin>" --run-dir runs/fork-proxy --model <provider/model> --proxy
   bun run script/measure-usage.ts report runs/fork-proxy
   ```
6. Mechanics check without a provider (any time): add `--stub` to steps 2-3 and
   run `report`/`diff` over the produced dirs.

Record the run dirs and the diff output with the R10-005 amendment. Residual
provider nondeterminism (even at temperature 0) shows up as small per-turn
deltas — that is inherent and expected in the diff.