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
`XDG_*`/`OPENCODE_*` cannot leak into a leg. A fixed credential list
(`OPENROUTER_API_KEY`, `FIRECRAWL_API_KEY`, `FAL_AI_API_KEY`, `N8N_MCP_TOKEN`)
passes through when set in the operator shell — identical for both legs, never
written to the run config, the manifest, or the repo. Prompts chain into one
session via `opencode run --continue` in the fresh data dir. The operator's
global config at `$HOME/.config/opencode` is still read by both legs — keep it
neutral, and run both legs from the same shell.

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
## Campaign mode (R13-002/003/006)

For robust fork-vs-upstream comparisons — interleaved runs, medians, verdict
isolation — drive a whole campaign from one spec file instead of single pairs:

```
bun run script/reference-workload.ts --campaign <spec.json> --out <dir> [--stub]
```

A **phase** is a workload variant (`name` + `model` + optional `mcp` config
fragment merged into the generated `opencode.json`); a **leg** is a shape
(`fork`/`upstream`, optional `proxy: true` attribution leg). Spec example:

```json
{
  "phases": [
    { "name": "base", "model": "<provider/model>" },
    {
      "name": "mcp",
      "model": "<provider/model>",
      "mcp": { "firecrawl": { "type": "remote", "url": "https://mcp.firecrawl.dev", "headers": { "Authorization": "Bearer {env:FIRECRAWL_API_KEY}" } } }
    }
  ],
  "legs": [
    { "shape": "fork", "bin": "<fork bin>" },
    { "shape": "upstream", "bin": "opencode" },
    { "shape": "fork", "bin": "<fork bin>", "proxy": true }
  ],
  "runs": 3,
  "seed": 42,
  "pin": "binding"
}
```

- `runs` — N per (phase × leg), default 3, minimum 2 (never single runs; N=1
  is refused at validation).
- `seed` — optional. When omitted a random 32-bit seed is drawn and recorded
  in `campaign.json`; the schedule (per phase, legs × N Fisher–Yates-shuffled,
  executed strictly sequentially) reproduces exactly from the recorded seed.
  Sequential execution is deliberate: interleaving neutralizes run-order
  confounds such as account-level cache warmth.
- `pin` — optional `"binding" | "advisory"`, set as
  `OPENCODE_PIN_BINDING_VERDICT` on every leg (upstream ignores it). Pin when
  comparing payloads so the run does not measure the verdict probe's
  cache/table state instead; unpinned legs report whatever the probe actually
  resolved.

Out-dir layout: `campaign.json` (spec echo + seed + per-run schedule/status,
written progressively and aborted on the first failure — a half-run leg would
poison its medians) plus one v1-layout run dir per run
(`<NN>-<phase>-<shape>[-proxy]/`). Capture env is set on fork-shape legs only:
real upstream binaries ignore the flag either way, and the stub obeys it, so
the upstream capture-lessness is simulated faithfully offline.

Report:

```
bun run script/measure-usage.ts campaign <dir> [-o out.json]
```

- **Medians, never single runs.** Per leg and phase: cold-start input (first
  provider turn — cache-cold by construction, the primary volume metric),
  session input (Σ per turn of `input + cacheRead + cacheWrite` — re-billing
  and account warmth move tokens between columns, never out of the sum), and
  payload chars (capture-bearing fork legs). Each column is a median with
  dispersion (`median`/`min`/`max` + the raw per-run `values`).
- **Verdict column + flip semantics (R13-003).** A comparison pair is
  `comparable` only when the fork side's runs record exactly one identical
  verdict (each manifest's `verdicts`, derived from the captures' per-turn
  `meta.verdict`). Upstream legs (no verdict machinery) and fork-proxy legs
  (kill-switched — the verdict axis is inert) are exempt: their column is null
  by design. A flip across runs, a mixed run (≥2 verdicts), or an
  unrecordable fork verdict surfaces as `comparable: false` with a warning
  naming the run dirs — medians stay emitted, labeled incomparable, never
  silently folded into a delta.
- **Binary identity (R13-005).** Every manifest records what produced the run
  (`<bin> --version` — fork commit or upstream release; `"stub"` in stub
  mode), so post-hoc analysis can never confuse which shape generated a dump.
- **Cache-collapse flags** in the per-leg diagnostics are a heuristic
  (`cacheRead < 0.5 × previous turn's total input`) — diagnostic only,
  expected to fire after deliberate prompt-base append batches; never a
  headline metric.

Campaign mode shares the single-run isolation rules (allowlisted child env,
credential passthrough, fresh data dir) — run all legs from the same shell.
Real-model campaigns stay offline from CI (credentials): run them ad hoc from
an operator shell, exactly like the single-pair procedure above, which remains
the quick-check path.

## Prompt authority (R13-004)

The prompt list (`prompts.ts`) consists of owner-authored committed literals —
never runtime-generated or AI-regenerated per session. Editing a prompt (or
the fixture repo) is an explicit repo change: it changes the workload digest
pinned in every `manifest.json`, thereby invalidating all recorded runs
compared against older digests — that is the mechanism working by design, not
an accident to avoid. Prompts that mutate the fixture must stay
self-contained: effects visible only within their own turn, so later prompts'
context remains structurally comparable across legs.
