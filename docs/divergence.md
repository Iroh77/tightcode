# Divergence registry — TightCode vs `anomalyco/opencode`

Every file **modified** relative to upstream, with why + last merged upstream commit. Files not listed here must stay byte-identical to upstream (`01 brownfield-conventions.md`).

**Permanent rule (upstream merges):** every upstream workflow file that emerges at merge is deleted unless it runs on a standard runner (`ubuntu-latest`) **without** secrets. Upstream's battery (publish/deploy/containers/bots/blacksmith) validates the upstream project, not this fork; the fork's own `ci.yml` + gate cover its claims.

Last upstream merge: `bbd72fb8b` (dev, scaffold, 2026-09-05).

| File | Why it diverges | Last merged upstream commit |
|---|---|---|
| `AGENTS.md` | TightCode router section appended (upstream contributor conventions kept above) | `bbd72fb8b` |
| `README.md` | Fork README (upstream README versions remain in `README.*.md` translations) | `bbd72fb8b` |
| `.gitignore` | `Second Cerveau` symlink line appended | `bbd72fb8b` |
| `packages/session-ui/src/v2/components/prompt-input/index.tsx` | One character: `\200B` (legacy octal escape) → `\u200B`. Upstream lint (oxc) treats the octal as a parse error; TS (tsgo) tolerates it. Temporary divergence — the day upstream fixes it, this row's diff disappears. | `bbd72fb8b` |
| `.github/workflows/*` (26 files) | Deleted wholesale (see permanent rule above); `ci.yml` replaces them. Delete/modify conflicts at merge are resolved with `git rm` (30 s). | `bbd72fb8b` |

## TightCode-only additive files (do not exist upstream — safe at merge)

`opencode.json`, `CHANGELOG.md`, `docs/code-map.md`, `docs/divergence.md`, `docs/agents/*`, `scripts/verify.sh`, `.opencode/agent/researcher-code.md`, `.github/workflows/ci.yml`.