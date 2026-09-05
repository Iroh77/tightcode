# Divergence registry — TightCode vs `anomalyco/opencode`

Every file **modified** relative to upstream, with why + last merged upstream commit. Files not listed here must stay byte-identical to upstream (`01 brownfield-conventions.md`).

Last upstream merge: `bbd72fb8b` (dev, scaffold, 2026-09-05).

| File | Why it diverges | Last merged upstream commit |
|---|---|---|
| `AGENTS.md` | TightCode router section appended (upstream contributor conventions kept above) | `bbd72fb8b` |
| `README.md` | Fork README (upstream README versions remain in `README.*.md` translations) | `bbd72fb8b` |
| `.gitignore` | `Second Cerveau` symlink line appended | `bbd72fb8b` |

## TightCode-only additive files (do not exist upstream — safe at merge)

`opencode.json`, `CHANGELOG.md`, `docs/code-map.md`, `docs/divergence.md`, `docs/agents/*`, `scripts/verify.sh`, `.opencode/agent/researcher-code.md`.