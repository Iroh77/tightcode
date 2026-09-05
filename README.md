# TightCode

A leaner, more stable fork of [OpenCode](https://opencode.ai) (`anomalyco/opencode`) focused on **reducing baseline token usage** and improving **LLM compatibility** — while keeping the OpenCode Desktop experience.

## Why

The reference third-party plugin (`opencode-lazy-loading`) intercepts tool schemas at the plugin layer: it saves on tools only, depends on the model driving a `load_tool()` indirection, and resets every message. TightCode targets the **core context-building** (system prompt, agents, skills, instructions, tool schemas, compaction) so savings are larger, more stable across models, and shared by both the CLI and Desktop.

## Status

Early — scaffold of `anomalyco/opencode@bbd72fb8b` (`dev`). No releases yet.

## Development

- Upstream Bun workspace: `bun install`, upstream scripts (`bun run lint`, `bun run typecheck`, tests per package).
- Verification gate: `./scripts/verify.sh` — typecheck + lint, tests per touched package (`./scripts/verify.sh <package>`).
- **Divergence model:** see `docs/divergence.md`. Never edit a file that isn't registered there — it must stay byte-identical to upstream so merges stay cheap.
- Upstream merges: merge `upstream/dev` on a regular cadence; run the gate before and after; update the registry.

## License

MIT (upstream, kept).