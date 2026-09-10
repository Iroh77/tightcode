# Divergence registry — TightCode vs `anomalyco/opencode`

Every file **modified** relative to upstream, with why + last merged upstream commit. Files not listed here must stay byte-identical to upstream (`01 brownfield-conventions.md`).

**Permanent rule (upstream merges):** every upstream workflow file that emerges at merge is deleted unless it runs on a standard runner (`ubuntu-latest`) **without** secrets. Upstream's battery (publish/deploy/containers/bots/blacksmith) validates the upstream project, not this fork; the fork's own `ci.yml` + gate cover its claims.

Last upstream merge: `bbd72fb8b` (dev, scaffold, 2026-09-05).

| File | Why it diverges | Last merged upstream commit |
|---|---|---|
| `AGENTS.md` | TightCode router section appended (upstream contributor conventions kept above) | `bbd72fb8b` |
| `README.md` | Fork README (upstream localized `README.*.md` translations removed — they presented OpenCode's branding/badges and routed users to upstream; a single-maintainer fork keeps English-only) | `bbd72fb8b` |
| `README.*.md` (21 files) | Deleted wholesale (upstream translations: OpenCode branding, links/badges to upstream). New upstream translations at merge are deleted. | `bbd72fb8b` |
| `SECURITY.md` | Fork reporting flow: points to `Iroh77/tightcode` Security Advisory + upstream-routing note (threat model kept — still factually true). | `bbd72fb8b` |
| `.gitignore` | `Second Cerveau` symlink line appended | `bbd72fb8b` |
| `packages/session-ui/src/v2/components/prompt-input/index.tsx` | One character: `\200B` (legacy octal escape) → `\u200B`. Upstream lint (oxc) treats the octal as a parse error; TS (tsgo) tolerates it. Temporary divergence — the day upstream fixes it, this row's diff disappears. | `bbd72fb8b` |
| `.github/workflows/*` (26 files) | Deleted wholesale (see permanent rule above); `ci.yml` replaces them. Delete/modify conflicts at merge are resolved with `git rm` (30 s). | `bbd72fb8b` |
| `packages/opencode/src/effect/runtime-flags.ts` | R00-013: `disableLazyTools` flag (`OPENCODE_DISABLE_LAZY_TOOLS`) added (tool-lazy-loading kill-switch, SC-3). | `bbd72fb8b` |
| `packages/opencode/src/session/llm.ts` | R12-008: `StreamInput.system` fork-extended `string[]` → `SystemBlock[]` (decision tool-lazy-loading-01 §4); live layer yields/wires `PromptBase` into `LLMRequestPrep.prepare`. | `bbd72fb8b` |
| `packages/opencode/src/session/llm/request.ts` | R12-008/R12-007: `PrepareInput.system` → `SystemBlock[]` + `promptBase`; reconcile → render of frozen blocks before the upstream join; `small`/kill-switch bypass. | `bbd72fb8b` |
| `packages/opencode/src/session/prompt.ts` | R12-008/SC-2: run loop assembles tagged `SystemBlock[]` (per-server `mcp:<server>` via `SystemPrompt.mcpBlocks`); `structured_output` stays a per-turn block. | `bbd72fb8b` |
| `packages/opencode/src/session/system.ts` | SC-2: `mcpBlocks` added (per-server sections, permissibility filter); upstream `mcp` refactored onto the shared `serverSection` helper — output bytes unchanged (system.test.ts green). | `bbd72fb8b` |
| `packages/opencode/test/provider/transform.test.ts` | `promptBase` passthrough stub added to a `LLMRequestPrep.prepare` call (new required input). | `bbd72fb8b` |
| `packages/opencode/test/session/llm.test.ts` | `StreamInput.system` fixtures converted to `SystemBlock[]` (14 sites). | `bbd72fb8b` |
| `packages/opencode/test/session/llm-native-recorded.test.ts` | `StreamInput.system` fixture converted to `SystemBlock[]`. | `bbd72fb8b` |

## TightCode-only additive files (do not exist upstream — safe at merge)

`opencode.json`, `CHANGELOG.md`, `docs/code-map.md`, `docs/divergence.md`, `docs/agents/*`, `scripts/verify.sh`, `.opencode/agent/researcher-code.md`, `.github/workflows/ci.yml`, `packages/opencode/src/session/llm/prompt-base.ts`, `packages/opencode/test/session/prompt-base.test.ts`, `packages/opencode/test/session/llm-request-prep.test.ts`, `packages/opencode/src/session/tool-listing.ts`, `packages/opencode/test/session/tool-listing.test.ts`.