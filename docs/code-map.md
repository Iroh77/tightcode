# Code map — TightCode

Committed map of the real code structure. Updated when the code changes, never left stale (`00 main.md`).

Layout: upstream `anomalyco/opencode` Bun workspace — the structure IS upstream's (`packages/*`). This map records where TightCode-specific behavior lives; see `docs/divergence.md` for the file-level registry.

Design source: `Second Cerveau/1 PROJETS/tightcode/ARCHITECTURE/basic-design.md` (V1 runtime target; payload-first principle).

| Zone | Location | Notes |
|---|---|---|
| Static slimming | `packages/opencode/src/session/instruction.ts`, `packages/opencode/src/session/system.ts` | `ContextSlimmer` transforms edited in at producers: instructions root-wins, mcp_instructions truncation, skills non-verbose |
| Tool listing policy | `packages/opencode/src/session/tools.ts` (`SessionTools.resolve`: universe + seeds + listing-view AITool record), `packages/opencode/src/session/tool-listing.ts` (`ToolListing.shape`/`render`), `packages/opencode/src/permission/index.ts` | `ToolListing` (pure): permissible-only filter, eager/deferred seeds (full facts), listing view with truncate100 + placeholder schema (advisory) or full schema (binding), MCP prefix-once on the group's first-written entry. Wiring (ticket 03): `resolve` builds the universe per producer (source, sanitized MCP server) and returns `{ tools, seeds }`; kill-switch passes full upstream shapes through with empty seeds; binding-verdict wiring is ticket 05 |
| Prompt-base freeze | `packages/opencode/src/session/llm/prompt-base.ts` (`PromptBase`: `reconcileSystem`, `reconcileTools`, `render`), `packages/opencode/src/session/llm/request.ts` (`LLMRequestPrep.prepare` wiring: reconcile + impose projection at the payload + `small`/flag bypass), `packages/opencode/src/session/llm.ts` (`StreamInput.system`/`toolSeeds` fork-extension), `packages/opencode/src/session/prompt.ts` (tagged-block producer, seeds producer), `packages/opencode/src/session/system.ts` (`mcpBlocks` per-server sections) | R12-008/R12-007: first-write-wins per `(sessionID, provider/model/endpoint)` for system blocks AND tool entries, batched append-only reveal, frozen verdict mode, `structured_output` per-turn passthrough; impose projects frozen entries into the payload (names the per-turn record no longer produces stay listed with a clearly-failing execute); kill-switch `OPENCODE_DISABLE_LAZY_TOOLS` (`packages/opencode/src/effect/runtime-flags.ts`) |
| Binding verdict | `packages/opencode/src/session/binding-verdict.ts` (`BindingVerdict.resolve`/`observe`, `layerWith({ staticTable, probe })` seam) + XDG app data cache `binding-verdicts.json` | R12-010: static table → cross-session JSON cache (SC-4, tolerant IO) → adversarial probe (record_answer, 10s timeout) → binding on unresolvable; `observe` learns binding→advisory (sticky); memoized per (provider, model, endpoint) per instance; `classifyProbeResponse` pure parse seam |
| load_tool + fallback | `packages/opencode/src/tool/` (new tool file + registration), `packages/opencode/src/session/processor.ts` (error-path hook) | R12-004/005/006: load, re-serve after compaction, direct-call fallback |
| Prompt capture | `packages/opencode/src/session/llm/request.ts` (sink) + app data dumps | `PromptCapture`: exact payload per turn, credential-free, off by default |
| Kill-switches | `packages/opencode/src/effect/runtime-flags.ts` (pattern), per-axis `OPENCODE_*` flags | R00-013: flag set = upstream-identical per axis |
| Measurement | `scripts/` (committed dev scripts) | Usage report from capture dumps + usage records; reference workload comparison |
| (filled by later steps) | | Detailed-design and implementation zones append here |
