# Changelog

All notable changes are documented here. Format: [Keep a Changelog](https://keepachangelog.com).

Releases are tagged `vX.Y.Z` (git tag); sections are dated: `## [x.y.z] - YYYY-MM-DD`.

## [Unreleased]

Scaffold — fork of `anomalyco/opencode@bbd72fb8b` (dev) as TightCode.

### Added (tool lazy loading — R12)

- `feat(core-lazy)`: prompt-base freeze skeleton (ticket 01) — tagged `SystemBlock[]` assembled by the run loop; `PromptBase.Service` freezes system blocks per (session, provider/model/endpoint) with batched append-only reveals; frozen blocks rendered byte-identically to upstream at request prep; `small` turns bypass; `OPENCODE_DISABLE_LAZY_TOOLS` kill-switch restores upstream per-turn behavior (R12-008 partial, R00-013 axis flag).
- `feat(core-lazy)`: tool-listing pure policy (ticket 02) — `ToolListing.shape` drops blanket-denied tools and classifies the eager set (`shell`/`read`/`load_tool`) vs deferred as full-fact seeds; `ToolListing.render` projects the listing view (truncate100 descriptions, constant placeholder schema under advisory, full schema under binding, MCP `"{server}: "` prefix once per group on the group's first-written entry); typed `MalformedToolEntryError` on malformed universe entries (R12-001/002/003/009).
- `feat(core-lazy)`: binding-verdict resolver (ticket 04) — `BindingVerdict.Service` resolves the schema-binding verdict per (provider, model, endpoint) before a session's first request: in-code static table → cross-session JSON cache (`Global.data/binding-verdicts.json`, SC-4 tolerant IO) → one adversarial probe (`record_answer`, `minimum: 100`, 10s timeout) → binding whenever nothing resolves; `observe` records schema-violation proof as sticky advisory for future sessions (R12-010 resolver half; wiring in ticket 05).
- `feat(core-lazy)`: binding mode in the listing (ticket 05) — `SessionTools.resolve` resolves the verdict (lazy branch only; kill-switch skips resolution entirely) and shapes the per-turn listing view with it; the verdict travels as `StreamInput.toolVerdict` into the frozen prompt base, whose mode freezes at first write so mid-session learning never re-renders written entries; binding writes every deferred tool's full input schema in place of the placeholder (schema-eager, description-deferred), advisory keeps the constant placeholder; missing verdict defaults conservative binding (R12-010 + R12-009 closed; upstream CLI/loop tests keep their scripted-reply choreography — the probe is answered off the books by the test LLM server, registered divergence).
- `feat(core-lazy)`: `load_tool` core built-in (ticket 06) — always eager, bulk-capable (dedupe, per-name output lines, never fatal); first load serves the full description + schema (advisory) or description only (binding, R12-010), repeats answer a short confirmation, and a load whose output left the model-visible history re-serves in full (`time.compacted` respected — R12-004 amendment 1); delivery markers ride the persisted tool-part metadata (`load_tool: {tools: [...]}`) so resume/compaction need no extra session state; unknown/eager names get clear per-name errors; kill-switch removes the registration (R12-004/005).
- `feat(core-lazy)`: direct-call fallback + behavioral learning (ticket 07) — deferred tools execute when called without a prior load; a typed execution failure of a not-yet-delivered tool appends the full schema to the error output and marks the part loaded (durable marker, running metadata preserved into the error state); permission rejections and aborts stay upstream; `InvalidArgumentsError` additionally feeds `BindingVerdict.observe` (binding→advisory, sticky, future sessions); one wrap point at the shaped-pass confluence covers built-in, resource and MCP wrappers uniformly with `processor.ts` byte-identical upstream (R12-006).

### Fixed (tool lazy loading — R12)

- `fix(core-lazy)`: eager set keyed to production registry ids (ticket 08 discovery, R12-002 amendment 1) — `ToolListing.EAGER` listed the shell tool as `"shell"` while the registry exposes it as `"bash"` (`ShellID.ToolID`, kept upstream for plugin/permission compatibility), so every real session payload deferred the shell tool the requirement makes eager; the set is now `bash`/`read`/`load_tool` and the seam fixtures use production ids.

### Added (tool lazy loading — R12, integration)

- `test(core-lazy)`: end-to-end integration test (ticket 08) — a real prompt loop against the scripted fixture provider proves the feature's core promise at the outgoing payload across module seams: system+tools prefix byte-stable across consecutive turns (R12-008), deferred entries stay truncated-description + placeholder even after the schema reached the model via history (R12-003/005), `load_tool` full serve → confirmations and an unloaded failing call recovering with the schema in its error output (R12-004/006), binding-mode session listing full schemas with description-only serves (R12-010).