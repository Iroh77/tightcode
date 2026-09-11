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