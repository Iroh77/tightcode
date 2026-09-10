# Changelog

All notable changes are documented here. Format: [Keep a Changelog](https://keepachangelog.com).

Releases are tagged `vX.Y.Z` (git tag); sections are dated: `## [x.y.z] - YYYY-MM-DD`.

## [Unreleased]

Scaffold — fork of `anomalyco/opencode@bbd72fb8b` (dev) as TightCode.

### Added (tool lazy loading — R12)

- `feat(core-lazy)`: prompt-base freeze skeleton (ticket 01) — tagged `SystemBlock[]` assembled by the run loop; `PromptBase.Service` freezes system blocks per (session, provider/model/endpoint) with batched append-only reveals; frozen blocks rendered byte-identically to upstream at request prep; `small` turns bypass; `OPENCODE_DISABLE_LAZY_TOOLS` kill-switch restores upstream per-turn behavior (R12-008 partial, R00-013 axis flag).