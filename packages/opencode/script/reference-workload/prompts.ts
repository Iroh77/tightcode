// Fixed reference-workload prompt list (R10-004): read-only prompts over the
// committed fixture repo, executed identically on upstream and fork. The
// workload identity (prompts + fixture content) is digested by the driver and
// pinned in RunManifest.promptsDigest — measure-usage diff refuses runs with
// differing digests. Changing either input invalidates recorded baseline runs.

export const prompts = [
  "List every file in this repository and summarize the project in one sentence.",
  "Read src/math.ts and explain what the fib function does.",
  "Read README.md and report the exact version number mentioned in it.",
  "Which file defines the THEME constant, and what are its keys?",
]