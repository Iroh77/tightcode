import path from "path"

// R11-001: with an instruction file at the worktree root, ancestor matches
// strictly between the session cwd and the worktree root add no signal — keep
// the cwd-level match (nearest signal wins) and the worktree-root match, drop
// the rest. Input is findUp's nearest-first output. No worktree-root file ⇒
// upstream stacking unchanged.
export function rootWins(input: { matches: string[]; file: string; directory: string; worktree: string }): string[] {
  const last = input.matches.at(-1)
  if (!last || path.resolve(last) !== path.resolve(path.join(input.worktree, input.file))) return input.matches
  const first = input.matches[0]
  if (
    first &&
    path.resolve(first) !== path.resolve(last) &&
    path.resolve(first) === path.resolve(path.join(input.directory, input.file))
  ) {
    return [first, last]
  }
  return [last]
}

export * as ContextSlimmer from "./context-slimmer"
