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

const MCP_INSTRUCTIONS_BOUND = 250

// R11-002: keep the "what is this server for" signal, drop boilerplate —
// word-boundary cut with the "..." inside the budget (unlike tool-listing's
// truncate100, whose ellipsis sits outside its bound). The 248-unit head
// leaves room for a word cut + ellipsis within the 250-unit budget.
export function mcpInstructions(text: string): string {
  if (text.length <= MCP_INSTRUCTIONS_BOUND) return text
  const head = text.slice(0, MCP_INSTRUCTIONS_BOUND - 2)
  const cut = head.lastIndexOf(" ")
  const base = cut > 0 ? head.slice(0, cut) : text.slice(0, MCP_INSTRUCTIONS_BOUND - 3)
  return base + "..."
}

export * as ContextSlimmer from "./context-slimmer"
