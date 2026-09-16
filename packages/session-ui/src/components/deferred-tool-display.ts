// R12-012 presentation unwrap: a deferred_tool part attributes to the inner
// tool via the unwrap metadata key (state.metadata.deferred_tool.tool).
// Pending parts carry no metadata yet — they keep the wrapper name until the
// running update arrives.
export function deferredToolName(tool: string, metadata: Record<string, unknown> | undefined): string {
  if (tool !== "deferred_tool") return tool
  const key = metadata?.deferred_tool
  if (!key || typeof key !== "object" || Array.isArray(key)) return tool
  const inner = (key as Record<string, unknown>).tool
  return typeof inner === "string" && inner ? inner : tool
}

// R12-012 Amendment 1: a deferred_tool part's renderer input must be the
// inner tool's arguments. Two stored shapes exist (researched on the shared
// DB, 2026-09-16): the wrapper envelope {name, args} on pending/live parts,
// and — after the processor's completion rebuild — the already-parsed inner
// args object directly. Returns the inner args object: envelope args parse
// through JSON.parse; a non-envelope object passes through as-is; anything
// else (missing, array, primitive, envelope with unparseable args) returns
// undefined — the caller degrades to the generic renderer. Callers gate on
// part.tool === "deferred_tool" before calling. Accepted edge: an inner tool
// whose own schema is {name: string, args: string} with object-parseable args
// is indistinguishable from the envelope — the parsed args win.
export function deferredToolInput(input: Record<string, unknown> | undefined): Record<string, any> | undefined {
  if (!input || typeof input !== "object" || Array.isArray(input)) return undefined
  if (typeof input.name === "string" && typeof input.args === "string") {
    try {
      const parsed: unknown = JSON.parse(input.args)
      if (!parsed || typeof parsed !== "object" || Array.isArray(parsed)) return undefined
      return parsed as Record<string, any>
    } catch {
      return undefined
    }
  }
  return input as Record<string, any>
}
