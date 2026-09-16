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

// R12-012 Amendment 1: a deferred_tool part's renderer input is the inner
// tool's parsed arguments, not the wrapper envelope. Returns the parsed args
// object iff input is the wrapper envelope (name string + args string whose
// JSON parses to a non-array object); undefined otherwise. Callers gate on
// part.tool === "deferred_tool" before calling.
export function deferredToolInput(input: Record<string, unknown> | undefined): Record<string, any> | undefined {
  if (!input || typeof input.name !== "string" || typeof input.args !== "string") return undefined
  try {
    const parsed: unknown = JSON.parse(input.args)
    if (!parsed || typeof parsed !== "object" || Array.isArray(parsed)) return undefined
    return parsed as Record<string, any>
  } catch {
    return undefined
  }
}
