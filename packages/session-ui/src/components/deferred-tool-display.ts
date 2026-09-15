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
