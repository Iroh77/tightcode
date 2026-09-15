export function webSearchProviderLabel(provider: unknown) {
  if (provider === "parallel") return "Parallel Web Search"
  if (provider === "exa") return "Exa Web Search"
  return "Web Search"
}

export function toolDisplayMetadata(state: unknown): Record<string, unknown> {
  if (!state || typeof state !== "object" || Array.isArray(state)) return {}
  if (!("status" in state) || state.status === "pending") return {}
  if (!("structured" in state) || !state.structured || typeof state.structured !== "object") return {}
  if (Array.isArray(state.structured)) return {}
  return state.structured as Record<string, unknown>
}

// R12-012 presentation unwrap: a deferred_tool part attributes to the inner
// tool via the unwrap metadata key. Pending parts carry no metadata yet —
// they keep the wrapper name until the running update arrives.
export function deferredToolName(tool: string, metadata: Record<string, unknown> | undefined): string {
  if (tool !== "deferred_tool") return tool
  const key = metadata?.deferred_tool
  if (!key || typeof key !== "object" || Array.isArray(key)) return tool
  const inner = (key as Record<string, unknown>).tool
  return typeof inner === "string" && inner ? inner : tool
}
