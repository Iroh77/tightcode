import type { JSONSchema7 } from "@ai-sdk/provider"
import DESCRIPTION from "./deferred_tool.txt"

// R12-012: the wrapper's single eager meta-tool, defined once with a closed,
// deployment-uniform schema (R12-009 — identical shape across provider
// families). The seed enters the listing through SessionTools.resolve on
// binding+wrapper sessions only; the dispatch executor and the AITool
// construction live at that confluence (ticket 19), so this file owns the
// frozen definition facts the seed and the catalog consumers read.
export const DEFERRED_TOOL_SCHEMA: JSONSchema7 = {
  type: "object",
  properties: {
    name: { type: "string", description: "Deferred tool name exactly as listed in the <deferred_tools> catalog" },
    args: { type: "string", description: "JSON string holding the tool's arguments object" },
  },
  required: ["name", "args"],
  additionalProperties: false,
}

export const DEFERRED_TOOL_DESCRIPTION = DESCRIPTION
