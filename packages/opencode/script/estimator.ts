import { readFileSync } from "fs"
import path from "path"
import { Tiktoken } from "js-tiktoken"
import o200kBase from "js-tiktoken/ranks/o200k_base"
import { Tokenizer } from "@huggingface/tokenizers"

// R10-007: real GLM/DeepSeek tokenizers by model-id prefix, js-tiktoken o200k as
// reference. Assets are committed and hash-pinned (assets/pins.json); loaded
// lazily at first estimate, memoized per process (SC-5).

export type AdapterKind = "glm" | "deepseek" | "o200k" | "o200k-fallback"

export type Estimator = {
  readonly modelID: string
  readonly adapter: AdapterKind
  estimate(text: string): number
}

type Pins = Record<"glm" | "deepseek", { sha256: string; source: string; revision: string; bytes: number }>

type GptByteLevelAsset = {
  added_tokens?: Array<{ content: string; special?: boolean }>
  pre_tokenizer?: { pretokenizers?: Array<{ pattern?: { Regex?: string } }> }
  model: { vocab: Record<string, number> }
}

type HfAddedTokens = { added_tokens?: Array<{ special?: boolean }> }

const defaultAssetsDir = path.join(import.meta.dir, "estimator", "assets")

// GPT-2 bytes_to_unicode, inverted: each byte-unicode char of a ByteLevel vocab
// token maps back to the raw byte the rank table must key on.
const byteDecoder = (() => {
  const printable: number[] = []
  for (let b = 0; b < 256; b++) if ((b >= 33 && b <= 126) || (b >= 161 && b <= 172) || (b >= 174 && b <= 255)) printable.push(b)
  let n = 0
  const enc = Array.from({ length: 256 }, () => 0)
  for (const b of printable) enc[b] = b
  for (let b = 0; b < 256; b++) if (!printable.includes(b)) enc[b] = 256 + n++
  return new Map<string, number>(enc.map((code, byte) => [String.fromCodePoint(code), byte]))
})()

let o200k: Tiktoken | undefined
const o200kEncode = (): ((text: string) => number) => {
  if (!o200k) {
    // Special tokens are never added (payload content only): strip the map so
    // encode never throws on special-token-shaped text.
    o200k = new Tiktoken({ ...o200kBase, special_tokens: {} }, {})
  }
  return (text) => o200k!.encode(text).length
}

const sha256Hex = (data: Buffer): string => {
  const hasher = new Bun.CryptoHasher("sha256")
  hasher.update(data)
  return hasher.digest("hex")
}

const loadPins = (assetsDir: string): Pins => JSON.parse(readFileSync(path.join(assetsDir, "pins.json"), "utf8"))

// Preferred adapter construction (proven against the HF python oracle): the HF
// BPE vocab (byte-unicode space) inverted back to raw bytes becomes the rank
// table, and the asset's own pre-tokenizer regex drives the split — same
// algorithm js-tiktoken runs for o200k. Limitation: non-special added tokens
// (e.g. "<think>") are not rank-representable here — a regex piece can only
// match them when it is a single punctuation run — so they count as BPE pieces;
// the native DeepSeek path matches them exactly.
const glmRanks = (asset: GptByteLevelAsset) => {
  const special = new Set(asset.added_tokens?.filter((t) => t.special).map((t) => t.content) ?? [])
  const tokens = Object.entries(asset.model.vocab)
    .filter(([token]) => !special.has(token))
    .sort((a, b) => a[1] - b[1])
    .map(([token]) => Buffer.from(Array.from(token, (code) => byteDecoder.get(code)!)).toString("base64"))
  const pattern = asset.pre_tokenizer?.pretokenizers?.find((p) => p.pattern?.Regex)?.pattern?.Regex
  if (!pattern) throw new Error("glm tokenizer.json has no pre-tokenizer regex")
  return { pat_str: pattern, special_tokens: {}, bpe_ranks: `x 0 ${tokens.join(" ")}` }
}

// Fallback variant (named by context-observability-03): the composed multi-regex
// pre-tokenizer cannot be expressed by js-tiktoken's single patStr, so DeepSeek
// runs on the native bindings. Special tokens are never added (payload content
// only); non-special added tokens stay — they are part of the real vocabulary.
const deepseekTokenizer = (asset: HfAddedTokens) =>
  new Tokenizer({ ...asset, added_tokens: asset.added_tokens?.filter((t) => !t.special) }, {})

const checkPin = (name: "glm" | "deepseek", assetsDir: string, prefix = "asset"): void => {
  const bytes = readFileSync(path.join(assetsDir, name, "tokenizer.json"))
  const pin = loadPins(assetsDir)[name]
  if (bytes.length !== pin.bytes) throw new Error(`${prefix} ${name}: size mismatch (${bytes.length} != ${pin.bytes})`)
  const hash = sha256Hex(bytes)
  if (hash !== pin.sha256) throw new Error(`${prefix} ${name}: hash mismatch (${hash} != ${pin.sha256})`)
}

const loadReal = (name: "glm" | "deepseek", assetsDir: string): { adapter: AdapterKind; encode: (text: string) => number } => {
  checkPin(name, assetsDir)
  const bytes = readFileSync(path.join(assetsDir, name, "tokenizer.json"))
  if (name === "glm") {
    const tiktoken = new Tiktoken(glmRanks(JSON.parse(bytes.toString("utf8"))), {})
    return { adapter: "glm", encode: (text) => tiktoken.encode(text).length }
  }
  const tokenizer = deepseekTokenizer(JSON.parse(bytes.toString("utf8")))
  return { adapter: "deepseek", encode: (text) => tokenizer.encode(text).ids.length }
}

const fallback = (name: string, error: unknown): { adapter: AdapterKind; encode: (text: string) => number } => {
  console.error(`estimator: ${name} tokenizer asset unavailable, falling back to o200k:`, error instanceof Error ? error.message : error)
  return { adapter: "o200k-fallback", encode: o200kEncode() }
}

const memo = new Map<string, Estimator>()

export const forModel = (modelID: string, opts?: { assetsDir?: string }): Estimator => {
  const assetsDir = opts?.assetsDir ?? defaultAssetsDir
  const key = `${assetsDir}\u0000${modelID}`
  const hit = memo.get(key)
  if (hit) return hit
  const segment = (modelID.split("/").pop() ?? modelID).toLowerCase()
  const kind: "glm" | "deepseek" | "o200k" = segment.startsWith("glm") ? "glm" : segment.startsWith("deepseek") ? "deepseek" : "o200k"
  let state: { adapter: AdapterKind; encode: (text: string) => number } | undefined
  const estimator: Estimator = {
    modelID,
    // Pre-estimate: the selected kind (prediction). Post-estimate: the pinned
    // fact — a fallback downgrades it, so consumers asserting engagement must
    // read it after the first estimate (R10-008).
    get adapter() {
      return state?.adapter ?? kind
    },
    estimate(text: string) {
      if (!state) {
        state =
          kind === "o200k"
            ? { adapter: "o200k", encode: o200kEncode() }
            : (() => {
                try {
                  return loadReal(kind, assetsDir)
                } catch (error) {
                  return fallback(kind, error)
                }
              })()
      }
      return state.encode(text)
    },
  }
  memo.set(key, estimator)
  return estimator
}

export const verifyAssets = (assetsDir: string = defaultAssetsDir): void => {
  for (const name of ["glm", "deepseek"] as const) checkPin(name, assetsDir, "estimator asset")
}

// --- Rendered-format parsing (SC-5 owner; R10-006) ---
// Sole owner of the rendered system-prompt format: parseSystemBlocks segments
// the joined system text by the producer anchors in canonical SC-2 order;
// parseSkillsListing splits a skills segment into headers + per-skill items in
// both Skill.fmt modes. Render-format drift fails in the parser tests, which
// pin committed fixture strings cut from the real renders.

export type SystemSegment =
  | { label: "base" | "environment" | "instructions" | "skills" | "structured_output" | "user_system" | "other"; text: string }
  | { label: `mcp:${string}`; server: string; text: string }
  | { label: "mcpInstructionsTags"; text: string } // the <mcp_instructions> wrapper lines
  | { label: "catalog" | `catalog:${string}`; text: string } // per <deferred_tools…> block
  | { label: "skillsItem"; name: string; text: string } // derived via parseSkillsListing, not emitted here

type RegionKind =
  | { kind: "base" | "environment" | "instructions" | "skills" | "structured_output" | "other" }
  | { kind: "mcp"; server: string }
  | { kind: "tags" }
  | { kind: "catalog"; server: string | undefined }

const ENV_ANCHOR = "You are powered by the model named "
const INSTRUCTIONS_ANCHOR = "Instructions from: "
const SKILLS_ANCHOR = "Skills provide specialized instructions and workflows for specific tasks."
const STRUCTURED_ANCHOR = "IMPORTANT: The user has requested structured output."
// SC-2: the environment block spans the <env> block AND the references
// section; the references header re-anchors the region after </env>.
const REFERENCES_ANCHOR = "Project references provide additional directories that can be accessed when relevant."
const MCP_OPEN = "<mcp_instructions>"
const MCP_CLOSE = "</mcp_instructions>"
const CATALOG_CLOSE = "</deferred_tools>"
const CATALOG_OPEN = /^<deferred_tools(?: server="([^"]*)")?>$/
const SERVER_SECTION = /^ {2}<server name="([^"]*)">$/

// Byte spans of each line: line i occupies [starts[i], starts[i+1] or end).
const lineSpans = (text: string) => {
  const lines = text.split("\n")
  const starts: number[] = []
  let offset = 0
  for (const line of lines) {
    starts.push(offset)
    offset += line.length + 1
  }
  return { lines, starts, endOf: (line: number) => (line + 1 < lines.length ? starts[line + 1] : text.length) }
}

// Anchors are line-initial (SC-2 render); an anchor-like line inside block
// content may shift labels but never totals — documented limitation.
export const parseSystemBlocks = (system: string[]): { segments: SystemSegment[]; text: string } => {
  const text = system.join("\n")
  const { lines, starts, endOf } = lineSpans(text)

  const segments: SystemSegment[] = []
  let open: RegionKind & { start: number } = { kind: "base", start: 0 }

  const close = (end: number, final = false) => {
    if (open.start >= end) return
    const slice = text.slice(open.start, end)
    if (open.kind === "mcp") segments.push({ label: `mcp:${open.server}`, server: open.server, text: slice })
    else if (open.kind === "tags") segments.push({ label: "mcpInstructionsTags", text: slice })
    else if (open.kind === "catalog")
      segments.push({ label: open.server === undefined ? "catalog" : `catalog:${open.server}`, text: slice })
    else if (open.kind === "other") segments.push({ label: final ? "user_system" : "other", text: slice })
    else segments.push({ label: open.kind, text: slice })
  }
  // End-delimited regions hand over to a pending gap: text after it is other
  // (mid-text) or user_system (trailing) — user_system only at text end.
  const closeToGap = (end: number) => {
    close(end)
    open = { kind: "other", start: end }
  }
  const reopen = (region: RegionKind, start: number) => {
    close(start)
    open = { ...region, start }
  }
  // Annotated return type keeps the full union: direct open.kind reads get
  // control-flow-narrowed across the else-if chain and TS2367 fires.
  const regionKind = (): RegionKind["kind"] => open.kind

  for (let i = 0; i < lines.length; i++) {
    const line = lines[i]
    if (line.startsWith(ENV_ANCHOR)) reopen({ kind: "environment" }, starts[i])
    else if (line === REFERENCES_ANCHOR) reopen({ kind: "environment" }, starts[i])
    else if (regionKind() === "environment" && line === "</env>") closeToGap(endOf(i))
    else if (line.startsWith(INSTRUCTIONS_ANCHOR)) reopen({ kind: "instructions" }, starts[i])
    else if (line === SKILLS_ANCHOR) reopen({ kind: "skills" }, starts[i])
    else if (line.startsWith(STRUCTURED_ANCHOR)) {
      // Single-line block: the anchor line is the whole region.
      reopen({ kind: "structured_output" }, starts[i])
      closeToGap(endOf(i))
    } else if (CATALOG_OPEN.test(line)) reopen({ kind: "catalog", server: CATALOG_OPEN.exec(line)?.[1] }, starts[i])
    else if (line === MCP_OPEN) reopen({ kind: "tags" }, starts[i])
    else if ((regionKind() === "tags" || regionKind() === "mcp") && SERVER_SECTION.test(line))
      reopen({ kind: "mcp", server: SERVER_SECTION.exec(line)![1] }, starts[i])
    else if ((regionKind() === "tags" || regionKind() === "mcp") && line === MCP_CLOSE) reopen({ kind: "tags" }, starts[i])
    else if (regionKind() === "catalog" && line === CATALOG_CLOSE) closeToGap(endOf(i))
  }
  close(text.length, true)
  return { segments, text }
}

export const parseSkillsListing = (skillsText: string): { headers: string; items: Array<{ name: string; text: string }> } => {
  const { lines, starts, endOf } = lineSpans(skillsText)

  // Item line ranges, non-verbose "- **name**: …" bullets and verbose
  // "  <skill>" … "  </skill>" blocks (both Skill.fmt modes).
  const ranges: Array<{ first: number; last: number; name: string }> = []
  for (let i = 0; i < lines.length; i++) {
    const bullet = /^- \*\*(.+?)\*\*:/.exec(lines[i])
    if (bullet) {
      ranges.push({ first: i, last: i, name: bullet[1] })
      continue
    }
    if (lines[i] === "  <skill>") {
      let name = ""
      let last = i
      while (last < lines.length) {
        const nameMatch = /^ {4}<name>(.*)<\/name>$/.exec(lines[last])
        if (nameMatch) name = nameMatch[1]
        if (lines[last] === "  </skill>") break
        last++
      }
      ranges.push({ first: i, last: Math.min(last, lines.length - 1), name })
      i = last
    }
  }

  // Each item spans to the next item's start; the last item ends at its own
  // final line, so a trailing wrapper tag (</available_skills>) stays in
  // headers. headers = everything outside the item spans.
  const items = ranges.map((range, index) => ({
    name: range.name,
    text: skillsText.slice(starts[range.first], index + 1 < ranges.length ? starts[ranges[index + 1].first] : endOf(range.last)),
  }))
  const head = ranges.length > 0 ? skillsText.slice(0, starts[ranges[0].first]) : skillsText
  const tail =
    ranges.length > 0 ? skillsText.slice(endOf(ranges.at(-1)!.last)) : ""
  return { headers: head + tail, items }
}

export * as Estimator from "./estimator"

const main = async (argv: string[]): Promise<number> => {
  const [cmd] = argv
  if (cmd === "verify") {
    verifyAssets()
    console.log("estimator assets verified")
    return 0
  }
  console.error("usage: bun run script/estimator.ts verify")
  return 1
}

if (import.meta.main) {
  try {
    process.exit(await main(process.argv.slice(2)))
  } catch (error) {
    console.error(error instanceof Error ? error.message : error)
    process.exit(1)
  }
}
