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
