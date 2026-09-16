import path from "path"

process.env.OPENCODE_DB = ":memory:"
process.env.NPM_CONFIG_AUDIT = "false"
process.env.OPENCODE_MODELS_PATH = path.join(import.meta.dir, "plugin", "fixtures", "models-dev.json")
process.env.OPENCODE_DISABLE_MODELS_FETCH = "true"
// Hermetic client identity (R00-015 environment parity): models-dev captures
// USER_AGENT at import, so a desktop-launched session env (OPENCODE_CLIENT=desktop)
// would leak into it. "cli" is the flag's own default (flag.ts).
process.env.OPENCODE_CLIENT = "cli"
