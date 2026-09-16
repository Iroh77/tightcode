#!/usr/bin/env bash
# TightCode local verification gate — brownfield: wraps the UPSTREAM toolchain.
# Green before a ticket closes (00 main.md).
set -euo pipefail
cd "$(dirname "$0")/.."

# Deterministic gate environment, matching CI (ubuntu-latest): upstream tests
# assert English CLI/git output and 022 umask-derived file modes; a localized
# desktop (fr) or group-write umask fails them without any code defect.
export LC_ALL=C
export LANG=C
umask 022

echo "== bun install =="
bun install --frozen-lockfile

echo "== lint (oxlint) =="
bun run lint

echo "== typecheck (turbo) =="
bun run typecheck

echo "== runtime-agnostic check (R00-016) =="
# Server-executed code must not touch the Bun global unguarded: the desktop
# embeds the server under Electron's Node runtime (R00-015). Bun-only CLI
# entrypoints, test files, and the sanctioned `typeof Bun` guard are exempt
# (basic-design-04).
violations="$(grep -rn 'Bun\.' packages/opencode/src packages/core/src \
  --include='*.ts' \
  | grep -v '\.test\.' \
  | grep -v 'src/cli/' \
  | grep -v 'typeof Bun' \
  || true)"
if [[ -n "$violations" ]]; then
  echo "unguarded Bun global reference(s) in server-executed code (R00-016):"
  echo "$violations"
  exit 1
fi

echo "== tests =="
# Upstream has no root test suite — tests run per package:
#   ./scripts/verify.sh core       → bun test --cwd packages/core
#   ./scripts/verify.sh opencode   → bun test --cwd packages/opencode
# Run `./scripts/verify.sh` with no args to typecheck+lint only.
PACKAGE="${1:-}"
if [[ -n "$PACKAGE" ]]; then
  bun test --cwd "packages/$PACKAGE"
else
  echo "typecheck+lint green. Tests are per package: ./scripts/verify.sh <package>"
fi