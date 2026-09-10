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