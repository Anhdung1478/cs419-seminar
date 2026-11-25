#!/usr/bin/env bash
# Apply patches/*.patch to the upstream submodule. Safe to re-run: already-applied patches are skipped.
set -euo pipefail
REPO="$(cd "$(dirname "$0")/.." && pwd)"
UP="$REPO/third_party/RLT4Reranking"
shopt -s nullglob
for p in "$REPO"/patches/*.patch; do
  if git -C "$UP" apply --reverse --check "$p" 2>/dev/null; then
    echo "already applied: $(basename "$p")"
  else
    git -C "$UP" apply "$p"
    echo "applied: $(basename "$p")"
  fi
done
