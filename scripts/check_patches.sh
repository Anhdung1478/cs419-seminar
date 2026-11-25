#!/usr/bin/env bash
# Verify patches/*.patch apply in order to a fresh upstream checkout and reproduce the patched submodule tree.
set -euo pipefail
REPO="$(cd "$(dirname "$0")/.." && pwd)"; UP="$REPO/third_party/RLT4Reranking"
T=$(mktemp -d)/fresh
git -C "$UP" worktree add -q --detach "$T" HEAD
git -C "$T" apply "$REPO"/patches/*.patch
if diff -r -q -x .git "$T/rlt" "$UP/rlt"; then echo "FRESH_APPLY_MATCHES"; else echo "MISMATCH"; fi
git -C "$UP" worktree remove --force "$T"
