#!/usr/bin/env bash
# One-off: write patches/0003-extension-hooks.patch = current upstream edits on top of patches 0001-0002.
set -euo pipefail
REPO="$(cd "$(dirname "$0")/.." && pwd)"; UP="$REPO/third_party/RLT4Reranking"
T=$(mktemp -d); trap 'rm -rf "$T"' EXIT
git -C "$UP" worktree add -q --detach "$T/base" HEAD
git -C "$T/base" apply "$REPO/patches/0001-labels-elif-and-per-query-evaluator.patch" "$REPO/patches/0002-dataset-drop-unused-imports.patch"
out="$REPO/patches/0003-extension-hooks.patch"; : > "$out"
for f in rlt/supervised_rlt/main.py rlt/supervised_rlt/dataset.py; do
  (cd "$T" && diff -u --label "a/$f" --label "b/$f" "base/$f" "$UP/$f" >> "$out") || true
done
git -C "$UP" worktree remove --force "$T/base"
# check: base + 0001 + 0002 + 0003 == current tree
git -C "$UP" stash -q && git -C "$UP" apply "$REPO"/patches/000[123]-*.patch && git -C "$UP" stash drop -q || true
git -C "$UP" diff --stat
