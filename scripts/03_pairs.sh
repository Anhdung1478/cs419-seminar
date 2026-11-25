#!/usr/bin/env bash
# Step 4: RankLLaMA inputs for 3 retrievers x 2 years -> $D/runs/rankllama_input/ (upload to Kaggle).
set -euo pipefail
source "$(dirname "$0")/env.sh"
OUT="$D/runs/rankllama_input"; mkdir -p "$OUT"

jobs=()
for S in 19 20; do
  for R in bm25 splade-pp-ed-pytorch repllama; do
    run="dl-$S-passage.run-original-$R-1000"
    jobs+=(--job "$D/tevatron/dl$S.jsonl.gz" "$D/runs/$run.txt" "$OUT/rerank_input.$run.jsonl")
  done
done
python -u "$REPO/src/prepare_pairs.py" --corpus "$D/tevatron/corpus.jsonl.gz" "${jobs[@]}"
wc -l "$OUT"/*.jsonl   # gate 2: 43,000 (DL19) / 54,000 (DL20) per file
