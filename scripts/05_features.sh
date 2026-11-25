#!/usr/bin/env bash
# Step 7 (PC CPU, runs while Kaggle does RankLLaMA). Needs WSL memory >= 12 GB (doc2vec ~7 GB).
set -euo pipefail
source "$(dirname "$0")/env.sh"
F="$D/features"
python -u "$REPO/src/features_lowmem.py" --mode prep    --index_path "$D/collection/msmarco.tsv" --output_path "$F"
python -u "$REPO/src/features_lowmem.py" --mode tfidf   --output_path "$F"
python -u "$REPO/src/features_lowmem.py" --mode doc2vec --output_path "$F" --vector_size 128

runs=(); qrels=()
for S in 19 20; do
  for R in bm25 splade-pp-ed-pytorch repllama; do
    runs+=("$D/runs/dl-$S-passage.run-original-$R-1000.txt"); qrels+=("$D/qrels/dl-$S-passage.qrels.txt")
  done
done
python -u "$REPO/src/features_lowmem.py" --mode infer --output_path "$F" --seq_len 1000 \
  --run_path "${runs[@]}" --qrels_path "${qrels[@]}"
ls -la "$F"/*.feature-*.json
