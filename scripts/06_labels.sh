#!/usr/bin/env bash
# Step 9: re-ranking labels (README 2.5; patched reranking_labels.py). Needs RankLLaMA runs from Kaggle
# copied into $D/runs/ and MonoT5 runs from step 6.
set -euo pipefail
source "$(dirname "$0")/env.sh"
for S in 19 20; do
  for pair in bm25:rankllama splade-pp-ed-pytorch:rankllama repllama:rankllama bm25:monot5; do
    R=${pair%%:*}; M=${pair##*:}
    base="$D/runs/dl-$S-passage.run-original-$R-1000"
    out="$D/labels/dl-$S-passage.label-original-$R-1000.$M-1000-ndcg@10.json"
    [ -s "$out" ] && { echo "have $out"; continue; }
    # gate 5: the log line "oracle ndcg_cut_10" should match the paper's Oracle
    python -u "$UP/rlt/reranking_labels.py" --retrieval_run_path "$base.txt" \
      --reranking_run_path "$base-$M-1000.txt" --qrels_path "$D/qrels/dl-$S-passage.qrels.txt" \
      --metric ndcg@10 --seq_len 1000 --output_path "$D/labels" 2>&1 | grep -E "oracle|fixed k=1000|optimal"
  done
done
