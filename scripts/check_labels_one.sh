#!/usr/bin/env bash
# Quick label check for one pipeline (gate 5). Usage: bash scripts/check_labels_one.sh 19 bm25 rankllama
set -euo pipefail
source "$(dirname "$0")/env.sh"
S=$1; R=$2; M=$3
base="$D/runs/dl-$S-passage.run-original-$R-1000"
python -u "$UP/rlt/reranking_labels.py" --retrieval_run_path "$base.txt" --reranking_run_path "$base-$M-1000.txt" \
  --qrels_path "$D/qrels/dl-$S-passage.qrels.txt" --metric ndcg@10 --seq_len 1000 --output_path "$D/labels" 2>&1 \
  | grep -aE "oracle|fixed k=1,|fixed k=1000|optimal fixed"
