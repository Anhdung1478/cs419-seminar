#!/usr/bin/env bash
# Step 3: first-stage runs (README 2.2), then gate 1 (w/o re-ranking nDCG@10 vs paper).
set -euo pipefail
source "$(dirname "$0")/env.sh"

for S in 19 20; do
  Q="$D/queries/dl-$S-passage.queries-original.tsv"

  out="$D/runs/dl-$S-passage.run-original-bm25-1000.txt"
  [ -s "$out" ] || python -m pyserini.search.lucene --threads 16 --batch-size 128 \
    --index msmarco-v1-passage-full --topics "$Q" --output "$out" \
    --bm25 --k1 0.9 --b 0.4 --hits 1000

  out="$D/runs/dl-$S-passage.run-original-splade-pp-ed-pytorch-1000.txt"
  [ -s "$out" ] || python -m pyserini.search.lucene --threads 16 --batch-size 128 \
    --index msmarco-v1-passage-splade-pp-ed --encoder naver/splade-cocondenser-ensembledistil \
    --topics "$Q" --output "$out" --hits 1000 --impact

  out="$D/runs/dl-$S-passage.run-original-repllama-1000.txt"
  [ -s "$out" ] || python -u "$UP/format.py" --input_path "$D/runs/run.repllama.psg.dl$S.txt" \
    --output_path "$out" --ranker_name repllama
done

# Gate 1 (paper: BM25 .506/.480, SPLADE++ .731/.720; RepLLaMA see results_in_papers/table3)
for S in 19 20; do
  for R in bm25 splade-pp-ed-pytorch repllama; do
    python "$REPO/src/eval_run.py" --run "$D/runs/dl-$S-passage.run-original-$R-1000.txt" \
      --qrels "$D/qrels/dl-$S-passage.qrels.txt"
  done
done
