#!/usr/bin/env bash
# E1 - seed variance of supervised RLT (only 43/54 training queries).
# All 12 supervised configs x {BM25, SPLADE++}->RankLLaMA x {DL19->DL20, DL20->DL19} x SEEDS.
set -euo pipefail
source "$(dirname "$0")/env.sh"; source "$REPO/scripts/lib_rlt.sh"
SEEDS=${SEEDS:-"42 1 2 3 4"}
for SEED in $SEEDS; do
  t0=$(date +%s)
  for R in original-bm25-1000 original-splade-pp-ed-pytorch-1000; do
    for S in 19 20; do
      T=$(other $S)
      for c in "${CONFIGS[@]}"; do
        set -- $c
        train_infer "$1" "$2" "$D/features" "$R" "$S" "$D/features" "$R" "$T" \
          "./checkpoint/e1/seed$SEED" "./output/e1/seed$SEED" "$SEED" || true
      done
    done
  done
  echo "seed $SEED done in $(( $(date +%s) - t0 ))s"
  python "$REPO/src/eval_ext.py" --out_root ./output/e1/seed$SEED --labels "$D/labels" \
    --csv "$REPO/results/ext/e1.csv" --meta exp=E1 seed=$SEED variant=raw
done
python "$REPO/src/summarize_ext.py" e1 "$REPO/results/ext/e1.csv" "$REPO/results/ext/e1_summary.md"
