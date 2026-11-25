#!/usr/bin/env bash
# E4 - add the cheap MonoT5 score as an extra item feature for predicting RankLLaMA cutoffs.
# All 12 supervised configs x 3 RankLLaMA pipelines x both directions, with and without the feature
# (both run here, same seeds, so the comparison is like-for-like).
set -euo pipefail
source "$(dirname "$0")/env.sh"; source "$REPO/scripts/lib_rlt.sh"
python "$REPO/src/feature_variants.py" monot5 --in_dir "$D/features" --out_dir "$D/features_monot5" --runs_dir "$D/runs"
SEEDS=${SEEDS:-"42"}
for SEED in $SEEDS; do
  for V in base monot5; do
    F="$D/features"; export RLT_EXTRA_FEATURES=""
    [ $V = monot5 ] && { F="$D/features_monot5"; export RLT_EXTRA_FEATURES=monot5; }
    t0=$(date +%s)
    for R in original-bm25-1000 original-splade-pp-ed-pytorch-1000 original-repllama-1000; do
      for S in 19 20; do
        T=$(other $S)
        for c in "${CONFIGS[@]}"; do
          set -- $c
          train_infer "$1" "$2" "$F" "$R" "$S" "$F" "$R" "$T" "./checkpoint/e4/$V/seed$SEED" "./output/e4/$V/seed$SEED" $SEED || true
        done
      done
    done
    echo "seed $SEED variant $V done in $(( $(date +%s) - t0 ))s"
    python "$REPO/src/eval_ext.py" --out_root ./output/e4/$V/seed$SEED --labels "$D/labels" \
      --csv "$REPO/results/ext/e4.csv" --meta exp=E4 seed=$SEED variant=$V
  done
done
python "$REPO/src/summarize_ext.py" e4 "$REPO/results/ext/e4.csv" "$REPO/results/ext/e4_summary.md"
