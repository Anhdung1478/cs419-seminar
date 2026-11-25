#!/usr/bin/env bash
# E3 - cross-retriever transfer of supervised RLT, raw vs per-query z-scored retrieval scores.
# Train on {BM25, SPLADE++}->RankLLaMA (DL19 or DL20), test on the other year for all 3 retrievers
# (same retriever = in-domain baseline). Seed 42.
set -euo pipefail
source "$(dirname "$0")/env.sh"; source "$REPO/scripts/lib_rlt.sh"
python "$REPO/src/feature_variants.py" zscore --in_dir "$D/features" --out_dir "$D/features_zscore"
SEED=42
for V in raw zscore; do
  F="$D/features"; [ $V = zscore ] && F="$D/features_zscore"
  t0=$(date +%s)
  for RS in original-bm25-1000 original-splade-pp-ed-pytorch-1000; do
    for S in 19 20; do
      T=$(other $S)
      for c in "${CONFIGS[@]}"; do
        set -- $c
        for RT in original-bm25-1000 original-splade-pp-ed-pytorch-1000 original-repllama-1000; do
          train_infer "$1" "$2" "$F" "$RS" "$S" "$F" "$RT" "$T" "./checkpoint/e3/$V" "./output/e3/$V" $SEED || true
        done
      done
    done
  done
  echo "variant $V done in $(( $(date +%s) - t0 ))s"
  python "$REPO/src/eval_ext.py" --out_root ./output/e3/$V --labels "$D/labels" \
    --csv "$REPO/results/ext/e3.csv" --meta exp=E3 seed=$SEED variant=$V
done
python "$REPO/src/summarize_ext.py" e3 "$REPO/results/ext/e3.csv" "$REPO/results/ext/e3_summary.md"
