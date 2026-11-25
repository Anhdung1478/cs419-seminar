#!/usr/bin/env bash
# Tiny local smoke test of the extension pipeline (2 configs, BM25->RankLLaMA, DL19->DL20, raw/zscore/monot5).
# Needs labels for dl-19 and dl-20 bm25 rankllama (scripts/check_labels_one.sh).
set -euo pipefail
source "$(dirname "$0")/env.sh"; source "$REPO/scripts/lib_rlt.sh"
T=./smoke_ext; rm -rf "$T/out" "$REPO/results/ext_smoke"; mkdir -p "$T"
R=original-bm25-1000
python "$REPO/src/feature_variants.py" zscore --in_dir "$D/features" --out_dir "$D/features_zscore"
python "$REPO/src/feature_variants.py" monot5 --in_dir "$D/features" --out_dir "$D/features_monot5" --runs_dir "$D/runs"
for V in raw zscore monot5; do
  F="$D/features"; export RLT_EXTRA_FEATURES=""
  [ $V = zscore ] && F="$D/features_zscore"
  [ $V = monot5 ] && { F="$D/features_monot5"; export RLT_EXTRA_FEATURES=monot5; }
  for c in "attncut rankllama-1000-ndcg@10-eet-alpha-0.001-beta1"; do
    set -- $c
    t0=$(date +%s)
    train_infer "$1" "$2" "$F" $R 19 "$F" $R 20 "$T/ck/$V" "$T/out/$V" 42
    echo "  $V $1: $(( $(date +%s) - t0 ))s"
  done
  python "$REPO/src/eval_ext.py" --out_root $T/out/$V --labels "$D/labels" --csv "$REPO/results/ext_smoke/s.csv" \
    --meta exp=SMOKE seed=42 variant=$V
done
cat "$REPO/results/ext_smoke/s.csv"
