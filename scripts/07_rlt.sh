#!/usr/bin/env bash
# Steps 10-11: unsupervised + supervised RLT (README 3.1, 3.2) for 4 pipelines, DL19 <-> DL20.
# Supervised outputs/checkpoints go under seed<SEED>/ so later seeds (E1) don't overwrite them.
#   BICUT=all|skip|only   (BiCut trains ~16 min per model, CPU-bound; it can run as a separate CPU job)
#   UNSUP=1|0             run the unsupervised methods (default 1, skipped when BICUT=only)
# Only epoch 100 is inferred (the epoch the paper reports) and other epochs' checkpoints are deleted after
# training: main.py saves ~7 MB per epoch, ~54 GB for all models, which filled Kaggle's disk.
set -euo pipefail
source "$(dirname "$0")/env.sh"
SEED=${SEED:-42}; BICUT=${BICUT:-all}; UNSUP=${UNSUP:-1}; [ "$BICUT" = only ] && UNSUP=0
export RLT_INFER_EPOCHS=100
FEAT="$D/features"; LAB="$D/labels"; QR="$D/qrels"
RETRIEVERS=(original-bm25-1000 original-splade-pp-ed-pytorch-1000 original-repllama-1000)
rerankers_for() { [ "$1" = original-bm25-1000 ] && echo "rankllama monot5" || echo "rankllama"; }
other() { [ "$1" = 19 ] && echo 20 || echo 19; }

# ---- unsupervised (3.1)
if [ "$UNSUP" = 1 ]; then
for R in "${RETRIEVERS[@]}"; do
  for S in 19 20; do
    T=$(other $S)
    python -u "$UP/rlt/unsupervised_rlt.py" --name fixed --feature_path "$FEAT/dl-$S-passage.feature-$R.json" --output_path ./output
    python -u "$UP/rlt/unsupervised_rlt.py" --name surprise --feature_path "$FEAT/dl-$S-passage.feature-$R.json" --output_path ./output
    for M in $(rerankers_for $R); do
      python -u "$UP/rlt/unsupervised_rlt.py" --name oracle \
        --test_labels_path "$LAB/dl-$S-passage.label-$R.$M-1000-ndcg@10.json" --output_path ./output
      for B in 0 1 2; do
        python -u "$UP/rlt/unsupervised_rlt.py" --name greedy --feature_path "$FEAT/dl-$S-passage.feature-$R.json" \
          --train_labels_path "$LAB/dl-$T-passage.label-$R.$M-1000-ndcg@10-eet-alpha-0.001-beta$B.json" --output_path ./output
      done
    done
  done
done
fi

# ---- supervised (3.2): train on DL$S, infer epoch 100 on the other year
CK="./checkpoint/seed$SEED"; OUT="./output/seed$SEED"
common=(--epoch_num 100 --interval 1 --seq_len 1000 --batch_size 64 --binarise_qrels --random_seed "$SEED")
sup() { # name ckname train-args... -- infer-args...
  local name=$1 ckname=$2; shift 2
  local tr=() te=(); while [ "$1" != -- ]; do tr+=("$1"); shift; done; shift; te=("$@")
  python -u "$UP/rlt/supervised_rlt/main.py" --name "$name" --checkpoint_path "$CK" "${tr[@]}" "${common[@]}"
  find "$CK/$ckname" -name '*.pkl' ! -name '100.pkl' -delete
  python -u "$UP/rlt/supervised_rlt/main.py" --name "$name" --checkpoint_path "$CK" "${te[@]}" "${common[@]}" \
    --checkpoint_name "$ckname" --output_path "$OUT" --infer
}
for R in "${RETRIEVERS[@]}"; do
  for S in 19 20; do
    T=$(other $S)
    if [ "$BICUT" != skip ]; then
      for A in 0.4 0.5 0.6; do
        sup bicut "dl-$S-passage.$R.bicut.alpha$A" \
          --feature_path "$FEAT/dl-$S-passage.feature-$R.json" --qrels_path "$QR/dl-$S-passage.qrels.txt" --alpha $A -- \
          --feature_path "$FEAT/dl-$T-passage.feature-$R.json" --qrels_path "$QR/dl-$T-passage.qrels.txt" --alpha $A
      done
    fi
    [ "$BICUT" = only ] && continue
    for M in $(rerankers_for $R); do
      for B in 0 1 2; do
        metric="$M-1000-ndcg@10-eet-alpha-0.001-beta$B"
        for model in choppy attncut mmoecut; do
          sup $model "dl-$S-passage.$R.$model.$metric" \
            --feature_path "$FEAT/dl-$S-passage.feature-$R.json" --label_path "$LAB/dl-$S-passage.label-$R.$metric.json" \
            --qrels_path "$QR/dl-$S-passage.qrels.txt" -- \
            --feature_path "$FEAT/dl-$T-passage.feature-$R.json" --label_path "$LAB/dl-$T-passage.label-$R.$metric.json" \
            --qrels_path "$QR/dl-$T-passage.qrels.txt"
        done
      done
    done
  done
done
