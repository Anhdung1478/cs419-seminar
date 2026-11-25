# Shared helper for the extension experiments (source after env.sh).
# train_infer NAME PARAM SRC_FEAT_DIR SRC_RET SRC_YEAR TGT_FEAT_DIR TGT_RET TGT_YEAR CKPT_DIR OUT_DIR SEED
#   NAME  : bicut | choppy | attncut | mmoecut
#   PARAM : BiCut alpha (0.4) or label metric (rankllama-1000-ndcg@10-eet-alpha-0.001-beta1)
# Trains once per (checkpoint dir, setup) - skipped if epoch 100 exists - then infers epoch 100 on the target.
export RLT_INFER_EPOCHS=100
QR="$D/qrels"; LAB="$D/labels"
train_infer() {
  local name=$1 p=$2 fs=$3 rs=$4 s=$5 ft=$6 rt=$7 t=$8 ck=$9 out=${10} seed=${11}
  local common=(--seq_len 1000 --epoch_num 100 --interval 1 --batch_size 64 --binarise_qrels --random_seed "$seed")
  local tr=() te=() ckname
  if [ "$name" = bicut ]; then
    tr=(--alpha "$p"); te=(--alpha "$p"); ckname="dl-$s-passage.$rs.bicut.alpha$p"
  else
    tr=(--label_path "$LAB/dl-$s-passage.label-$rs.$p.json")
    te=(--label_path "$LAB/dl-$t-passage.label-$rt.$p.json"); ckname="dl-$s-passage.$rs.$name.$p"
  fi
  if [ ! -s "$ck/$ckname/100.pkl" ]; then
    python -u "$UP/rlt/supervised_rlt/main.py" --name "$name" --checkpoint_path "$ck" \
      --feature_path "$fs/dl-$s-passage.feature-$rs.json" --qrels_path "$QR/dl-$s-passage.qrels.txt" \
      "${tr[@]}" "${common[@]}" > /dev/null 2>&1 || { echo "TRAIN FAILED $ckname"; return 1; }
    # main.py saves every epoch (~7 MB each for AttnCut/Choppy/MtCut); keep only the one we infer
    find "$ck/$ckname" -name '*.pkl' ! -name '100.pkl' -delete
  fi
  python -u "$UP/rlt/supervised_rlt/main.py" --name "$name" --checkpoint_path "$ck" \
    --feature_path "$ft/dl-$t-passage.feature-$rt.json" --qrels_path "$QR/dl-$t-passage.qrels.txt" \
    "${te[@]}" "${common[@]}" --checkpoint_name "$ckname" --output_path "$out" --infer > /dev/null 2>&1 \
    || { echo "INFER FAILED $ckname -> $rt"; return 1; }
}
# supervised configs: "name param". BiCut is left out by default: its loss is a pure-Python double loop over all
# 1000 items of every list (~10 s/epoch, ~17 min per training vs ~1 min for the others), and it ignores the
# re-ranking labels. INCLUDE_BICUT=1 adds it back.
CONFIGS=()
if [ "${INCLUDE_BICUT:-0}" = 1 ]; then for A in 0.4 0.5 0.6; do CONFIGS+=("bicut $A"); done; fi
for M in choppy attncut mmoecut; do for B in 0 1 2; do CONFIGS+=("$M rankllama-1000-ndcg@10-eet-alpha-0.001-beta$B"); done; done
other() { [ "$1" = 19 ] && echo 20 || echo 19; }
