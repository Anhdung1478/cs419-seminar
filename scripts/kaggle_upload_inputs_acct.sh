#!/usr/bin/env bash
# Upload the 6 RankLLaMA input files as private dataset <user>/rlt-rankllama-inputs on another Kaggle account.
# Usage: bash scripts/kaggle_upload_inputs_acct.sh ~/.kaggle2
set -euo pipefail
CFG="${1:?config dir, e.g. ~/.kaggle2}"
REPO="$(cd "$(dirname "$0")/.." && pwd)"
USER_=$(python3 -c "import json,sys; print(json.load(open(sys.argv[1]))['username'])" "$CFG/kaggle.json")
SRC="$REPO/work/datasets/msmarco-v1-passage/runs/rankllama_input"
U="$REPO/work/kaggle_upload/rankllama-inputs-$USER_"; mkdir -p "$U"
cp "$SRC"/rerank_input.*.jsonl "$U/"
cat > "$U/dataset-metadata.json" <<EOF
{"title": "RLT RankLLaMA inputs (DL19/DL20)", "id": "$USER_/rlt-rankllama-inputs", "licenses": [{"name": "other"}],
 "description": "Private. RankLLaMA re-rank inputs for reproducing RLT4Reranking. MS MARCO terms apply."}
EOF
source "$HOME/miniforge3/etc/profile.d/conda.sh"; conda activate kag
export KAGGLE_CONFIG_DIR="$CFG"
kaggle datasets status "$USER_/rlt-rankllama-inputs" >/dev/null 2>&1 || kaggle datasets create -p "$U"
until kaggle datasets status "$USER_/rlt-rankllama-inputs" | grep -q ready; do sleep 10; done
echo "$USER_/rlt-rankllama-inputs ready"
