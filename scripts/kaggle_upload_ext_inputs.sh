#!/usr/bin/env bash
# Upload rlt-code and rlt-e-inputs (features, labels, qrels, MonoT5 runs) as private datasets on one account.
# Usage: bash scripts/kaggle_upload_ext_inputs.sh <config dir> <labels dir>
set -euo pipefail
CFG="$1"; LABELS="$2"
REPO="$(cd "$(dirname "$0")/.." && pwd)"; D="$REPO/work/datasets/msmarco-v1-passage"
USER_=$(python3 -c "import json,sys; print(json.load(open(sys.argv[1]))['username'])" "$CFG/kaggle.json")
source "$HOME/miniforge3/etc/profile.d/conda.sh"; conda activate kag
export KAGGLE_CONFIG_DIR="$CFG"
up() { # dir slug title
  cat > "$1/dataset-metadata.json" <<EOF
{"title": "$3", "id": "$USER_/$2", "licenses": [{"name": "other"}], "description": "Private. cs419 RLT seminar."}
EOF
  if kaggle datasets status "$USER_/$2" >/dev/null 2>&1; then kaggle datasets version -p "$1" -m update >/dev/null
  else kaggle datasets create -p "$1" >/dev/null; fi
}
C="$REPO/work/kaggle_upload/code-$USER_"; rm -rf "$C"; mkdir -p "$C"
cp "$REPO"/scripts/*.sh "$REPO"/src/*.py "$REPO"/patches/*.patch "$C/"
up "$C" rlt-code "RLT seminar code"
E="$REPO/work/kaggle_upload/e-inputs-$USER_"; rm -rf "$E"; mkdir -p "$E"
cp "$D"/features/dl-*-passage.feature-original-*-1000.json "$LABELS"/dl-*-passage.label-*.json \
   "$D"/qrels/dl-*-passage.qrels.txt "$D"/runs/dl-*-passage.run-original-*-monot5-1000.txt "$E/"
up "$E" rlt-e-inputs "RLT extension inputs"
for s in rlt-code rlt-e-inputs; do until kaggle datasets status "$USER_/$s" | grep -q ready; do sleep 10; done; done
echo "$USER_: rlt-code + rlt-e-inputs ready ($(ls "$E" | wc -l) input files)"
