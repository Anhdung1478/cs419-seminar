#!/usr/bin/env bash
# Upload first-stage runs, qrels, queries, needed passages and our scripts as the private Kaggle dataset
# anhdung147/rlt-runs (input of kaggle/features and kaggle/monot5). Re-run to publish a new version.
set -euo pipefail
source "$(dirname "$0")/env.sh"
U="$WORK/kaggle_upload/rlt-runs"; mkdir -p "$U"
cp "$D"/runs/dl-*-passage.run-original-{bm25,splade-pp-ed-pytorch,repllama}-1000.txt "$U/"
cp "$D"/qrels/dl-*-passage.qrels.txt "$D"/queries/dl-*-passage.queries-original.tsv "$U/"
cp "$D/collection/msmarco.needed.tsv" "$U/"   # built by scripts/04_monot5.sh (passages in the 6 runs)
cp "$REPO/src/features_lowmem.py" "$REPO/src/monot5_rerank.py" "$REPO/src/eval_run.py" "$U/"
cat > "$U/dataset-metadata.json" <<'EOF'
{"title": "RLT first-stage runs (DL19/DL20)", "id": "anhdung147/rlt-runs", "licenses": [{"name": "other"}],
 "description": "Private. BM25 / SPLADE++ / RepLLaMA top-1000 runs, TREC DL19/DL20 passage queries+qrels, MS MARCO passages in those runs, and scripts, for reproducing RLT4Reranking. MS MARCO terms apply."}
EOF
source "$HOME/miniforge3/etc/profile.d/conda.sh"; conda activate kag
if kaggle datasets status anhdung147/rlt-runs >/dev/null 2>&1; then
  kaggle datasets version -p "$U" -m "add queries, passages, monot5 script"
else
  kaggle datasets create -p "$U"
fi
