#!/usr/bin/env bash
# Push kaggle/rankllama to one Kaggle account with an explicit per-GPU split (T4 x2).
# Usage: bash scripts/kaggle_push_rankllama.sh <kaggle config dir> '<python list of 2 lists of run names>'
# e.g.   bash scripts/kaggle_push_rankllama.sh ~/.kaggle \
#          '[["dl-20-passage.run-original-bm25-1000","dl-19-passage.run-original-bm25-1000"],["dl-20-passage.run-original-splade-pp-ed-pytorch-1000"]]'
set -euo pipefail
CFG="$1"; QUEUES="$2"; SLUG="${3:-rlt-rankllama}"   # 3rd arg: kernel slug (second job on the same account)
REPO="$(cd "$(dirname "$0")/.." && pwd)"
USER_=$(python3 -c "import json,sys; print(json.load(open(sys.argv[1]))['username'])" "$CFG/kaggle.json")
T="$REPO/work/kaggle_push/$SLUG-$USER_"; rm -rf "$T"; mkdir -p "$T"
python3 - "$REPO/kaggle/rankllama/rankllama_rerank.py" "$T/rankllama_rerank.py" "$QUEUES" <<'EOF'
import sys
src, dst, queues = sys.argv[1:]
code = open(src).read()
assert "\nQUEUES = None\n" in code
open(dst, "w").write(code.replace("\nQUEUES = None\n", f"\nQUEUES = {queues}\n"))
EOF
sed -e "s#anhdung147/#$USER_/#g" -e "s#$USER_/rlt-rankllama\"#$USER_/$SLUG\"#" -e "s#\"title\": \"rlt-rankllama\"#\"title\": \"$SLUG\"#" \
  "$REPO/kaggle/rankllama/kernel-metadata.json" > "$T/kernel-metadata.json"
source "$HOME/miniforge3/etc/profile.d/conda.sh"; conda activate kag
export KAGGLE_CONFIG_DIR="$CFG"
kaggle kernels push -p "$T" --accelerator NvidiaTeslaT4
