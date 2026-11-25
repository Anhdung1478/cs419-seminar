#!/usr/bin/env bash
# Push an extension job to a Kaggle account (GPU T4 x2 by default; ACCEL=none for a CPU session).
# Usage: bash scripts/kaggle_push_ext.sh <config dir> <slug> '<STEPS python list>' '<ENV python dict>'
# e.g.   bash scripts/kaggle_push_ext.sh /root/.kaggle2 rlt-e1-a '["e1.sh"]' '{"SEEDS": "42 1 2"}'
set -euo pipefail
CFG="$1"; SLUG="$2"; STEPS="$3"; ENVJ="${4:-{\}}"
REPO="$(cd "$(dirname "$0")/.." && pwd)"
USER_=$(python3 -c "import json,sys; print(json.load(open(sys.argv[1]))['username'])" "$CFG/kaggle.json")
T="$REPO/work/kaggle_push/$SLUG-$USER_"; rm -rf "$T"; mkdir -p "$T"
python3 - "$REPO/kaggle/ext/ext_job.py" "$T/ext_job.py" "$STEPS" "$ENVJ" <<'EOF'
import sys
src, dst, steps, env = sys.argv[1:]
code = open(src).read()
code = code.replace('\nSTEPS = ["e1.sh"]\n', f"\nSTEPS = {steps}\n").replace("\nENV = {}\n", f"\nENV = {env}\n")
open(dst, "w").write(code)
EOF
cat > "$T/kernel-metadata.json" <<EOF
{"id": "$USER_/$SLUG", "title": "$SLUG", "code_file": "ext_job.py", "language": "python", "kernel_type": "script",
 "is_private": true, "enable_gpu": $([ "${ACCEL:-}" = none ] && echo false || echo true), "enable_tpu": false, "enable_internet": true,
 "dataset_sources": ["$USER_/rlt-code", "$USER_/rlt-e-inputs"],
 "competition_sources": [], "kernel_sources": [], "model_sources": []}
EOF
source "$HOME/miniforge3/etc/profile.d/conda.sh"; conda activate kag
export KAGGLE_CONFIG_DIR="$CFG"
if [ "${ACCEL:-}" = none ]; then kaggle kernels push -p "$T" 2>&1 | tail -1
else kaggle kernels push -p "$T" --accelerator "${ACCEL:-NvidiaTeslaT4}" 2>&1 | tail -1; fi
