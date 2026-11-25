#!/usr/bin/env bash
# Step 12: evaluate all pipelines and compare with the paper -> results/reproduction/*.md
set -euo pipefail
source "$(dirname "$0")/env.sh"
python "$REPO/src/evaluate_all.py" --repo "$REPO" --seed "${SEED:-42}"
