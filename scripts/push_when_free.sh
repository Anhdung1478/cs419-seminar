#!/usr/bin/env bash
# Wait until one of the given kernels (same account) finishes, then push an extension job into the freed GPU slot.
# Usage: bash scripts/push_when_free.sh <config dir> <slug> '<STEPS>' '<ENV>' owner/k1 owner/k2 ...
CFG="$1"; SLUG="$2"; STEPS="$3"; ENVJ="$4"; shift 4
source "$HOME/miniforge3/etc/profile.d/conda.sh"; conda activate kag
export KAGGLE_CONFIG_DIR="$CFG"
while true; do
  for k in "$@"; do
    s=$(kaggle kernels status "$k" 2>&1 | tail -1)
    case "$s" in *COMPLETE*|*ERROR*|*CANCEL*) echo "$(date +%H:%M) $k finished: ${s##*.}"; break 2;; esac
  done
  sleep 120
done
bash "$(dirname "$0")/kaggle_push_ext.sh" "$CFG" "$SLUG" "$STEPS" "$ENVJ"
