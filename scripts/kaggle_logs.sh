#!/usr/bin/env bash
# Show key lines of a Kaggle kernel's live log + status.  Usage: bash scripts/kaggle_logs.sh <user/kernel> [config dir]
source "$HOME/miniforge3/etc/profile.d/conda.sh"; conda activate kag
[ -n "${2:-}" ] && export KAGGLE_CONFIG_DIR="$2"
kaggle kernels logs "$1" 2>&1 | grep -aE "SMOKE|OOM|queues|\] done|Error|error|base model|Traceback|COMPLETE" | tail -12
kaggle kernels status "$1" 2>&1 | tail -1
