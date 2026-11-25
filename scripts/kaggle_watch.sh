#!/usr/bin/env bash
# Print one line whenever a Kaggle kernel's status changes; exit when all are finished.
# Args: owner/slug, optionally prefixed with a config dir for another account: /root/.kaggle2:kanphm/rlt-rankllama
# Usage: bash scripts/kaggle_watch.sh anhdung147/rlt-features /root/.kaggle2:kanphm/rlt-rankllama
source "$HOME/miniforge3/etc/profile.d/conda.sh"; conda activate kag
declare -A last
first=1   # QUIET=1: record the starting statuses without printing (for re-arming a watch)
while true; do
  done_all=1
  for arg in "$@"; do
    if [[ "$arg" == *:* ]]; then cfg="${arg%%:*}"; k="${arg#*:}"; else cfg="$HOME/.kaggle"; k="$arg"; fi
    s=$(KAGGLE_CONFIG_DIR="$cfg" kaggle kernels status "$k" 2>&1 | tail -1 | sed -E 's/.*status "?KernelWorkerStatus\.([A-Z_]+)"?.*/\1/')
    if [ "${last[$k]:-}" != "$s" ]; then
      [ "${QUIET:-0}" = 1 ] && [ $first = 1 ] || echo "$(date +%H:%M) $k: $s"
      last[$k]=$s
    fi
    case "$s" in COMPLETE|ERROR|CANCEL*) ;; *) done_all=0 ;; esac
  done
  [ $done_all = 1 ] && exit 0
  first=0
  sleep 120
done
