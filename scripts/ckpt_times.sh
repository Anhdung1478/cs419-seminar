#!/usr/bin/env bash
# Print training start/end times (epoch 1 and 100 checkpoint mtimes) for each checkpoint dir under $1.
for d in "$1"/*/; do
  printf '%s  %s -> %s\n' "$(basename "$d")" "$(date -r "$d/001.pkl" +%T 2>/dev/null)" "$(date -r "$d/100.pkl" +%T 2>/dev/null)"
done
