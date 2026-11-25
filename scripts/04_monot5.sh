#!/usr/bin/env bash
# Step 6 (PC GPU). BM25 = reproduction; SPLADE++/RepLLaMA = ahead of time for E4.
# Memory: reading the 3 GB msmarco.tsv fills WSL's page cache (which Windows can't reclaim quickly),
# so first extract the ~211k needed passages into a small TSV once, then drop the cache.
set -euo pipefail
source "$(dirname "$0")/env.sh"
SMALL="$D/collection/msmarco.needed.tsv"
if [ ! -s "$SMALL" ]; then
  python - "$D" "$SMALL" <<'EOF'
import sys
D, out = sys.argv[1], sys.argv[2]
need = set()
for s in ("19", "20"):
    for r in ("bm25", "splade-pp-ed-pytorch", "repllama"):
        with open(f"{D}/runs/dl-{s}-passage.run-original-{r}-1000.txt") as f:
            need.update(line.split()[2] for line in f)
with open(f"{D}/collection/msmarco.tsv") as f, open(out, "w") as w:
    for line in f:
        if line.split("\t", 1)[0] in need:
            w.write(line)
print(f"{len(need):,} passages -> {out}")
EOF
fi
sync; echo 1 > /proc/sys/vm/drop_caches || true

for R in bm25 splade-pp-ed-pytorch repllama; do
  for S in 19 20; do
    run="$D/runs/dl-$S-passage.run-original-$R-1000.txt"
    [ -s "${run%.txt}-monot5-1000.txt" ] && { echo "have ${run%.txt}-monot5-1000.txt"; continue; }
    python -u "$REPO/src/monot5_rerank.py" --query_path "$D/queries/dl-$S-passage.queries-original.tsv" \
      --run_path "$run" --qrels_path "$D/qrels/dl-$S-passage.qrels.txt" --collection "$SMALL" --k 1000
    # gate 4 (BM25: paper Fixed-k(1000) .730 / .713)
    python "$REPO/src/eval_run.py" --run "${run%.txt}-monot5-1000.txt" --qrels "$D/qrels/dl-$S-passage.qrels.txt"
  done
done
