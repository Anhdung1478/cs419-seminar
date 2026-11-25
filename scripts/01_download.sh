#!/usr/bin/env bash
# Step 2: raw data + prebuilt indexes. Idempotent (skips files that already exist).
set -euo pipefail
source "$(dirname "$0")/env.sh"

get() { # url dest
  [ -s "$2" ] && { echo "have $2"; return; }
  echo "get  $2"; curl -fL --retry 3 -o "$2.part" "$1" && mv "$2.part" "$2"
}
MS=https://msmarco.z22.web.core.windows.net/msmarcoranking

# Queries / qrels (README 2.1, same final names)
for y in 2019:19 2020:20; do
  Y=${y%%:*}; S=${y##*:}
  q="$D/queries/dl-$S-passage.queries-original.tsv"
  [ -s "$q" ] || { get "$MS/msmarco-test$Y-queries.tsv.gz" "$q.gz"; gunzip -f "$q.gz"; }
  get "https://trec.nist.gov/data/deep/${Y}qrels-pass.txt" "$D/qrels/dl-$S-passage.qrels.txt"
done

# MS MARCO collection (features step)
if [ ! -s "$D/collection/msmarco.tsv" ]; then
  get "$MS/collection.tar.gz" "$D/collection/collection.tar.gz"
  tar -xzf "$D/collection/collection.tar.gz" -C "$D/collection/"
  mv "$D/collection/collection.tsv" "$D/collection/msmarco.tsv"
  rm "$D/collection/collection.tar.gz"
fi

# Tevatron corpus (with titles) + DL queries: what RankLLaMA's prepare_rerank_file.py used
HF=https://huggingface.co/datasets
mkdir -p "$D/tevatron"
get "$HF/Tevatron/msmarco-passage-corpus/resolve/main/corpus.jsonl.gz" "$D/tevatron/corpus.jsonl.gz"
get "$HF/Tevatron/msmarco-passage/resolve/main/dl19.jsonl.gz" "$D/tevatron/dl19.jsonl.gz"
get "$HF/Tevatron/msmarco-passage/resolve/main/dl20.jsonl.gz" "$D/tevatron/dl20.jsonl.gz"

# RepLLaMA runs shared by the original author (README 2.2.3)
get "https://www.dropbox.com/scl/fi/byty1lk2um36imz0788yd/run.repllama.psg.dl19.txt?rlkey=615ootx2mia42cxdilp4tvqzh&dl=1" "$D/runs/run.repllama.psg.dl19.txt"
get "https://www.dropbox.com/scl/fi/drgg9vj8mxe3qwayggj9o/run.repllama.psg.dl20.txt?rlkey=22quuq5wzvn6ip0c5ml6ad5cs&dl=1" "$D/runs/run.repllama.psg.dl20.txt"

# Prebuilt pyserini indexes (download into $PYSERINI_CACHE)
python - <<'EOF'
from pyserini.search.lucene import LuceneSearcher, LuceneImpactSearcher
s = LuceneSearcher.from_prebuilt_index("msmarco-v1-passage-full"); print("bm25 index docs:", s.num_docs)
LuceneImpactSearcher.from_prebuilt_index("msmarco-v1-passage-splade-pp-ed", "naver/splade-cocondenser-ensembledistil"); print("splade index OK")
EOF
echo DOWNLOAD_DONE
