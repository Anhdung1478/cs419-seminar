"""Build RankLLaMA re-rank inputs (one JSONL per run) exactly like upstream
tevatron/examples/rankllama/prepare_rerank_file.py, without its broken HF split loader
and without holding all 8.8M passages in RAM.

Each output line: {"docid","title","text","score","query_id","query"}; queries not in the
Tevatron dl19/dl20 query file are skipped (as upstream does), depth-limited to --depth.

Usage:
  python prepare_pairs.py --corpus corpus.jsonl.gz \
      --job dl19.jsonl.gz run.txt out.jsonl [--job ...]
"""
import argparse
import gzip
import json
from collections import defaultdict


def read_run(path, depth):
    res = defaultdict(list)
    with open(path) as f:
        for line in f:
            qid, _, pid, _, score, _ = line.split()
            if len(res[qid]) < depth:
                res[qid].append((pid, float(score)))
    return res


def read_queries(path):
    with gzip.open(path, "rt") as f:
        return {(e := json.loads(l))["query_id"]: e["query"] for l in f}


ap = argparse.ArgumentParser()
ap.add_argument("--corpus", required=True)
ap.add_argument("--job", nargs=3, action="append", required=True,
                metavar=("QUERIES_JSONL_GZ", "RUN", "OUT_JSONL"))
ap.add_argument("--depth", type=int, default=1000)
args = ap.parse_args()

jobs = []
needed = set()
for qpath, run_path, out in args.job:
    queries = read_queries(qpath)
    run = {q: v for q, v in read_run(run_path, args.depth).items() if q in queries}
    jobs.append((queries, run, out))
    needed.update(pid for lst in run.values() for pid, _ in lst)
print(f"{len(jobs)} runs, {len(needed):,} unique passages needed")

corpus = {}
with gzip.open(args.corpus, "rt") as f:
    for line in f:
        e = json.loads(line)
        if e["docid"] in needed:
            corpus[e["docid"]] = {"docid": e["docid"], "title": e["title"], "text": e["text"]}
missing = needed - corpus.keys()
assert not missing, f"{len(missing)} docids missing from corpus, e.g. {next(iter(missing))}"

for queries, run, out in jobs:
    n = 0
    with open(out, "w") as f:
        for qid, lst in run.items():
            for pid, score in lst:
                rec = dict(corpus[pid], score=score, query_id=qid, query=queries[qid])
                f.write(json.dumps(rec) + "\n")
                n += 1
    print(f"{out}: {len(run)} queries, {n:,} lines, "
          f"empty titles {sum(1 for l in run.values() for p, _ in l if not corpus[p]['title']):,}")
