"""Derived RLT feature files for the extension experiments (same file names, so upstream path parsing still works).

  zscore : E3 - replace each list's retrieval "score" by its per-query z-score. Raw BM25, SPLADE++ (impact) and
           RepLLaMA (dot product) scores live on very different scales, and upstream feeds them unnormalised.
  monot5 : E4 - add key "monot5": MonoT5 log P(true) for every item, aligned with the feature list's docids.

Usage:
  python feature_variants.py zscore --in_dir F --out_dir F_norm
  python feature_variants.py monot5 --in_dir F --out_dir F_e4 --runs_dir RUNS
"""
import argparse
import glob
import json
import os

import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("kind", choices=["zscore", "monot5"])
ap.add_argument("--in_dir", required=True)
ap.add_argument("--out_dir", required=True)
ap.add_argument("--runs_dir")
a = ap.parse_args()
os.makedirs(a.out_dir, exist_ok=True)

for path in sorted(glob.glob(f"{a.in_dir}/*.feature-*.json")):
    name = os.path.basename(path)  # dl-19-passage.feature-original-bm25-1000.json
    feats = json.load(open(path))
    if a.kind == "zscore":
        for f in feats.values():
            s = np.asarray(f["score"], dtype=float)
            f["score"] = ((s - s.mean()) / (s.std() or 1.0)).tolist()
    else:
        ds, rest = name.split(".", 1)
        retriever = rest.removeprefix("feature-").removesuffix(".json")
        run = f"{a.runs_dir}/{ds}.run-{retriever}-monot5-1000.txt"
        mono = {}
        with open(run) as r:
            for line in r:
                qid, _, docid, _, score, _ = line.split()
                mono.setdefault(qid, {})[docid] = float(score)
        for qid, f in feats.items():
            f["monot5"] = [mono[qid][d] for d in f["docid"]]
    with open(f"{a.out_dir}/{name}", "w") as w:
        json.dump(feats, w)
    print(a.kind, name, len(feats), "queries")
