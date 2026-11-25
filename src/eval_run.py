"""nDCG@10 (graded qrels, judged queries only) + run shape checks for a TREC run file."""
import argparse
from collections import Counter

import pytrec_eval

ap = argparse.ArgumentParser()
ap.add_argument("--run", required=True)
ap.add_argument("--qrels", required=True)
ap.add_argument("--depth", type=int, default=1000)
args = ap.parse_args()

with open(args.qrels) as f:
    qrels = pytrec_eval.parse_qrel(f)
with open(args.run) as f:
    run = pytrec_eval.parse_run(f)

judged = {q: d for q, d in run.items() if q in qrels}
sizes = Counter(len(d) for d in judged.values())
scores = pytrec_eval.RelevanceEvaluator(qrels, {"ndcg_cut_10"}).evaluate(judged)
ndcg = sum(s["ndcg_cut_10"] for s in scores.values()) / len(scores)

short = sum(n for size, n in sizes.items() if size < args.depth)
print(f"{args.run.split('/')[-1]:<60} judged_q={len(judged):>3}  "
      f"short_lists={short}  nDCG@10={ndcg:.3f}")
