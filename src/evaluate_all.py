"""Evaluate all RLT cutoff files per pipeline and compare with the paper.

For every (dataset, retriever, reranker) it scores, with the upstream rlt/evaluation.py `evaluation()`:
  ours        : our k-files (unsupervised in output/, supervised epoch-100 in output/seed<S>/)
  paper@ours  : the paper's own k-files (results_in_papers/) against OUR labels  (gate 6)
and reads the paper's reported numbers from results_in_papers/<table>/result.*.
Writes results/reproduction/<pipeline>.md and prints a summary.
"""
import argparse
import ast
import glob
import logging
import os
import re
import sys

ap = argparse.ArgumentParser()
ap.add_argument("--repo", required=True)
ap.add_argument("--seed", type=int, default=42)
ap.add_argument("--epoch", default="100")
args = ap.parse_args()

UP = f"{args.repo}/third_party/RLT4Reranking"
WORK = f"{args.repo}/work"
sys.path.insert(0, f"{UP}/rlt")
from evaluation import evaluation  # noqa: E402  (upstream)

logging.getLogger().setLevel(logging.WARNING)

PIPELINES = [  # (paper table dir, retriever, reranker)
    ("table1_bm25_rankllama", "bm25", "rankllama"),
    ("table2_splade_rankllama", "splade-pp-ed-pytorch", "rankllama"),
    ("table3_repllama_rankllma", "repllama", "rankllama"),
    ("table4_bm25_monot5", "bm25", "monot5"),
]


def relevant(name, pipe, reranker):
    """k-file belongs to this reranker's table? (fixed/surprise/bicut are reranker-agnostic)"""
    method = name[len(pipe) + 1:]  # e.g. greedy-ckpt-dl-20-passage.original-bm25-1000.rankllama-1000-...
    if method.startswith(("fixed-", "surprise", "bicut")):
        return True
    return f"{reranker}-1000-" in method


def paper_numbers(table_dir, ds, ret, reranker):
    out = {}
    for path in glob.glob(f"{UP}/results_in_papers/{table_dir}/result.{ds}.original-{ret}-1000.*"):
        for line in open(path):
            m = re.match(r"(\S+):\s+(\{.*\})\s*$", line)  # "<kfile>.<reranker-metric>:<padding> {dict}"
            if not m:
                continue
            kfile, metric = m.group(1).rsplit(".", 1)
            if metric == f"{reranker}-1000-ndcg@10":
                out[kfile] = ast.literal_eval(m.group(2))
    return out


def fmt(d):
    return f"{d['avg. k']:>7.1f} / {d['ndcg@10']:.3f}" if d else "      – / –"


os.makedirs(f"{args.repo}/results/reproduction", exist_ok=True)
for table_dir, ret, reranker in PIPELINES:
    lines = [f"# {ret} → {reranker}  (avg. k / nDCG@10)\n",
             "| k-file | ours | paper (reported) | paper k-files on our labels |", "|---|---|---|---|"]
    for ds in ("dl-19-passage", "dl-20-passage"):
        labels = f"{WORK}/datasets/msmarco-v1-passage/labels/{ds}.label-original-{ret}-1000.{reranker}-1000-ndcg@10.json"
        if not os.path.exists(labels):
            lines.append(f"| {ds}: labels missing | | | |")
            continue
        pipe = f"{ds}.original-{ret}-1000"
        ours = sorted(glob.glob(f"{WORK}/output/{pipe}/{pipe}.*"))
        ours += sorted(glob.glob(f"{WORK}/output/seed{args.seed}/{pipe}/{pipe}.*-{args.epoch}"))
        ours = [p for p in ours if relevant(os.path.basename(p), pipe, reranker)]
        reported = paper_numbers(table_dir, ds, ret, reranker)
        for p in ours:
            name = os.path.basename(p)
            mine = evaluation(p, labels)
            paper_k = f"{UP}/results_in_papers/{table_dir}/{name}"
            on_ours = evaluation(paper_k, labels) if os.path.exists(paper_k) else None
            lines.append(f"| `{name.replace(pipe + '.', '')}` | {fmt(mine)} | {fmt(reported.get(name))} | {fmt(on_ours)} |")
    md = "\n".join(lines) + "\n"
    open(f"{args.repo}/results/reproduction/{table_dir}.md", "w").write(md)
    print(md)
