"""Condense results/reproduction/table*.md into a per-method-family summary of |ours - paper|.
Usage: python repro_stats.py <results/reproduction dir>"""
import glob
import os
import re
import sys

import pandas as pd

d = sys.argv[1]
rows = []
for path in sorted(glob.glob(f"{d}/table*.md")):
    table = os.path.basename(path).removesuffix(".md")
    year = None
    for line in open(path):
        m = re.match(r"\| `(.+?)` \|\s+([\d.]+) / ([\d.]+) \|\s+([\d.–]+) / ([\d.–]+) \|", line)
        if not m:
            continue
        name, k, nd, pk, pn = m.groups()
        if name == "fixed-1":
            year = "DL19" if year is None else "DL20"
        if pn == "–":
            continue
        fam = re.match(r"(fixed|greedy|oracle|surprise|bicut|choppy|attncut|mmoecut)", name).group(1)
        rows.append(dict(table=table, year=year, family=fam, name=name, k=float(k), ndcg=float(nd),
                         paper_k=float(pk), paper_ndcg=float(pn)))
df = pd.DataFrame(rows)
df["d_ndcg"] = (df.ndcg - df.paper_ndcg).round(3)
df["d_k"] = (df.k - df.paper_k).round(1)
g = df.groupby("family").agg(n=("d_ndcg", "size"), mean_abs_d_ndcg=("d_ndcg", lambda s: s.abs().mean()),
                             max_abs_d_ndcg=("d_ndcg", lambda s: s.abs().max()),
                             exact_ndcg=("d_ndcg", lambda s: (s == 0).mean()),
                             mean_abs_d_k=("d_k", lambda s: s.abs().mean())).round(3)
worst = df.reindex(df.d_ndcg.abs().sort_values(ascending=False).index).head(8)
md = ["# Reproduction vs paper: summary\n",
      "Per method family over all 4 tables x 2 years (nDCG@10 and avg. k differences; exact = share of rows equal to the paper at 3 decimals).\n",
      g.to_markdown(), "", "## Largest nDCG@10 gaps\n",
      worst[["table", "year", "name", "k", "ndcg", "paper_k", "paper_ndcg", "d_ndcg"]].to_markdown(index=False), ""]
open(f"{d}/summary.md", "w").write("\n".join(md))
print("\n".join(md))
