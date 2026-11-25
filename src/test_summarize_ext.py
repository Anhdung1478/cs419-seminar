"""Synthetic-data check for summarize_ext.py (all three modes). Run: python test_summarize_ext.py <tmpdir>"""
import itertools
import os
import random
import subprocess
import sys

import pandas as pd

tmp = sys.argv[1]
os.makedirs(tmp, exist_ok=True)
here = os.path.dirname(os.path.abspath(__file__))
rets = ["original-bm25-1000", "original-splade-pp-ed-pytorch-1000", "original-repllama-1000"]
cfgs = [(m, f"beta{b}") for m in ("choppy", "attncut", "mmoecut") for b in (0, 1, 2)]
rng = random.Random(0)


def row(**kw):
    return {"avg_k": rng.uniform(10, 900), "ndcg10": rng.uniform(0.6, 0.8), "n_q": 43, **kw}


rows = {"e1": [], "e3": [], "e4": []}
for seed, r, (m, p), s in itertools.product([42, 1, 2], rets[:2], cfgs, ["dl-19-passage", "dl-20-passage"]):
    rows["e1"].append(row(exp="E1", seed=seed, variant="raw", method=m, param=p, train_ds=s, train_ret=r, test_ds=s, test_ret=r))
for v, rs, rt, (m, p), s in itertools.product(["raw", "zscore"], rets[:2], rets, cfgs, ["dl-20-passage"]):
    rows["e3"].append(row(exp="E3", seed=42, variant=v, method=m, param=p, train_ds="dl-19-passage", train_ret=rs, test_ds=s, test_ret=rt))
for v, r, (m, p), s in itertools.product(["base", "monot5"], rets, cfgs, ["dl-19-passage", "dl-20-passage"]):
    rows["e4"].append(row(exp="E4", seed=42, variant=v, method=m, param=p, train_ds=s, train_ret=r, test_ds=s, test_ret=r))
for k, rs in rows.items():
    pd.DataFrame(rs).to_csv(f"{tmp}/{k}.csv", index=False)
    subprocess.run([sys.executable, f"{here}/summarize_ext.py", k, f"{tmp}/{k}.csv", f"{tmp}/{k}.md"],
                   check=True, stdout=subprocess.DEVNULL)
    print(k, "OK", os.path.getsize(f"{tmp}/{k}.md"), "bytes")
