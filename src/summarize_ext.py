"""Markdown summaries for the extension experiments.  Usage: summarize_ext.py {e1,e3,e4} <csv> <out.md>"""
import sys

import pandas as pd

kind, csv_path, out = sys.argv[1:]
df = pd.read_csv(csv_path)
df["config"] = df["method"] + " " + df["param"].astype(str)
short = {"original-bm25-1000": "BM25", "original-splade-pp-ed-pytorch-1000": "SPLADE++",
         "original-repllama-1000": "RepLLaMA"}
for c in ("train_ret", "test_ret"):
    df[c] = df[c].map(short).fillna(df[c])
lines = []

if kind == "e1":
    g = df.groupby(["test_ret", "test_ds", "config"])
    t = g.agg(seeds=("seed", "nunique"), k_mean=("avg_k", "mean"), k_std=("avg_k", "std"),
              ndcg_mean=("ndcg10", "mean"), ndcg_std=("ndcg10", "std"),
              ndcg_min=("ndcg10", "min"), ndcg_max=("ndcg10", "max")).reset_index().round(3)
    lines += ["# E1 - seed variance of supervised RLT (RankLLaMA, nDCG@10)\n", t.to_markdown(index=False), ""]
    lines += ["## Spread summary", f"- mean nDCG@10 std across seeds: {t['ndcg_std'].mean():.3f}",
              f"- max nDCG@10 range (max-min) across seeds: {(t['ndcg_max'] - t['ndcg_min']).max():.3f}",
              f"- mean avg-k std across seeds: {t['k_std'].mean():.1f}", ""]
    # Is the best method (per beta) the same for every seed?
    for (ret, ds, beta), sub in df[df.method != "bicut"].groupby(["test_ret", "test_ds", "param"]):
        winners = sub.loc[sub.groupby("seed")["ndcg10"].idxmax(), "method"].value_counts().to_dict()
        lines.append(f"- {ret} {ds} {beta}: best method per seed -> {winners}")

elif kind == "e3":
    for v, sub in df.groupby("variant"):
        p = sub.pivot_table(index=["train_ret", "test_ds", "config"], columns="test_ret",
                            values="ndcg10").round(3)
        lines += [f"# E3 - cross-retriever transfer, scores={v} (nDCG@10 on test retriever)\n", p.to_markdown(), ""]
    # transfer gap: model trained on another retriever vs model trained on the test retriever (same variant)
    ind = df[df.train_ret == df.test_ret].set_index(["variant", "test_ds", "config", "test_ret"])["ndcg10"]
    tr = df[df.train_ret != df.test_ret].copy()
    tr["in_domain"] = [ind.get((v, d, c, r)) for v, d, c, r in zip(tr.variant, tr.test_ds, tr.config, tr.test_ret)]
    tr["gap"] = tr["ndcg10"] - tr["in_domain"]
    gap = tr.dropna(subset=["gap"]).groupby(["variant", "train_ret", "test_ret"])["gap"].agg(["mean", "min", "count"])
    lines += ["## Transfer gap = nDCG@10(trained on other retriever) - nDCG@10(trained on test retriever)\n",
              gap.round(3).to_markdown(), ""]
    k = df.pivot_table(index=["variant", "train_ret"], columns="test_ret", values="avg_k", aggfunc="mean").round(1)
    lines += ["## Mean avg. k by train/test retriever\n", k.to_markdown(), ""]

elif kind == "e4":
    p = df.pivot_table(index=["test_ret", "test_ds", "config"], columns="variant",
                       values=["ndcg10", "avg_k"]).round(3)
    p[("delta", "ndcg10")] = (p[("ndcg10", "monot5")] - p[("ndcg10", "base")]).round(3)
    p[("delta", "avg_k")] = (p[("avg_k", "monot5")] - p[("avg_k", "base")]).round(1)
    lines += ["# E4 - MonoT5 score as an extra RLT feature (RankLLaMA pipelines)\n", p.to_markdown(), ""]
    d = p["delta"].reset_index()
    d["method"] = d["config"].str.split().str[0]
    lines += ["## Mean change (monot5 - base) by method and retriever\n",
              d.groupby(["test_ret", "method"])[["ndcg10", "avg_k"]].mean().round(3).to_markdown(), ""]

open(out, "w").write("\n".join(lines) + "\n")
print("\n".join(lines))
