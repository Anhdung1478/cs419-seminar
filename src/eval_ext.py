"""Score extension k-files (epoch 100) with upstream rlt/evaluation.py and append rows to a CSV.

k-file name: <test_ds>.<test_ret>.<method>-ckpt-<train_ds>.<train_ret>.<method>.<param>-<epoch>
Each is scored against the TEST pipeline's RankLLaMA nDCG@10 labels.
"""
import argparse
import csv
import glob
import logging
import os
import sys

ap = argparse.ArgumentParser()
ap.add_argument("--out_root", required=True)
ap.add_argument("--labels", required=True)
ap.add_argument("--csv", required=True)
ap.add_argument("--epoch", default="100")
ap.add_argument("--meta", nargs="*", default=[], help="key=value columns added to every row")
a = ap.parse_args()

sys.path.insert(0, os.path.join(os.environ["UP"], "rlt"))
from evaluation import evaluation  # noqa: E402  (upstream)

logging.getLogger().setLevel(logging.WARNING)
meta = dict(kv.split("=", 1) for kv in a.meta)
FIELDS = list(meta) + ["method", "param", "train_ds", "train_ret", "test_ds", "test_ret", "avg_k", "ndcg10", "n_q"]

rows = []
for path in sorted(glob.glob(f"{a.out_root}/*/*-ckpt-*-{a.epoch}")):
    name = os.path.basename(path)
    test, ckpt = name.split("-ckpt-", 1)
    test_ds, test_ret, method = test.split(".", 2)
    train_ds, train_ret, _, rest = ckpt.split(".", 3)
    param = rest.rsplit("-", 1)[0]
    param = param if method == "bicut" else "beta" + param.rsplit("beta", 1)[1]
    labels = f"{a.labels}/{test_ds}.label-{test_ret}.rankllama-1000-ndcg@10.json"
    r = evaluation(path, labels)
    rows.append({**meta, "method": method, "param": param, "train_ds": train_ds, "train_ret": train_ret,
                 "test_ds": test_ds, "test_ret": test_ret, "avg_k": r["avg. k"], "ndcg10": r["ndcg@10"],
                 "n_q": r["# q"]})

os.makedirs(os.path.dirname(a.csv), exist_ok=True)
new = not os.path.exists(a.csv)
with open(a.csv, "a", newline="") as f:
    w = csv.DictWriter(f, fieldnames=FIELDS)
    if new:
        w.writeheader()
    w.writerows(rows)
print(f"{len(rows)} rows -> {a.csv}")
