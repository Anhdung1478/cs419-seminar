"""Kaggle job (GPU T4): run extension experiment scripts (scripts/e1.sh, e3.sh, e4.sh) on the reproduction's data.

Inputs (datasets on the running account):
  <user>/rlt-code      our scripts/, src/, patches/ (flat)
  <user>/rlt-e-inputs  feature JSONs, label JSONs, qrels, MonoT5 runs (flat)
STEPS and ENV are filled in by scripts/kaggle_push_ext.sh.
Outputs: results/ext/*.csv|md, logs, ext_outputs.tar.gz (k-files).
"""
import glob
import os
import shutil
import subprocess

STEPS = ["e1.sh"]
ENV = {}
UPSTREAM_COMMIT = "df012641062f9b8afde6e5b8266667f87816b242"
REPO = "/kaggle/tmp/repo"
W = "/kaggle/working"
D = f"{REPO}/work/datasets/msmarco-v1-passage"


def sh(cmd, log=None, env=None):
    print("+", cmd, flush=True)
    kw = dict(shell=True, check=True, executable="/bin/bash", env={**os.environ, **(env or {})})
    if log:
        with open(f"{W}/{log}", "w") as f:
            subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, **kw)
    else:
        subprocess.run(cmd, **kw)


def find(pattern):
    hits = sorted(glob.glob(f"/kaggle/input/**/{pattern}", recursive=True))
    assert hits, f"missing input: {pattern}"
    return hits


code = os.path.dirname(find("eval_ext.py")[0])
for sub in ("scripts", "src", "patches"):
    os.makedirs(f"{REPO}/{sub}", exist_ok=True)
for f in glob.glob(f"{code}/*"):
    n = os.path.basename(f)
    shutil.copy(f, f"{REPO}/{'scripts' if n.endswith('.sh') else 'patches' if n.endswith('.patch') else 'src'}/{n}")
up = f"{REPO}/third_party/RLT4Reranking"
if not os.path.isdir(up):
    sh(f"git clone -q --filter=blob:none --no-checkout https://github.com/ChuanMeng/RLT4Reranking {up} && "
       f"git -C {up} sparse-checkout set --no-cone '/*' '!/results_in_papers/robust04/' && "
       f"git -C {up} checkout -q {UPSTREAM_COMMIT}")
sh(f"bash {REPO}/scripts/apply_patches.sh")
sh("pip install -q pytrec-eval-terrier more-itertools attrs tabulate")

for sub in ("runs", "qrels", "features", "labels"):
    os.makedirs(f"{D}/{sub}", exist_ok=True)
for pat, sub in (("dl-*-passage.feature-original-*-1000.json", "features"), ("dl-*-passage.label-*.json", "labels"),
                 ("dl-*-passage.qrels.txt", "qrels"), ("dl-*-passage.run-original-*-monot5-1000.txt", "runs")):
    for f in find(pat):
        # Kaggle strips "@" from uploaded file names (ndcg@10 -> ndcg10); restore the upstream names
        shutil.copy(f, f"{D}/{sub}/{os.path.basename(f).replace('-ndcg10', '-ndcg@10')}")

failed = []
for step in STEPS:
    try:
        sh(f"bash {REPO}/scripts/{step}", log=f"{step}.log", env=ENV)
    except subprocess.CalledProcessError:
        failed.append(step)  # keep going: earlier CSV rows / k-files are still saved below

if os.path.isdir(f"{REPO}/results"):
    shutil.copytree(f"{REPO}/results", f"{W}/results", dirs_exist_ok=True)
sh(f"tar -czf {W}/ext_outputs.tar.gz -C {REPO}/work output")
for f in glob.glob(f"{W}/results/ext/*_summary.md"):
    print(open(f).read())
assert not failed, f"failed steps (see their logs): {failed}"
