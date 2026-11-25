"""Kaggle job (CPU, internet on, no GPU quota): RLT item features with src/features_lowmem.py.

Runs on Kaggle because tokenising 8.8M passages + doc2vec (~7 GB) does not fit next to Windows on a 16 GB PC.
Input dataset anhdung147/rlt-runs: 6 first-stage runs, 2 qrels, features_lowmem.py.
Outputs (/kaggle/working): dl-XX-passage.feature-original-<retriever>-1000.json (6 files) + features.log
"""
import glob
import os
import shutil
import subprocess
import sys
import time

T = "/kaggle/tmp/w"
D = f"{T}/datasets/msmarco-v1-passage"  # keep upstream layout: scripts parse names from paths
OUT = "/kaggle/working"
for sub in ("collection", "runs", "qrels", "features"):
    os.makedirs(f"{D}/{sub}", exist_ok=True)
log = open(f"{OUT}/features.log", "a")


def sh(cmd):
    print("+", cmd, flush=True)
    t0 = time.time()
    subprocess.run(cmd, shell=True, check=True, stdout=log, stderr=subprocess.STDOUT)
    msg = f"done in {time.time() - t0:.0f}s: {cmd[:120]}"
    print(msg, flush=True)
    log.write(msg + "\n")
    log.flush()


src = glob.glob("/kaggle/input/**/features_lowmem.py", recursive=True)[0]
inp = os.path.dirname(src)
for f in glob.glob(f"{inp}/*.run-original-*-1000.txt"):
    shutil.copy(f, f"{D}/runs/")
for f in glob.glob(f"{inp}/*.qrels.txt"):
    shutil.copy(f, f"{D}/qrels/")
shutil.copy(src, T)
runs = sorted(glob.glob(f"{D}/runs/*.txt"))
assert len(runs) == 6, runs

sh('pip install -q "gensim>=4.3.3" pytrec-eval-terrier')
if not os.path.exists(f"{D}/collection/msmarco.tsv"):
    sh(f"curl -fsSL https://msmarco.z22.web.core.windows.net/msmarcoranking/collection.tar.gz "
       f"| tar -xz -C {D}/collection && mv {D}/collection/collection.tsv {D}/collection/msmarco.tsv")

py = f"{sys.executable} -u {T}/features_lowmem.py --output_path {D}/features --workers 4"
sh(f"{py} --mode prep --index_path {D}/collection/msmarco.tsv")
os.remove(f"{D}/collection/msmarco.tsv")  # free disk
sh(f"{py} --mode tfidf")
sh(f"{py} --mode doc2vec --vector_size 128")
qrels = [f"{D}/qrels/{os.path.basename(r).split('.')[0]}.qrels.txt" for r in runs]
sh(f"{py} --mode infer --seq_len 1000 --run_path {' '.join(runs)} --qrels_path {' '.join(qrels)}")

for f in glob.glob(f"{D}/features/*.feature-*.json"):
    shutil.copy(f, OUT)
print(sorted(os.listdir(OUT)))
