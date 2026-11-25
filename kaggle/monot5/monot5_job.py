"""Kaggle job (GPU T4 x2): MonoT5 re-ranking of all 6 first-stage runs with src/monot5_rerank.py.
BM25 runs = reproduction (paper Table 4); SPLADE++/RepLLaMA runs = prepared for extension E4.
Two queues in parallel, one per T4. Outputs: dl-XX-passage.run-original-<ret>-1000-monot5-1000.txt + logs.
"""
import glob
import os
import shutil
import subprocess
import sys
import time
from multiprocessing import Process

inp = os.path.dirname(glob.glob("/kaggle/input/**/monot5_rerank.py", recursive=True)[0])
W = "/kaggle/working"
R = "/kaggle/tmp/runs"
os.makedirs(R, exist_ok=True)
os.makedirs(f"{W}/logs", exist_ok=True)
subprocess.run('pip install -q pytrec-eval-terrier "transformers>=4.36,<5" sentencepiece', shell=True, check=True)

jobs = []
for run in sorted(glob.glob(f"{inp}/dl-*-passage.run-original-*-1000.txt")):
    shutil.copy(run, R)
    name = os.path.basename(run)
    s = name.split(".")[0]  # dl-19-passage
    jobs.append((f"{R}/{name}", f"{inp}/{s}.queries-original.tsv", f"{inp}/{s}.qrels.txt"))
assert len(jobs) == 6, jobs


def queue(gpu, mine):
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu))
    for run, q, qrels in mine:
        name = os.path.basename(run).removesuffix(".txt")
        t0 = time.time()
        with open(f"{W}/logs/{name}-monot5.log", "w") as log:
            subprocess.run([sys.executable, "-u", f"{inp}/monot5_rerank.py", "--query_path", q, "--run_path", run,
                            "--qrels_path", qrels, "--collection", f"{inp}/msmarco.needed.tsv", "--k", "1000",
                            "--batch_size", "64"], env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
            out = f"{R}/{name}-monot5-1000.txt"
            shutil.copy(out, W)
            res = subprocess.run([sys.executable, f"{inp}/eval_run.py", "--run", out, "--qrels", qrels],
                                 capture_output=True, text=True).stdout
            log.write(res)
        print(f"[gpu{gpu}] {name}: {time.time() - t0:.0f}s  {res.strip()}", flush=True)


procs = [Process(target=queue, args=(g, jobs[g::2])) for g in (0, 1)]
for p in procs:
    p.start()
for p in procs:
    p.join()
assert all(p.exitcode == 0 for p in procs), "a MonoT5 queue failed; see logs/"
print(sorted(os.listdir(W)))
