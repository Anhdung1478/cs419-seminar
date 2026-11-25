"""Kaggle job (GPU T4 x2, internet on): RankLLaMA re-ranking with the UPSTREAM script, unchanged.

Inputs (attached to the kernel):
  - dataset with rerank_input.dl-XX-passage.run-original-<retriever>-1000.jsonl (built by scripts/03_pairs.sh)
  - base weights + tokenizer: Kaggle model metaresearch/llama-2 (7b-hf) if attached, otherwise downloaded
    from the ungated HF mirror NousResearch/Llama-2-7b-hf (no HF token needed)
Outputs (/kaggle/working):
  - dl-XX-passage.run-original-<retriever>-1000-rankllama-1000.txt   (one per input file)
  - timings.json, logs/*.log

Flow: setup -> smoke test (500 pairs) -> 2 queues in parallel, one per T4, each running
upstream reranker_inference.py over its files (batch 32, falls back to 16/8 on OOM; the README's 64
assumes a bigger GPU, and batch size does not change the scores).
"""
import glob
import json
import os
import subprocess
import sys
import time

UPSTREAM_COMMIT = "df012641062f9b8afde6e5b8266667f87816b242"
ADAPTER = "castorini/rankllama-v1-7b-lora-passage"
# Per-GPU work split, filled in by scripts/kaggle_push_rankllama.sh when the work is shared across accounts:
# [[run-name substrings for GPU0], [... for GPU1]]. None = all 6 files, balanced automatically.
QUEUES = None
W = "/kaggle/working"
os.makedirs(f"{W}/logs", exist_ok=True)


def sh(cmd, **kw):
    print("+", cmd, flush=True)
    subprocess.run(cmd, shell=True, check=True, **kw)


# ---------------------------------------------------------------- setup
t_setup = time.time()
up = "/kaggle/tmp/RLT4Reranking"
if not os.path.isdir(up):
    sh(f"git clone -q https://github.com/ChuanMeng/RLT4Reranking {up} && git -C {up} checkout -q {UPSTREAM_COMMIT}")
# Upstream was written for transformers 4.36 / peft 0.6, but Kaggle's image is Python 3.13 + numpy 2
# (downgrading numpy breaks the preinstalled jax that transformers imports). transformers 4.46 is the
# newest 4.x line that still has every API the upstream code uses (prepare_for_model, file_utils,
# TrainingArguments fields) and supports numpy 2 / Python 3.13. Keep Kaggle's numpy.
sh('pip install -q "transformers==4.46.3" "peft==0.13.2" "datasets==3.6.0" "huggingface_hub<1.0" "accelerate>=0.26,<2"')
sh(f"pip install -q --no-deps -e {up}/tevatron")
if sys.version_info >= (3, 13):  # data.py line 1 imports cgitb (removed in 3.13; unused)
    sh(f"sed -i '/^from cgitb import text$/d' {up}/tevatron/examples/rankllama/data.py")

from huggingface_hub import list_repo_files, snapshot_download  # noqa: E402

# Base Llama-2-7B: Kaggle model metaresearch/llama-2 (7b-hf) if attached, else the ungated HF mirror
# NousResearch/Llama-2-7b-hf (same weights/tokenizer as gated meta-llama/Llama-2-7b-hf).
base = sorted(glob.glob("/kaggle/input/llama-2/**/config.json", recursive=True))
base = [os.path.dirname(p) for p in base if "7b" in p and "chat" not in p]
if base:
    base = base[0]
else:
    mirror = "NousResearch/Llama-2-7b-hf"
    weights = "*.safetensors" if any(f.endswith(".safetensors") for f in list_repo_files(mirror)) else "*.bin"
    base = snapshot_download(mirror, local_dir="/kaggle/tmp/llama-2-7b-hf",
                             allow_patterns=["*.json", "tokenizer.model", weights])
print("base model:", base)

# Llama-2's config has use_cache=true, so every forward pass also stores keys/values for all 32 layers
# (~1.6 GB at batch 16). That does not fit next to the 12.6 GiB fp16 weights on a 14.6 GiB T4.
# RankLLaMA scores each pair with one forward pass and never reads the cache, so turning it off
# leaves the scores unchanged. Use a writable copy of the config (Kaggle inputs are read-only).
local_base = "/kaggle/tmp/llama-2-7b-base"
os.makedirs(local_base, exist_ok=True)
for f in os.listdir(base):
    dst = f"{local_base}/{f}"
    if f != "config.json" and not os.path.exists(dst):
        os.symlink(f"{base}/{f}", dst)
cfg_base = json.load(open(f"{base}/config.json"))
cfg_base["use_cache"] = False
json.dump(cfg_base, open(f"{local_base}/config.json", "w"))
base = local_base
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

adapter = snapshot_download(ADAPTER, local_dir="/kaggle/tmp/rankllama-adapter")
cfg_path = f"{adapter}/adapter_config.json"
cfg = json.load(open(cfg_path))
print("adapter base was:", cfg["base_model_name_or_path"])
cfg["base_model_name_or_path"] = base  # same Llama-2-7b weights, local copy
json.dump(cfg, open(cfg_path, "w"))

inputs = sorted(glob.glob("/kaggle/input/**/rerank_input.*.jsonl", recursive=True))
assert len(inputs) == 6, inputs
setup_s = time.time() - t_setup


def run_one(gpu, in_path, out_path, log_path, batch_sizes=(32, 16, 8)):
    for bs in batch_sizes:
        cmd = [sys.executable, "-u", "reranker_inference.py", "--output_dir=temp",
               "--model_name_or_path", adapter, "--tokenizer_name", base,
               "--encode_in_path", in_path, "--fp16", "--per_device_eval_batch_size", str(bs),
               "--q_max_len", "32", "--p_max_len", "164", "--dataset_name", "json",
               "--encoded_save_path", out_path]
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu))
        t0 = time.time()
        with open(log_path, "a") as log:
            rc = subprocess.run(cmd, cwd=f"{up}/tevatron/examples/rankllama", env=env,
                                stdout=log, stderr=subprocess.STDOUT).returncode
        if rc == 0:
            return {"seconds": round(time.time() - t0, 1), "batch_size": bs}
        if "OutOfMemoryError" not in open(log_path).read():
            raise RuntimeError(f"{in_path} failed (rc={rc}); see {log_path}")
        print(f"OOM at batch {bs} on {in_path}, retrying smaller", flush=True)
    raise RuntimeError(f"{in_path}: OOM even at batch {batch_sizes[-1]}")


def out_name(in_path):
    return os.path.basename(in_path).removeprefix("rerank_input.").removesuffix(".jsonl") + "-rankllama-1000.txt"


# ---------------------------------------------------------------- smoke test
smoke_in = "/kaggle/tmp/smoke.jsonl"
with open(inputs[0]) as f, open(smoke_in, "w") as g:
    for _, line in zip(range(500), f):
        g.write(line)
smoke = run_one(0, smoke_in, "/kaggle/tmp/smoke.txt", f"{W}/logs/smoke.log")
lines = open("/kaggle/tmp/smoke.txt").read().split("\n")[:-1]
assert len(lines) == 500 and all(len(l.split()) == 6 for l in lines), "smoke output malformed"
print(f"SMOKE OK: 500 pairs in {smoke['seconds']}s (batch {smoke['batch_size']})", flush=True)

# ---------------------------------------------------------------- 2 queues, one per T4
# Balance ~145k pairs per GPU: DL19 files have 43k lines, DL20 54k.
if QUEUES:
    queues = [[next(p for p in inputs if f"rerank_input.{n}.jsonl" in p) for n in q] for q in QUEUES]
else:
    by_size = sorted(inputs, key=lambda p: -sum(1 for _ in open(p)))
    queues = [[], []]
    for p in by_size:  # greedy balance
        min(queues, key=lambda q: sum(sum(1 for _ in open(x)) for x in q)).append(p)
n_gpus = len(subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True).stdout.strip().splitlines())
if n_gpus < 2:  # e.g. a single L4 (24 GB): run everything on GPU 0, one file after another
    queues = [sum(queues, [])]
print(f"{n_gpus} GPU(s); queues:", [[os.path.basename(p) for p in q] for q in queues], flush=True)

# Run each queue in its own process so the two GPUs work concurrently.
import multiprocessing as mp  # noqa: E402


def queue_main(gpu, files, result_q):
    res = {}
    try:
        for p in files:
            name = out_name(p)
            res[name] = run_one(gpu, p, f"{W}/{name}", f"{W}/logs/{name}.log")
            print(f"[gpu{gpu}] done {name}: {res[name]}", flush=True)
    except Exception as e:  # report instead of hanging the parent's rq.get()
        res[f"gpu{gpu}_error"] = repr(e)
    result_q.put(res)


def gpu_mem_used(gpu):
    out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits",
                          "-i", str(gpu)], capture_output=True, text=True).stdout
    return int(out.strip() or 0)


rq = mp.Queue()
procs = []
for gpu, files in enumerate(queues):
    if not files:
        continue
    p = mp.Process(target=queue_main, args=(gpu, files, rq))
    p.start()
    procs.append(p)
    # Stagger: each process holds the 13.5 GB fp16 model in CPU RAM while loading/merging LoRA.
    # Wait until GPU `gpu` has the model before starting the next one, so RAM never holds two copies.
    t0 = time.time()
    while gpu_mem_used(gpu) < 12000 and p.is_alive() and time.time() - t0 < 1800:
        time.sleep(15)

timings = {"setup_seconds": round(setup_s, 1), "smoke": smoke}
for _ in procs:
    timings.update(rq.get())
for p in procs:
    p.join()
json.dump(timings, open(f"{W}/timings.json", "w"), indent=1)
print(json.dumps(timings, indent=1))
errors = {k: v for k, v in timings.items() if k.endswith("_error")}
assert not errors, f"queue failures (finished files are still saved): {errors}"
