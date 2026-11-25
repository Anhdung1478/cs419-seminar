# cs419-seminar

CS419 seminar on **Ranked List Truncation for Large Language Model-based Re-Ranking** (SIGIR 2024).

- Paper: https://dl.acm.org/doi/10.1145/3626772.3657864
- Reference code: https://github.com/ChuanMeng/RLT4Reranking

See [docs/seminar-requirements.md](docs/seminar-requirements.md) for the seminar requirements and
[docs/experiment-plan.md](docs/experiment-plan.md) for the experiment plan.

## Reproduction pipeline

The upstream code is a git submodule at `third_party/RLT4Reranking`, pinned to `df01264`. It is checked out
sparsely (Robust04 results excluded, because their filenames contain `*`). Clone with:

```bash
# inside WSL (Linux git; Windows git cannot check out the upstream tree)
git clone https://github.com/Anhdung1478/cs419-seminar && cd cs419-seminar
git submodule init && git -C third_party/RLT4Reranking sparse-checkout set --no-cone '/*' '!/results_in_papers/robust04/'
git submodule update
```

Run the PC steps inside WSL2 from the repo root. Data, caches and outputs go to `work/`, which is gitignored.
Heavy steps ran on Kaggle; each `kaggle/<job>/` folder is a pushable kernel (Kaggle CLI 2.x, env `kag`, from
`scripts/kaggle_cli_setup.sh`). Kaggle API tokens are never stored in the repo: they live in WSL `~/.kaggle/`.

| Step | Where it ran | How |
|---|---|---|
| Env (conda `rlt`, Py 3.10, JDK 11) + upstream patches | PC | `bash scripts/00_setup.sh` |
| Data + pyserini indexes | PC | `bash scripts/01_download.sh` |
| BM25 / SPLADE++ / RepLLaMA runs + gate 1 | PC | `bash scripts/02_retrieve.sh` |
| RankLLaMA inputs → private Kaggle dataset | PC | `bash scripts/03_pairs.sh`, `scripts/kaggle_upload_*.sh` |
| RankLLaMA re-ranking (upstream script, 6 runs) | Kaggle GPU T4×2 | `scripts/kaggle_push_rankllama.sh` → `kaggle/rankllama/` |
| MonoT5 re-ranking (6 runs) | Kaggle GPU T4×2 | `kaggle/monot5/` (PC alternative: `scripts/04_monot5.sh`) |
| Features (tf-idf, doc2vec; low-memory rewrite) | Kaggle CPU | `kaggle/features/` (PC alternative: `scripts/05_features.sh`) |
| Labels (+ Oracle checks) | PC | `bash scripts/06_labels.sh` / `scripts/check_labels_one.sh` |
| Unsupervised + supervised RLT | Kaggle GPU (+ CPU for BiCut) | `scripts/kaggle_push_ext.sh` running `07_rlt.sh` via `kaggle/ext/ext_job.py` |
| Evaluation vs paper → `results/reproduction/` | PC | `bash scripts/08_eval.sh`, `src/repro_stats.py` |

Results: [results/reproduction/summary.md](results/reproduction/summary.md). Checks, deviations from the original
setup and the compute log: [docs/reproduction-notes.md](docs/reproduction-notes.md).

Changes to the upstream code are kept as `patches/*.patch`; `scripts/apply_patches.sh` applies them, and
`scripts/check_patches.sh` verifies them on a fresh checkout. Our replacements for scripts that no longer run are
in `src/`; each file's docstring says what it replaces and why.
