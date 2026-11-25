# Reproduction notes: RLT for LLM-based re-ranking (Meng et al., SIGIR 2024)

Scope: TREC DL19 + DL20 with these pipelines:
- BM25 → RankLLaMA
- SPLADE++ → RankLLaMA
- RepLLaMA → RankLLaMA
- BM25 → MonoT5

Out of scope: LeCut and Robust04.

## 1. Checks against the paper

| Gate | Quantity | DL19 ours / paper | DL20 ours / paper |
|---|---|---|---|
| Retrieval | BM25 nDCG@10 (no re-ranking) | .506 / .506 | .480 / .480 |
| | SPLADE++ | .731 / .731 | .720 / .720 |
| | RepLLaMA (author's runs) | .738 / .739 | .719 / .719 |
| Re-ranking | BM25 → MonoT5, Fixed-k(1000) | .730 / .730 | .713 / .713 |
| | BM25 → RankLLaMA, Fixed-k(1000) | .747 / .747 | .762 / .762 |
| | SPLADE++ → RankLLaMA | .768 / .768 | .768 / .768 |
| | RepLLaMA → RankLLaMA | .763 / .763 | .768 / .768 |
| Labels | Oracle, BM25 → RankLLaMA (avg. k) | .785 (156.9) / .785 (157) | .804 (228.6) / .804 (229) |
| | Oracle, SPLADE++ → RankLLaMA | .810 (17.4) / .810 (17) | .820 (27.7) / .820 (28) |
| | Oracle, RepLLaMA → RankLLaMA | .794 (23.3) / .794 (23.33) | .799 (42.5) / .799 (42.46) |

The RLT method tables are in `results/reproduction/*.md`, produced by `scripts/08_eval.sh`. `results/reproduction/summary.md` condenses them.

| Method family | Rows (4 tables × 2 years) | Mean \|Δ nDCG@10\| vs paper | Max \|Δ\| | Rows identical to paper |
|---|---|---|---|---|
| Fixed-k, Greedy-k, Surprise, Oracle | 96 | 0.000 | 0.000 | 100% |
| BiCut | 24 | 0.003 | 0.013 | 42% |
| AttnCut | 24 | 0.002 | 0.005 | 33% |
| MtCut | 24 | 0.005 | 0.022 | 21% |
| Choppy | 24 | 0.007 | 0.044 | 25% |

- **Labels are identical to the authors'.** Scoring the paper's own per-query cutoff files (`results_in_papers/`) against our labels reproduces the paper's numbers exactly. So every remaining gap comes from the trained models (training randomness, and doc2vec features we re-trained ourselves).
- **The largest gaps are a collapse to k=1, and they mirror each other.** Choppy β=2 on SPLADE++: our DL20 model predicts k=1 (no re-ranking, .720 vs paper .764), while the paper's DL19 model collapsed and ours did not (.762 vs .731). This instability motivates E1.

## 2. Deviations from the original setup and why

| Area | Original | Ours | Why / effect |
|---|---|---|---|
| pyserini | 0.20.0 | 0.22.1 | 0.20 has no SPLADE++ ED prebuilt index. BM25 numbers identical. |
| pytrec_eval | pytrec-eval 0.5 | pytrec-eval-terrier 0.5.6 | Same module and trec_eval; ships wheels, so no compiler is needed. |
| gensim | 3.2.0 | 4.3.3 (PC) / ≥4.3 (Kaggle) | 3.2 cannot be installed on Python ≥3.7. Same API. |
| Features | `rlt/features.py` keeps 8.8M passages in RAM (45–60 GB) | `src/features_lowmem.py` streams to disk; sparse tf-idf cosine; Doc2Vec in `corpus_file` mode | Same preprocessing, hyperparameters (128-d, 40 epochs, min_count 2) and output JSON. Doc2Vec is stochastic either way, so the features are not bit-identical to the authors' (which were never released). |
| RankLLaMA input | `prepare_rerank_file.py` loads HF `dl19`/`dl20` splits (removed upstream) | `src/prepare_pairs.py` reads the same Tevatron files directly | Identical JSONL fields; titles kept (`-` placeholder as in Tevatron). |
| RankLLaMA run | transformers 4.36, peft 0.6, A100-class GPU, batch 64 | Upstream `reranker_inference.py` unchanged, on Kaggle T4: transformers 4.46.3, peft 0.13.2, batch 32 | Kaggle image is Python 3.13 + numpy 2. All 6 runs reproduce the paper's Fixed-k(1000) exactly. |
| | Base `meta-llama/Llama-2-7b-hf` (gated) | Ungated mirror `NousResearch/Llama-2-7b-hf` | Same weights and tokenizer. |
| | `use_cache` true (KV cache) | `use_cache=false` in the base config | The KV cache does not fit on a 15 GB T4. Each pair is a single forward pass, so scores are unchanged. |
| MonoT5 | `monot5.py` via pygaggle (silently on CPU, all 200 queries) | `src/monot5_rerank.py`: same scoring (log P("true"), "Query: … Document: … Relevant:", 512 tokens), GPU, judged queries only | BM25 → MonoT5 reproduces .730/.713 exactly. |
| Labels | `reranking_labels.py` crashes on `ndcg@10` (`if` vs `elif`) | Patch 0001; also builds the evaluator once per query (same numbers) | — |
| Supervised RLT | — | Patch 0002 drops unused pyserini/BertTokenizer imports | — |
| Supervised RLT | — | Patch 0003: env-gated hooks `RLT_INFER_EPOCHS` (infer only epoch 100, used by `07_rlt.sh`) and `RLT_EXTRA_FEATURES` (extra feature columns, unused here) | Epoch 100 is the epoch the paper reports. Kaggle's disk cannot hold all 100 checkpoints of every model (~54 GB). |

## 3. Compute log

| Job | Where | Time |
|---|---|---|
| Features (tokenise 17 min, tf-idf 8 min, doc2vec 4.1 h, infer 25 min) | Kaggle CPU | 5.0 h |
| MonoT5, 6 runs | Kaggle T4×2 | ~45 min |
| RankLLaMA, 6 runs (291k pairs, about 7.2 pairs/s per T4) | Kaggle T4×2 ×2 jobs | ~3.8 h wall-clock |
| Failed attempts (numpy/jax clash, KV-cache OOM), each caught by the 500-pair smoke test | Kaggle T4 | ~20 min |
