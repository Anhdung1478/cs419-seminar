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
| Supervised RLT | — | Patch 0003: env-gated hooks `RLT_INFER_EPOCHS` (infer only epoch 100, used by `07_rlt.sh`) and `RLT_EXTRA_FEATURES` (extra feature columns, used by E4) | Epoch 100 is the epoch the paper reports. Kaggle's disk cannot hold all 100 checkpoints of every model (~54 GB). |

## 3. Extension design notes

- **BiCut** is excluded from E1/E3/E4. Its loss is a pure-Python double loop over every item of every list: about 10 s/epoch and about 16 min per training, against 35 s for AttnCut. It also ignores the re-ranking labels.
- Extensions infer only **epoch 100**, which is the epoch the paper reports.
- **E1 seeds:** 42 (the paper's default), 1, 2, 3, 4.

## 3b. E3 result: cross-retriever transfer (seed 42, `results/ext/e3_summary.md`)

The transfer gap below is nDCG@10 of a model trained on another retriever, minus a model trained on the test retriever. Avg. k is shown alongside.

| Train → test | Raw scores: gap (mean / worst) | Z-scored scores: gap (mean / worst) |
|---|---|---|
| BM25 → SPLADE++ | −0.023 / −0.053 | **+0.002 / −0.010** |
| SPLADE++ → BM25 | −0.000 / −0.118 | −0.046 / −0.125 |

- With raw scores, BM25-trained models applied to SPLADE++ often collapse to k≈1, giving no re-ranking (.731/.720), or jump to k≈1000. SPLADE++ impact scores are far outside the BM25 training range.
- Per-query z-scoring removes the BM25 → SPLADE++ gap and also lowers avg. k: on SPLADE++ from 446 to 374.
- The reverse direction looks fine with raw scores only because those models re-rank almost everything (mean avg. k 562 on BM25), which gives full quality at nearly full cost. With z-scores they cut much earlier (avg. k ≈ 138) and lose quality on BM25.
- **Takeaway:** the score feature is retriever-specific. Normalising it fixes the collapses, but a cutoff learned on a strong retriever still cuts too early for a weak one.

## 3c. E1 result: seed variance of supervised RLT (`results/ext/e1_summary.md`)

Setup: Choppy, AttnCut and MtCut × β∈{0,1,2}; BM25 and SPLADE++ → RankLLaMA; both train→test directions; seeds 42, 1, 2, 3, 4. That is 180 runs.

| Quantity | Value |
|---|---|
| Mean std of nDCG@10 across seeds | 0.007 |
| Largest max−min nDCG@10 across seeds | 0.050 (Choppy β=2, SPLADE++) |
| Mean std of avg. k across seeds | 144 items |
| (pipeline, year, β) groups where the best method changes with the seed | 7 / 12 |

- With 43/54 training queries, the seed spread is as large as the reproduction gaps and as the differences between methods in the paper's tables. **Comparisons between single-seed supervised methods are not reliable**, so they should be reported as mean ± std.
- MtCut on SPLADE++ DL20 predicts k≈1000 for every β and every seed. It ignores the efficiency side of EET entirely.

## 3d. E4 result: MonoT5 score as an extra RLT feature (`results/ext/e4_summary.md`)

Cost model, measured on Kaggle T4s: MonoT5-base scores ~58 pairs/s and RankLLaMA-7B ~7.2 pairs/s. So scoring all 1000 items with MonoT5 costs about the same as re-ranking **124 items** with RankLLaMA.

| Pipeline | Δ nDCG@10 (mean over methods) | Δ avg. k | Pays off after the MonoT5 cost? |
|---|---|---|---|
| RepLLaMA → RankLLaMA | −0.002 … +0.015 | −268 (AttnCut), −182 (MtCut) | Yes (net −144 / −58 items) |
| BM25 → RankLLaMA | −0.007 … +0.007 | −60 … +8 | No |
| SPLADE++ → RankLLaMA | −0.018 … +0.008 | −85 … +155 | No |

- Every nDCG change is within the seed noise from E1 (std 0.007, range up to 0.05). The only effect clearly above noise is the avg. k reduction on RepLLaMA (≈2σ of the avg. k std).

## 4. Compute log

| Job | Where | Time |
|---|---|---|
| Features (tokenise 17 min, tf-idf 8 min, doc2vec 4.1 h, infer 25 min) | Kaggle CPU | 5.0 h |
| MonoT5, 6 runs | Kaggle T4×2 | ~45 min |
| RankLLaMA, 6 runs (291k pairs, about 7.2 pairs/s per T4) | Kaggle T4×2 ×2 jobs | ~3.8 h wall-clock |
| Failed attempts (numpy/jax clash, KV-cache OOM), each caught by the 500-pair smoke test | Kaggle T4 | ~20 min |
