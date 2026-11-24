# Experiment Plan: RLT for LLM-based Re-Ranking (PC vs Kaggle)

Paper: Meng et al., *Ranked List Truncation for Large Language Model-based Re-Ranking* (SIGIR 2024)
Code: https://github.com/ChuanMeng/RLT4Reranking

## 1. Hardware

| | Your PC | Kaggle (free) |
|---|---|---|
| GPU | RTX 2060 SUPER, **8 GB** | 2× T4 (16 GB each) or 1× P100 (16 GB) |
| RAM | 16 GB | ~29 GB |
| CPU | Xeon E5-2680 v4 (14C/28T) | 4 vCPU |
| Disk | D: 133 GB free | ~20 GB saved output (`/kaggle/working`) |
| Limits | none | 30 GPU-h/week, 12 h per session |
| OS | Windows + **WSL2 Ubuntu (already installed)** | Linux |

**How to split the work:** a 7B model in fp16 is about 13.5 GB, so it can't fit on 8 GB. Steps that run a 7B model go to Kaggle. Everything else runs on your PC, which has the large collection, the indexes and more CPU cores.

## 2. Pipeline split

| # | Step (README section) | Where | Why / notes |
|---|---|---|---|
| 0 | Environment setup | **PC (WSL2)** | The README recommends Linux. Use conda with Python 3.10. Pyserini needs Java 21. Don't use Windows Python 3.14, which is too new for pyserini, faiss and torch pins. Make a 2nd env for PyGaggle. |
| 1 | Download MS MARCO passage, DL19/DL20 queries and qrels (2.1) | **PC** | collection.tsv is ~3 GB. Keep data on `D:` (`/mnt/d/...`) because C: has only 44 GB free. |
| 2 | BM25 retrieval, top-1000 (2.2) | **PC** (CPU) | Pyserini prebuilt index, 97 queries in total, takes minutes. |
| 3 | SPLADE++ retrieval (2.2) | **PC** | Query encoding on CPU or GPU is cheap; the prebuilt impact index is several GB. |
| 4 | RepLLaMA runs (2.2) | **PC** | Just download the Dropbox runs and run `format.py`. No model needed. |
| 5 | `prepare_rerank_file.py` | **PC** | Writes query + passage text for top-1000. It's small (tens of MB), so **upload it to Kaggle as a private Dataset**. Then Kaggle never needs the 3 GB collection. |
| 6 | **RankLLaMA-7B re-ranking**, 3 retrievers × 2 datasets (2.3) | **Kaggle GPU (T4)** | ~300k query–passage pairs. Fits a 16 GB T4 in fp16; drop batch size to 8–16 if you run out of memory. Rough estimate: 3–5 GPU-h. Download the run files back to the PC (they're small). |
| 7 | MonoT5-base re-ranking, BM25 only (2.3) | **PC GPU** | 220M parameters, so it fits 8 GB easily. Use the PyGaggle env. |
| 8 | Features: tf-idf and doc2vec, then `--mode infer` (2.4) | **PC first**, Kaggle CPU as fallback | Fitting tf-idf and doc2vec on 8.8M passages may exceed 16 GB RAM. Watch it with `htop`. If it runs out of memory, run a Kaggle **CPU** notebook (~29 GB RAM, which doesn't use GPU quota). |
| 9 | LeCut RepLLaMA embeddings (`embedding.py`) | **Kaggle GPU** | 7B encoder over ~97k passages at 512 tokens. Rough estimate: 1–2 GPU-h. |
| 10 | Training labels (`reranking_labels.py`) (2.5) | **PC** | CPU only, just runs evaluation. |
| 11 | Unsupervised RLT: fixed-k, greedy-k, surprise, oracle (3.1) | **PC** | CPU, takes seconds. |
| 12 | Supervised RLT: BiCut, Choppy, AttnCut, MMoECut, LeCut (3.2) | **PC GPU** | Tiny models on 43/54 training queries. 100 epochs takes minutes. |
| 13 | Evaluation + `plots.ipynb` (3.3, 4) | **PC** | |

**Scope cut:** skip **Robust04 + RankLLaMA**. It uses `p_max_len 2048` over 250 × 1000 docs, which would use about a week of Kaggle quota or more. Focus on DL19/DL20 to reproduce the paper. Mention Robust04 in the report only by citing the paper's numbers.

**Kaggle checklist:**
- `meta-llama/Llama-2-7b-hf` (the tokenizer/base for RankLLaMA) is **gated**. Accept the Meta license on Hugging Face and add `HF_TOKEN` as a Kaggle Secret.
- Pick **GPU T4 x2**, not P100 (T4 is faster for fp16). Use `device_map="auto"` if one T4 isn't enough.
- Write runs to `/kaggle/working` and checkpoint each (retriever, dataset) run separately, because sessions get killed at 12 h.
- Set `HF_HOME=/kaggle/tmp` (or similar) so the 13 GB model doesn't fill the saved-output quota.

**Expected Kaggle usage:** about 5–8 GPU-h for the reproduction, which fits in one week's 30 h with room for extensions.

## 3. Data flow

```
PC (WSL2)                                   Kaggle
─────────                                   ──────
download data → BM25/SPLADE++/RepLLaMA runs
prepare_rerank_file.py ──(upload Dataset)──► RankLLaMA re-rank (T4)
                                            RepLLaMA embeddings (T4)
run files + embeddings ◄──(download output)──┘
MonoT5 re-rank (2060S)
features → labels → unsupervised/supervised RLT → eval → plots
```

## 4. Limitation / improvement experiments (pick 2–3)

Ordered from cheapest to most expensive. All except E5 and E6 reuse the runs above and run on the **PC**.

| ID | Hypothesis / limitation | Experiment | Where |
|---|---|---|---|
| E1 | **Tiny training sets**: supervised RLT is trained on only 43 or 54 queries, so results may be unstable. | Re-train each supervised method with 5–10 random seeds. Report mean ± std of nDCG@10 and avg-k. Check whether the differences between methods hold up. | PC |
| E2 | **Avg-k ignores overhead**: the cost metric leaves out feature extraction, e.g. doc2vec inference. | Measure wall-clock time of the RLT step against the re-ranking time it saves (use RankLLaMA timing from Kaggle). Report the real speedup. | PC + Kaggle timings |
| E3 | **Transfer across retrievers**: does a cutoff model trained on one retriever's lists work on another's? | Train on BM25→RankLLaMA and test on SPLADE++/RepLLaMA→RankLLaMA, and the reverse. | PC |
| E4 | **Cascade features**: RLT uses only first-stage features. | Add cheap MonoT5 scores, already computed, as item features to predict the cutoff for the expensive RankLLaMA. Compare to the original features. | PC |
| E5 | **Cheaper per-item cost instead of fewer items**: is truncation better than quantization? | Run RankLLaMA in 4-bit on your 2060S (bitsandbytes) for one setting, e.g. BM25 DL19. Compare the effectiveness/cost curve with RLT. | PC GPU (slow but fits) |
| E6 | **Listwise LLM re-rankers**: the paper only studies pointwise ones. | RankZephyr-7B (rank_llm, sliding window) over the top-100 for BM25 DL19/DL20. Check whether RLT conclusions hold when cost is counted in windows. | Kaggle (2× T4) |

Recommended set for the seminar: **E1 + E3 + E4**. All three are cheap, and each gives a clear "limitation → improvement" story. Add E6 only if you have Kaggle quota left.

## 5. Order of work

1. Set up the PC (WSL2 envs) and do steps 1–5, then upload to Kaggle.
2. Start the RankLLaMA notebook on Kaggle (step 6). **While it runs**, do MonoT5 and features (steps 7–8) on the PC.
3. Kaggle step 9 (LeCut embeddings).
4. PC steps 10–13: reproduce the paper's DL19/DL20 tables and check them against the paper.
5. Run the extension experiments, then write up the slides and report.
