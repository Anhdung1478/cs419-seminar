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
| 6 | **RankLLaMA-7B re-ranking**, 3 retrievers × 2 datasets (2.3) | **Kaggle GPU (T4)** | ~300k query–passage pairs. Fits a 16 GB T4 in fp16; drop batch size to 8–16 if you run out of memory. Rough estimate: 4–8 GPU-h (see §6). Download the run files back to the PC (they're small). |
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

**Expected Kaggle usage:** about 7–13 GPU-h for the reproduction (see §6), which fits in one week's 30 h with room for extensions.

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

## 5b. Scope decision: run only the parts the extensions need

The repo doesn't publish re-ranked runs, features, labels or checkpoints. The extensions need all of these, so most of the original pipeline has to run anyway. Run it **once**, limited to what E1, E3 and E4 need. That one run also serves as the reproduction.

| Part | Keep? | Why |
|---|---|---|
| BM25 → RankLLaMA, DL19 + DL20 | **Keep** | Main setting. Use it to check your numbers against the paper. |
| SPLADE++ → RankLLaMA | **Keep** | E3 needs at least two retrievers. |
| RepLLaMA → RankLLaMA | Optional | Adds a dense retriever to E3. The runs download for free, but re-ranking costs +1.5–3 GPU-h. |
| BM25 → MonoT5 | **Keep** | Runs on the PC, and E4 needs it. |
| LeCut (RepLLaMA embeddings) | Skip | Saves 1.5–3 GPU-h. Cite the paper's LeCut numbers instead. |
| Robust04 | Skip | Too expensive. |

**Sanity check before the extensions:** compare your oracle, fixed-k and one supervised method (e.g. AttnCut) on BM25→RankLLaMA with the paper's table. They should match to within about ±0.01 nDCG@10. If they don't, fix the setup first.

**Trimmed cost:** ~3.5–7 Kaggle GPU-h (2 retrievers), compared with 7–13 h for the full pipeline.

## 6. Time estimates

These are back-of-envelope numbers, not measurements. They rest on these assumptions:
- **RankLLaMA-7B fp16 on one T4: ~8–12 query–passage pairs/s.** Each pair is ~180 tokens, about 2.4 TFLOP, at ~20–25 effective TFLOPS. Two runs go in parallel, one per GPU, on a 2× T4 session.
- **MonoT5-base on the 2060S: ~50–150 pairs/s.**
- **Pair counts:** 43 queries (DL19) + 54 queries (DL20) = 97 queries × 1000 docs = 97k pairs per retriever.
- **Kaggle quota is counted per session hour,** not per GPU.

**Calibrate early.** Time the first 1000 RankLLaMA pairs. If the rate isn't ~10 pairs/s, scale every Kaggle number below by (10 ÷ your rate).

Quota checked 2026-11-24: 30 GPU-h, 0 used. The quota resets 2026-11-28, so you get a second 30 h window soon after.

### 6.1 Reproduction (DL19 + DL20)

| Step | Where | Compute time | Notes |
|---|---|---|---|
| Environment setup (conda, pyserini, tevatron, pygaggle) | PC | — | **3–6 h of your time**, mostly debugging dependencies |
| Download collection + prebuilt indexes (BM25 full, SPLADE++) | PC | 0.5–1.5 h | depends on your bandwidth; ~10 GB in total |
| BM25 + SPLADE++ retrieval, RepLLaMA formatting | PC | 10–20 min | |
| `prepare_rerank_file.py` + upload Kaggle Dataset | PC | 15–30 min | |
| **RankLLaMA re-ranking, 3 × 97k = 291k pairs** | Kaggle 2× T4 | **4–8 h** | ~8–16 GPU-h if the model must be split across both T4s (no parallel runs) |
| Model download + load overhead (~3 sessions) | Kaggle | 0.5–1 h | 13.5 GB per session |
| **RepLLaMA embeddings for LeCut** (~97k passages) | Kaggle 2× T4 | **1.5–3 h** | |
| MonoT5 re-ranking (BM25, 97k pairs) | PC GPU | 20–60 min | |
| tf-idf + doc2vec fit on 8.8M passages | PC CPU | 1.5–4 h | ×3–5 on Kaggle CPU (4 cores) if it runs out of memory |
| Feature inference for 4 pipelines × 2 datasets | PC CPU | 20–60 min | |
| Training labels (nDCG@10 at every k, 8 runs) | PC CPU | 10–30 min | |
| Unsupervised RLT | PC | < 5 min | |
| Supervised RLT (~100–150 trainings, ~1 step per epoch) | PC GPU | 0.5–2 h | can run overnight with a script |
| Evaluation + plots | PC | 15–30 min | |
| **Total** | | **Kaggle ≈ 7–13 GPU-h · PC ≈ 5–10 h compute** | **calendar ≈ 3–5 days** |

### 6.2 Extension experiments

| ID | Extra compute | Where | Your time (coding + analysis) | Notes |
|---|---|---|---|---|
| E1 seed variance | reduced: 5 seeds × 6 methods × 2 pipelines × 2 directions × 1 β ≈ **1 h**; full: 3–8 h | PC GPU | 2–3 h | runs unattended overnight |
| E2 real wall-clock cost | ~30 min PC + **~0 Kaggle** if you add timers to step 6 now (otherwise ~15 min re-run) | PC + Kaggle | 2–4 h | **put `time.perf_counter()` around batches in the RankLLaMA notebook before running it** |
| E3 cross-retriever transfer | **< 30 min** (reuse the trained models; inference only) | PC | 2–4 h | score scales differ (BM25 vs SPLADE vs dense), so try z-score normalization too |
| E4 MonoT5 scores as features | 40–120 min MonoT5 on SPLADE/RepLLaMA lists + ~1 h retraining | PC GPU | 3–5 h | change `features.py` and the input dimensions; count MonoT5's own cost fairly |
| E5 4-bit RankLLaMA on 2060S | **2–4 h per dataset** (~3–6 pairs/s) | PC GPU | 1–2 h | run in WSL; 8-bit (~7 GB) won't fit on 8 GB |
| E6 RankZephyr listwise RLT | **4–10 GPU-h** (cutoffs k ∈ {20…100}, ~25 windows per query, vLLM on 2× T4) | Kaggle | 4–8 h | add 1–2 h for vLLM/Kaggle version issues; may need the second quota window |

**Recommended set (E1 + E3 + E4):** ~3–5 h of PC compute, 0 Kaggle hours, ~1.5–2 days of your time.
**All extensions:** adds ~10–20 h of PC compute and 4–10 Kaggle GPU-h.
**Kaggle total, worst case** (reproduction + E6): ~23 GPU-h, which still fits in one 30 h window.
