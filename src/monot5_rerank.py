"""MonoT5 re-ranking with the same scoring and output as upstream monot5.py (pygaggle MonoT5),
but on GPU, without pygaggle/pyserini, judged queries only.

Score = log P("true") from log_softmax over the ("false","true") logits of the first decoded token,
input "Query: {q} Document: {d} Relevant:", max 512 tokens (pygaggle defaults).
Passage text comes from msmarco.tsv (identical to the pyserini index "contents" field).
Output: <run minus .txt>-monot5-<k>.txt, lines "qid Q0 docid rank score monoT5-<model>".
"""
import argparse
import time

import pytrec_eval
import torch
from tqdm import tqdm
from transformers import T5ForConditionalGeneration, T5Tokenizer

ap = argparse.ArgumentParser()
ap.add_argument("--query_path", required=True)
ap.add_argument("--run_path", required=True)
ap.add_argument("--qrels_path", required=True)
ap.add_argument("--collection", required=True)
ap.add_argument("--model", default="castorini/monot5-base-msmarco")
ap.add_argument("--k", type=int, default=1000)
ap.add_argument("--batch_size", type=int, default=32)
args = ap.parse_args()

queries = {}
with open(args.query_path) as f:
    for line in f:
        qid, qtext = line.split("\t")
        queries[qid] = qtext.strip()
with open(args.qrels_path) as f:
    judged = set(pytrec_eval.parse_qrel(f))
with open(args.run_path) as f:
    run = {q: d for q, d in pytrec_eval.parse_run(f).items() if q in judged}
run = {q: sorted(d.items(), key=lambda x: x[1], reverse=True)[: args.k] for q, d in run.items()}

needed = {pid for lst in run.values() for pid, _ in lst}
passages = {}
with open(args.collection) as f:
    for line in f:
        pid, text = line.rstrip("\n").split("\t", 1)
        if pid in needed:
            passages[pid] = text
assert needed <= passages.keys()

tok = T5Tokenizer.from_pretrained(args.model)
model = T5ForConditionalGeneration.from_pretrained(args.model).cuda().eval()
false_id, true_id = tok.encode("false")[0], tok.encode("true")[0]  # 6136, 1176 as in pygaggle
start = torch.full((1, 1), model.config.decoder_start_token_id, device="cuda")

out_path = f"{args.run_path.removesuffix('.txt')}-monot5-{args.k}.txt"
t0 = time.perf_counter()
with open(out_path, "w") as w, torch.inference_mode():
    for qid, lst in tqdm(run.items(), desc="queries"):
        scores = []
        for i in range(0, len(lst), args.batch_size):
            batch = lst[i : i + args.batch_size]
            texts = [f"Query: {queries[qid]} Document: {passages[pid]} Relevant:" for pid, _ in batch]
            enc = tok(texts, max_length=512, truncation=True, padding="longest", return_tensors="pt").to("cuda")
            logits = model(**enc, decoder_input_ids=start.expand(len(batch), 1)).logits[:, 0, :]
            logp = torch.log_softmax(logits[:, [false_id, true_id]], dim=-1)[:, 1]
            scores.extend(logp.tolist())
        ranked = sorted(zip((pid for pid, _ in lst), scores), key=lambda x: x[1], reverse=True)
        for rank, (pid, s) in enumerate(ranked, 1):
            w.write(f"{qid} Q0 {pid} {rank} {s} monoT5-{args.model}\n")
print(f"wrote {out_path} in {time.perf_counter() - t0:.0f}s")
