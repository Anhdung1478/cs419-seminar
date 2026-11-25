"""Low-memory equivalent of upstream rlt/features.py (same preprocessing, models and output JSON).

The upstream script keeps all 8.8M preprocessed passages as Python objects (~45-60 GB peak for
tf-idf, ~30 GB for doc2vec). Here everything is streamed through text files on disk:

  prep    : msmarco.tsv -> tokens.txt (one doc per line, tokens with corpus freq > 1) + docids.txt
            (preprocess_doc identical to upstream; parallel over CPU cores)
  tfidf   : gensim Dictionary over tokens.txt (streamed) -> dictionary.dic;
            TfidfModel(dictionary=...) -> tfidf_model (same df/N as upstream's TfidfModel(bow_corpus))
  doc2vec : Doc2Vec(vector_size=128, min_count=2, epochs=40, corpus_file=tokens.txt, workers=N)
            -> doc2vec_model_128 (corpus_file mode = same algorithm, multi-threaded; doc tag = line no.)
  infer   : per run, upstream feature JSON <ds>.feature-<retriever>.json
            {qid: {pos, docid, score, doc_len, unique_num, tfidf_sim, doc2vec_sim}};
            tf-idf cosine computed on sparse vectors (same values as upstream's dense lists).
"""
import argparse
import json
import logging
import math
import os
from collections import Counter
from multiprocessing import Pool

import gensim
import numpy as np
import pytrec_eval
from tqdm import tqdm

logging.basicConfig(format="%(asctime)s : %(levelname)s : %(message)s", level=logging.INFO)


# ---------------------------------------------------------------- upstream helpers (verbatim logic)
def preprocess_doc(text):
    text = text.replace("\t", " ").replace("\n", " ").replace("\r", " ")
    text = gensim.parsing.preprocessing.strip_punctuation(text)
    text = gensim.parsing.preprocessing.remove_stopwords(text)
    text = gensim.parsing.preprocessing.stem(text)
    return [token for token in text.split()]


def cos_sim(x, y):
    num = x.dot(y.T)
    denom = np.linalg.norm(x) * np.linalg.norm(y)
    sim = (num / denom) if denom != 0 else 0
    return sim if not np.isnan(sim) else 0


def neighbor_sim(vec_list, sim=cos_sim):
    sim_list = [[0, sim(vec_list[0], vec_list[1])]]
    for i in range(1, len(vec_list) - 1):
        sim_list.append([sim(vec_list[i], vec_list[i - 1]), sim(vec_list[i], vec_list[i + 1])])
    sim_list.append([sim(vec_list[-2], vec_list[-1]), 0])
    return sim_list


def sparse_cos(x, y):
    """cos_sim of the dense vectors upstream builds from gensim's [(id, weight)] lists."""
    num = sum(v * y[i] for i, v in x.items() if i in y)
    denom = math.sqrt(sum(v * v for v in x.values())) * math.sqrt(sum(v * v for v in y.values()))
    sim = (num / denom) if denom != 0 else 0
    return sim if not math.isnan(sim) else 0


def _prep_chunk(lines):
    out = []
    for line in lines:
        docid, text = line.strip().split("\t")
        out.append((docid, preprocess_doc(text)))
    return out


def _chunks(f, n):
    buf = []
    for line in f:
        buf.append(line)
        if len(buf) == n:
            yield buf
            buf = []
    if buf:
        yield buf


class TokenLines:
    """Streams tokens.txt as lists of tokens (restartable, for gensim)."""

    def __init__(self, path):
        self.path = path

    def __iter__(self):
        with open(self.path) as f:
            for line in f:
                yield line.split()


# ---------------------------------------------------------------- modes
def mode_prep(a):
    raw, toks, ids = (os.path.join(a.output_path, n) for n in ("tokens_raw.txt", "tokens.txt", "docids.txt"))
    if os.path.exists(toks):
        print("have", toks)
        return
    freq = Counter()
    n_docs = 0
    with open(a.index_path) as f, open(raw, "w") as w_raw, open(ids, "w") as w_ids, Pool(a.workers) as pool:
        for chunk in tqdm(pool.imap(_prep_chunk, _chunks(f, 20000)), desc="preprocess (x20k docs)"):
            for docid, tokens in chunk:
                freq.update(tokens)
                w_raw.write(" ".join(tokens) + "\n")
                w_ids.write(docid + "\n")
                n_docs += 1
    print(f"# doc: {n_docs}\n# vocab (initial): {len(freq)}")
    # filter out tokens only appearing once (upstream: frequency[token] > 1)
    with open(raw) as r, open(toks + ".part", "w") as w:
        for line in tqdm(r, total=n_docs, desc="filter freq>1"):
            w.write(" ".join(t for t in line.split() if freq[t] > 1) + "\n")
    os.replace(toks + ".part", toks)
    os.remove(raw)


def mode_tfidf(a):
    dic_path = os.path.join(a.output_path, "dictionary.dic")
    tfidf_path = os.path.join(a.output_path, "tfidf_model")
    if os.path.exists(tfidf_path):
        print("have", tfidf_path)
        return
    dictionary = gensim.corpora.Dictionary(TokenLines(os.path.join(a.output_path, "tokens.txt")))
    print(dictionary)
    dictionary.save(dic_path)
    # Same idf as TfidfModel(list_of_bow): df and num_docs come from the identical corpus.
    gensim.models.TfidfModel(dictionary=dictionary).save(tfidf_path)


def mode_doc2vec(a):
    path = os.path.join(a.output_path, f"doc2vec_model_{a.vector_size}")
    if os.path.exists(path):
        print("have", path)
        return
    model = gensim.models.doc2vec.Doc2Vec(
        corpus_file=os.path.join(a.output_path, "tokens.txt"),
        vector_size=a.vector_size, min_count=a.min_count, epochs=a.epoch, workers=a.workers)
    model.save(path)
    print(model.infer_vector(["only", "you", "can", "prevent", "forest", "fires"]))


def mode_infer(a):
    # Load all runs first, so the corpus is scanned once for every needed doc.
    runs = []
    for run_path, qrels_path in zip(a.run_path, a.qrels_path):
        ds = run_path.split("/")[-1].split(".")[0]
        retriever = "-".join(run_path.split("/")[-1].split(".")[1].split("-")[1:])
        out = os.path.join(a.output_path, f"{ds}.feature-{retriever}.json")
        if os.path.exists(out):
            print("have", out)
            continue
        with open(run_path) as r:
            run = pytrec_eval.parse_run(r)
        with open(qrels_path) as r:
            qrels = pytrec_eval.parse_qrel(r)
        keep = {}
        for qid, d in run.items():
            if qid not in qrels:
                continue
            if len(d) < a.seq_len:
                print(f"throw query{qid} because it has only {len(d)} items")
                continue
            keep[qid] = sorted(d.items(), key=lambda x: x[1], reverse=True)
        runs.append((out, keep))
    if not runs:
        return

    needed = {docid for _, keep in runs for lst in keep.values() for docid, _ in lst}
    tokens = {}
    with open(os.path.join(a.output_path, "docids.txt")) as fi, open(os.path.join(a.output_path, "tokens.txt")) as ft:
        for docid, line in zip(fi, ft):
            docid = docid.rstrip("\n")
            if docid in needed:
                tokens[docid] = line.split()
    assert needed <= tokens.keys(), f"{len(needed - tokens.keys())} docs missing from tokens.txt"

    dictionary = gensim.corpora.Dictionary.load(os.path.join(a.output_path, "dictionary.dic"))
    tfidf_model = gensim.models.TfidfModel.load(os.path.join(a.output_path, "tfidf_model"))
    d2v = gensim.models.doc2vec.Doc2Vec.load(os.path.join(a.output_path, f"doc2vec_model_{a.vector_size}"))

    for out, keep in runs:
        feature = {}
        for qid, ranked_list in tqdm(keep.items(), desc=os.path.basename(out)):
            f = feature[qid] = {k: [] for k in ("pos", "docid", "score", "doc_len", "unique_num")}
            tfidf_list, d2v_list = [], []
            for idx, (docid, score) in enumerate(ranked_list):
                doc = tokens[docid]
                bow_doc = dictionary.doc2bow(doc)
                assert len(doc) == sum(c for _, c in bow_doc)
                f["pos"].append(idx)
                f["docid"].append(docid)
                f["score"].append(score)
                f["doc_len"].append(len(doc))
                f["unique_num"].append(len(bow_doc))
                tfidf_list.append(dict(tfidf_model[bow_doc]))
                d2v_list.append(np.array(d2v.infer_vector(doc).tolist()))
            f["tfidf_sim"] = neighbor_sim(tfidf_list, sim=sparse_cos)
            f["doc2vec_sim"] = neighbor_sim(d2v_list)
        with open(out, "w") as w:
            w.write(json.dumps(feature, default=float))
        print("wrote", out)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--mode", required=True, choices=["prep", "tfidf", "doc2vec", "infer"])
    p.add_argument("--vector_size", type=int, default=128)
    p.add_argument("--epoch", type=int, default=40)
    p.add_argument("--min_count", type=int, default=2)
    p.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 4) - 2))
    p.add_argument("--index_path")
    p.add_argument("--run_path", nargs="*", default=[])
    p.add_argument("--qrels_path", nargs="*", default=[])
    p.add_argument("--output_path", required=True)
    p.add_argument("--seq_len", type=int, default=1000)
    args = p.parse_args()
    {"prep": mode_prep, "tfidf": mode_tfidf, "doc2vec": mode_doc2vec, "infer": mode_infer}[args.mode](args)
