import os, re, math, json, random, argparse
from collections import Counter
from statistics import mean, pstdev

import numpy as np

# deps
from datasets import load_dataset
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder

# utils 
def set_seed(s):
    random.seed(s); np.random.seed(s)

def tok(s: str):
    return re.findall(r"[A-Za-z0-9_]+", (s or "").lower())

def mrr_at_k(preds, labels, k=10):
    s = 0.0
    for p, y in zip(preds, labels):
        r = next((i+1 for i, d in enumerate(p[:k]) if d == y), None)
        s += 0.0 if r is None else 1.0 / r
    return s / max(1, len(labels))

def ndcg_at_k(preds, labels, k=10):
    # binary relevance (1 if hit anywhere in top-k)
    def dcg(p, y, k):
        for i, d in enumerate(p[:k], 1):
            if d == y:
                return 1.0 / math.log2(i + 1)
        return 0.0
    return sum(dcg(p, y, k) for p, y in zip(preds, labels)) / max(1, len(labels))

def recall_at_k(preds, labels, k=50):
    hit = sum(1 for p, y in zip(preds, labels) if y in p[:k])
    return hit / max(1, len(labels))

def ensure_dir(p): 
    os.makedirs(p, exist_ok=True)

# tiny Bayes GenIR
class BayesGenIR:
    def __init__(self, alpha=1.0, temperature=1.0):
        self.alpha = float(alpha)
        self.temperature = float(temperature)
        self.doc_ids = []
        self.doc_token_counts = {}
        self.total_tokens = {}
        self.doc_priors = {}
        self.vocab = set()

    def fit(self, train_pairs, corpus):
        df = Counter([p["doc_id"] for p in train_pairs])
        self.doc_ids = sorted(set(df.keys()))
        total_pairs = sum(df.values()) or 1
        self.doc_priors = {d: df[d] / total_pairs for d in self.doc_ids}
        self.doc_token_counts = {d: Counter() for d in self.doc_ids}

        for d in self.doc_ids:
            r = corpus.get(d, {})
            t = f"{r.get('title','')} {r.get('text','')}"
            for w in tok(t):
                self.doc_token_counts[d][w] += 1
                self.vocab.add(w)

        for ex in train_pairs:
            for w in tok(ex["query"]):
                self.doc_token_counts[ex["doc_id"]][w] += 1
                self.vocab.add(w)

        self.total_tokens = {d: sum(c.values()) for d, c in self.doc_token_counts.items()}

    def _log_p_t_given_d(self, t, d):
        cnt = self.doc_token_counts[d][t]
        V = len(self.vocab) + 1
        return math.log((cnt + self.alpha) / (self.total_tokens[d] + self.alpha * V))

    def predict(self, query, topk=10):
        toks = tok(query)
        raw = {}
        for d in self.doc_ids:
            s = math.log(self.doc_priors.get(d, 1e-12))
            for w in toks:
                s += self._log_p_t_given_d(w, d)
            raw[d] = s
        # softmax with temperature
        T = max(1e-6, self.temperature)
        mx = max(raw.values()) if raw else 0.0
        exps = {k: math.exp((v - mx) / T) for k, v in raw.items()}
        Z = sum(exps.values()) or 1.0
        probs = {k: exps[k] / Z for k in exps}
        ranked = sorted(probs.items(), key=lambda x: -x[1])[:topk]
        return [d for d, _ in ranked], [probs[d] for d, _ in ranked]

def entropy(p):
    p = np.array(p) + 1e-12
    p = p / p.sum()
    return float(-(p * np.log(p)).sum())

#pipeline
def load_fiqa_subset(max_docs=1200, max_pairs=1800):
    ds = load_dataset("BeIR/fiqa")
    d_corpus = ds["corpus"]
    # corpus rows may be dicts with keys 'doc_id','title','text'
    corpus = {}
    for i, r in enumerate(d_corpus):
        if len(corpus) >= max_docs: break
        did = str(r.get("doc_id") or r.get("_id") or i)
        corpus[did] = {"title": r.get("title", "") or "", "text": r.get("text", "") or ""}
    keep = set(corpus.keys())

    # build (query, doc_id) positive pairs from qrels
    d_queries = ds["queries"]
    qrels = ds["qrels"]
    pairs = []
    for r in qrels:
        qid = str(r.get("query-id") or r.get("qid") or r.get("query_id") or r.get("query"))
        did = str(r.get("corpus-id") or r.get("doc_id") or r.get("corpus_id") or r.get("docid"))
        rel = float(r.get("score", 0))
        if rel > 0 and did in keep:
            # queries can be list-like in this HF dump
            if isinstance(d_queries, list) and qid.isdigit() and int(qid) < len(d_queries):
                qtext = d_queries[int(qid)]["text"]
            else:
                try:
                    qtext = next(x["text"] for x in d_queries if str(x.get("id", x.get("qid"))) == qid)
                except StopIteration:
                    qtext = None
            if qtext:
                pairs.append({"query": qtext, "doc_id": did})
    random.shuffle(pairs)
    pairs = pairs[:max_pairs] if max_pairs else pairs
    # split dev/train (20% dev)
    n_dev = max(100, int(0.2 * len(pairs)))
    dev = pairs[:n_dev]; train = pairs[n_dev:]
    return corpus, train, dev

def build_bm25(corpus):
    doc_ids = []
    docs_tok = []
    for did, r in corpus.items():
        doc_ids.append(did)
        docs_tok.append(tok(f"{r.get('title','')} {r.get('text','')}"))
    return doc_ids, BM25Okapi(docs_tok)

def run_once(seed, max_docs=1200, max_pairs=1800, topk=10):
    set_seed(seed)
    corpus, train, dev = load_fiqa_subset(max_docs=max_docs, max_pairs=max_pairs)
    doc_ids, bm25 = build_bm25(corpus)
    ce = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

    # Bayes model (generative scoring over IDs)
    bayes = BayesGenIR(alpha=1.0, temperature=1.0)
    seen = sorted({x["doc_id"] for x in train})
    if not seen:  # fallback اگر train خیلی کوچک شد
        seen = doc_ids[: min(500, len(doc_ids))]
        train = [{"query": "dummy", "doc_id": d} for d in seen]
    bayes.fit(train, {d: corpus[d] for d in seen})

    # eval set
    queries = [x["query"] for x in dev]
    labels  = [x["doc_id"] for x in dev]

    # BM25
    bm25_preds = []
    for q in queries:
        scores = bm25.get_scores(tok(q))
        idx = np.argsort(-scores)[:topk]
        bm25_preds.append([doc_ids[i] for i in idx])

    # BM25 → CrossEncoder
    bm25_ce_preds = []
    for q in queries:
        scores = bm25.get_scores(tok(q))
        idx = np.argsort(-scores)[:min(50, len(scores))]
        cands = [(doc_ids[i], f"{corpus[doc_ids[i]]['title']} {corpus[doc_ids[i]]['text']}") for i in idx]
        ce_scores = ce.predict([(q, t) for _, t in cands], batch_size=32)
        ords = np.argsort(-np.array(ce_score_
