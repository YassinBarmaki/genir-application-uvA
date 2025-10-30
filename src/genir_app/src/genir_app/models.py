import re, json, math, numpy as np
from typing import List, Tuple, Dict
from collections import Counter

def _simple_tokenize(text: str) -> List[str]:
    text = re.sub(r"[^A-Za-z0-9\- ]", " ", str(text or "").lower())
    return [t for t in re.split(r"\s+", text.strip()) if t]

# BayesGenIR 
class BayesGenIR:
    def __init__(self, alpha=1.0, temperature=1.0, synonyms: Dict[str, List[str]] = None):
        self.alpha=float(alpha); self.temperature=float(temperature); self.synonyms=synonyms or {}
        self.doc_ids=[]; self.doc_token_counts={}; self.total_tokens={}; self.doc_priors={}; self.vocab=set()
    def _expand_syn(self, toks: List[str]) -> List[str]:
        inv={}
        for k, vs in (self.synonyms or {}).items():
            for v in vs: inv[v]=k
        out=toks[:]
        for t in toks:
            if t in self.synonyms: out+=self.synonyms[t]
            if t in inv: out.append(inv[t])
        seen=set(); uniq=[]
        for t in out:
            if t not in seen:
                uniq.append(t); seen.add(t)
        return uniq
    def fit(self, train_pairs: List[dict], corpus: Dict[str, dict]):
        df = Counter([p["doc_id"] for p in train_pairs])
        self.doc_ids = sorted(set(df.keys()))
        total = sum(df.values()) or 1
        self.doc_priors = {d: df[d]/total for d in self.doc_ids}
        self.doc_token_counts = {d: Counter() for d in self.doc_ids}
        for d in self.doc_ids:
            t=(corpus.get(d,{}).get("title","")+" "+corpus.get(d,{}).get("text",""))
            for tok in _simple_tokenize(t):
                self.doc_token_counts[d][tok]+=1; self.vocab.add(tok)
        for ex in train_pairs:
            for tok in _simple_tokenize(ex["query"]):
                self.doc_token_counts[ex["doc_id"]][tok]+=1; self.vocab.add(tok)
        self.total_tokens = {d: sum(c.values()) for d,c in self.doc_token_counts.items()}
    def _log_p_t_given_d(self, t, d):
        cnt=self.doc_token_counts[d][t]; V=len(self.vocab)+1
        return math.log((cnt + self.alpha) / (self.total_tokens[d] + self.alpha*V))
    def _softmax_T(self, scores: Dict[str, float]):
        T=max(1e-6, self.temperature); mx=max(scores.values())
        exps={k: math.exp((v-mx)/T) for k,v in scores.items()}
        Z=sum(exps.values()) or 1.0; return {k: exps[k]/Z for k in scores}
    def predict(self, query: str, topk=5):
        toks=self._expand_syn(_simple_tokenize(query)); raw={}; contribs={}
        for d in self.doc_ids:
            s=math.log(self.doc_priors.get(d,1e-12)); c={"<PRIOR>": s}
            for t in toks:
                lt=self._log_p_t_given_d(t,d); s+=lt; c[t]=c.get(t,0.0)+lt
            raw[d]=s; contribs[d]=c
        probs=self._softmax_T(raw)
        ranked=sorted(probs.items(), key=lambda x:x[1], reverse=True)[:topk]
        return [(d,p,contribs[d]) for d,p in ranked]

#BM25 
try:
    from rank_bm25 import BM25Okapi
    _HAS_RBM=True
except Exception:
    _HAS_RBM=False

class BM25Index:
    def __init__(self, corpus: dict, tokenizer=_simple_tokenize, k1=1.5, b=0.75):
        self.tokenize=tokenizer; self.doc_ids=[]; self.docs_tok=[]
        for d,rec in corpus.items():
            txt=f"{rec.get('title','')} {rec.get('text','')}"
            self.doc_ids.append(d)
            self.docs_tok.append(self.tokenize(txt))
        self.k1=float(k1); self.b=float(b)
        if _HAS_RBM:
            self.bm25=BM25Okapi(self.docs_tok); self._fallback=False
        else:
            self._fallback=True
            from collections import Counter
            self.N=len(self.docs_tok)
            self.avgdl=np.mean([len(t) for t in self.docs_tok]) if self.docs_tok else 1.0
            self.tf=[Counter(t) for t in self.docs_tok]
            df=Counter()
            for toks in self.docs_tok:
                for w in set(toks): df[w]+=1
            self.idf={w: np.log(1 + (self.N - df[w] + 0.5)/(df[w] + 0.5)) for w in df}
    def _score_doc_fallback(self, q_toks, idx):
        tf=self.tf[idx]; dl=sum(tf.values()) or 1; score=0.0
        for t in q_toks:
            if t not in tf or t not in self.idf: continue
            freq=tf[t]
            num=freq*(self.k1+1.0)
            den=freq + self.k1*(1.0 - self.b + self.b*dl/self.avgdl)
            score += self.idf[t]*(num/den)
        return float(score)
    def search(self, query: str, topk=50):
        q_toks=self.tokenize(query)
        if not self.doc_ids: return []
        if not q_toks: return [(self.doc_ids[i], 0.0) for i in range(min(topk, len(self.doc_ids)))]
        if not getattr(self, "_fallback", True):
            scores=self.bm25.get_scores(q_toks)
            idx=np.argsort(-scores)[:topk]
            return [(self.doc_ids[i], float(scores[i])) for i in idx]
        else:
            scores=[self._score_doc_fallback(q_toks, i) for i in range(len(self.docs_tok))]
            idx=np.argsort(-np.array(scores))[:topk]
            return [(self.doc_ids[i], float(scores[i])) for i in idx]

# Cross-Encoder rerank 
def ce_rerank(query: str, candidates: List[Tuple[str,str]], model_name="cross-encoder/ms-marco-MiniLM-L-6-v2"):
    try:
        from sentence_transformers import CrossEncoder
        ce=CrossEncoder(model_name)
        pairs=[(query, txt) for _, txt in candidates]
        scores=ce.predict(pairs, batch_size=32)
        ranked=sorted(zip(candidates, scores), key=lambda x:-x[1])
        return [(docid, float(s)) for ((docid,_), s) in ranked]
    except Exception:
        return [(docid, 0.0) for (docid, _) in candidates]

# Seq2DocID (NCI)
try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except Exception:
    TORCH_AVAILABLE = False

class NCIClassifier(nn.Module if TORCH_AVAILABLE else object):
    def __init__(self, num_docs:int, qvocab_size:int, hidden=256):
        if not TORCH_AVAILABLE:
            raise RuntimeError("PyTorch not available")
        super().__init__()
        self.qemb = nn.Embedding(qvocab_size, hidden)
        self.gru  = nn.GRU(hidden, hidden, batch_first=True)
        self.proj = nn.Linear(hidden, num_docs)
    def forward(self, q_tokens):  # (B,T)
        x,_ = self.gru(self.qemb(q_tokens))
        h = x.mean(dim=1)
        return self.proj(h)

def build_query_vocab(train_pairs: List[dict], min_freq=1)->Dict[str,int]:
    cnt=Counter()
    for ex in train_pairs:
        for t in _simple_tokenize(ex["query"]): cnt[t]+=1
    vocab={"<unk>":0}
    for t,c in cnt.items():
        if c>=min_freq and t not in vocab:
            vocab[t]=len(vocab)
    return vocab
