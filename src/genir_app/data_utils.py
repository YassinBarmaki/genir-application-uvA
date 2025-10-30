import os, re, json, random, datetime
from typing import Dict, List, Iterable

def _write_jsonl(path, rows: Iterable[dict]):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

def _load_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)

def _simple_tokenize(text: str) -> List[str]:
    text = re.sub(r"[^A-Za-z0-9\- ]", " ", str(text or "").lower())
    return [t for t in re.split(r"\s+", text.strip()) if t]

def build_tiny_dataset(out_dir: str):
    corpus = [
        {"doc_id":"D001","title":"Neural Indexing with DSI","text":"Sequence models generating identifiers for retrieval.","timestamp":"2019-06-01"},
        {"doc_id":"D002","title":"OOD Robustness","text":"Handling query shift with synonyms and smoothing.","timestamp":"2020-03-10"},
        {"doc_id":"D003","title":"Low-Resource Learning","text":"Few-shot and online updates for dynamic corpora.","timestamp":"2020-12-15"},
        {"doc_id":"D004","title":"Reliability & Calibration","text":"ECE and temperature scaling for trustworthiness.","timestamp":"2018-09-21"},
        {"doc_id":"D005","title":"Transparency","text":"Token attributions and auditable likelihood traces.","timestamp":"2017-01-05"},
    ]
    train = [
        {"query":"neural indexing with dsi","doc_id":"D001"},
        {"query":"generate document id","doc_id":"D001"},
        {"query":"ood robustness","doc_id":"D002"},
        {"query":"distribution shift","doc_id":"D002"},
        {"query":"few shot low resource","doc_id":"D003"},
        {"query":"online update","doc_id":"D003"},
        {"query":"reliability calibration","doc_id":"D004"},
        {"query":"token attribution","doc_id":"D004"},
        {"query":"transparency","doc_id":"D005"},
        {"query":"auditable","doc_id":"D005"},
    ]
    dev = [
        {"query":"identifier generation","doc_id":"D001"},
        {"query":"robust to shift","doc_id":"D002"},
        {"query":"few-shot setting","doc_id":"D003"},
    ]
    _write_jsonl(f"{out_dir}/corpus.jsonl", corpus)
    _write_jsonl(f"{out_dir}/train.jsonl", train)
    _write_jsonl(f"{out_dir}/dev.jsonl", dev)
    return out_dir

def load_hf_or_beir(beir_name="fiqa", max_docs=1500, max_pairs=2000, out_dir="data/hf_fiqa"):

    try:
        from datasets import load_dataset
        ds = load_dataset(f"BeIR/{beir_name}")
        d_corpus = ds["corpus"]; d_queries = ds["queries"]; d_qrels = ds["qrels"]

        c_rows = []
        for r in d_corpus:
            c_rows.append({"doc_id": str(r.get("doc_id") or r.get("_id")),
                           "title": r.get("title",""),
                           "text": r.get("text",""),
                           "timestamp": str(r.get("date",""))})

        if max_docs: c_rows = c_rows[:max_docs]
        keep = set([r["doc_id"] for r in c_rows])

        pairs = []
        for r in d_qrels:
            qid = str(r.get("query-id") or r.get("qid") or r.get("query_id") or r.get("query"))
            did = str(r.get("corpus-id") or r.get("doc_id") or r.get("corpus_id") or r.get("docid"))
            rel = float(r.get("score", 0))
            if did in keep and rel > 0:
                qtext = None
                try:
                    if isinstance(d_queries, list) and qid.isdigit():
                        qtext = d_queries[int(qid)]["text"]
                    else:
                        qtext = next(x["text"] for x in d_queries if str(x.get("id", x.get("qid"))) == qid)
                except Exception:
                    pass
                if qtext:
                    pairs.append({"query": qtext, "doc_id": did})

        random.shuffle(pairs)
        pairs = pairs[:max_pairs] if max_pairs else pairs
        n_dev = max(100, int(0.2*len(pairs)))
        dev = pairs[:n_dev]; train = pairs[n_dev:]

        os.makedirs(out_dir, exist_ok=True)
        _write_jsonl(f"{out_dir}/corpus.jsonl", c_rows)
        _write_jsonl(f"{out_dir}/train.jsonl", train)
        _write_jsonl(f"{out_dir}/dev.jsonl", dev)
        return out_dir
    except Exception:
        return None

def ensure_dataset(source="hf", name="fiqa", out_dir="data/hf_fiqa", tiny_dir="data/tiny", max_docs=1500, max_pairs=2000) -> str:
    path = load_hf_or_beir(beir_name=name, max_docs=max_docs, max_pairs=max_pairs, out_dir=out_dir)
    if path: return path
    return build_tiny_dataset(tiny_dir)

def load_corpus(data_dir: str) -> Dict[str, dict]:
    return {r["doc_id"]: r for r in _load_jsonl(f"{data_dir}/corpus.jsonl")}

def apply_time_wall(corpus: Dict[str, dict], cutoff: str) -> Dict[str, dict]:
    try:
        dt_cut = datetime.datetime.strptime(cutoff, "%Y-%m-%d").date()
    except Exception:
        return corpus
    out = {}
    for did, rec in corpus.items():
        ts = rec.get("timestamp","")
        ok = True
        if ts:
            try:
                d = datetime.datetime.strptime(ts[:10], "%Y-%m-%d").date()
                ok = (d <= dt_cut)
            except Exception:
                ok = True
        if ok:
            out[did] = rec
    return out

def prepare_pairs(data_dir: str, split="dev"):
    return list(_load_jsonl(f"{data_dir}/{split}.jsonl"))
