import os, argparse, random, sys, pathlib
# import 
sys.path.append(str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from rich import print
from rich.table import Table
from rich.console import Console

from genir_app.data_utils import ensure_dataset, load_corpus, prepare_pairs, apply_time_wall
from genir_app.models import BayesGenIR, BM25Index, NCIClassifier, build_query_vocab
from genir_app.eval_utils import acc_top1, mrr_at_k, to_markdown_table, entropy
from genir_app.manifests import write_manifest

console = Console()

def eval_bayes(model, dev_pairs, topk=5):
    labels, preds, conf = [], [], []
    for ex in dev_pairs:
        ranked = model.predict(ex["query"], topk=topk)
        preds.append([d for d,_,__ in ranked]); labels.append(ex["doc_id"])
        conf.append(ranked[0][1] if ranked else 0.0)
    return {"top1": acc_top1(preds, labels), "mrr@5": mrr_at_k(preds, labels, 5), "conf_entropy": entropy(conf)}

def eval_bm25(bm25, dev_pairs, topk=5):
    labels, preds = [], []
    for ex in dev_pairs:
        top = bm25.search(ex["query"], topk=topk)
        preds.append([d for d,_ in top]); labels.append(ex["doc_id"])
    return {"top1": acc_top1(preds, labels), "mrr@5": mrr_at_k(preds, labels, 5)}

def eval_nci(model, qv, docs, dev_pairs, device="cpu", topk=5):
    import torch
    labels, preds, conf = [], [], []
    for ex in dev_pairs:
        ids=[qv.get(t,0) for t in ex["query"].lower().split()][:24]
        X=torch.tensor([ids or [0]], dtype=torch.long, device=device)
        with torch.no_grad():
            logits=model(X)
            probs=torch.softmax(logits,-1).squeeze(0).detach().cpu().numpy()
        idx = probs.argsort()[::-1][:topk]
        preds.append([docs[i] for i in idx]); labels.append(ex["doc_id"])
        conf.append(float(probs[idx[0]]) if len(idx) else 0.0)
    return {"top1": acc_top1(preds, labels), "mrr@5": mrr_at_k(preds, labels, 5), "conf_entropy": entropy(conf)}

def train_nci(train_pairs, docs, hidden=384, epochs=6, lr=1e-3, device="cpu"):
    import torch, torch.nn as nn
    qv = build_query_vocab(train_pairs, min_freq=1)
    did = {d:i for i,d in enumerate(docs)}
    model = NCIClassifier(len(docs), len(qv), hidden=hidden).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    crit= nn.CrossEntropyLoss()

    def encode_batch(batch):
        T = max(1, max(len(ex["query"].split()) for ex in batch))
        X=[]
        for ex in batch:
            ids=[qv.get(t,0) for t in ex["query"].lower().split()]
            if not ids: ids=[0]
            ids = ids + [0]*(T-len(ids))
            X.append(ids)
        X=torch.tensor(X, dtype=torch.long, device=device)
        y=torch.tensor([did[ex["doc_id"]] for ex in batch], dtype=torch.long, device=device)
        return X,y

    B=64
    for ep in range(1, epochs+1):
        random.shuffle(train_pairs); loss_sum=0.0; steps=0
        for i in range(0, len(train_pairs), B):
            batch=train_pairs[i:i+B]
            X,y = encode_batch(batch)
            logits = model(X)
            loss = crit(logits,y)
            opt.zero_grad(); loss.backward(); opt.step()
            loss_sum += loss.item(); steps+=1
        print(f"[cyan][NCI][/cyan] epoch {ep} loss={loss_sum/max(1,steps):.4f}")
    return model, qv

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", type=str, default="fiqa")
    p.add_argument("--max-docs", type=int, default=3000)
    p.add_argument("--max-pairs", type=int, default=4000)
    p.add_argument("--models", nargs="+", default=["BM25","Bayes","NCI"])
    p.add_argument("--topk", type=int, default=5)
    p.add_argument("--alpha", type=float, default=1.0)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--seeds", nargs="+", type=int, default=[42,123,2025])
    p.add_argument("--run-timewall", action="store_true")
    p.add_argument("--timewall-cutoff", type=str, default="")
    p.add_argument("--epochs", type=int, default=6)      # برای NCI
    p.add_argument("--hidden", type=int, default=384)     # برای NCI
    p.add_argument("--device", type=str, default="auto", choices=["auto","cpu","cuda"])
    args = p.parse_args()

    # device
    device="cpu"
    if args.device=="cuda":
        device="cuda"
    elif args.device=="auto":
        try:
            import torch
            device="cuda" if torch.cuda.is_available() else "cpu"
        except Exception:
            device="cpu"
    print(f"[yellow]Device: {device}[/yellow]")

    all_rows = []
    headers = ["seed"]

    if "BM25" in args.models: headers += ["BM25_top1","BM25_mrr@5"]
    if "Bayes" in args.models: headers += ["Bayes_top1","Bayes_mrr@5","Bayes_confH"]
    if "NCI" in args.models:   headers += ["NCI_top1","NCI_mrr@5","NCI_confH"]

    for seed in args.seeds:
        random.seed(seed)
        data_dir = ensure_dataset("hf", args.dataset, out_dir=f"data/hf_{args.dataset}", tiny_dir="data/tiny", max_docs=args.max_docs, max_pairs=args.max_pairs)
        corpus = load_corpus(data_dir)
        removed=0
        if args.run_timewall and args.timewall_cutoff:
            c2 = apply_time_wall(corpus, args.timewall_cutoff); removed = len(corpus)-len(c2); corpus=c2

        train = prepare_pairs(data_dir, "train")
        dev   = prepare_pairs(data_dir, "dev")
        if args.run_timewall and args.timewall_cutoff:
            keep=set(corpus.keys())
            train=[r for r in train if r["doc_id"] in keep]
            dev  =[r for r in dev   if r["doc_id"] in keep]

        row = {"seed": seed}

        # BM25
        if "BM25" in args.models:
            bm25 = BM25Index(corpus)
            bm_m = eval_bm25(bm25, dev, topk=args.topk)
            row["BM25_top1"]  = bm_m["top1"]
            row["BM25_mrr@5"] = bm_m["mrr@5"]

        # Bayes
        if "Bayes" in args.models:
            syn={"distribution":["shift","distro"], "identifier":["id","docid","doc-id"], "robustness":["robust"], "few":["few-shot","fewshot"]}
            bay = BayesGenIR(alpha=args.alpha, temperature=args.temperature, synonyms=syn)
            bay.fit(train, corpus)
            by_m = eval_bayes(bay, dev, topk=args.topk)
            row["Bayes_top1"]  = by_m["top1"]
            row["Bayes_mrr@5"] = by_m["mrr@5"]
            row["Bayes_confH"] = by_m["conf_entropy"]

        # NCI (Seq2DocID)
        if "NCI" in args.models:
            try:
                docs = sorted({ex["doc_id"] for ex in train})
                model, qv = train_nci(train, docs, hidden=args.hidden, epochs=args.epochs, lr=1e-3, device=device)
                nci_m = eval_nci(model, qv, docs, dev, device=device, topk=args.topk)
                row["NCI_top1"]  = nci_m["top1"]
                row["NCI_mrr@5"] = nci_m["mrr@5"]
                row["NCI_confH"] = nci_m["conf_entropy"]
            except Exception as e:
                print(f"[red]NCI skipped:[/red] {e}")

        # Manifest
        write_manifest(f"manifests/{args.dataset}_seed{seed}_run_summary.json",
                       {"seed": seed, "dataset": args.dataset,
                        "timewall": args.timewall_cutoff if args.run_timewall else None,
                        "removed_docs": removed, "metrics": {k:v for k,v in row.items() if k!='seed'}})
        print(f"[green]✓ seed {seed} done[/green]")

        all_rows.append(row)

    # Save markdown table
    os.makedirs("artifacts", exist_ok=True)
    md = to_markdown_table(all_rows, headers)
    with open("artifacts/results_summary.md","w",encoding="utf-8") as f:
        f.write("# Results Summary\n\n"+md+"\n")

    # Pretty console table
    table = Table(show_header=True, header_style="bold magenta")
    for h in headers: table.add_column(h)
    for r in all_rows:
        table.add_row(*[str(r.get(h,'')) for h in headers])
    console.print(table)

if __name__ == "__main__":
    main()
