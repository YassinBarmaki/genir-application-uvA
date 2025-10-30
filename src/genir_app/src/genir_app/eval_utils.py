from typing import List, Dict
import numpy as np
from tabulate import tabulate

def mrr_at_k(preds: List[List[str]], labels: List[str], k=5):
    s=0.0
    for p,y in zip(preds,labels):
        try: r = p[:k].index(y)+1; s += 1.0/r
        except ValueError: pass
    return s/max(1,len(labels))

def acc_top1(preds: List[List[str]], labels: List[str]):
    return sum(1 for p,y in zip(preds,labels) if (p[0] if p else None)==y)/max(1,len(labels))

def entropy(p):
    p = np.array(p)+1e-12
    p = p/np.sum(p)
    return float(-np.sum(p*np.log(p)))

def to_markdown_table(rows: List[Dict[str, str]], headers: List[str]) -> str:
    return tabulate(rows, headers=headers, tablefmt="github", floatfmt=".4f")
