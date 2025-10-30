Why this repo for GenIR @ IRLab (UvA)
This is a continuation line of my MSc thesis & current RAG work: a hybrid GenIR sandbox with
BM25→CrossEncoder, Bayes (generative scoring over doc IDs), and a Seq2DocID (NCI) head.
It targets the five open challenges in the vacancy:

- Out-of-distribution: entropy-based OOD flag at query-time (bins + ECE chart).
- Low-resource: FiQA subset + synonym expansion + temperature calibration.
- Reliability: time-wall audit (no post-date leakage) + signed JSON manifests.
- Dynamic settings: BM25 candidate pool + rerank + online-friendly priors.
- Transparency: token-level attributions for Bayes and explicit evidence logs.





#GenIR_ Multi-Seed + OOD + Time-Wall (IRLab-ready)

Hybrid Retrieval (BM25 + Bayes) +  Seq2DocID (NCI) + PyTorch.
- OOD  confidence.
- Time-Wall  leakage.
- Manifests (JSON)  reproducibility.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
CPU:
pip install torch --index-url https://download.pytorch.org/whl/cpu
CUDA 12.1 :
pip install torch --index-url https://download.pytorch.org/whl/cu121

python scripts/run_experiments.py \
  --dataset fiqa --max-docs 3000 --max-pairs 4000 \
  --seeds 42 123 2025 \
  --models BM25 Bayes NCI \
  --epochs 6 --hidden 384 --device auto \
  --run-timewall --timewall-cutoff 2020-12-31

bash scripts/run.sh
