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
