#!/usr/bin/env bash
set -e
python scripts/run_experiments.py \
  --dataset fiqa --max-docs 3000 --max-pairs 4000 \
  --seeds 42 123 2025 \
  --models BM25 Bayes NCI \
  --epochs 6 --hidden 384 --device auto \
  --run-timewall --timewall-cutoff 2020-12-31
