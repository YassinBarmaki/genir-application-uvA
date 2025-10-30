#!/usr/bin/env bash
set -e
python -m venv .venv
source .venv/Scripts/activate 2>/dev/null || source .venv/bin/activate
pip install -r requirements.txt
python scripts/run_experiments.py --seeds 42 123 2025 --timewall 2020-12-31
