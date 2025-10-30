import argparse
from datetime import date
print("GenIR experiment runner (skeleton).")
parser = argparse.ArgumentParser()
parser.add_argument("--seeds", nargs="+", default=["42","123","2025"])
parser.add_argument("--timewall", default=str(date.today()))
args = parser.parse_args()
print("Seeds:", args.seeds)
print("Time-wall:", args.timewall)
# TODO: call into src/genir_app/... modules and write manifests/results_summary.md
