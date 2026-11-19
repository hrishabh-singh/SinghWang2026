"""Collect the test-set metrics of all trained classifiers into one table (Fig. 4A).

Usage
    python compare.py [--results results]       ->  results/model_comparison.csv, ranked by cross-validated AUC
"""
import argparse
import json
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", type=Path, default=HERE / "results")
    a = ap.parse_args()
    rows = []
    for f in sorted(a.results.glob("*/metrics.json")):
        m = json.loads(f.read_text())
        row = {"model": m["model"], "cv_auc": m["cross_validation"]["auc_ovr_macro"]}
        for k in ("auc_ovr_macro", "macro_f1", "balanced_accuracy", "accuracy"):
            row[f"test_{k}"] = m["test"][k]
        row.update({f"test_auc_class{c}": v for c, v in enumerate(m["test"]["auc_per_class"])})
        rows.append(row)
    if not rows:
        raise SystemExit(f"no results found in {a.results}; run train.py first")
    table = pd.DataFrame(rows).sort_values("cv_auc", ascending=False)
    table.to_csv(a.results / "model_comparison.csv", index=False)
    print(table.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
