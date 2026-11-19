"""Spearman correlation of every Library 1 descriptor with the binding free energy, separately for the two regimes.

Usage
    python spearman_by_regime.py [--data ../data/library1/library1_dataset.csv] [--out results] [--top 5]

Regimes (Results 2, Fig. 2C-D): La-Gd (large ions, n = 63 measurements) and Tb-Lu (small ions, n = 123).
Descriptors: all columns of the Library 1 table except identifiers/targets, the hydration-shell columns (1st/2nd
shell distance and count) and the DBAASP/mutation columns that no model uses (EXCLUDED below).
Outputs: spearman_<regime>.csv with rho and p for every descriptor (sorted by rho), and the strongest positive and
negative correlates printed to the console (the ten bars of Fig. 2C/D use --top 5).
"""
import argparse
from pathlib import Path

import pandas as pd
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parent.parent
REGIMES = {"La-Gd": ["La", "Ce", "Pr", "Nd", "Pm", "Sm", "Eu", "Gd"],
           "Tb-Lu": ["Tb", "Dy", "Ho", "Er", "Tm", "Yb", "Lu"]}
NON_DESCRIPTORS = ["PEPS", "Ln", "del_G", "del_del_G", "mask1", "sequence", "seq_code", "weights", "del_Gsolv_Qiao"]
NON_DESCRIPTORS += [c for c in pd.read_csv(ROOT / "data/library1/library1_dataset.csv", nrows=0).columns
                    if c.startswith("BOSPCA")]                       # ESM-2 components are not descriptors here
EXCLUDED = ["1st_dist", "1st_N", "2nd_dist", "2nd_N", "Penetration Depth", "Tilt Angle",
            "ngle Subtended by the Hydrophobic Residues", "∆Hydrophobicity", "∆Gsolvation", "∆VdwRadius",
            "∆charge", "∆polarity"]


def correlations(df, ions):
    sub = df[df["Ln"].isin(ions)]
    feats = [c for c in df.columns if c not in NON_DESCRIPTORS + EXCLUDED]
    rows = []
    for c in feats:
        res = spearmanr(sub[c], sub["del_G"])
        rows.append((c, res.statistic, res.pvalue))
    out = pd.DataFrame(rows, columns=["descriptor", "rho", "p"]).dropna()   # constant columns give NaN
    return len(sub), out.sort_values("rho", ascending=False).reset_index(drop=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, default=ROOT / "data/library1/library1_dataset.csv")
    ap.add_argument("--out", type=Path, default=Path(__file__).resolve().parent / "results")
    ap.add_argument("--top", type=int, default=5)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(a.data)
    for name, ions in REGIMES.items():
        n, r = correlations(df, ions)
        r.to_csv(a.out / f"spearman_{name}.csv", index=False)
        top = pd.concat([r.head(a.top), r.tail(a.top)])
        print(f"\n{name} (n = {n}): strongest correlates of dG\n{top.round(3).to_string(index=False)}")


if __name__ == "__main__":
    main()
