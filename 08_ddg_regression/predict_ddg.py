"""Predict ddG(Ln/Tb) for peptides outside Library 1 and compare with experiment (Fig. S7, Table S5).

Usage
    python predict_ddg.py [--features descriptors] [--results results]
    (requires results/<features>/model.joblib from train_ddg.py, ../data/external/efhand_experimental_ddG.csv from
     ../06_efhand_experimental_values/competition_to_dG.py, and ../data/external/lbt3_n5d_experimental_ddG.csv)

External peptides
    EF1R ... EF4Rmod   reverse EF-hand peptides (ref. 20), experimental ddG from the competition data
    LBT_N5D            LBT3 with the N5D substitution (ref. 26)
Input features for these peptides are in ../data/external/*_regression_features.csv (ion descriptors, DBAASP
descriptors, solubility score and the ESM-2 components projected with ../05_esm2_features).

Reported for each peptide: MSE and R2 against experiment (ions with both values), and the slopes of linear fits of ddG
against the Shannon radius (CN 9) for Tb-Lu (small ions) and La-Gd (large ions); the pooled MSE over the five EF-hand
peptides is printed at the end.
Output: results/<features>/external_predictions.csv and external_metrics.csv
"""
import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

import ddg_model as D

HERE = Path(__file__).resolve().parent
EXT = D.ROOT / "data/external"
SMALL = ["Tb", "Dy", "Ho", "Er", "Tm", "Yb", "Lu"]
LARGE = ["La", "Ce", "Pr", "Nd", "Pm", "Sm", "Eu", "Gd"]


def slope(df, col, ions, radius):
    s = df[df.Ion.isin(ions)].dropna(subset=[col])
    return np.polyfit(s.Ion.map(radius), s[col], 1)[0] if len(s) >= 2 else np.nan


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--features", default="descriptors", choices=list(D.FEATURE_SETS))
    ap.add_argument("--results", type=Path, default=HERE / "results")
    a = ap.parse_args()
    out = a.results / a.features
    model = joblib.load(out / "model.joblib")
    radius = dict(pd.read_csv(D.ROOT / "data/reference/shannon_radii_cn9.tsv", sep="\t").iloc[:, :2].values)

    feats = pd.concat([pd.read_csv(EXT / "efhand_regression_features.csv"),
                       pd.read_csv(EXT / "lbt3_n5d_regression_features.csv")], ignore_index=True)
    feats["ddG_pred"] = D.predict(model, feats[D.FEATURE_SETS[a.features]])
    ef = pd.read_csv(EXT / "efhand_experimental_ddG.csv").rename(columns={"ID": "Peptide", "Ln": "Ion"})
    n5d = pd.read_csv(EXT / "lbt3_n5d_experimental_ddG.csv").rename(columns={"Ln": "Ion"}).assign(Peptide="LBT_N5D")
    exp = pd.concat([ef[["Peptide", "Ion", "ddG_vs_Tb", "ddG_lo", "ddG_hi"]], n5d], ignore_index=True)
    pred = feats[["Peptide", "Ion", "ddG_pred"]].merge(exp, on=["Peptide", "Ion"], how="left") \
        .rename(columns={"ddG_vs_Tb": "ddG_exp"})
    pred.to_csv(out / "external_predictions.csv", index=False)

    rows = []
    for pep, g in pred.groupby("Peptide", sort=False):
        b = g.dropna(subset=["ddG_exp"])
        err = b.ddG_pred - b.ddG_exp
        rows.append({"peptide": pep, "n_ions": len(b), "mse": (err ** 2).mean(),
                     "r2": 1 - (err ** 2).sum() / ((b.ddG_exp - b.ddG_exp.mean()) ** 2).sum(),
                     "slope_small_exp": slope(b, "ddG_exp", SMALL, radius),
                     "slope_small_pred": slope(b, "ddG_pred", SMALL, radius),
                     "slope_large_exp": slope(b, "ddG_exp", LARGE, radius),
                     "slope_large_pred": slope(b, "ddG_pred", LARGE, radius)})
    m = pd.DataFrame(rows)
    m.to_csv(out / "external_metrics.csv", index=False)
    print(m.round(3).to_string(index=False))
    efs = pred[pred.Peptide != "LBT_N5D"].dropna(subset=["ddG_exp"])
    print(f"pooled MSE over the five EF-hand peptides: {((efs.ddG_pred - efs.ddG_exp) ** 2).mean():.3f} (kcal/mol)^2")


if __name__ == "__main__":
    main()
