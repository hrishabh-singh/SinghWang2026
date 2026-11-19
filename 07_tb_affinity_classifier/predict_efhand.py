"""Tb(III) binding free energy of the five reverse EF-hand peptides from the trained classifier (Fig. 4E).

Usage
    python predict_efhand.py [--model extra_trees] [--results results]
    (requires results/<model>/model.joblib from train.py and ../data/external/efhand_experimental_Tb_dG.csv from
     ../06_efhand_experimental_values/competition_to_dG.py)

Method (Methods)
    The classifier gives class probabilities p_k for the four affinity classes. Each class is represented by the
    median Tb(III) dG of its training sequences (classes 0-2 contain a single dG value each; Table S2), and
        dG_pred = sum_k p_k dG_k,     spread = sqrt(sum_k p_k (dG_k - dG_pred)^2)
    The prediction is compared with the experimental dG obtained from the published competition data.

Output: results/<model>/efhand_Tb_dG_predictions.csv and the mean absolute error printed to the console.
"""
import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

import models as M

HERE = Path(__file__).resolve().parent


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="extra_trees", choices=list(M.MODELS))
    ap.add_argument("--results", type=Path, default=HERE / "results")
    ap.add_argument("--features", type=Path, default=M.ROOT / "data/external/efhand_dbaasp_descriptors.csv")
    ap.add_argument("--experimental", type=Path, default=M.ROOT / "data/external/efhand_experimental_Tb_dG.csv")
    a = ap.parse_args()

    model = joblib.load(a.results / a.model / "model.joblib")
    train = M.load(M.TRAIN_CSV)
    classes = np.unique(train[M.LABEL])
    class_dG = train.groupby(M.LABEL)["G_Kcal_mol"].median().reindex(classes).to_numpy()

    ef = M.load(a.features)
    p = M.predict_proba(model, ef[M.FEATURES].to_numpy(float), classes)
    pred = p @ class_dG
    spread = np.sqrt((p * (class_dG[None, :] - pred[:, None]) ** 2).sum(axis=1))

    out = pd.DataFrame({"ID": ef["ID"], "predicted_class": classes[p.argmax(axis=1)],
                        **{f"p_class{c}": p[:, k] for k, c in enumerate(classes)},
                        "dG_predicted": pred, "dG_predicted_spread": spread})
    exp = pd.read_csv(a.experimental)[["ID", "dG", "dG_lo", "dG_hi"]].rename(
        columns={"dG": "dG_experimental", "dG_lo": "dG_experimental_lo", "dG_hi": "dG_experimental_hi"})
    out = out.merge(exp, on="ID", how="left")
    out["abs_error"] = (out.dG_predicted - out.dG_experimental).abs()
    out.to_csv(a.results / a.model / "efhand_Tb_dG_predictions.csv", index=False)
    print("class dG (kcal/mol):", np.round(class_dG, 3).tolist())
    print(out[["ID", "dG_experimental", "dG_predicted", "dG_predicted_spread", "abs_error"]].round(3).to_string(index=False))
    print(f"mean absolute error: {out.abs_error.mean():.2f} kcal/mol")


if __name__ == "__main__":
    main()
