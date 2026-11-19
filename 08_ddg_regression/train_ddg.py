"""Train the ddG(Ln/Tb) XGBoost regressor on Library 1.

Usage
    python train_ddg.py --features descriptors                              # models behind the reported numbers
    python train_ddg.py --features descriptors_esm2                         # ESM-2 variant, as reported
    python train_ddg.py --features descriptors --hyperparameters selected   # grid-selected values (see ddg_model.py)
    python train_ddg.py --features descriptors --hyperparameters search     # rerun the grid search, then use its pick
    (--n-jobs sets the grid-search workers; 1,944 settings x 5 folds x 5 seeds ran on an 8-core HPC node)

Procedure (Methods)
    1. Grid search (GRID in ddg_model.py) with five-fold cross-validation (KFold, shuffled) repeated for five seeds;
       each setting is scored by its mean squared error on z = asinh(ddG), averaged over the seeds, and the lowest
       mean MSE is selected. Sample weights are passed to every fit.
    2. Out-of-fold predictions of the selected setting (five folds, seed 1511), reported on the ddG scale.
    3. Final model refitted on all 186 measurements -> results/<features>/model.joblib.
    --hyperparameters reported (default) reproduces the published models exactly; see REPORTED in ddg_model.py.

Note: rows of the same peptide (measured with different ions) can fall in both training and validation folds, so the
out-of-fold scores are optimistic; the external peptides (predict_ddg.py) are the stricter test.
"""
import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, KFold

import ddg_model as D

HERE = Path(__file__).resolve().parent


def grid_search(X, z, w, n_jobs):
    grid = {f"xgb__{k}": v for k, v in D.GRID.items()}
    per_seed = []
    for seed in D.GRID_SEEDS:
        gs = GridSearchCV(D.pipeline(seed=seed), grid, scoring="neg_mean_squared_error",
                          cv=KFold(D.N_FOLDS, shuffle=True, random_state=seed), n_jobs=n_jobs, refit=False)
        gs.fit(X, z, xgb__sample_weight=w)
        res = pd.DataFrame(gs.cv_results_)
        per_seed.append(pd.DataFrame({"params": [json.dumps({k.replace("xgb__", ""): v for k, v in p.items()},
                                                             sort_keys=True) for p in res["params"]],
                                      f"mse_seed{seed}": -res["mean_test_score"]}).set_index("params"))
        print(f"  seed {seed}: best MSE(z) {per_seed[-1].iloc[:, 0].min():.5f}")
    table = pd.concat(per_seed, axis=1)
    table["mean_mse"] = table.mean(axis=1)
    table["std_mse"] = table.drop(columns="mean_mse").std(axis=1, ddof=0)
    table = table.sort_values("mean_mse", kind="stable")
    return json.loads(table.index[0]), table


def out_of_fold(params, X, z, w):
    z_oof = np.zeros_like(z)
    for tr, va in KFold(D.N_FOLDS, shuffle=True, random_state=D.OOF_SEED).split(X):
        m = D.pipeline(**params).fit(X.iloc[tr], z[tr], xgb__sample_weight=w[tr])
        z_oof[va] = m.predict(X.iloc[va])
    return np.sinh(z_oof)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--features", required=True, choices=list(D.FEATURE_SETS))
    ap.add_argument("--hyperparameters", choices=["reported", "selected", "search"], default="reported")
    ap.add_argument("--n-jobs", type=int, default=-1)
    ap.add_argument("--out", type=Path, default=HERE / "results")
    a = ap.parse_args()
    out = a.out / a.features
    out.mkdir(parents=True, exist_ok=True)

    df, X, y = D.load_training(a.features)
    z, w = np.arcsinh(y), D.density_weights(y)
    print(f"[{a.features}] {len(y)} measurements, {X.shape[1]} features")
    if a.hyperparameters == "search":
        params, table = grid_search(X, z, w, a.n_jobs)
        table.to_csv(out / "grid_search_over_seeds.csv")
    else:
        params = D.hyperparameters(a.features, a.hyperparameters)
    print(f"[{a.features}] hyperparameters: {params}")

    y_oof = out_of_fold(params, X, z, w)
    pd.DataFrame({"PEPS": df.PEPS, "Ln": df.Ln, "ddG": y, "ddG_oof": y_oof}).to_csv(out / "oof_predictions.csv",
                                                                                  index=False)
    model = D.pipeline(**params).fit(X, z, xgb__sample_weight=w)
    joblib.dump(model, out / "model.joblib")
    gain = model.named_steps["xgb"].get_booster().get_score(importance_type="gain")
    pd.Series({c: gain.get(f"f{i}", gain.get(c, 0.0)) for i, c in enumerate(X.columns)}, name="gain") \
        .sort_values(ascending=False).to_csv(out / "gain_importance.csv")
    summary = {"features": a.features, "feature_columns": list(X.columns),
               "hyperparameter_mode": a.hyperparameters, "hyperparameters": params,
               "oof_mse": mean_squared_error(y, y_oof), "oof_r2": r2_score(y, y_oof)}
    (out / "metrics.json").write_text(json.dumps(summary, indent=2))
    print(f"[{a.features}] out-of-fold MSE {summary['oof_mse']:.4f}  R2 {summary['oof_r2']:.3f}")


if __name__ == "__main__":
    main()
