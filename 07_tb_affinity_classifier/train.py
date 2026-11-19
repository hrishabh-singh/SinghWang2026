"""Train and evaluate the Tb(III)-affinity classifiers on Library 2 (Fig. 4A-D).

Usage
    python train.py --model extra_trees                 # full grid search (as in the paper), then evaluation
    python train.py --model extra_trees --selected      # skip the grid search; use selected_hyperparameters.json
    python train.py --model all --selected              # all seven models
    models: extra_trees, xgboost, random_forest, decision_tree, svm_rbf, svm_linear, logistic_regression

Procedure (Methods)
    1. Grid search on the training set (29,633 sequences); each setting is scored by the one-vs-rest macro AUC in
       stratified five-fold cross-validation (shuffled, seed 42).
    2. Out-of-fold probabilities of the selected model on the same five folds (solid ROC curves).
    3. Refit on the whole training set and predict the held-out test set (7,408 sequences; dashed ROC curves).
    4. Mean |SHAP| feature importance of the refitted model on 400 random training sequences.

Outputs in results/<model>/
    metrics.json (selected hyperparameters, cross-validation and test metrics), grid_search.csv (without --selected),
    oof_probabilities.csv, test_probabilities.csv, shap_importance.csv, model.joblib
"""
import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

import models as M

HERE = Path(__file__).resolve().parent


def run(model_name, use_selected, out_root, shap=True):
    train, test = M.load(M.TRAIN_CSV), M.load(M.TEST_CSV)
    X, y = M.xy(train)
    Xt, yt = M.xy(test)
    classes = np.unique(y)
    out = out_root / model_name
    out.mkdir(parents=True, exist_ok=True)
    print(f"[{model_name}] train {len(y)}, test {len(yt)}, classes {classes.tolist()}")

    if use_selected:
        params = M.selected_params(model_name)
    else:
        params, grid = M.grid_search(model_name, X, y, classes)
        grid.to_csv(out / "grid_search.csv", index=False)
    print(f"[{model_name}] hyperparameters: {params}")

    oof = M.out_of_fold(M.build(model_name, params), X, y, classes)
    model = M.build(model_name, params).fit(X, y)
    test_proba = M.predict_proba(model, Xt, classes)

    cols = [f"p_class{c}" for c in classes]
    pd.concat([train[["LBT", "Sequence", M.LABEL]], pd.DataFrame(oof, columns=cols)], axis=1) \
        .to_csv(out / "oof_probabilities.csv", index=False)
    pd.concat([test[["LBT", "Sequence", M.LABEL]], pd.DataFrame(test_proba, columns=cols)], axis=1) \
        .to_csv(out / "test_probabilities.csv", index=False)
    summary = {"model": model_name, "features": M.FEATURES, "hyperparameters": params,
               "n_train": int(len(y)), "n_test": int(len(yt)),
               "cross_validation": M.metrics(y, oof, classes), "test": M.metrics(yt, test_proba, classes)}
    (out / "metrics.json").write_text(json.dumps(summary, indent=2))
    joblib.dump(model, out / "model.joblib")
    if shap:
        M.shap_importance(model, X).to_csv(out / "shap_importance.csv", index=False)
    t = summary["test"]
    print(f"[{model_name}] test AUC {t['auc_ovr_macro']:.4f}  macro-F1 {t['macro_f1']:.4f}  "
          f"balanced accuracy {t['balanced_accuracy']:.4f}  accuracy {t['accuracy']:.4f}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, choices=["all", *M.MODELS])
    ap.add_argument("--selected", action="store_true", help="use the hyperparameters selected in the paper")
    ap.add_argument("--no-shap", action="store_true")
    ap.add_argument("--out", type=Path, default=HERE / "results")
    a = ap.parse_args()
    for name in (M.MODELS if a.model == "all" else [a.model]):
        run(name, a.selected, a.out, shap=not a.no_shap)


if __name__ == "__main__":
    main()
