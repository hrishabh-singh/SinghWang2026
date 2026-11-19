"""Shared definitions for the Library 2 Tb(III)-affinity classifiers: data, models, hyperparameter grids, metrics, SHAP.

Library 2: Imperiali LBT sequences with Tb(III) dG = RT ln K_D. Four affinity classes from the dG cutoffs
-7.155, -7.8765 and -9.0745 kcal/mol (class 0 weakest, class 3 strongest). Fixed 80/20 split (29,633 / 7,408).
Inputs: the nine DBAASP sequence descriptors (FEATURES).
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import ParameterGrid, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

ROOT = Path(__file__).resolve().parent.parent
TRAIN_CSV = ROOT / "data/library2/library2_train.csv"
TEST_CSV = ROOT / "data/library2/library2_test.csv"
SELECTED = Path(__file__).resolve().parent / "selected_hyperparameters.json"

FEATURES = ["Normalized Hydrophobic Moment", "Normalized Hydrophobicity", "Net Charge", "Isoelectric Point",
            "Disordered Conformation Propensity", "Linear Moment", "Propensity to in vitro Aggregation",
            "Amphiphilicity Index", "Propensity to PPII coil"]
LABEL = "label"
CLASS_CUTOFFS = (-7.155, -7.8765, -9.0745)      # kcal/mol, documentation of how 'label' was assigned
SEED = 42
N_FOLDS = 5


# ---- models and grids (exhaustive grid search; scaled inputs for the linear and kernel models) ----------------------
def _scaled(model):
    return Pipeline([("scaler", StandardScaler()), ("model", model)])


MODELS = {
    "extra_trees": (lambda p: ExtraTreesClassifier(random_state=SEED, n_jobs=-1, **p),
                    {"n_estimators": [200, 400], "max_depth": [None, 5, 10], "min_samples_split": [2, 5],
                     "min_samples_leaf": [1, 2], "max_features": ["sqrt", None], "class_weight": [None, "balanced"]}),
    "random_forest": (lambda p: RandomForestClassifier(random_state=SEED, n_jobs=-1, **p),
                      {"n_estimators": [200, 400], "max_depth": [None, 5, 10], "min_samples_split": [2, 5],
                       "min_samples_leaf": [1, 2], "max_features": ["sqrt", None],
                       "class_weight": [None, "balanced"]}),
    "xgboost": (lambda p: __import__("xgboost").XGBClassifier(random_state=SEED, n_jobs=-1, tree_method="hist",
                                                              objective="multi:softprob", eval_metric="mlogloss",
                                                              num_class=4, **p),
                {"n_estimators": [200, 400], "max_depth": [2, 3, 4], "learning_rate": [0.03, 0.1],
                 "subsample": [0.8, 1.0], "colsample_bytree": [0.8, 1.0], "reg_lambda": [1.0, 5.0],
                 "min_child_weight": [1, 3]}),
    "decision_tree": (lambda p: DecisionTreeClassifier(random_state=SEED, **p),
                      {"max_depth": [None, 3, 5, 8, 12], "min_samples_split": [2, 5, 10],
                       "min_samples_leaf": [1, 2, 5], "class_weight": [None, "balanced"],
                       "criterion": ["gini", "entropy"]}),
    "svm_rbf": (lambda p: _scaled(SVC(kernel="rbf", probability=True, random_state=SEED, **p)),
                {"C": [0.5, 1.0, 5.0, 10.0, 50.0], "gamma": ["scale", 0.1, 0.05, 0.01],
                 "class_weight": [None, "balanced"]}),
    "svm_linear": (lambda p: _scaled(SVC(kernel="linear", probability=True, random_state=SEED, **p)),
                   {"C": [0.1, 1.0, 10.0, 50.0, 100.0], "class_weight": [None, "balanced"]}),
    "logistic_regression": (lambda p: _scaled(LogisticRegression(random_state=SEED, max_iter=2000, **p)),
                            {"penalty": ["l2"], "C": [0.01, 0.1, 1.0, 10.0, 100.0], "solver": ["lbfgs", "liblinear"],
                             "class_weight": [None, "balanced"]}),
}


def build(model_name, params):
    return MODELS[model_name][0](params)


def selected_params(model_name):
    """Hyperparameters selected by the grid search reported in the paper."""
    return json.loads(SELECTED.read_text())[model_name]


# ---- data -----------------------------------------------------------------------------------------------------------
def load(path):
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]              # one source header carries a trailing space
    return df


def xy(df):
    return df[FEATURES].to_numpy(float), df[LABEL].to_numpy(int)


# ---- cross-validation, grid search, metrics -------------------------------------------------------------------------
def folds():
    return StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)


def predict_proba(model, X, classes):
    """Class probabilities in the order of `classes`."""
    p = model.predict_proba(X)
    out = np.zeros((len(X), len(classes)))
    for j, c in enumerate(model.classes_):
        out[:, np.searchsorted(classes, c)] = p[:, j]
    return out


def ovr_auc(y, proba, classes):
    return roc_auc_score(y, proba, labels=list(classes), multi_class="ovr", average="macro")


def out_of_fold(model, X, y, classes):
    """Out-of-fold class probabilities from stratified five-fold cross-validation."""
    proba = np.zeros((len(y), len(classes)))
    for tr, va in folds().split(X, y):
        proba[va] = predict_proba(clone(model).fit(X[tr], y[tr]), X[va], classes)
    return proba


def grid_search(model_name, X, y, classes, log=print):
    """Exhaustive grid search; score = mean one-vs-rest macro AUC over the five folds."""
    records = []
    grid = list(ParameterGrid(MODELS[model_name][1]))
    for i, params in enumerate(grid, 1):
        aucs = [ovr_auc(y[va], predict_proba(build(model_name, params).fit(X[tr], y[tr]), X[va], classes), classes)
                for tr, va in folds().split(X, y)]
        records.append({"params": json.dumps(params), "cv_auc_mean": np.mean(aucs), "cv_auc_std": np.std(aucs),
                        **{f"fold{k}_auc": a for k, a in enumerate(aucs, 1)}})
        log(f"  [{i}/{len(grid)}] {params}  AUC {np.mean(aucs):.5f}")
    table = pd.DataFrame(records)
    best = json.loads(table.loc[table.cv_auc_mean.idxmax(), "params"])   # first of any ties, in grid order
    return best, table.sort_values("cv_auc_mean", ascending=False, kind="stable")


def metrics(y, proba, classes):
    pred = classes[proba.argmax(axis=1)]
    return {"auc_ovr_macro": ovr_auc(y, proba, classes),
            "auc_per_class": [roc_auc_score(y == c, proba[:, k]) for k, c in enumerate(classes)],
            "macro_f1": f1_score(y, pred, average="macro"),
            "balanced_accuracy": balanced_accuracy_score(y, pred),
            "accuracy": accuracy_score(y, pred),
            "macro_precision": precision_score(y, pred, average="macro", zero_division=0),
            "macro_recall": recall_score(y, pred, average="macro", zero_division=0),
            "confusion_matrix": confusion_matrix(y, pred, labels=classes).tolist()}


# ---- SHAP ------------------------------------------------------------------------------------------------------------
def shap_importance(model, X_train, n_background=200, n_explain=400):
    """Mean |SHAP| per feature over n_explain random training rows (averaged over classes).
    Tree models use the tree explainer, linear models the linear explainer, SVMs the kernel explainer."""
    import shap
    rng = np.random.default_rng(SEED)
    bg = rng.choice(len(X_train), size=min(n_background, len(X_train)), replace=False)
    ex = rng.choice(len(X_train), size=min(n_explain, len(X_train)), replace=False)
    est, Xt = model, X_train
    if isinstance(model, Pipeline):
        est, Xt = model.named_steps["model"], model.named_steps["scaler"].transform(X_train)
    if isinstance(est, (ExtraTreesClassifier, RandomForestClassifier, DecisionTreeClassifier)) or \
            est.__class__.__name__.startswith("XGB"):
        values = shap.TreeExplainer(est).shap_values(Xt[ex])
    elif isinstance(est, LogisticRegression):
        values = shap.LinearExplainer(est, Xt[bg]).shap_values(Xt[ex])
    else:
        values = shap.KernelExplainer(est.predict_proba, Xt[bg]).shap_values(Xt[ex], nsamples=200)
    arr = np.abs(np.asarray(values))
    feat_axis = [i for i, n in enumerate(arr.shape) if n == len(FEATURES)][-1]
    mean_abs = arr.mean(axis=tuple(i for i in range(arr.ndim) if i != feat_axis))
    return pd.DataFrame({"feature": FEATURES, "mean_abs_shap": mean_abs,
                         "relative": mean_abs / mean_abs.sum()}).sort_values("mean_abs_shap", ascending=False)
