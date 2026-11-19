"""Shared definitions for the ddG(Ln/Tb) regression on Library 1.

Target   del_del_G = dG(peptide, Ln) - dG(peptide, Tb), kcal/mol (186 measurements)
Model    StandardScaler -> XGBoost regressor (pseudo-Huber loss, hist trees) trained on z = asinh(ddG);
         predictions are returned as sinh(z).
Weights  inverse-frequency sample weights from 10 quantile bins of ddG (normalised to mean 1), so that the few large
         ddG values (mostly the largest ions) are not swamped by the many small ones.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

ROOT = Path(__file__).resolve().parent.parent
LIBRARY1 = ROOT / "data/library1/library1_dataset.csv"
TARGET = "del_del_G"

# REE-cation descriptors: solvation free energy, Lennard-Jones sigma/epsilon, first/second hydration-shell
# distance and coordination number
ION = ["del_Gsolv", "Ln-sigma", "Ln-epsilon", "1st_dist", "1st_N", "2nd_dist", "2nd_N"]
# DBAASP peptide descriptors (the nine used for Library 2) and the CamSol-type solubility score
PEPTIDE = ["Normalized Hydrophobic Moment", "Normalized Hydrophobicity", "Net Charge", "Isoelectric Point",
           "Disordered Conformation Propensity", "Linear Moment", "Propensity to in vitro Aggregation",
           "Amphiphilicity Index", "Propensity to PPII coil", "solubility_score"]
ESM2 = [f"BOSPCA{i}_ESM2" for i in range(1, 16)]
FEATURE_SETS = {
    "descriptors": ION + PEPTIDE,                   # model reported in the paper (Fig. S7, Table S5)
    "descriptors_esm2": ION + PEPTIDE + ESM2,       # variant with 15 ESM-2 principal components
}

GRID = {"learning_rate": [0.03, 0.05, 0.1], "max_depth": [3, 5, 7], "min_child_weight": [3, 5, 7],
        "gamma": [0.0, 0.2], "colsample_bytree": [0.5, 0.7], "subsample": [0.8, 1.0],
        "n_estimators": [300, 600, 1000], "huber_slope": [0.5, 1.0, 1.5]}
GRID_SEEDS = [42, 1713, 671, 331, 519]
OOF_SEED = 1511
FINAL_SEED = 42
N_FOLDS = 5
WEIGHT_BINS = 10
SELECTED = Path(__file__).resolve().parent / "selected_hyperparameters.json"
# Settings of the models behind the numbers reported in the paper (OOF metrics, Fig. S7, Table S5). In the original
# training script the grid-selected values were passed to the pipeline under doubled 'xgb__xgb__' names, which XGBoost
# stores but does not apply; only huber_slope (set in the constructor) took effect, so these models used the XGBoost
# defaults otherwise (100 trees, maximum depth 6, learning rate 0.3, no row/column subsampling).
REPORTED = {"huber_slope": 0.5}


def pipeline(seed=FINAL_SEED, **params):
    xgb = XGBRegressor(objective="reg:pseudohubererror", tree_method="hist", n_jobs=0, random_state=seed)
    xgb.set_params(**params)
    return Pipeline([("scaler", StandardScaler()), ("xgb", xgb)])


def density_weights(y, n_bins=WEIGHT_BINS):
    """Inverse-frequency weights over quantile bins of y, normalised to mean 1."""
    edges = np.unique(np.percentile(y, np.linspace(0, 100, n_bins + 1)))
    if len(edges) < 3:
        edges = np.linspace(y.min(), y.max(), n_bins + 1)
    ids = np.digitize(y, edges[1:-1], right=True)
    freq = np.bincount(ids, minlength=len(edges) - 1).astype(float)
    freq[freq == 0] = 1.0
    w = 1.0 / freq[ids]
    return w / w.mean()


def load_training(feature_set):
    df = pd.read_csv(LIBRARY1)
    return df, df[FEATURE_SETS[feature_set]], df[TARGET].to_numpy(float)


def hyperparameters(feature_set, which):
    """'reported': settings of the published models; 'selected': grid-search selection (selected_hyperparameters.json)."""
    return dict(REPORTED) if which == "reported" else json.loads(SELECTED.read_text())[feature_set]


def predict(model, X):
    return np.sinh(model.predict(X))
