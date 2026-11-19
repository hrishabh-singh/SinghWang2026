"""ESM-2 sequence features: first-token (BOS) embedding reduced to 15 principal components.

Usage
    # Fit the scaler and PCA on the unique Library 1 sequences and write the 15 components for every unique sequence
    python esm2_features.py fit [--data ../data/library1/library1_dataset.csv] [--artifacts artifacts]
    # Project new sequences with the saved scaler and PCA
    python esm2_features.py project --sequences YIDTNNDGWYEGDELLA DKDGDGYISAAE --out new_pcs.csv
    python esm2_features.py project --csv peptides.csv --seq-column sequence --out new_pcs.csv

Method (Methods; Table S3 note)
    Model     facebook/esm2_t33_650M_UR50D (650M parameters, 33 layers, 1,280-dimensional hidden states)
    Input     sequence with every non-letter removed (e.g. terminal '-CONH2' notation), upper case
    Feature   last hidden state of the first (BOS/<cls>) token, 1,280 numbers per sequence
    Reduction StandardScaler followed by PCA(15, random_state=0), fitted on the unique sequences of Library 1 only
              (so peptides measured with many ions are not over-weighted); every row of a peptide gets the same PCs.
    The fitted artifacts used in the paper are in artifacts/ (bos_esm2_k15_scaler.joblib, bos_esm2_k15_pca.joblib);
    'fit' overwrites them only when --artifacts points to the same folder.
"""
import argparse
import re
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from transformers import AutoModel, AutoTokenizer

HERE = Path(__file__).resolve().parent
MODEL = "facebook/esm2_t33_650M_UR50D"
K_PCS = 15
TAG = f"bos_esm2_k{K_PCS}"


def clean_sequence(raw):
    """Keep letters only, upper case (drops terminal-group notation and spaces)."""
    return re.sub(r"[^A-Za-z]", "", str(raw)).upper()


class ESM2:
    def __init__(self, model_name=MODEL):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, do_lower_case=False)
        self.model = AutoModel.from_pretrained(model_name, add_pooling_layer=False).to(self.device).eval()

    @torch.no_grad()
    def bos(self, seq):
        """Last-layer embedding of the first token, shape (1280,)."""
        inputs = {k: v.to(self.device) for k, v in self.tokenizer(seq, return_tensors="pt").items()}
        return self.model(**inputs).last_hidden_state[0, 0].cpu().numpy()


def bos_matrix(sequences):
    esm = ESM2()
    return np.stack([esm.bos(s) for s in sequences])


def pc_table(sequences, Z):
    return pd.DataFrame(Z, columns=[f"BOSPCA{i}_ESM2" for i in range(1, Z.shape[1] + 1)]).assign(
        sequence=sequences)[["sequence"] + [f"BOSPCA{i}_ESM2" for i in range(1, Z.shape[1] + 1)]]


def fit(data, artifacts):
    seqs = list(dict.fromkeys(clean_sequence(s) for s in pd.read_csv(data)["sequence"]))   # unique, first-seen order
    X = bos_matrix(seqs)
    scaler = StandardScaler().fit(X)
    pca = PCA(n_components=K_PCS, random_state=0).fit(scaler.transform(X))
    artifacts.mkdir(parents=True, exist_ok=True)
    joblib.dump(scaler, artifacts / f"{TAG}_scaler.joblib")
    joblib.dump(pca, artifacts / f"{TAG}_pca.joblib")
    print(f"{len(seqs)} unique sequences; variance explained by {K_PCS} PCs: {pca.explained_variance_ratio_.sum():.3f}")
    return pc_table(seqs, pca.transform(scaler.transform(X)))


def project(sequences, artifacts):
    scaler = joblib.load(artifacts / f"{TAG}_scaler.joblib")
    pca = joblib.load(artifacts / f"{TAG}_pca.joblib")
    seqs = [clean_sequence(s) for s in sequences]
    return pc_table(seqs, pca.transform(scaler.transform(bos_matrix(seqs))))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("fit")
    f.add_argument("--data", type=Path, default=HERE.parent / "data/library1/library1_dataset.csv")
    f.add_argument("--artifacts", type=Path, default=HERE / "artifacts_refit")
    f.add_argument("--out", type=Path, default=Path("library1_esm2_pcs.csv"))
    p = sub.add_parser("project")
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--sequences", nargs="+")
    src.add_argument("--csv", type=Path)
    p.add_argument("--seq-column", default="sequence")
    p.add_argument("--artifacts", type=Path, default=HERE / "artifacts")
    p.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()

    if a.cmd == "fit":
        table = fit(a.data, a.artifacts)
    else:
        seqs = a.sequences if a.sequences else pd.read_csv(a.csv)[a.seq_column].tolist()
        table = project(seqs, a.artifacts)
    table.to_csv(a.out, index=False)
    print(f"wrote {len(table)} rows -> {a.out}")


if __name__ == "__main__":
    main()
