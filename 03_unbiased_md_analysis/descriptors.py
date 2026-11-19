"""Conformational and interaction-energy descriptors of one peptide-Ln(III) complex from its unbiased run.

Usage
    python descriptors.py RUN_DIR/analysis [--peptide LBT1-CONH2 --ion Tb] [--out descriptors.csv]

Reads the files written by gromacs_analysis.sh and returns one row with the descriptor columns of the Library 1 table
(data/library1/library1_dataset.csv):
    angle1..angle12, angle<k>_std   backbone N-CA-C angle of loop residue k+2 (residues 3-14), mean and SD
    phi1..phi12, psi1..psi12 (+_std) backbone torsions of the same residues, mean and SD
    SASA                           mean peptide solvent-accessible surface area (nm^2)
    <A>-<B>, <A>-<B>.1             mean and SD of the short-range interaction energy Coul-SR + LJ-SR (kJ/mol) between
                                   energy groups A, B in {Ln, r3, r5, r7, r9, r11, r14, SOL}; r<i> = residue i
                                   (Coulomb: plain cut-off 2.35 nm; Lennard-Jones: production settings, 1.0 nm)
All averages are taken from 100 ns to the end of the 200 ns production run (the window used for the .xvg files).
"""
import argparse
import re
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

LOOP_RESIDUES = list(range(3, 15))                     # residues 3-14 -> descriptor index k = residue - 2
GROUPS = ["Ln", "r3", "r5", "r7", "r9", "r11", "r14", "SOL"]


def read_xvg(path):
    """Numeric columns and legend labels of a GROMACS .xvg file."""
    labels, rows = [], []
    for line in Path(path).read_text().splitlines():
        if re.match(r'@ s\d+ legend "', line):
            labels.append(line.split('"')[1])
        elif line and not line.startswith(("#", "@")):
            rows.append([float(x) for x in line.split()])
    return np.array(rows), labels


def angle_descriptors(path):
    """N-CA-C angles: gmx angle -all writes time, average, then one column per residue."""
    data, _ = read_xvg(path)
    data = data[np.round(data[:, 0]).astype(int) % 10 == 0]          # one frame every 10 ps
    out = {}
    for k, res in enumerate(LOOP_RESIDUES, start=1):
        col = data[:, 2 + (res - 1)]                                  # residue 1 is column 2
        out[f"angle{k}"], out[f"angle{k}_std"] = col.mean(), col.std()
    return out


def torsion_descriptors(path):
    """phi/psi from gmx rama (lines: phi psi RES-n)."""
    rows = []
    for line in Path(path).read_text().splitlines():
        if line and not line.startswith(("#", "@")):
            phi, psi, name = line.split()[:3]
            rows.append((float(phi), float(psi), int(name.split("-")[-1])))
    d = pd.DataFrame(rows, columns=["phi", "psi", "res"])
    out = {}
    for k, res in enumerate(LOOP_RESIDUES, start=1):
        s = d[d.res == res]
        for t in ("phi", "psi"):
            out[f"{t}{k}"], out[f"{t}{k}_std"] = s[t].mean(), s[t].std(ddof=0)
    return out


def _term(data, labels, kind, a, b, path):
    cols = [i + 1 for i, lab in enumerate(labels) if lab in (f"{kind}-SR:{a}-{b}", f"{kind}-SR:{b}-{a}")]
    if len(cols) != 1:
        raise ValueError(f"{kind}-SR:{a}-{b} not found in {path}")
    return data[:, cols[0]]


def energy_descriptors(coul_path, lj_path):
    """Mean and SD over frames of E = Coul-SR + LJ-SR for every pair of energy groups.

    Coul-SR comes from the rerun with plain Coulomb cut-off at 2.35 nm (energy_coul.xvg) and LJ-SR from the rerun with
    the production settings (energy_lj.xvg); both reruns cover the same frames."""
    coul, lab_c = read_xvg(coul_path)
    lj, lab_l = read_xvg(lj_path)
    if coul.shape[0] != lj.shape[0]:
        raise ValueError("the two energy files have different numbers of frames")
    out = {}
    for a, b in combinations(GROUPS, 2):
        e = _term(coul, lab_c, "Coul", a, b, coul_path) + _term(lj, lab_l, "LJ", a, b, lj_path)
        out[f"{a}-{b}"], out[f"{a}-{b}.1"] = e.mean(), e.std()
    return out


def sasa_descriptor(path):
    data, _ = read_xvg(path)
    return {"SASA": data[:, 1].mean()}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("analysis_dir", type=Path)
    ap.add_argument("--peptide", default="")
    ap.add_argument("--ion", default="")
    ap.add_argument("--out", type=Path)
    a = ap.parse_args()
    d = a.analysis_dir
    row = {"PEPS": a.peptide, "Ln": a.ion}
    row.update(angle_descriptors(d / "angles.xvg"))
    row.update(torsion_descriptors(d / "rama.xvg"))
    row.update(sasa_descriptor(d / "sasa.xvg"))
    row.update(energy_descriptors(d / "energy_coul.xvg", d / "energy_lj.xvg"))
    df = pd.DataFrame([row])
    df.to_csv(a.out or d / "descriptors.csv", index=False)
    print(df.T.to_string(header=False))


if __name__ == "__main__":
    main()
