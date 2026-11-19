"""Coordination number (Eq. 6) and ion-oxygen contacts from ion-oxygen distances.

Usage
    # 1. distances of one new unbiased run (files ion_O_<atom>.xvg written by gromacs_analysis.sh) -> distance table
    python coordination.py collect RUN_DIR/analysis --ion Tb --out tb_distances.csv
    # 2. coordination states and contact occupancy from a distance table
    python coordination.py summarize ../data/md_derived/unbiased_LBT1-CONH2_distances.csv.gz
    python coordination.py summarize ../data/md_derived/metadynamics_Tb_Lu_distances.csv.gz --by-state

Definitions (Methods)
    CN   = sum_i s(r_i),  s(r) = 1 / (1 + ((r - d0) / r0)^16) for r > d0 (1 otherwise), over the ten ligating oxygens
           (D3 OD1/OD2, N5 OD1, D7 OD1/OD2, W9 O, E11 OE1/OE2, E14 OE1/OE2); r0 = 0.27 nm, d0 = 0.03 nm. This is the
           PLUMED COORDINATION switching function with NN = 16, MM = 32 used as CV1 in the metadynamics.
    state n         frames with n - 1/2 <= CN < n + 1/2
    contact         ion-oxygen distance below 0.30 nm
    --by-state      contact occupancy within each coordination state (states with at least 200 frames), as in Fig. S2;
                    metadynamics frames are used as sampled, without reweighting
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

# Atom numbers in the LBT1-CONH2 topology -> site names used in the tables
OXYGENS = {51: "D3_OD1", 52: "D3_OD2", 77: "N5_OD1", 105: "D7_OD1", 106: "D7_OD2", 139: "W9_O",
           172: "E11_OE1", 173: "E11_OE2", 206: "E14_OE1", 207: "E14_OE2"}
R0, D0 = 0.27, 0.03          # nm
CONTACT_NM = 0.30
MIN_FRAMES_PER_STATE = 200


def switching(r, r0=R0, d0=D0):
    """PLUMED rational switching function with NN = 16, MM = 32: (1 - x^16) / (1 - x^32) = 1 / (1 + x^16)."""
    x = np.clip((np.asarray(r) - d0) / r0, 0.0, None)
    return 1.0 / (1.0 + x ** 16)


def coordination_number(distances, r0=R0, d0=D0):
    """CN of every frame from an (n_frames, 10) array of ion-oxygen distances in nm."""
    return switching(distances, r0, d0).sum(axis=1)


def collect(analysis_dir, ion):
    """Distance table (ion, time_ns, d_<site>..., cn) from the ion_O_<atom>.xvg files of one run."""
    cols, time = {}, None
    for atom, site in OXYGENS.items():
        d = np.loadtxt(Path(analysis_dir) / f"ion_O_{atom}.xvg", comments=("#", "@"))
        time = d[:, 0] if time is None else time[: len(d)]
        cols[f"d_{site}"] = d[:, 1]
    n = min(len(v) for v in cols.values())
    df = pd.DataFrame({k: v[:n] for k, v in cols.items()})
    df.insert(0, "time_ns", time[:n] / 1000.0)
    df.insert(0, "ion", ion)
    df["cn"] = coordination_number(df[[f"d_{s}" for s in OXYGENS.values()]].to_numpy())
    return df


def distance_columns(df):
    return [c for c in df.columns if c.startswith("d_")]


def summarize(df, by_state=False):
    """Fraction of time in each CN state and contact occupancy (%) of each oxygen, per ion."""
    df = df.copy()
    df["state"] = np.floor(df["cn"] + 0.5).astype(int)
    states = (df.groupby("ion")["state"].value_counts(normalize=True).mul(100).rename("percent_time")
              .reset_index().sort_values(["ion", "state"]))
    keys = ["ion", "state"] if by_state else ["ion"]
    grouped = df.groupby(keys)
    occ = grouped[distance_columns(df)].apply(lambda g: (g < CONTACT_NM).mean() * 100)
    occ["frames"] = grouped.size()
    if by_state:
        occ = occ[occ["frames"] >= MIN_FRAMES_PER_STATE]
    occ.columns = [c.replace("d_", "contact_%_") for c in occ.columns]
    return states, occ.reset_index()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("collect")
    c.add_argument("analysis_dir", type=Path)
    c.add_argument("--ion", required=True)
    c.add_argument("--out", type=Path, required=True)
    s = sub.add_parser("summarize")
    s.add_argument("table", type=Path)
    s.add_argument("--by-state", action="store_true")
    a = ap.parse_args()

    if a.cmd == "collect":
        df = collect(a.analysis_dir, a.ion)
        df.to_csv(a.out, index=False)
        print(f"{len(df)} frames, mean CN {df.cn.mean():.2f} -> {a.out}")
    else:
        states, occ = summarize(pd.read_csv(a.table), a.by_state)
        pd.set_option("display.width", 200)
        print(states.pivot(index="ion", columns="state", values="percent_time").fillna(0).round(1), "\n")
        print(occ.round(1).to_string(index=False))


if __name__ == "__main__":
    main()
