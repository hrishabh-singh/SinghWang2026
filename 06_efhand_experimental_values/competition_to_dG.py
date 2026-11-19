"""Experimental Tb(III) dG and ddG(Ln/Tb) of the five reverse EF-hand peptides from published competition data.

Usage
    python competition_to_dG.py [--boxes ../data/external/efhand_competition_boxplots] [--out ../data/external]

Data (Gutenthaler-Tietze et al., ref. 20): each peptide (10 uM) was equilibrated with Eu(III) and a second lanthanide
(40 uM each); the ratio R of free to complexed Eu(III), normalised to the Eu-only sample, was reported as box plots.
The medians and quartiles of R were digitised from those plots (data/external/efhand_competition_boxplots/).

Conversion (Methods)
    For every ion, K_D(Ln) follows from the mass balance of competitive 1:1 binding at the stated totals, using the
    published K_D of the 1:1 Eu(III) complex; the Eu-only reference is 80 uM Eu + 10 uM peptide. 1:2 peptide:metal
    complexes are neglected. dG = RT ln K_D (298 K), ddG(Ln/Tb) = dG(Ln) - dG(Tb).
    Uncertainty: Monte Carlo with 20,000 draws, R ~ Normal(median, IQR/1.349) (floor 0.004, about one pixel) and
    K_D(Eu) ~ Normal(mean, SD) truncated at 0.02 uM; reported values are the median and the distances to the 16th and
    84th percentiles.

Outputs
    efhand_experimental_Tb_dG.csv   ID, peptide, dG, dG_lo, dG_hi           (Fig. 4E)
    efhand_experimental_ddG.csv     ID, peptide, Ln, ddG_vs_Tb, ddG_lo, ddG_hi (Fig. S7A-E, Table S5)
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RT = 1.987204258e-3 * 298.0                     # kcal/mol
N_DRAWS, SEED = 20_000, 7
PEPTIDE_UM, METAL_UM = 10.0, 40.0
# K_D of the 1:1 Eu(III) complex (uM, mean and SD) from ref. 20 (Table 1; EF4-R_mod from the text)
KD_EU = {"EF1-R": (0.50, 0.11), "EF2-R": (0.59, 0.15), "EF3-R": (0.32, 0.09), "EF4-R": (2.5, 0.5),
         "EF4-R_mod": (0.150, 0.050)}
ID = {"EF1-R": "EF1R", "EF2-R": "EF2R", "EF3-R": "EF3R", "EF4-R": "EF4R", "EF4-R_mod": "EF4Rmod"}


def free_peptide_reference(kd_eu):
    """Free peptide (uM) in the Eu-only sample: 80 uM Eu + 10 uM peptide, 1:1 binding."""
    eu = 2 * METAL_UM
    b = eu + PEPTIDE_UM + kd_eu
    complex_ = (b - np.sqrt(b * b - 4 * eu * PEPTIDE_UM)) / 2
    return PEPTIDE_UM - complex_


def kd_ln(R, kd_eu):
    """K_D of the competing lanthanide (uM) from the ratio R by the 1:1 competitive mass balance."""
    p_free = free_peptide_reference(kd_eu) / R
    eu_free = METAL_UM / (1 + p_free / kd_eu)
    eu_bound = METAL_UM - eu_free
    ln_bound = PEPTIDE_UM - p_free - eu_bound
    ln_bound = np.where(ln_bound > 1e-6, ln_bound, np.nan)          # physically impossible draws are dropped
    return (METAL_UM - ln_bound) * p_free / ln_bound


def convert(box_dir):
    rng = np.random.default_rng(SEED)
    rows = []
    for pep, (k0, ks) in KD_EU.items():
        box = pd.read_csv(box_dir / f"{pep}_boxes.csv").set_index("Ln")
        sigma = np.maximum((box.q3 - box.q1) / 1.349, 0.004)
        R = {ln: rng.normal(box["median"][ln], sigma[ln], N_DRAWS) for ln in box.index}
        kd_eu = rng.normal(k0, ks, N_DRAWS)
        kd_eu = np.where(kd_eu > 0.02, kd_eu, np.nan)
        dG = {ln: RT * np.log(kd_ln(R[ln], kd_eu) * 1e-6) for ln in box.index}
        for ln in box.index:
            g = np.nanpercentile(dG[ln], [16, 50, 84])
            h = np.nanpercentile(dG[ln] - dG["Tb"], [16, 50, 84])
            tb = ln == "Tb"
            rows.append(dict(ID=ID[pep], peptide=pep, Ln=ln, dG=g[1], dG_lo=g[1] - g[0], dG_hi=g[2] - g[1],
                             ddG_vs_Tb=h[1], ddG_lo=0.0 if tb else h[1] - h[0], ddG_hi=0.0 if tb else h[2] - h[1]))
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--boxes", type=Path, default=ROOT / "data/external/efhand_competition_boxplots")
    ap.add_argument("--out", type=Path, default=ROOT / "data/external")
    a = ap.parse_args()
    d = convert(a.boxes)
    d[d.Ln == "Tb"][["ID", "peptide", "dG", "dG_lo", "dG_hi"]].to_csv(a.out / "efhand_experimental_Tb_dG.csv", index=False)
    d[["ID", "peptide", "Ln", "ddG_vs_Tb", "ddG_lo", "ddG_hi"]].to_csv(a.out / "efhand_experimental_ddG.csv", index=False)
    print(d[d.Ln == "Tb"][["peptide", "dG", "dG_lo", "dG_hi"]].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
