# Data

All energies are in kcal/mol unless noted; distances in nm.

## `library1/library1_dataset.csv` — Library 1 (186 peptide–REE measurements)

40 LBT peptides (38 distinct sequences; the –CONH2 and –COO⁻ forms share a sequence) measured with up to 13 lanthanides. One row per peptide–ion pair.

| Columns | Meaning |
|---|---|
| `PEPS`, `Ln`, `sequence` | peptide name, lanthanide, sequence |
| `del_G` | binding free energy ΔG = RT ln K_D (298 K) |
| `del_del_G` | ΔΔG(Ln/Tb) = ΔG(Ln) − ΔG(Tb) of the same peptide (regression target) |
| `del_Gsolv`, `Ln-sigma`, `Ln-epsilon` | ion solvation free energy (kJ/mol) and Lennard-Jones σ (nm), ε (kJ/mol) of the ion model |
| `1st_dist`, `1st_N`, `2nd_dist`, `2nd_N` | first/second hydration-shell distance (Å) and water count of the free ion (MD) |
| `Normalized Hydrophobic Moment` … `Propensity to PPII coil` | DBAASP sequence descriptors |
| `ddG`, `ddG_H2O`, `dTm` | AUTOMUTE predicted stability changes |
| `solubility_score` | sequence solubility score (used by the ΔΔG regression) |
| `angle1`–`angle12` (+`_std`) | mean (SD) backbone N–Cα–C angle of residue k+2 (residues 3–14), unbiased MD, 100–200 ns |
| `phi1`–`phi12`, `psi1`–`psi12` (+`_std`) | mean (SD) backbone torsions of the same residues |
| `SASA` | mean peptide solvent-accessible surface area (nm²) |
| `Ln-r3` … `r14-SOL` | mean non-bonded energy (Coul-SR + LJ-SR, kJ/mol) between the ion (`Ln`), ligating residues (`r3` … `r14`) and water (`SOL`) |
| `Ln-r3.1` … `r14-SOL.1` | standard deviation of the same energies over the trajectory (see note) |
| `BOSPCA1_ESM2` … `BOSPCA15_ESM2` | 15 principal components of the ESM-2 first-token embedding (`05_esm2_features`) |
| `del_Gsolv_Qiao`, `∆…` columns, `Penetration Depth`, `Tilt Angle`, `ngle Subtended…`, `mask1`, `weights`, `seq_code` | carried from the source table; not used by any analysis here |

Note on the energy SD columns (`*.1`): in at least 27 rows (all of LBT3-COO⁻, the LBT-n peptides, and a few others) these were computed over a 30,000-row buffer that held 1,001 frames and zeros elsewhere, which inflates them to about 0.18 |mean| (≈ 0.02–0.05 |mean| elsewhere). The table is kept as used in the paper; `03_unbiased_md_analysis/descriptors.py` computes the SD over the actual frames.

## `library2/` — Library 2 (37,041 Imperiali sequences, Tb³⁺)

`library2_train.csv` (29,633) and `library2_test.csv` (7,408): the fixed 80/20 split used in the paper.
Columns: `LBT` (library ID), `Sequence`, `Kd_nM`, `G_Kcal_mol` (ΔG from K_D), the DBAASP descriptors, `label` (affinity class 0–3 from the ΔG cutoffs −7.155, −7.8765, −9.0745 kcal/mol; 3 = strongest) and `binary_label` (not used). The classifiers use the nine descriptors listed in `07_tb_affinity_classifier/models.py`; `Penetration Depth`, `Tilt Angle` and `Angle Subtended…` are not used.

## `external/` — peptides outside the training libraries

| File | Content |
|---|---|
| `external_peptides.csv` | names and sequences of the reverse EF-hand peptides (EF1R … EF4Rmod; ref. 20) and LBT3-N5D (ref. 26) |
| `efhand_dbaasp_descriptors.csv` | DBAASP descriptors of the EF-hand peptides (classifier input) |
| `efhand_regression_features.csv`, `lbt3_n5d_regression_features.csv` | regression inputs per peptide–ion pair (ion descriptors, DBAASP descriptors, solubility score, ESM-2 components; other ESM columns are not used) |
| `efhand_competition_boxplots/` | medians and quartiles of the published competition ratios, digitised from the box plots of ref. 20 |
| `efhand_experimental_Tb_dG.csv`, `efhand_experimental_ddG.csv` | experimental Tb³⁺ ΔG and ΔΔG(Ln/Tb) with uncertainties (output of `06_efhand_experimental_values`) |
| `lbt3_n5d_experimental_ddG.csv` | experimental ΔΔG(Ln/Tb) of LBT3-N5D |

## `md_derived/` — ion–oxygen distances

- `unbiased_LBT1-CONH2_distances.csv.gz`: unbiased LBT1-CONH2 runs with 12 lanthanides; distances from the ion to the ten ligating oxygens every 100 ps, and CN (Eq. 6) (Fig. S1).
- `metadynamics_Tb_Lu_distances.csv.gz`: the same distances plus the nearest water oxygen in the Tb³⁺ and Lu³⁺ metadynamics walkers every 100 ps; frames are not reweighted (Fig. S2).

## `reference/shannon_radii_cn9.tsv`

Shannon effective ionic radii of nine-coordinate Ln³⁺ (Å), used for all radius axes and slope fits.
