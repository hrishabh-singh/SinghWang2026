# SinghWang2026

Simulation inputs, analysis code and machine-learning code for

> H. Singh, Y. Wang, J. Marmorstein, S. Crane, M. J. Lancaster, E. J. Petersson, C. de la Fuente-Nunez, K. J. Stebe, R. Radhakrishnan. **Selective Lanthanide-Binding Peptides: Molecular Insights into Binding Affinity and Selectivity** (2026).

The study combines unbiased MD and well-tempered multiple-walker metadynamics of the lanthanide-binding tag LBT1 (YIDTNNDGWYEGDELLA; PDB 1TJB) with descriptor analysis and machine learning on two peptide libraries: Library 1 (186 peptide–REE measurements) and Library 2 (37,041 Imperiali sequences with Tb³⁺ affinities).

## Contents

| Folder | What it does | Paper |
|---|---|---|
| `01_md_setup/` | Force field (CHARMM36 Jul2021 + Qiao lanthanide LJ parameters, `lanthanides-2.itp`), LBT1-CONH2 structures, topology, `.mdp` files and `run_unbiased.sh` (minimisation → 5 ns equilibration → 200 ns production → energy-group reruns) | Methods: MD |
| `02_metadynamics/inputs/` | PLUMED inputs as run (Tb, Lu), metadynamics `.mdp`, starting structures, topologies, `run_metadynamics.sh` (multiple walkers, shared bias) | Fig. 1, Methods |
| `02_metadynamics/analysis/` | `reweight.py`: hill reconstruction, per-walker bias history, ITRE reweighting on restart segments, free-energy surfaces and CN-state populations with standard errors | Fig. 1C–F, Eq. 7 |
| `02_metadynamics/results/` | Populations, reweighting logs and surfaces for La, Tb, Lu (outputs of `reweight.py`) | Fig. 1 |
| `03_unbiased_md_analysis/` | `gromacs_analysis.sh` (angles, torsions, ion–O distances, SASA, pairwise energies); `descriptors.py` (Library 1 MD descriptors); `coordination.py` (Eq. 6 CN, CN states, contact occupancy) | Figs. 2, S1–S3, S5 |
| `04_descriptor_correlations/` | Spearman correlation of every Library 1 descriptor with ΔG, separately for La–Gd and Tb–Lu | Fig. 2C–D |
| `05_esm2_features/` | ESM-2 (`esm2_t33_650M_UR50D`) first-token embedding → StandardScaler + PCA(15) fitted on the unique Library 1 sequences; projection of new sequences; fitted artifacts | Table S3, regression variant |
| `06_efhand_experimental_values/` | Experimental Tb³⁺ ΔG and ΔΔG(Ln/Tb) of the reverse EF-hand peptides from the published competition data (1:1 mass balance, Monte Carlo errors) | Fig. 4E, Fig. S7, Table S5 |
| `07_tb_affinity_classifier/` | Seven classifiers on nine DBAASP descriptors (grid search on stratified 5-fold OvR AUC), test metrics, SHAP, EF-hand ΔG as the class-probability-weighted mean | Fig. 4 |
| `08_ddg_regression/` | XGBoost ΔΔG(Ln/Tb) regressor (asinh target, pseudo-Huber loss, density weights), with or without ESM-2 components; predictions for EF-hand peptides and LBT3-N5D | Fig. S7, Table S5 |
| `data/` | Library 1 and 2 tables, external-peptide inputs and experimental values, derived ion–oxygen distances, ionic radii ([data/README.md](data/README.md)) | |

Every script starts with a docstring giving its usage, method and outputs; `python <script> -h` prints it.

## Installation

Python ≥ 3.12 (the paper used 3.12.7):

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

The simulations used GROMACS 2018.6 patched with PLUMED 2.5.1. The analysis scripts need only Python, apart from `gromacs_analysis.sh`, which calls `gmx`.

## Running the analyses

Run these from inside each folder. Outputs go to a `results/` subfolder, which git ignores.

```bash
# Metadynamics (needs the COLVARS.<walker> files of a run)
python 02_metadynamics/analysis/reweight.py --system Tb --run-dir <run dir> --n-walkers 4
# Unbiased MD descriptors of a new run
bash 03_unbiased_md_analysis/gromacs_analysis.sh <run dir> Tb
python 03_unbiased_md_analysis/descriptors.py <run dir>/analysis
python 03_unbiased_md_analysis/coordination.py summarize data/md_derived/unbiased_LBT1-CONH2_distances.csv.gz
# Descriptor correlations (Fig. 2C-D)
python 04_descriptor_correlations/spearman_by_regime.py
# ESM-2 components of new sequences
python 05_esm2_features/esm2_features.py project --csv data/external/external_peptides.csv --out pcs.csv
# Experimental EF-hand values
python 06_efhand_experimental_values/competition_to_dG.py
# Library 2 classifiers (drop --selected to rerun the grid search)
python 07_tb_affinity_classifier/train.py --model all --selected
python 07_tb_affinity_classifier/compare.py
python 07_tb_affinity_classifier/predict_efhand.py
# ΔΔG regression
python 08_ddg_regression/train_ddg.py --features descriptors
python 08_ddg_regression/predict_ddg.py --features descriptors
```

Expected values (this code, checked against the paper):

- Metadynamics: Tb³⁺ P₉ = 0.42 ± 0.09, Lu³⁺ P₈ = 0.56 ± 0.11, La³⁺ P₁₀ = 0.27 ± 0.07 (12, 9 and 18 segments; `02_metadynamics/results/`).
- ExtraTrees: test OvR AUC 0.9999, macro-F1 0.994, balanced accuracy 0.991. EF-hand Tb³⁺ ΔG mean absolute error 0.58 kcal/mol.
- ΔΔG regression: EF-hand pooled MSE 0.111 (descriptors) and 0.125 (with ESM-2). LBT3-N5D MSE 0.060 with R² = 0.21 (descriptors) and 0.076 (with ESM-2).

## Notes

- **Not included.** Raw trajectories, HILLS and COLVARS files are not included because of their size (tens of GB); they are available from the authors on request. `02_metadynamics/results/` holds the outputs derived from them.
- **Metadynamics inputs.** `plumed_Tb.dat` and `plumed_Lu.dat` are the archived walker inputs. The La run used the same protocol with six walkers, but its input files were not archived. CV2 is the Cα RMSD of residues 3–14: the reference PDB carries unit displacement weights on those atoms. The equilibration `.mdp` in `01_md_setup` is the one used for the unbiased runs.
- **Library 1 variant structures.** Variants were built from 1TJB with SCWRL4, and the LLA extension with Discovery Studio. Only the LBT1-CONH2 structures are included.
- **ΔΔG regression hyperparameters.** In the original training script, the grid-selected values were passed to the estimator under doubled `xgb__xgb__` names. XGBoost stored but did not apply them, so the published models used XGBoost's default tree settings with `huber_slope` = 0.5. `--hyperparameters reported` (the default) reproduces those models exactly. `--hyperparameters selected` applies the selection listed in Methods, and `--hyperparameters search` reruns the search. See `08_ddg_regression/ddg_model.py`.
- **Library 2 split.** The 80/20 split is provided as the two CSV files used in the paper.
- **Validation and leakage.** The Library 1 cross-validation splits by measurement, so the same peptide can appear in both training and validation folds. The external peptides are the stricter test.

## License

Penn non-commercial research license: see [LICENSE](LICENSE) (the same text as [LICENSE.docx](LICENSE.docx)). For commercial use, contact the Penn Center for Innovation.
