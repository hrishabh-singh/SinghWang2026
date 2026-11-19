#!/usr/bin/env bash
# GROMACS post-processing of one unbiased run (as produced by 01_md_setup/run_unbiased.sh).
#
#   ./gromacs_analysis.sh RUN_DIR ION       e.g.  ./gromacs_analysis.sh ../01_md_setup/runs/LBT1-CONH2_Tb Tb
#
# Writes RUN_DIR/analysis/:
#   angles.xvg        backbone N-CA-C angle of every residue (gmx angle -all), 100-200 ns
#   rama.xvg          backbone phi/psi of every residue (gmx rama), 100-200 ns
#   ion_O_<atom>.xvg  distance between the ion and each of the ten ligating oxygens, whole run, every 100 ps
#   sasa.xvg          solvent-accessible surface area of the peptide, 100-200 ns
#   energy_coul.xvg   Coul-SR between all pairs of energy groups (ion, residues 3/5/7/9/11/14, water), 100-200 ns
#   energy_lj.xvg     LJ-SR between the same pairs, 100-200 ns
# Then run:  python descriptors.py RUN_DIR/analysis   and   python coordination.py unbiased RUN_DIR/analysis
set -euo pipefail
RUN=${1:?run directory}
ION=${2:?ion symbol, e.g. Tb}
GMX=${GMX:-gmx}
B=100000; E=200000                                   # analysis window (ps)
# Ligating oxygens of LBT1-CONH2 (atom numbers of the topology): D3 OD1/OD2, N5 OD1, D7 OD1/OD2, W9 O,
# E11 OE1/OE2, E14 OE1/OE2. The ion is atom 261.
OXYGENS=(51 52 77 105 106 139 172 173 206 207)
ION_ATOM=261

cd "$RUN" && mkdir -p analysis && cd analysis

# Backbone angles: default group 4 ("Backbone": N, CA, C of every residue) read as consecutive triplets
echo q | $GMX make_ndx -f ../md2.gro -o index.ndx
echo 4 | $GMX angle -f ../md2.trr -n index.ndx -all -ov angles.xvg -b $B -e $E

# Backbone torsions
$GMX rama -s ../md2.tpr -f ../md2.trr -o rama.xvg -b $B -e $E

# Ion-oxygen distances (every frame of the trajectory, i.e. every 100 ps)
for a in "${OXYGENS[@]}"; do
    $GMX distance -s ../md2.tpr -f ../md2.trr -select "atomnr $ION_ATOM plus atomnr $a" -oall "ion_O_${a}.xvg"
done

# Peptide SASA (100 ns to the end)
echo 1 | $GMX sasa -s ../md2.tpr -f ../md2.trr -o sasa.xvg -b $B

# Pairwise energies from the two reruns: every Coul-SR and LJ-SR term between energy groups, from 100 ns on
for kind in coulomb lj; do
    terms=$(echo 0 | $GMX energy -f ../rerun_${kind}.edr 2>&1 | grep -oE '[0-9]+ +(Coul|LJ)-SR:[A-Za-z0-9]+-[A-Za-z0-9]+' | awk '{print $1}')
    printf '%s\n0\n' "$terms" | $GMX energy -f ../rerun_${kind}.edr -s ../rerun_${kind}.tpr -b $B -o energy_${kind/coulomb/coul}.xvg
done
echo "Analysis files in $RUN/analysis"
