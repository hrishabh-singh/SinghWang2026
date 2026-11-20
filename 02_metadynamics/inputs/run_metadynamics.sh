#!/usr/bin/env bash
# Well-tempered multiple-walker metadynamics of LBT1-CONH2 + Ln(III) with GROMACS 2018.6 patched with PLUMED 2.5.1.
#
#   ./run_metadynamics.sh ION N_WALKERS        e.g.  ./run_metadynamics.sh Tb 4   (La: 6 walkers, Lu: 4)
#
# Creates runs/<ION>/{a,b,c,...}, one directory per walker, each with plumed.dat, and starts all walkers in one
# MPI job sharing a single bias (WALKERS_MPI). Continue a run with the same command after adding RESTART to the
# plumed.dat files and appending '-cpi md2.cpt' to mdrun.
#   plumed_Tb.dat   archived input of the Tb run
#   plumed_Lu.dat   archived input of the Lu run
# The La run inputs were not archived; it was run with the same protocol, six walkers.
set -euo pipefail
ION=${1:?ion, e.g. Tb}
NW=${2:?number of walkers}
GMX=${GMX:-gmx_mpi}
MPIRUN=${MPIRUN:-mpirun}
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SETUP="$HERE/../../01_md_setup"
PLUMED_IN="$HERE/plumed_${ION}.dat"; [[ -f "$PLUMED_IN" ]] || PLUMED_IN="$HERE/plumed_Tb.dat"
START="$HERE/structures/start_LBT1-CONH2_${ION}.gro"; [[ -f "$START" ]] || START="$HERE/structures/start_LBT1-CONH2_Tb.gro"
TOP="$HERE/topology/topol_metad_${ION}.top"; [[ -f "$TOP" ]] || TOP="$HERE/topology/topol_metad_Tb.top"

RUN="$HERE/runs/$ION"; mkdir -p "$RUN" && cd "$RUN"
ln -sfn "$SETUP/forcefield/charmm36-jul2021.ff" charmm36-jul2021.ff
cp "$HERE/topology/posre.itp" "$SETUP/structures/LBT1-1TJB-ref.pdb" .
sed "s/^Tb3+ /${ION}3+ /" "$TOP" > topol.top
$GMX grompp -f "$HERE/metadynamics.mdp" -c "$START" -p topol.top -o md2.tpr -maxwarn 1

WALKERS=()
for i in $(seq 0 $((NW - 1))); do
    w=$(printf "\\$(printf '%03o' $((97 + i)))")          # a, b, c, ...
    mkdir -p "$w" && cp md2.tpr "$w/" && cp "$PLUMED_IN" "$w/plumed.dat"
    WALKERS+=("$w")
done
$MPIRUN -np "$NW" $GMX mdrun -s md2.tpr -deffnm md2 -multidir "${WALKERS[@]}" -plumed plumed.dat
