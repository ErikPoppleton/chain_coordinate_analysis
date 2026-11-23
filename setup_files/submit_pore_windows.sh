#!/bin/bash -l
# Per-umbrella-window pore occupancy: chain-coordinate rerun of every window,
# then analysis/pore_density.py --windows all for this one system.
#
# Submit from a {ff}_{lipid}/ directory (the one holding umbrella0..23/ and
# submit_pull.sh), with REPO pointing at this repository's clone:
#
#     cd ~/Simulations/charmm_DOPC && sbatch --export=ALL,REPO=$REPO $REPO/setup_files/submit_pore_windows.sh
#
# Writes umbrella{w}/rerunerr.log (skipped where one already exists, e.g.
# umbrella23) and $REPO/data/poredens_windows/poredens{,_counts}_{ff}_{lipid}_wNN.npz.
#SBATCH -p interactive
#SBATCH -t 1:59:00
#SBATCH -n 1
#SBATCH -c 24
#SBATCH --mem 64000
#SBATCH -J POREWIN
#SBATCH -e porewin.%j.err
#SBATCH -o porewin.%j.out

set -u
: "${REPO:?set REPO to the chain_coordinate_analysis clone}"
NPROC=${SLURM_CPUS_PER_TASK:-24}

SYS=$(basename "$PWD")              # {ff}_{lipid}
SIMS=$(dirname "$PWD")
FF=${SYS%%_*}
LIPID=${SYS#*_}

module load gcc/8 cuda/11.6
export GMX=/u/epoppleton/software/gromacs-chain-coordinate/build/bin/gmx_mpi
source ~/software/gromacs-chain-coordinate/build/scripts/GMXRC.bash

# The production CV settings (same as submit_umbrella.sh), so the rerun log's
# per-frame pore centre and xi are the ones the windows were restrained on.
read X < <(grep "FREEZE_CYLINDER_X=" submit_pull.sh | cut -d "=" -f 2)
read Y < <(grep "FREEZE_CYLINDER_Y=" submit_pull.sh | cut -d "=" -f 2)
export SLICE_COORD_N=27
export SLICE_COORD_D=0.1
export SLICE_COORD_CYL_R=0.9
export SLICE_COORD_FREEZE_CYLINDER_XIREF=0.8
export SLICE_COORD_FREEZE_CYLINDER_X=$X
export SLICE_COORD_FREEZE_CYLINDER_Y=$Y
export OMP_NUM_THREADS=1

# 1. one single-threaded rerun per window, NPROC at a time.  The CV prints its
#    per-frame table (centre, xiChain) to stderr -> rerunerr.log.
for w in $(seq 0 23); do
    d=umbrella$w
    if [ -s $d/rerunerr.log ] && grep -q "Cylinder center" $d/rerunerr.log; then
        echo "$d: rerunerr.log present, skipping rerun"; continue
    fi
    for f in md.xtc md.tpr md.gro; do
        [ -e $d/$f ] || echo "WARNING $d: missing $f"
    done
    ( cd $d && $GMX mdrun -rerun md.xtc -s md.tpr -deffnm rerun -ntomp 1 \
          > rerunout.log 2> rerunerr.log && echo "$d: rerun done" ) &
    while [ "$(jobs -rp | wc -l)" -ge "$NPROC" ]; do sleep 5; done
done
wait

# 2. maps + per-frame in-cylinder counts (+ xi) for every window.
cd "$REPO"
PYTHONNOUSERSITE=1 conda run --no-capture-output -n lipid \
    python analysis/pore_density.py --sims-root "$SIMS" \
    --forcefields "$FF" --lipids "$LIPID" --windows all --nprocs "$NPROC"
