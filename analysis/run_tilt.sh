#!/bin/zsh
# Drive the whole tilt-modulus pipeline over every force field x lipid x
# replicate found in traj/ (traj/{ff}_{lipid}.gro + traj/{ff}_{lipid}{rep}.xtc).
#
# Two conda envs are needed: `lipid64` (MDAnalysis) for tilt_convert.py and the
# OpenStructure python for tilt_modulus.py.  For each replicate we convert the
# gro+xtc to an OST topology+DCD, compute chi (appended to data/tilt_moduli.json,
# keyed {ff}_{lipid}_{rep}), then delete the bulky DCD.  Results accumulate, so
# the batch is resumable; pass FRESH=1 to start data/tilt_moduli.json clean.
#
#   ./analysis/run_tilt.sh                 # all systems, all reps
#   FFS="amber" LIPIDS="DOPC" ./analysis/run_tilt.sh   # a subset
set -u
cd "$(dirname "$0")/.."

OST_PY=/Users/erikpoppleton/miniconda3/envs/openstructure/bin/python
FFS=${FFS:-"charmm amber slipids"}
LIPIDS=${LIPIDS:-"DOPC DMPC DMPG"}
REPS=${REPS:-"1 2 3"}
STRIDE=${STRIDE:-1}
DENS_STRIDE=${DENS_STRIDE:-10}

[[ ${FRESH:-0} == 1 ]] && rm -f data/tilt_moduli.json

for ff in ${=FFS}; do
  for lip in ${=LIPIDS}; do
    top=traj/${ff}_${lip}.gro
    [[ -f $top ]] || { echo "skip ${ff}_${lip}: no $top"; continue; }
    for rep in ${=REPS}; do
      xtc=traj/${ff}_${lip}${rep}.xtc
      base=data/tilt_${ff}_${lip}_${rep}
      [[ -f $xtc ]] || { echo "skip ${ff}_${lip}_${rep}: no $xtc"; continue; }
      echo "==== ${ff}_${lip}_${rep} ===="
      PYTHONNOUSERSITE=1 conda run -n lipid64 python analysis/tilt_convert.py \
        "$top" "$xtc" -o "$base" || { echo "  convert FAILED"; continue; }
      $OST_PY analysis/tilt_modulus.py "$base" \
        --stride "$STRIDE" --density-stride "$DENS_STRIDE" || echo "  modulus FAILED"
      rm -f "${base}.dcd"     # bulky; keep the small topology .pdb
    done
  done
done
echo "==== done; results in data/tilt_moduli.json ===="
