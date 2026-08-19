#!/bin/bash
#SBATCH -p small
#SBATCH -t 1-00:00:00
#SBATCH -n 1
#SBATCH -c 10
#SBATCH --mem 32000
#SBATCH -J WHAM
#SBATCH -e wham.%j.err
#SBATCH -o wham.%j.out

module load gcc/8 cuda/11.6
export GMX=/u/epoppleton/software/gromacs-chain-coordinate/build/bin/gmx_mpi
source ~/software/gromacs-chain-coordinate/build/scripts/GMXRC.bash

export OMP_NUM_THREADS=10

$GMX wham -it tpr_files.dat -ix pullx_files.dat -o profile.xvg -hist hist.xvg -unit kJ -ac -oiact iac.xvg -bs-method traj -bsres bs.xvg
