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

# The published profiles: default -auto range.  gmx wham then drops every sample
# equal to the histogram maximum (ibin == bins in gmx_wham.cpp), which for the
# chain coordinate is the saturated xi = 1 -- the profile therefore covers xi < 1,
# which is what we want: xi = 1 collects every fully formed pore at one point, so
# a "bin" there is a point mass whose free energy depends on the bin width.
$GMX wham -it tpr_files.dat -ix pullx_files.dat -o profile.xvg -hist hist.xvg -unit kJ -ac -oiact iac.xvg

# The published umbrella histograms (SI) come from a separate run with an explicit
# range, so the xi = 1 samples are counted (in a 1.000-1.005 bin); its profile is
# not used -- its last bin is that point mass (empty for DOPC).
# $GMX wham -it tpr_files.dat -ix pullx_files.dat -o profile_range.xvg -hist hist.xvg -unit kJ -ac -oiact iac_range.xvg -min 0 -max 1.005 -bins 201
