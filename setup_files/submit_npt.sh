#!/bin/bash -l
# Standard output and error:
#SBATCH -o slurm.%j.out
#SBATCH -e slurm.%j.err
# Job Name:
#SBATCH -J JOBNAME
# Queue (Partition):
#SBATCH --partition=gpu

# Number of nodes and MPI tasks per node:
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=10
#SBATCH --threads-per-core=1

# Request say 32 GB of main Memory per node in Units of MB:
#SBATCH --mem=32000

# GPU stuff
#SBATCH --constraint="gpu"
#SBATCH --gres=gpu:1

# Wall clock limit:
#SBATCH --time=24:00:00

# Run the program:
# For pinning threads correctly:
export GMX_GPU_PME_DECOMPOSITION=1

module load gcc/8 cuda/11.6;
export GMX=/u/epoppleton/software/gromacs-chain-coordinate/build/bin/gmx_mpi;
source ~/software/gromacs-chain-coordinate/build/scripts/GMXRC.bash;
read X < <(cat ../submit_pull.sh | grep "FREEZE_CYLINDER_X=" | cut -d "=" -f 2);
read Y < <(cat ../submit_pull.sh | grep "FREEZE_CYLINDER_Y=" | cut -d "=" -f 2);
export SLICE_COORD_N=27;
export SLICE_COORD_D=0.1;
export SLICE_COORD_CYL_R=0.9;
export SLICE_COORD_FREEZE_CYLINDER_XIREF=0.8
export SLICE_COORD_FREEZE_CYLINDER_X=$X;
export SLICE_COORD_FREEZE_CYLINDER_Y=$Y;

srun $n_tasks $GMX mdrun -s npt.tpr -deffnm npt -maxh 23.76 -pin on -bonded cpu -nb gpu -pme gpu -ntomp 10 
