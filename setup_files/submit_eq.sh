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

module load gromacs/2026.3

srun gmx_mpi mdrun -deffnm md -cpi md.cpt -maxh 23.76 -pin on -bonded cpu -nb gpu -pme gpu -ntomp 10 
