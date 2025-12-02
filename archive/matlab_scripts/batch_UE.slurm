#!/bin/bash

#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=32G
#SBATCH --time=96:00:00

#SBATCH --mail-user=s.e.ivings@sheffield.ac.uk
#SBATCH --mail-type=FAIL

module load MATLAB/2023b

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export SLURM_EXPORT_ENV=ALL

dos2unix TA_UE.m

matlab -nodesktop -nosplash -r TA_UE