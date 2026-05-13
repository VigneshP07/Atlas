#!/bin/bash

# Parameters
#SBATCH --cpus-per-task=10
#SBATCH --error=/scratch/abircs/sudipta/atlas/experiments/%j_0_log.err
#SBATCH --gpus-per-node=2
#SBATCH --job-name=atlas_job
#SBATCH --job-name=submitit
#SBATCH --mem=60GB
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --open-mode=append
#SBATCH --output=/scratch/abircs/sudipta/atlas/experiments/%j_0_log.out
#SBATCH --partition=gpu
#SBATCH --signal=USR2@90
#SBATCH --time=60
#SBATCH --wckey=submitit

# command
export SUBMITIT_EXECUTOR=slurm
srun --unbuffered --output /scratch/abircs/sudipta/atlas/experiments/%j_%t_log.out --error /scratch/abircs/sudipta/atlas/experiments/%j_%t_log.err /scratch/abircs/condaenvs/atlas-env-new/bin/python -u -m submitit.core._submit /scratch/abircs/sudipta/atlas/experiments
