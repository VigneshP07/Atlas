#!/bin/bash

# Parameters
#SBATCH --comment='A100 job'
#SBATCH --cpus-per-task=10
#SBATCH --error=/scratch/abircs/sudipta/atlas/experiments/%j/%j_0_log.err
#SBATCH --gpus-per-node=2
#SBATCH --job-name=atlas_train
#SBATCH --mem=160GB
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=2
#SBATCH --open-mode=append
#SBATCH --output=/scratch/abircs/sudipta/atlas/experiments/%j/%j_0_log.out
#SBATCH --partition=gpu
#SBATCH --signal=USR2@120
#SBATCH --time=1439
#SBATCH --wckey=submitit

# command
export SUBMITIT_EXECUTOR=slurm
srun --unbuffered --output /scratch/abircs/sudipta/atlas/experiments/%j/%j_%t_log.out --error /scratch/abircs/sudipta/atlas/experiments/%j/%j_%t_log.err /scratch/abircs/condaenvs/atlas-env-new/bin/python -u -m submitit.core._submit /scratch/abircs/sudipta/atlas/experiments/%j
