#!/bin/bash
#SBATCH --job-name=atlas_train
#SBATCH --partition=gpu
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1          # One srun task per node (torchrun will spawn GPU procs)
#SBATCH --gpus-per-node=2            # GPUs per node
#SBATCH --cpus-per-task=12
#SBATCH --time=23:50:00
#SBATCH --output=/scratch/abircs/sudipta/atlas/BatchLogs/%j/out.log
#SBATCH --error=/scratch/abircs/sudipta/atlas/BatchLogs/%j/err.log
#SBATCH --chdir=/scratch/abircs/sudipta/atlas
#SBATCH --comment="A100 job"

# ---------- configuration ----------
NNODES=${SLURM_JOB_NUM_NODES}
NPROC_PER_NODE=2                     # Number of GPUs per node
MASTER_PORT=29500
CONFIG="/scratch/abircs/sudipta/atlas/configs/test.yaml"
PYTHON="/scratch/abircs/condaenvs/atlas-env-new/bin/python"
# -----------------------------------

# Create job-specific log directory
mkdir -p /scratch/abircs/sudipta/atlas/BatchLogs/$SLURM_JOB_ID

# MASTER_ADDR = first node in the list
MASTER_ADDR=$(scontrol show hostnames "$SLURM_NODELIST" | head -n 1)

echo "=============================================="
echo " SLURM Job ID       : $SLURM_JOB_ID"
echo " Nodes allocated    : $NNODES"
echo " GPUs per node      : $NPROC_PER_NODE"
echo " MASTER_ADDR        : $MASTER_ADDR"
echo " MASTER_PORT        : $MASTER_PORT"
echo " Config file        : $CONFIG"
echo " Log directory      : /scratch/abircs/sudipta/atlas/BatchLogs/$SLURM_JOB_ID"
echo "=============================================="

# Activate your conda env
source ~/.bashrc
conda activate atlas-env-new

# Launch torchrun via srun
srun --nodes=$NNODES --ntasks-per-node=1 --cpus-per-task=$SLURM_CPUS_PER_TASK \
  $PYTHON -m torch.distributed.run \
    --nnodes=$NNODES \
    --nproc_per_node=$NPROC_PER_NODE \
    --node_rank=$SLURM_NODEID \
    --master_addr=$MASTER_ADDR \
    --master_port=$MASTER_PORT \
    main.py $CONFIG
