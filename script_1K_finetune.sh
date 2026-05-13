#!/bin/bash
#SBATCH --job-name=atlas-dist-finetune        # Job name
#SBATCH --nodes=4                   # Number of nodes
#SBATCH --ntasks-per-node=1          # A single task (the Python script) per node
#SBATCH --gres=gpu:2                 # Number of GPUs per node
#SBATCH --cpus-per-task=8           # Number of CPU cores per task
#SBATCH --partition=gpu              # Slurm partition
#SBATCH --output=/scratch/abircs/sudipta/atlas/Output_Logs/atlas-dist-finetune-%j.out  # Standard output log file
#SBATCH --error=/scratch/abircs/sudipta/atlas/Output_Logs/atlas-dist-finetune-%j.err   # Standard error log file
#SBATCH --time=48:00:00              # Time limit
#SBATCH --exclusive                  # Allocate all resources on a node to this job

# -------------------------------------
# Distributed Setup
# -------------------------------------
# Get a list of the nodes assigned to the job
nodes_array=($(scontrol show hostnames "$SLURM_JOB_NODELIST"))

# Get the hostname of the first node (our master)
head_node=${nodes_array[0]}

# Get the IP address of the master node's Infiniband interface (ib0)
head_node_ip=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname -I | awk '{print $1}')

# Set environment variables for PyTorch Distributed
export MASTER_ADDR=$head_node_ip
export MASTER_PORT=29500
export WORLD_SIZE=$(($SLURM_NNODES * 2))
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

# For debugging NCCL issues
export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=0
export NCCL_SOCKET_IFNAME=ib0
export NCCL_P2P_DISABLE=0

echo "Nodes: ${nodes_array[@]}"
echo "Master node: ${head_node} (IP: ${MASTER_ADDR})"
echo "Master port: ${MASTER_PORT}"
echo "World size: ${WORLD_SIZE}"


source /home/abircs/.bashrc
conda activate /scratch/abircs/condaenvs/atlas-env-new



# cd /scratch/abircs/atlas
cd /scratch/abircs/sudipta/atlas

# -------------------------------------
# Launch the Distributed Training Job
# -------------------------------------
# srun \
#     sh -c '
#         torchrun \
#             --nnodes=$SLURM_NNODES \
#             --nproc_per_node=2 \
#             --node_rank=$SLURM_PROCID \
#             --master_addr=$MASTER_ADDR \
#             --master_port=$MASTER_PORT \
#             train.py configs/test.yaml
#         '

srun torchrun \
    --nnodes=$SLURM_NNODES \
    --nproc-per-node=2 \
    --rdzv-id=run \
    --rdzv-backend=c10d \
    --rdzv-conf timeout=3600 \
    --rdzv-endpoint=$MASTER_ADDR:$MASTER_PORT \
    main_finetune.py /scratch/abircs/sudipta/atlas/configs/imagenet1k_finetune.yaml


# srun torchrun \
#     --nnodes=$SLURM_NNODES \
#     --nproc-per-node=2 \
#     --rdzv-id=run \
#     --rdzv-backend=c10d \
#     --rdzv-endpoint=$MASTER_ADDR:$MASTER_PORT \
#     main.py configs/test_check.yaml