#!/bin/bash
#SBATCH -J Atlas_CUB200_FT_512
#SBATCH --output /home/others/abir_proj/codebase/Atlas/Output_Logs/atlas_cub200_ft_512.out
#SBATCH --error /home/others/abir_proj/codebase/Atlas/Output_Logs/atlas_cub200_ft_512.err
#SBATCH --partition=gpu_l40
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=20
#SBATCH --mem=128G
#SBATCH --time=2-00:05:00

cd /home/others/abir_proj/codebase/Atlas

source /home/others/abir_proj/miniconda3/etc/profile.d/conda.sh
conda activate /home/others/abir_proj/Env/atlas_env

export PYTHONPATH=/home/others/abir_proj/codebase/Atlas:$PYTHONPATH

# Use torchrun for local (single-node) multi-GPU training
# This avoids the --local-rank injection error
torchrun \
    --nproc_per_node=2 \
    --nnodes=1 \
    --master_addr=localhost \
    --master_port=39502 \
    main.py configs/atlas_cub200_ft_512.yaml