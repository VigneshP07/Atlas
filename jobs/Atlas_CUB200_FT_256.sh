#!/bin/bash
#SBATCH -J Atlas_CUB200_FT_256
#SBATCH --output /home/others/abir_proj/codebase/Atlas/Output_Logs/atlas_cub200_ft_256.out
#SBATCH --error /home/others/abir_proj/codebase/Atlas/Output_Logs/atlas_cub200_ft_256.err
#SBATCH --partition=gpu_h100
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=20
#SBATCH --mem=128G
#SBATCH --time=12:00:00

cd /home/others/abir_proj/codebase/Atlas

source /home/others/abir_proj/miniconda3/etc/profile.d/conda.sh
conda activate /home/others/abir_proj/Env/atlas_env

export PYTHONPATH=/home/others/abir_proj/codebase/Atlas:$PYTHONPATH

export MASTER_ADDR=localhost
export MASTER_PORT=28430
export WORLD_SIZE=1


python main.py configs/atlas_cub200_ft_256.yaml