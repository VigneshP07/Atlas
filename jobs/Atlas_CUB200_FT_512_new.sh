#!/bin/bash
#SBATCH -J Atlas_CUB200_FT_512_new
#SBATCH --output /home/others/abir_proj/codebase/Atlas/Output_Logs/atlas_cub200_ft_512_new.out
#SBATCH --error /home/others/abir_proj/codebase/Atlas/Output_Logs/atlas_cub200_ft_512_new.err
#SBATCH --partition=gpu_l40
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=20
#SBATCH --mem=128G
#SBATCH --time=9:00:00

cd /home/others/abir_proj/codebase/Atlas

source /home/others/abir_proj/miniconda3/etc/profile.d/conda.sh
conda activate /home/others/abir_proj/Env/atlas_env

export PYTHONPATH=/home/others/abir_proj/codebase/Atlas:$PYTHONPATH

export MASTER_ADDR=localhost
export MASTER_PORT=28440
export WORLD_SIZE=1


python main.py configs/atlas_cub200_ft_512_new.yaml