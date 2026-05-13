#!/bin/bash
#SBATCH -J Atlas_EuroSAT_FT_256
#SBATCH --output /home/others/abir_proj/codebase/Atlas/Output_Logs/atlas_eurosat_ft_256.out
#SBATCH --error /home/others/abir_proj/codebase/Atlas/Output_Logs/atlas_eurosat_ft_256.err
#SBATCH --partition=gpu_l40
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --mem=128G
#SBATCH --time=2-00:00:00

cd /home/others/abir_proj/codebase/Atlas

source /home/others/abir_proj/miniconda3/etc/profile.d/conda.sh
conda activate /home/others/abir_proj/Env/atlas_env

export PYTHONPATH=/home/others/abir_proj/codebase/Atlas:$PYTHONPATH

export MASTER_ADDR=localhost
export MASTER_PORT=29431
export WORLD_SIZE=1


python main.py configs/atlas_euro_ft_256.yaml