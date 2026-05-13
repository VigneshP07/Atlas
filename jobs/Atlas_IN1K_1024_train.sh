#!/bin/bash
#SBATCH -J Atlas_IN1K_train_1024
#SBATCH --output /raid/abircs/Vignesh_and_Indumouli_ATLAS/Codebase/Atlas/Output_Logs/atlas_in1k_1024_train.out
#SBATCH --error /raid/abircs/Vignesh_and_Indumouli_ATLAS/Codebase/Atlas/Output_Logs/atlas_in1k_1024_train.err
#SBATCH --partition=dgx2
#SBATCH --qos=gpu2
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=10
#SBATCH --gres=gpu:4
#SBATCH --time=2-00:00:00


source /scratch/apps/miniconda3/etc/profile.d/conda.sh

cd /raid/abircs/Vignesh_and_Indumouli_ATLAS/Codebase/Atlas

conda activate /raid/abircs/Vignesh_and_Indumouli_ATLAS/env/atlas_env

if [ "$(hostname)" == "dgxmaster.iitkgp.ac.in" ]; then
  # Path as seen from the master node
  export MY_LARGE_STORAGE="/dgx002/abircs"
else
  # Path as seen from any compute node
  export MY_LARGE_STORAGE="/raid/abircs"
fi

# Set all cache directories based on the correct storage path
export CACHE_DIR="$MY_LARGE_STORAGE/cache"
export HF_HOME="$CACHE_DIR/huggingface"
export TORCH_HOME="$CACHE_DIR/torch"
export PIP_CACHE_DIR="$CACHE_DIR/pip"

export PYTHONPATH=/home/abircs:$PYTHONPATH
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH



CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 --master_port=28428 main.py configs/atlas_1024_in1k.yaml
