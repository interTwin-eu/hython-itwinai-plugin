#!/bin/bash

# Job configuration
#SBATCH --job-name=installation
#SBATCH --account=s24r05-03-users
#SBATCH --partition=gpu
#SBATCH --time=00:25:00

#SBATCH --output=installation.out
#SBATCH --error=installation.err

# Resource allocation
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-node=1

ml --force purge
ml CMake/3.29.3-GCCcore-13.3.0
ml mpi4py/3.1.5
ml OpenMPI/4.1.6-GCC-13.2.0
ml cuDNN/8.9.7.29-CUDA-12.3.0
ml CUDA/12.6.0
ml NCCL/2.22.3-GCCcore-13.3.0-CUDA-12.6.0
ml Python/3.12.3-GCCcore-13.3.0

python -m venv .venv
source .venv/bin/activate

uv pip install --no-cache-dir -e .
curl -fsSL https://github.com/interTwin-eu/itwinai/raw/main/env-files/torch/install-horovod-deepspeed-cuda.sh | bash
