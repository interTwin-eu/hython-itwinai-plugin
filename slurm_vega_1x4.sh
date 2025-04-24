#!/bin/bash

# Job configuration
#SBATCH --job-name=ddp-1x4
#SBATCH --account=s24r05-03-users 
#SBATCH --partition=gpu
#SBATCH --time=00:30:00

#SBATCH --output=slurm_job_logs/ddp-1x4.out
#SBATCH --error=slurm_job_logs/ddp-1x4.err

# Resource allocation
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-gpu=8
#SBATCH --exclusive

# Pre-execution command
ml GCCcore/13.3.0 \
    CUDA/12.6.0 \
    UCX-CUDA/1.16.0-GCCcore-13.3.0-CUDA-12.6.0 \
    NCCL/2.22.3-GCCcore-13.3.0-CUDA-12.6.0 \
    Python/3.11.5-GCCcore-13.2.0 \
    CMake/3.24.3-GCCcore-11.3.0 \
    OpenMPI \
    mpi4py \
    OpenSSL/3
source .venv/bin/activate
export OMP_NUM_THREADS=4

# Job execution command
srun --cpu-bind=none --ntasks-per-node=1 \
bash -c "torchrun \
--log_dir='logs_torchrun' \
--nnodes=$SLURM_NNODES \
--nproc_per_node=$SLURM_GPUS_PER_NODE \
--rdzv_id=$SLURM_JOB_ID \
--rdzv_conf=is_host=\$(((SLURM_NODEID)) && echo 0 || echo 1) \
--rdzv_backend=c10d \
--rdzv_endpoint='$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)'i:29500 \
$(which itwinai) exec-pipeline \
strategy=ddp \
checkpoints_location=checkpoints_ddp"
