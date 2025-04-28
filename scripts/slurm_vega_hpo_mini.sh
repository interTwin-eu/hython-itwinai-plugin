#!/bin/bash

# Job configuration
#SBATCH --job-name=ddp-minimal
#SBATCH --account=s24r05-03-users 
#SBATCH --partition=gpu
#SBATCH --time=00:25:00

#SBATCH --output=slurm_job_logs/minimal-vega-ddp.out
#SBATCH --error=slurm_job_logs/minimal-vega-ddp.err

# Resource allocation
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-gpu=6

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
export HYDRA_FULL_ERROR=1
timestamp=$(date +%s)

# Job execution command
srun --cpu-bind=none --ntasks-per-node=1 \
bash -c "ray start \
--head \
--node-ip-address=localhost \
--port=7639 \
--num-cpus=6 \
--num-gpus=1 \
--dashboard-host=0.0.0.0 \
--dashboard-port=8265 \
&& \
itwinai exec-pipeline \
--config-path configuration_files \
--config-name vega_training \
num_workers_dataloader=2 \
experiment_name=minimal-ddp-${USER} \
experiment_run=minimal-ddp-${timestamp} \
+pipe_key=hpo"
