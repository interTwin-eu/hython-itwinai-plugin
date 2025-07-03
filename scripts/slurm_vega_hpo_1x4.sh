#!/bin/bash

# Job configuration
#SBATCH --job-name=ddp-1x4
#SBATCH --account=s24r05-03-users 
#SBATCH --partition=gpu
#SBATCH --time=00:30:00

#SBATCH --output=slurm_job_logs/vega-ddp-1x4.out
#SBATCH --error=slurm_job_logs/vega-ddp-1x4.err

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
timestamp=$(date +%s)
export OMP_NUM_THREADS=4
export HYDRA_FULL_ERROR=1

# Job execution command
srun --cpu-bind=none --ntasks-per-node=1 \
bash -c "ray start \
--head \
--node-ip-address=localhost \
--port=7639 \
--num-cpus=32 \
--num-gpus=4 \
--dashboard-host=0.0.0.0 \
--dashboard-port=8265 \
&& \
itwinai exec-pipeline \
--config-path configuration_files \
--config-name vega_training \
num_workers_dataloader=4 \
experiment_name=ddp-1x4-${USER} \
run_id=ddp-1x4-${timestamp} \
+pipe_key=hpo"
