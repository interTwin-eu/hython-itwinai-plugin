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
#SBATCH --gpus-per-node=2
#SBATCH --mem=128G
#SBATCH --cpus-per-gpu=8

# Pre-execution command
ml --force purge
ml CMake/3.29.3-GCCcore-13.3.0
ml mpi4py/3.1.5
ml OpenMPI/4.1.6-GCC-13.2.0
ml cuDNN/8.9.7.29-CUDA-12.3.0
ml CUDA/12.6.0
ml NCCL/2.22.3-GCCcore-13.3.0-CUDA-12.6.0
ml Python/3.12.3-GCCcore-13.3.0
ml OpenSSL/3

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
--num-cpus=8 \
--num-gpus=2 \
--dashboard-host=0.0.0.0 \
--dashboard-port=8265 \
&& \
itwinai exec-pipeline \
--config-path configuration_files \
--config-name vega_training \
num_workers_dataloader=4 \
experiment_name=minimal-ddp-${USER} \
run_name=minimal-ddp-${timestamp} \
+pipe_key=hpo"
