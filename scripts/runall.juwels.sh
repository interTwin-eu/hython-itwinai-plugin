#!/bin/bash

# --------------------------------------------------------------------------------------
# Part of the interTwin Project: https://www.intertwin.eu/
#
# Created by: Matteo Bunino
#
# Credit:
# - Matteo Bunino <matteo.bunino@cern.ch> - CERN
# - Linus Eickhoff <linus.maximilian.eickhoff@cern.ch> - CERN
# --------------------------------------------------------------------------------------

# Python virtual environment (no conda/micromamba)
PYTHON_VENV=".venv"

# Clear SLURM logs (*.out and *.err files)
read -p "Delete all existing scalability metrics and logs y/n?: " answer
if [[ "$answer" =~ ^[Yy]$ ]]; then
    rm -rf scalability-metrics logs_* checkpoints_* plots mllogs outputs ray_checkpoints
fi
mkdir -p logs_slurm

export HYDRA_FULL_ERROR=1

# DDP itwinai
DIST_MODE="ddp"
RUN_NAME="hython-juwels-runall-ddp"
NNODES=2
NGPUS_PER_NODE=4
TOT_GPUS=$(($NNODES * $NGPUS_PER_NODE))

TRAINING_CMD="itwinai exec-pipeline --config-path configuration_files --config-name juwels_training strategy=ddp run_name=hython-juwels-runall-ddp num_workers_per_trial=$TOT_GPUS"
sbatch --export=ALL,DIST_MODE="$DIST_MODE",RUN_NAME="$RUN_NAME",TRAINING_CMD="$TRAINING_CMD",PYTHON_VENV="$PYTHON_VENV" \
    --job-name="$RUN_NAME-n$N" \
    --output="logs_slurm/job-$RUN_NAME-n$N.out" \
    --error="logs_slurm/job-$RUN_NAME-n$N.err" \
    --nodes=$NNODES --gpus-per-node=$NGPUS_PER_NODE \
    ./scripts/slurm.juwels.sh

# # DeepSpeed itwinai
# DIST_MODE="deepspeed"
# RUN_NAME="hython-juwels-runall-deepspeed"
# TRAINING_CMD="itwinai exec-pipeline --config-path configuration_files --config-name juwels_training strategy=deepspeed run_name=hython-juwels-runall-deepspeed"
# sbatch --export=ALL,DIST_MODE="$DIST_MODE",RUN_NAME="$RUN_NAME",TRAINING_CMD="$TRAINING_CMD",PYTHON_VENV="$PYTHON_VENV" \
#     --job-name="$RUN_NAME-n$N" \
#     --output="logs_slurm/job-$RUN_NAME-n$N.out" \
#     --error="logs_slurm/job-$RUN_NAME-n$N.err" \
#     ./scripts/slurm.juwels.sh
#
# # Horovod itwinai
# DIST_MODE="horovod"
# RUN_NAME="hython-juwels-runall-horovod"
# TRAINING_CMD="itwinai exec-pipeline --config-path configuration_files --config-name juwels_training strategy=horovod run_name=hython-juwels-runall-horovod"
# sbatch --export=ALL,DIST_MODE="$DIST_MODE",RUN_NAME="$RUN_NAME",TRAINING_CMD="$TRAINING_CMD",PYTHON_VENV="$PYTHON_VENV" \
#     --job-name="$RUN_NAME-n$N" \
#     --output="logs_slurm/job-$RUN_NAME-n$N.out" \
#     --error="logs_slurm/job-$RUN_NAME-n$N.err" \
#     ./scripts/slurm.juwels.sh
#
