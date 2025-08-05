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
rm -rf logs_slurm checkpoints* mllogs* ray_checkpoints
mkdir logs_slurm
rm -rf logs_torchrun

export HYDRA_FULL_ERROR=1

# DDP itwinai
DIST_MODE="ddp"
RUN_NAME="hython-juwels-runall-ddp"
TRAINING_CMD="itwinai exec-pipeline --config-path configuration_files --config-name juwels_training strategy=ddp run_name=hython-juwels-runall-ddp"
sbatch --export=ALL,DIST_MODE="$DIST_MODE",RUN_NAME="$RUN_NAME",TRAINING_CMD="$TRAINING_CMD",PYTHON_VENV="$PYTHON_VENV" \
    --job-name="$RUN_NAME-n$N" \
    --output="logs_slurm/job-$RUN_NAME-n$N.out" \
    --error="logs_slurm/job-$RUN_NAME-n$N.err" \
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
