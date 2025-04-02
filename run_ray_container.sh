#!/bin/bash

# --------------------------------------------------------------------------------------
# Part of the interTwin Project: https://www.intertwin.eu/
#
# Created by: Linus Eickhoff
#
# Credit:
# - Matteo Bunino <matteo.bunino@cern.ch> - CERN
# - Linus Eickhoff <linus.maximilian.eickhoff@cern.ch> - CERN
# --------------------------------------------------------------------------------------

# shellcheck disable=all

# DISCLAIMER: 
# this script is here to support the development, so it may not be maintained and it may be a bit "rough".
# Do not mind it too much.

# Select on which system you want to run the tests
WHERE_TO_RUN=$1 #"jsc" # "vega"

# Global config
NUM_NODES=2
GPUS_PER_NODE=2

# HPC-wise config
if [[ $WHERE_TO_RUN == "jsc" ]]; then

    # JSC (venv)
    export CONTAINER_PATH="./hython.sif"
    # Path to shared filesystem that all the Ray workers can access. /tmp is a local filesystem path to each worker
    # This is only needed by tests. This is needed by the torch trainer to store and retrieve checkpoints
    export SHARED_FS_PATH="/p/project1/intertwin/eickhoff2/tmp"

    SLURM_SCRIPT="slurm_ray_container.sh"
    PARTITION="develbooster"

elif [[ $WHERE_TO_RUN == "vega" ]]; then

    # Vega (container)
    export CONTAINER_PATH="./hython.sif"
    # Path to shared filesystem that all the Ray workers can access. /tmp is a local filesystem path to each worker
    # This is only needed by tests. This is needed by the torch trainer to store and retrieve checkpoints
    export SHARED_FS_PATH="/ceph/hpc/data/st2301-itwin-users/tmp-mbunino2"

    SLURM_SCRIPT="slurm.vega.sh"
    PARTITION="gpu"

else
    echo "On what system are you running?"
    exit 1
fi

# Cleanup SLURM logs (*.out and *.err files) and other logs
rm -rf logs_slurm
mkdir logs_slurm
rm -rf logs_torchrun logs_mpirun logs_srun checkpoints


export DIST_MODE="ray"
export RUN_NAME="ray-eurac-train"
export COMMAND="itwinai exec-pipeline --config-path configuration_files --config-name training"
sbatch  \
    --job-name="$RUN_NAME-n$N" \
    --output="logs_slurm/job-$RUN_NAME-n$N.out" \
    --error="logs_slurm/job-$RUN_NAME-n$N.err" \
    --nodes=$NUM_NODES \
    --gpus-per-node=$GPUS_PER_NODE \
    --gres=gpu:$GPUS_PER_NODE \
    --partition=$PARTITION \
    $SLURM_SCRIPT

