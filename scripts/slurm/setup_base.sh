#!/bin/bash
# Shared base setup for all SLURM jobs

set -xeuo pipefail

# Display job info
scontrol show job "$SLURM_JOB_ID"

# Load environment
source .venv/bin/activate

# Directories and paths
# EXP_NAME: stable across resubmissions of the same experiment (checkpoints, validation)
# RUN_NAME: unique per SLURM submission (logs, wandb)
export EXP_NAME="${SLURM_JOB_NAME}"
export RUN_NAME="${SLURM_JOB_ID}"
export PROJECT_NAME=qval
# export WANDB_ENTITY=qval
export WANDB_PROJECT="${PROJECT_NAME}"
export WANDB__SERVICE_WAIT=300
export WANDB_OFFICIAL=1
export HYDRA_FULL_ERROR=1
export HDFS_DATA_PATH=datasets/
export HDFS_CHECKPOINT_PATH=.checkpoints/
export HDFS_LOG_PATH=.logs/

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE_PATH="${HDFS_LOG_PATH}/${EXP_NAME}_${RUN_NAME}_${TIMESTAMP}.log"

# Create directories if they don't exist
mkdir -p $HDFS_CHECKPOINT_PATH
mkdir -p $HDFS_LOG_PATH

echo "=========================================="
echo "Running experiment: ${EXP_NAME}"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Project: $PROJECT_NAME"
echo "Run name: $RUN_NAME"
echo "Node: $SLURM_NODELIST"
echo "Logs: $LOG_FILE_PATH"
echo "Detected n. of nodes: $SLURM_NNODES"
echo "Detected n. of CPUs: $SLURM_CPUS_ON_NODE"
echo "=========================================="
