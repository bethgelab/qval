#!/bin/bash
#SBATCH --job-name=vb-eval
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --nodes=1
#SBATCH --partition=<partition_name>
#SBATCH --time=00-00:20
#SBATCH --mem=64G
#SBATCH --output=.logs/%u-%x-%j.out
#SBATCH --error=.logs/%u-%x-%j.err
#SBATCH --requeue

CONFIG_FILE=${1:?"ERROR: You must provide a config file path. Job aborted."}

source scripts/slurm/setup_base.sh

echo "Running evaluation with config: ${CONFIG_FILE}"

python scripts/pipeline/evaluate.py \
    --config "${CONFIG_FILE}"
