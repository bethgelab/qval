#!/bin/bash
#SBATCH --job-name=vb-pred
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --nodes=1
#SBATCH --partition=<partition_name>
#SBATCH --time=00-00:30
#SBATCH --gres=gpu:1
#SBATCH --mem=512G
#SBATCH --output=.logs/%u-%x-%j.out
#SBATCH --error=.logs/%u-%x-%j.err
#SBATCH --requeue

CONFIG_FILE=${1:?"ERROR: You must provide a config file path. Job aborted."}

source scripts/slurm/setup_base.sh
source scripts/slurm/setup_gpu.sh

echo "Running prediction with config: ${CONFIG_FILE}"

python scripts/pipeline/predict.py \
    --config "${CONFIG_FILE}" \
    --tensor-parallel-size "${NUM_GPUS}"
