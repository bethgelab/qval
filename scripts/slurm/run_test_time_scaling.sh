#!/bin/bash
#SBATCH --job-name=vb-tts
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --nodes=1
#SBATCH --partition=<partition_name>
#SBATCH --time=00-01:00
#SBATCH --gres=gpu:1
#SBATCH --mem=512G
#SBATCH --output=.logs/%u-%x-%j.out
#SBATCH --error=.logs/%u-%x-%j.err
#SBATCH --requeue

CONFIG_FILE=${1:?"ERROR: You must provide a config file path. Job aborted."}
OUTPUT_DIR=${2:-"data/test_time_scaling"}

source scripts/slurm/setup_base.sh
source scripts/slurm/setup_gpu.sh

echo "Running test-time scaling with config: ${CONFIG_FILE}"

python scripts/pipeline/test_time_scaling.py \
    --config "${CONFIG_FILE}" \
    --output-dir "${OUTPUT_DIR}" \
    --tensor-parallel-size "${NUM_GPUS}"
