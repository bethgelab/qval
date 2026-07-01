#!/bin/bash
#SBATCH --job-name=build-imgs-tb
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --nodes=1
#SBATCH --partition=<partition_name>
#SBATCH --time=0-04:00
#SBATCH --mem=512G
#SBATCH --output=.logs/%u-%x-%j.out
#SBATCH --error=.logs/%u-%x-%j.err

set -euo pipefail

source .venv/bin/activate
export MAKEFLAGS="-j16"
: "${TMPDIR:?TMPDIR must be set to node-local scratch}"

# Point all tmp-ish locations at node-local scratch ($TMPDIR is set by SLURM
# to /scratch_local/<user>-<jobid>/ on this cluster, ~1TB). Default /tmp is
# a small tmpfs and blows up under --jobs 16 parallel sandboxes.
export APPTAINER_TMPDIR="$TMPDIR"
export SINGULARITY_TMPDIR="$APPTAINER_TMPDIR"
export APPTAINER_CACHEDIR="$TMPDIR/apptainer-cache"
export SINGULARITY_CACHEDIR="$APPTAINER_CACHEDIR"
mkdir -p "$APPTAINER_CACHEDIR"

python scripts/utils/stage_sif_images.py \
    --dataset-path ~/dev/OpenThoughts-TBLite \
    --sif-cache-dir .sif_cache_lite \
    --docker-archive-dir docker_archives/ \
    --apptainer-command singularity \
    --preinstall-config shared/configs/image_preinstalls/terminal_bench.yaml \
    --fakeroot \
    --jobs 16 \
    --force

SIF=$(ls .sif_cache_lite/*.sif | head -1)
singularity instance start --fakeroot --cleanenv --contain --no-home "$SIF" test-inst
singularity exec --cleanenv instance://test-inst bash -lc "echo hello"
singularity instance stop test-inst
