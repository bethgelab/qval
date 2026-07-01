#!/bin/bash
# Shared GPU setup for all SLURM jobs

nvidia-smi

# vLLM 0.10.2 cache directory configuration (required for training environment)
# Format: /scratch_local/<user_id>-<slurm_job_id>
# This prevents permission errors with vLLM 0.10.2
export TMPDIR="/scratch_local/${USER}-${SLURM_JOB_ID}/tmp"
export TORCH_COMPILE_CACHE_DIR="${TMPDIR}/torch_compile_cache"
export XDG_CACHE_HOME="/scratch_local/${USER}-${SLURM_JOB_ID}/.cache"
source scripts/slurm/setup_hf_cache.sh

# Create cache directories
mkdir -p "$TMPDIR"
mkdir -p "$TORCH_COMPILE_CACHE_DIR"
mkdir -p "$XDG_CACHE_HOME"

export NCCL_DEBUG=DEBUG

# ---- Prefer a readable user-owned CUDA toolkit for JIT compilation ----
# Some Ferranti nodes expose /usr/local/cuda in a way that is either not
# readable by jobs or points at an nvcc variant incompatible with the
# FlashInfer/vLLM kernels we JIT on first model load. The venv install scripts
# already assume a local CUDA 12.8.1 toolkit, so make runtime jobs prefer it.
CUDA_TOOLKIT_ROOT="$HOME/cuda-12.8.1"
if [ ! -x "${CUDA_TOOLKIT_ROOT}/bin/nvcc" ] && [ -n "${CUDA_HOME:-}" ]; then
    CUDA_TOOLKIT_ROOT="${CUDA_HOME}"
fi
if [ -x "${CUDA_TOOLKIT_ROOT}/bin/nvcc" ]; then
    export CUDA_HOME="${CUDA_TOOLKIT_ROOT}"
    export CUDA_PATH="${CUDA_TOOLKIT_ROOT}"
    export CUDACXX="${CUDA_TOOLKIT_ROOT}/bin/nvcc"
    export FLASHINFER_NVCC="${CUDA_TOOLKIT_ROOT}/bin/nvcc"
    export PATH="${CUDA_TOOLKIT_ROOT}/bin:${PATH}"
    export LD_LIBRARY_PATH="${CUDA_TOOLKIT_ROOT}/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
    if [ -d "${CUDA_TOOLKIT_ROOT}/include" ]; then
        export CPATH="${CUDA_TOOLKIT_ROOT}/include${CPATH:+:${CPATH}}"
    fi
else
    echo "WARNING: Expected CUDA toolkit not found at ${CUDA_TOOLKIT_ROOT}; falling back to existing CUDA environment." >&2
fi
if [ -x "${CUDA_TOOLKIT_ROOT}/bin/ptxas" ]; then
    export TRITON_PTXAS_PATH="${CUDA_TOOLKIT_ROOT}/bin/ptxas"
else
    echo "WARNING: ptxas not found at ${CUDA_TOOLKIT_ROOT}/bin/ptxas; Triton JIT may fail." >&2
fi
export FLASHINFER_WORKSPACE_BASE="/scratch_local/${USER}-${SLURM_JOB_ID}"
# ----------------------------------------------------------------------

echo "=========================================="
echo "GPU config: ${EXP_NAME}"
echo "=========================================="
echo "GPUs: $CUDA_VISIBLE_DEVICES"
echo "Torch compile cache: $TORCH_COMPILE_CACHE_DIR"
echo "Cache dir: $XDG_CACHE_HOME"
echo "HF home: $HF_HOME"
echo "HF hub cache: $HUGGINGFACE_HUB_CACHE"
echo "Qwen3.5 122B local model dir: $QVAL_QWEN35_122B_MODEL_DIR"
if [ -n "${HF_TOKEN:-}" ]; then
    echo "HF authentication: token available from environment"
elif [ -f "${HF_HOME}/token" ]; then
    echo "HF authentication: token file available under HF_HOME"
else
    echo "WARNING: No HF_TOKEN/HUGGING_FACE_HUB_TOKEN or ${HF_HOME}/token found; Hugging Face may rate-limit anonymous requests." >&2
fi
if [[ "${EXP_NAME:-}" == *qw35-122* ]] && [ ! -e "${QVAL_QWEN35_122B_MODEL_DIR}/config.json" ]; then
    echo "ERROR: Qwen3.5 122B local snapshot is missing at ${QVAL_QWEN35_122B_MODEL_DIR}." >&2
    echo "Submit scripts/slurm/download_qwen35_122b_a10b.sh before running 122B prediction jobs." >&2
    exit 1
fi
echo "Temp dir: $TMPDIR"
export NUM_GPUS="${SLURM_GPUS_ON_NODE:-1}"
echo "Detected n. of GPUs: $NUM_GPUS"
echo "=========================================="

# ---- Force NVIDIA stack; avoid ROCm conflicts ----
# VERL/Ray will set CUDA_VISIBLE_DEVICES per-actor; ROCR_* must be unset.
unset ROCR_VISIBLE_DEVICES
unset HIP_VISIBLE_DEVICES

# Make it very obvious in logs
echo "GPU env after cleanup:"
env | grep -E 'CUDA_VISIBLE_DEVICES|HIP_VISIBLE_DEVICES|ROCR_VISIBLE_DEVICES' || true
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# --------------------------------------------------

# ---- flashinfer libcudart preload ----
# The system CUDA at /usr/local/cuda is mode 711 (exec-only, not readable),
# so flashinfer's ctypes.CDLL preload of libcudart.so.12 fails. Point it at
# the readable pip-shipped copy from the nvidia-cuda-runtime-cu12 wheel.
NV_CUDART_DIR="$(python -c 'import nvidia.cuda_runtime, os; print(os.path.dirname(nvidia.cuda_runtime.__file__))')"
export CUDA_LIB_PATH="$NV_CUDART_DIR/lib"
echo "CUDA_LIB_PATH: $CUDA_LIB_PATH"
echo "CUDA_HOME: ${CUDA_HOME:-unset}"
echo "CUDACXX: ${CUDACXX:-unset}"
echo "TRITON_PTXAS_PATH: ${TRITON_PTXAS_PATH:-unset}"
echo "FLASHINFER_NVCC: ${FLASHINFER_NVCC:-unset}"
echo "FLASHINFER_WORKSPACE_BASE: ${FLASHINFER_WORKSPACE_BASE:-unset}"
command -v nvcc || true
nvcc --version || true
if [ -n "${TRITON_PTXAS_PATH:-}" ]; then
    "${TRITON_PTXAS_PATH}" --version || true
fi
if [ -n "${FLASHINFER_NVCC:-}" ]; then
    "${FLASHINFER_NVCC}" --version || true
fi
if [ -n "${CUDA_HOME:-}" ] && [ -r "${CUDA_HOME}/include/cuda_runtime.h" ]; then
    echo "CUDA headers: ${CUDA_HOME}/include/cuda_runtime.h"
else
    echo "WARNING: cuda_runtime.h is not readable from CUDA_HOME=${CUDA_HOME:-unset}" >&2
fi
# --------------------------------------

# ---- vLLM multiprocessing start method ----
# predict.py imports torch (via llenvs/qval) before vLLM spawns its
# EngineCore subprocess. Default fork then hits "Cannot re-initialize CUDA
# in forked subprocess". Force spawn so the child gets a clean CUDA state.
export VLLM_WORKER_MULTIPROC_METHOD=spawn
# ------------------------------------------

# ---- Disable flashinfer sampler (avoids one runtime JIT path) ----
# The top-k/top-p sampler can fall back to vLLM's native implementation, which
# avoids extra FlashInfer JIT work. Other FlashInfer kernels may still compile,
# so FLASHINFER_NVCC and FLASHINFER_WORKSPACE_BASE above force a readable CUDA
# toolkit and a per-job cache instead of stale home-cache ninja files.
export VLLM_USE_FLASHINFER_SAMPLER=0
# ----------------------------------------------------------------
