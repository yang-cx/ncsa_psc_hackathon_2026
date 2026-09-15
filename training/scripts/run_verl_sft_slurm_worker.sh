#!/usr/bin/env bash
# Launch one multi-node VERL SFT worker per Slurm node inside Podman-HPC.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
: "${SLURM_PROCID:?run this script through one-task-per-node srun}"
: "${MASTER_ADDR:?set MASTER_ADDR to the first allocated node}"

VERL_IMAGE="${VERL_IMAGE:-docker.io/verlai/verl@sha256:26b2b1de89333cfbddbf639be5f0ab9d2dbaed19ee5b4892765f59617671e915}"
HF_CACHE_HOST="${HF_CACHE_HOST:-${PSCRATCH:?}/qwen35-hf-cache}"
CONTAINER_TMP_HOST="${CONTAINER_TMP_HOST:-${PSCRATCH:?}/qwen35-container-tmp}"
NPROC_PER_NODE="${NPROC_PER_NODE:-4}"
NNODES="${NNODES:-${SLURM_NNODES:?}}"
MASTER_PORT="${MASTER_PORT:-29500}"

mkdir -p "$HF_CACHE_HOST" "$CONTAINER_TMP_HOST"

container_env=(
  -e HF_HOME=/hf_cache
  -e HF_HUB_CACHE=/hf_cache/hub
  -e HF_ASSETS_CACHE=/hf_cache/assets
  -e HF_XET_CACHE=/hf_cache/xet
  -e HF_HUB_DISABLE_XET=1
  -e UV_CACHE_DIR=/hf_cache/uv
  -e TMPDIR=/tmp
  -e "MODEL_PATH=${MODEL_PATH:?}"
  -e "VERL_DATA_DIR=${VERL_DATA_DIR:-/workspace/artifacts/native-sft/verl/main-agent-approved}"
  -e "NPROC_PER_NODE=$NPROC_PER_NODE"
  -e "NNODES=$NNODES"
  -e "NODE_RANK=$SLURM_PROCID"
  -e "MASTER_ADDR=$MASTER_ADDR"
  -e "MASTER_PORT=$MASTER_PORT"
  -e "TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-16}"
  -e "MICRO_BATCH_SIZE_PER_GPU=${MICRO_BATCH_SIZE_PER_GPU:-1}"
  -e "MAX_LENGTH=${MAX_LENGTH:-12288}"
  -e "ULYSSES_SEQUENCE_PARALLEL_SIZE=${ULYSSES_SEQUENCE_PARALLEL_SIZE:-1}"
  -e "USE_REMOVE_PADDING=${USE_REMOVE_PADDING:-false}"
  -e "LR=${LR:-1e-4}"
  -e "TOTAL_EPOCHS=${TOTAL_EPOCHS:-1}"
  -e "USE_PEFT=${USE_PEFT:-1}"
  -e "LORA_RANK=${LORA_RANK:-16}"
  -e "LORA_ALPHA=${LORA_ALPHA:-16}"
  -e "SAVE_DIR=${SAVE_DIR:?}"
  -e "PROJECT_NAME=${PROJECT_NAME:-trexfitter-native-sft}"
  -e "EXPERIMENT_NAME=${EXPERIMENT_NAME:-$(basename "$SAVE_DIR")}"
  -e "TEST_FREQ=${TEST_FREQ:-after_each_epoch}"
  -e "SAVE_FREQ=${SAVE_FREQ:-after_each_epoch}"
  -e "RESUME_MODE=${RESUME_MODE:-disable}"
  -e "HYDRA_RUN_ID=${HYDRA_RUN_ID:-${SLURM_JOB_ID}-${SLURM_STEP_ID:-batch}}"
  -e "PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
)

# Host networking is required so torch.distributed workers on other Slurm
# nodes can reach MASTER_ADDR:MASTER_PORT inside the containers.
podman-hpc run --rm --gpu --entrypoint= --ipc=host --network=host \
  -v "$REPO_ROOT":/workspace \
  -v "$HF_CACHE_HOST":/hf_cache \
  -v "$CONTAINER_TMP_HOST":/tmp \
  "${container_env[@]}" \
  -w /workspace \
  "$VERL_IMAGE" \
  /bin/bash training/scripts/run_verl_sft.sh
