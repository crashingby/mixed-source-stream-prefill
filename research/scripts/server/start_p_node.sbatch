#!/usr/bin/env bash
#SBATCH --job-name=vllm-p
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=research/scripts/logs/vllm-p-%j.out

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
cd "${REPO_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-${REPO_ROOT}/.venv/bin/python}"
MODEL_PATH="${MODEL_PATH:-${HOME}/Models/Qwen2.5-3B/}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-qwen2.5-3b}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8100}"
NIXL_SIDE_CHANNEL_PORT="${NIXL_SIDE_CHANNEL_PORT:-5600}"
# NIXL 会把这个地址交给远端节点主动发起握手。未显式指定时使用实际
# Slurm 运行节点的 hostname，避免跨节点 P/D 错把 localhost 发布给对端。
NIXL_SIDE_CHANNEL_HOST="${VLLM_NIXL_SIDE_CHANNEL_HOST:-$(hostname)}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.65}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-4096}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-16}"
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-1}"
if ! [[ "${TENSOR_PARALLEL_SIZE}" =~ ^[1-9][0-9]*$ ]]; then
  echo "TENSOR_PARALLEL_SIZE must be a positive integer, got: ${TENSOR_PARALLEL_SIZE}" >&2
  exit 2
fi
ATTENTION_BACKEND="${ATTENTION_BACKEND:-}"
ENABLE_KV_TRANSFER="${ENABLE_KV_TRANSFER:-1}"
if [[ "${ENABLE_KV_TRANSFER}" != "0" && "${ENABLE_KV_TRANSFER}" != "1" ]]; then
  echo "ENABLE_KV_TRANSFER must be 0 or 1, got: ${ENABLE_KV_TRANSFER}" >&2
  exit 2
fi
PROMPT_STREAM_UPDATE_COALESCING_CONFIG="${PROMPT_STREAM_UPDATE_COALESCING_CONFIG:-}"

ATTENTION_ARGS=()
if [[ -n "${ATTENTION_BACKEND}" ]]; then
  ATTENTION_ARGS=(--attention-backend "${ATTENTION_BACKEND}")
fi

COALESCING_ARGS=()
if [[ -n "${PROMPT_STREAM_UPDATE_COALESCING_CONFIG}" ]]; then
  COALESCING_ARGS=(
    --prompt-stream-update-coalescing-config
    "${PROMPT_STREAM_UPDATE_COALESCING_CONFIG}"
  )
fi

KV_TRANSFER_ARGS=()
if [[ "${ENABLE_KV_TRANSFER}" == "1" ]]; then
  KV_TRANSFER_ARGS=(
    --kv-transfer-config
    '{"kv_connector":"NixlConnector","kv_role":"kv_producer","kv_load_failure_policy":"fail"}'
  )
fi

echo "SLURM_JOB_ID=${SLURM_JOB_ID:-}"
echo "SLURM_JOB_NODELIST=${SLURM_JOB_NODELIST:-}"
echo "HOSTNAME=$(hostname)"
echo "HOST_IP=$(hostname -I || true)"
echo "ROLE=prefill-producer"
echo "OPENAI_PORT=${PORT}"
echo "NIXL_SIDE_CHANNEL_HOST=${NIXL_SIDE_CHANNEL_HOST}"
echo "NIXL_SIDE_CHANNEL_PORT=${NIXL_SIDE_CHANNEL_PORT}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-}"
echo "SLURM_JOB_GPUS=${SLURM_JOB_GPUS:-}"
echo "MODEL_PATH=${MODEL_PATH}"
echo "SERVED_MODEL_NAME=${SERVED_MODEL_NAME}"
echo "TENSOR_PARALLEL_SIZE=${TENSOR_PARALLEL_SIZE}"
echo "ATTENTION_BACKEND=${ATTENTION_BACKEND:-auto}"
echo "ENABLE_KV_TRANSFER=${ENABLE_KV_TRANSFER}"
echo "PROMPT_STREAM_UPDATE_COALESCING_CONFIG=${PROMPT_STREAM_UPDATE_COALESCING_CONFIG:-disabled}"

echo "=== nvidia-smi before vLLM startup ==="
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi
  nvidia-smi --query-gpu=index,uuid,memory.total,memory.used,memory.free \
    --format=csv
else
  echo "nvidia-smi not found"
fi

UCX_NET_DEVICES="${UCX_NET_DEVICES:-all}" \
VLLM_NIXL_SIDE_CHANNEL_HOST="${NIXL_SIDE_CHANNEL_HOST}" \
VLLM_NIXL_SIDE_CHANNEL_PORT="${NIXL_SIDE_CHANNEL_PORT}" \
"${PYTHON_BIN}" -m vllm.entrypoints.openai.api_server \
  --model "${MODEL_PATH}" \
  --served-model-name "${SERVED_MODEL_NAME}" \
  --trust-remote-code \
  --dtype auto \
  --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}" \
  --max-model-len "${MAX_MODEL_LEN}" \
  --max-num-seqs "${MAX_NUM_SEQS}" \
  --tensor-parallel-size "${TENSOR_PARALLEL_SIZE}" \
  --host "${HOST}" \
  --port "${PORT}" \
  --enforce-eager \
  --no-async-scheduling \
  "${ATTENTION_ARGS[@]}" \
  "${COALESCING_ARGS[@]}" \
  "${KV_TRANSFER_ARGS[@]}"
