#!/usr/bin/env bash
#SBATCH --job-name=vllm-d
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=research/scripts/logs/vllm-d-%j.out

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
cd "${REPO_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-${REPO_ROOT}/.venv/bin/python}"
MODEL_PATH="${MODEL_PATH:-${HOME}/Models/Qwen2.5-3B/}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-qwen2.5-3b}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8200}"
NIXL_SIDE_CHANNEL_PORT="${NIXL_SIDE_CHANNEL_PORT:-5601}"
# NIXL 会把这个地址交给远端节点主动发起握手。未显式指定时使用实际
# Slurm 运行节点的 hostname，避免跨节点 P/D 错把 localhost 发布给对端。
NIXL_SIDE_CHANNEL_HOST="${VLLM_NIXL_SIDE_CHANNEL_HOST:-$(hostname)}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.8}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-4096}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-64}"
ATTENTION_BACKEND="${ATTENTION_BACKEND:-CHUNK_SPARSE_ATTN}"
GENERATION_CONFIG="${GENERATION_CONFIG:-}"
PROMPT_STREAM_MASKED_PADDING="${PROMPT_STREAM_MASKED_PADDING:-0}"
if [[ "${PROMPT_STREAM_MASKED_PADDING}" != "0" && "${PROMPT_STREAM_MASKED_PADDING}" != "1" ]]; then
  echo "PROMPT_STREAM_MASKED_PADDING must be 0 or 1." >&2
  exit 2
fi
PADDING_ARGS=()
if [[ "${PROMPT_STREAM_MASKED_PADDING}" == "1" ]]; then
  # D 读取 P 的 KV 时必须延续同一 mask 与 RoPE 位置修正。
  ATTENTION_BACKEND="${ATTENTION_BACKEND:-CHUNK_SPARSE_ATTN}"
  # 与 P 保持相同的默认采样参数，避免 Qwen 模型默认的 repetition
  # penalty 把物理 padding token 误当成真实 prompt 历史。
  GENERATION_CONFIG="${GENERATION_CONFIG:-vllm}"
  PADDING_ARGS=(--prompt-stream-masked-padding)
fi

ATTENTION_ARGS=()
if [[ -n "${ATTENTION_BACKEND}" ]]; then
  ATTENTION_ARGS=(--attention-backend "${ATTENTION_BACKEND}")
fi

GENERATION_CONFIG_ARGS=()
if [[ -n "${GENERATION_CONFIG}" ]]; then
  GENERATION_CONFIG_ARGS=(--generation-config "${GENERATION_CONFIG}")
fi

echo "SLURM_JOB_ID=${SLURM_JOB_ID:-}"
echo "SLURM_JOB_NODELIST=${SLURM_JOB_NODELIST:-}"
echo "HOSTNAME=$(hostname)"
echo "HOST_IP=$(hostname -I || true)"
echo "ROLE=decode-consumer"
echo "OPENAI_PORT=${PORT}"
echo "NIXL_SIDE_CHANNEL_HOST=${NIXL_SIDE_CHANNEL_HOST}"
echo "NIXL_SIDE_CHANNEL_PORT=${NIXL_SIDE_CHANNEL_PORT}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-}"
echo "SLURM_JOB_GPUS=${SLURM_JOB_GPUS:-}"
echo "MODEL_PATH=${MODEL_PATH}"
echo "SERVED_MODEL_NAME=${SERVED_MODEL_NAME}"
echo "ATTENTION_BACKEND=${ATTENTION_BACKEND:-auto}"
echo "GENERATION_CONFIG=${GENERATION_CONFIG:-model-default}"
echo "PROMPT_STREAM_MASKED_PADDING=${PROMPT_STREAM_MASKED_PADDING}"

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
  --host "${HOST}" \
  --port "${PORT}" \
  --enforce-eager \
  --no-async-scheduling \
  "${ATTENTION_ARGS[@]}" \
  "${GENERATION_CONFIG_ARGS[@]}" \
  "${PADDING_ARGS[@]}" \
  --kv-transfer-config '{"kv_connector":"NixlConnector","kv_role":"kv_consumer","kv_load_failure_policy":"fail"}'
