#!/usr/bin/env bash
# 单个 Slurm task 启动一个 P 或 D；由 start_stable_kv_pair.sh 分配两张卡。
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$(pwd)}"
cd "${REPO_ROOT}"
PYTHON_BIN="${PYTHON_BIN:-${REPO_ROOT}/.venv/bin/python}"
# Python 子进程还会调用 ninja 等虚拟环境工具，绝对解释器路径不能替代 PATH。
export PATH="$(dirname "${PYTHON_BIN}"):${PATH}"
MODEL_PATH="${MODEL_PATH:-${HOME}/Models/Qwen2.5-3B}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-qwen2.5-3b}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.8}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-4096}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-64}"
PROMPT_STREAM_MASKED_PADDING="${PROMPT_STREAM_MASKED_PADDING:-1}"
ATTENTION_BACKEND="${ATTENTION_BACKEND:-CHUNK_SPARSE_ATTN}"
ROLE="${1:-${SLURM_PROCID:-}}"

if [[ "${PROMPT_STREAM_MASKED_PADDING}" != "1" ]] ||
  [[ "${ATTENTION_BACKEND}" != "CHUNK_SPARSE_ATTN" && "${ATTENTION_BACKEND}" != "TRITON_ATTN" ]]; then
  echo "Stable KV experiment requires masked padding and CHUNK_SPARSE_ATTN or TRITON_ATTN." >&2
  exit 2
fi
if [[ -z "${SLURM_JOB_ID:-}" ]]; then
  echo "Run under srun so Slurm assigns one GPU to each server task." >&2
  exit 2
fi
case "${ROLE}" in
  0|p)
    KV_ROLE=kv_producer
    OPENAI_PORT="${P_PORT:-8100}"
    SIDE_PORT="${P_SIDE_PORT:-5600}"
    ;;
  1|d)
    KV_ROLE=kv_consumer
    OPENAI_PORT="${D_PORT:-8200}"
    SIDE_PORT="${D_SIDE_PORT:-5601}"
    ;;
  *)
    echo "Role must be p/d or SLURM_PROCID=0/1, got ${ROLE}." >&2
    exit 2
    ;;
esac
# 两组对照均打开 connector 能力，只有逐请求 early_stable_kv_push 不同。
KV_CONFIG="{\"kv_connector\":\"NixlPushConnector\",\"kv_role\":\"${KV_ROLE}\",\"kv_load_failure_policy\":\"fail\",\"kv_connector_extra_config\":{\"stable_kv_push\":true}}"
export VLLM_NIXL_SIDE_CHANNEL_HOST="${VLLM_NIXL_SIDE_CHANNEL_HOST:-$(hostname)}"
export VLLM_NIXL_SIDE_CHANNEL_PORT="${SIDE_PORT}"
export UCX_NET_DEVICES="${UCX_NET_DEVICES:-all}"
export PYTHONUNBUFFERED=1
echo "HOSTNAME=$(hostname) SLURM_JOB_ID=${SLURM_JOB_ID} ROLE=${KV_ROLE}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-} PORT=${OPENAI_PORT} SIDE_PORT=${SIDE_PORT}"
echo "MODEL_PATH=${MODEL_PATH} GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION}"
echo "KV_TRANSFER_CONFIG=${KV_CONFIG}"

# 使用 CUDA 逻辑设备检查 Slurm 实际分到的 GPU，避免误查主机上的另一张卡。
# 这个检查进程退出后再启动服务，不保留额外的 CUDA context。
"${PYTHON_BIN}" - "${GPU_MEMORY_UTILIZATION}" <<'PY'
import sys

import torch

if torch.cuda.device_count() != 1:
    raise SystemExit("Each stable KV server task must see exactly one Slurm GPU.")
free_bytes, total_bytes = torch.cuda.mem_get_info(0)
fraction = float(sys.argv[1])
print(f"GPU={torch.cuda.get_device_name(0)} free={free_bytes / 2**30:.2f} GiB "
      f"total={total_bytes / 2**30:.2f} GiB requested={fraction:.2%}", flush=True)
if not 0 < fraction <= 1 or free_bytes < total_bytes * fraction:
    raise SystemExit("Assigned GPU does not have the requested free memory fraction.")
PY

exec "${PYTHON_BIN}" -m vllm.entrypoints.openai.api_server \
  --model "${MODEL_PATH}" \
  --served-model-name "${SERVED_MODEL_NAME}" \
  --host 0.0.0.0 \
  --port "${OPENAI_PORT}" \
  --dtype auto \
  --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}" \
  --max-model-len "${MAX_MODEL_LEN}" \
  --max-num-seqs "${MAX_NUM_SEQS}" \
  --tensor-parallel-size 1 \
  --attention-backend "${ATTENTION_BACKEND}" \
  --generation-config vllm \
  --prompt-stream-masked-padding \
  --enforce-eager \
  --no-async-scheduling \
  --kv-transfer-config "${KV_CONFIG}"
