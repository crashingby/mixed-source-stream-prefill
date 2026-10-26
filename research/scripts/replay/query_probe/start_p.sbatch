#!/usr/bin/env bash
#SBATCH --job-name=query-probe-p
#SBATCH --partition=gpu
#SBATCH --nodelist=g2c10
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=01:00:00
#SBATCH --output=research/scripts/replay/query_probe/results/p-%j.out

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
cd "${REPO_ROOT}"
PYTHON_BIN="${PYTHON_BIN:-${REPO_ROOT}/.venv/bin/python}"
MODEL_PATH="${MODEL_PATH:-${HOME}/Models/Qwen2.5-3B}"
# 功能回放使用贪心输出，避免采样器额外触发 FlashInfer JIT 编译。
export VLLM_USE_FLASHINFER_SAMPLER="${VLLM_USE_FLASHINFER_SAMPLER:-0}"

# 仅诊断作业启用 profiler；普通性能矩阵保持关闭。
EXTRA_ARGS=()
if [[ -n "${PROFILER_CONFIG:-}" ]]; then
  EXTRA_ARGS+=(--profiler-config "${PROFILER_CONFIG}")
fi

# 首版功能验证固定为单 GPU、eager、无 KV transfer；不需要 D 节点。
exec "${PYTHON_BIN}" -m vllm.entrypoints.openai.api_server \
  --model "${MODEL_PATH}" \
  --served-model-name qwen2.5-3b \
  --host 0.0.0.0 \
  --port "${PORT:-8110}" \
  --tensor-parallel-size 1 \
  --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION:-0.8}" \
  --max-model-len "${MAX_MODEL_LEN:-16384}" \
  --max-num-seqs "${MAX_NUM_SEQS:-8}" \
  --max-num-batched-tokens "${MAX_NUM_BATCHED_TOKENS:-1024}" \
  --attention-backend "${ATTENTION_BACKEND:-CHUNK_SPARSE_ATTN}" \
  --generation-config vllm \
  --prompt-stream-masked-padding \
  --enforce-eager \
  --no-async-scheduling "${EXTRA_ARGS[@]}"
