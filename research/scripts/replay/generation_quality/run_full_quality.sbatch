#!/usr/bin/env bash
#SBATCH --job-name=llama-quality
#SBATCH --partition=gpu
#SBATCH --nodelist=g2c13
#SBATCH --nodes=1
#SBATCH --ntasks=2
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:2
#SBATCH --mem=128G
#SBATCH --output=research/scripts/replay/generation_quality/results/full-quality-%j.out

# 一份 Slurm allocation 管理 P、D、proxy、预热和正式评分，脱离 SSH 后继续执行。
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
cd "${REPO_ROOT}"
PYTHON_BIN="${PYTHON_BIN:-${REPO_ROOT}/.venv/bin/python}"
export PATH="$(dirname "${PYTHON_BIN}"):${PATH}"
export MODEL_PATH="${MODEL_PATH:-${HOME}/Models/Llama-3.1-8B-Instruct}"
export MODEL="${MODEL:-llama-3.1-8b-instruct}"
export TOKENIZER_MODEL="${MODEL_PATH}"
export DATASET="${DATASET:-triviaqa}"
export MAX_REQUESTS="${MAX_REQUESTS:-1000}"
export MAX_CONCURRENCY="${MAX_CONCURRENCY:-4}"
export QPS="${QPS:-0}"
export PROBE_KEEP_RATIO="${PROBE_KEEP_RATIO:-0.5}"
export MODE_ORDER="${MODE_ORDER:-normal segmented sparse}"
export RUN_PREFIX="${RUN_PREFIX:-${DATASET}-llama-quality}"
export FAIL_FAST=1
export OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/research/scripts/replay/generation_quality/results/llama-${DATASET}-job-${SLURM_JOB_ID}}"
P_PORT="${P_PORT:-18100}"
D_PORT="${D_PORT:-18200}"
PROXY_PORT="${PROXY_PORT:-18000}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.8}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-16384}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-1024}"
JOB_TMP="${REPO_ROOT}/tmp/quality-${SLURM_JOB_ID}"
mkdir -p "${OUTPUT_DIR}" "${JOB_TMP}"
export TMPDIR="${REPO_ROOT}/tmp"
export TRITON_CACHE_DIR="${JOB_TMP}/triton"
export TORCHINDUCTOR_CACHE_DIR="${JOB_TMP}/inductor"
export VLLM_CACHE_ROOT="${JOB_TMP}/vllm"
export XDG_CACHE_HOME="${JOB_TMP}/cache"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1
export VLLM_USE_FLASHINFER_SAMPLER=0 VLLM_USE_V2_MODEL_RUNNER=0
export NO_PROXY="127.0.0.1,localhost,$(hostname)" no_proxy="127.0.0.1,localhost,$(hostname)"
export VLLM_NIXL_SIDE_CHANNEL_HOST=127.0.0.1
export PROXY_URL="http://127.0.0.1:${PROXY_PORT}"

PROCESSES=()
cleanup() {
  trap - EXIT INT TERM
  for pid in "${PROCESSES[@]}"; do
    kill "${pid}" 2>/dev/null || true
  done
  for pid in "${PROCESSES[@]}"; do
    wait "${pid}" 2>/dev/null || true
  done
  rm -rf -- "${JOB_TMP}"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

wait_ready() {
  local url="$1" pid="$2"
  for ((attempt=0; attempt<240; attempt++)); do
    kill -0 "${pid}" 2>/dev/null || { echo "Service exited: ${url}"; return 1; }
    if curl --noproxy '*' --max-time 3 -fsS "${url}" >/dev/null 2>&1; then
      return 0
    fi
    sleep 5
  done
  echo "Service startup timed out: ${url}"
  return 1
}

cp research/scripts/replay/generation_quality/run_full_quality.sbatch "${OUTPUT_DIR}/launch.sbatch"
echo "JOB=${SLURM_JOB_ID} HOST=$(hostname) MODEL=${MODEL_PATH} OUTPUT=${OUTPUT_DIR}"
echo "GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION} MAX_MODEL_LEN=${MAX_MODEL_LEN}"
echo "MAX_REQUESTS=${MAX_REQUESTS} MAX_CONCURRENCY=${MAX_CONCURRENCY} QPS=${QPS}"
SERVER_ARGS=(
  -m vllm.entrypoints.openai.api_server
  --model "${MODEL_PATH}" --served-model-name "${MODEL}"
  --host 127.0.0.1 --dtype auto --generation-config vllm
  --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}"
  --max-model-len "${MAX_MODEL_LEN}" --max-num-seqs "${MAX_CONCURRENCY}"
  --max-num-batched-tokens "${MAX_NUM_BATCHED_TOKENS}"
  --block-size 16 --tensor-parallel-size 1
  --enforce-eager --no-async-scheduling --prompt-stream-masked-padding
)

# 每个独占 step 只分配一张卡；由 Slurm 设置 CUDA_VISIBLE_DEVICES。
srun --exclusive --exact --nodes=1 --ntasks=1 --cpus-per-task=16 \
  --mem=64G --gpus-per-task=1 --gpu-bind=single:1 \
  env VLLM_NIXL_SIDE_CHANNEL_PORT=15600 \
  "${PYTHON_BIN}" "${SERVER_ARGS[@]}" --port "${P_PORT}" \
  --attention-backend CHUNK_SPARSE_ATTN \
  --kv-transfer-config '{"kv_connector":"NixlConnector","kv_role":"kv_producer","kv_load_failure_policy":"fail"}' \
  >"${OUTPUT_DIR}/p.log" 2>&1 &
PROCESSES+=("$!")
wait_ready "http://127.0.0.1:${P_PORT}/health" "${PROCESSES[0]}"
# 依次完成模型加载，减少共享环境的并发模块扫描与权重读取。
srun --exclusive --exact --nodes=1 --ntasks=1 --cpus-per-task=16 \
  --mem=64G --gpus-per-task=1 --gpu-bind=single:1 \
  env VLLM_NIXL_SIDE_CHANNEL_PORT=15601 \
  "${PYTHON_BIN}" "${SERVER_ARGS[@]}" --port "${D_PORT}" \
  --attention-backend TRITON_ATTN \
  --kv-transfer-config '{"kv_connector":"NixlConnector","kv_role":"kv_consumer","kv_load_failure_policy":"fail"}' \
  >"${OUTPUT_DIR}/d.log" 2>&1 &
PROCESSES+=("$!")
wait_ready "http://127.0.0.1:${D_PORT}/health" "${PROCESSES[1]}"

"${PYTHON_BIN}" research/scripts/server/proxy_server.py \
  --host 127.0.0.1 --port "${PROXY_PORT}" \
  --prefiller-hosts 127.0.0.1 --prefiller-ports "${P_PORT}" \
  --decoder-hosts 127.0.0.1 --decoder-ports "${D_PORT}" \
  >"${OUTPUT_DIR}/proxy.log" 2>&1 &
PROCESSES+=("$!")
wait_ready "${PROXY_URL}/openapi.json" "${PROCESSES[2]}"

# 先确认三种模式均能完成完整 P/D 生成，再开始 1000 请求的正式实验。
echo "Starting warmup."
MAX_REQUESTS=4 RUN_PREFIX=warmup \
  bash research/scripts/replay/generation_quality/run_generation_quality.sbatch \
  >"${OUTPUT_DIR}/warmup.log" 2>&1
echo "Warmup passed; starting formal quality replay."
bash research/scripts/replay/generation_quality/run_generation_quality.sbatch \
  >"${OUTPUT_DIR}/quality.log" 2>&1
echo "Quality replay and EM/F1 evaluation completed: ${OUTPUT_DIR}"
