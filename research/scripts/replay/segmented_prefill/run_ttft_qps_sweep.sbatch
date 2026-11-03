#!/usr/bin/env bash
#SBATCH --job-name=ttft-sweep
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=24:00:00
#SBATCH --output=research/scripts/logs/replay-ttft-sweep-%j.out

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
cd "${REPO_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-${REPO_ROOT}/.venv/bin/python}"
DATA_DIR="${DATA_DIR:-research/data/mix_source}"
OUTPUT_DIR="${OUTPUT_DIR:-research/scripts/replay/segmented_prefill/results}"
PROXY_URL="${PROXY_URL:-http://127.0.0.1:8000}"
MODEL="${MODEL:-qwen2.5-3b}"
TOKENIZER_MODEL="${TOKENIZER_MODEL:-${MODEL_PATH:-}}"
TOKENIZER_TRUST_REMOTE_CODE="${TOKENIZER_TRUST_REMOTE_CODE:-0}"
if [[ -z "${TOKENIZER_MODEL}" ]]; then
  echo "TOKENIZER_MODEL (or MODEL_PATH) must be set." >&2
  exit 2
fi

MAX_REQUESTS="${MAX_REQUESTS:-512}"
QPS_LIST="${QPS_LIST:-0.5 1 2 4}"
ARRIVAL_PROCESS="${ARRIVAL_PROCESS:-poisson}"
RANDOM_SEED="${RANDOM_SEED:-2026}"
MAX_CONCURRENCY="${MAX_CONCURRENCY:-32}"

APPEND_INITIAL_DELAY="${APPEND_INITIAL_DELAY:-0.0}"
UPDATE_INITIAL_DELAY="${UPDATE_INITIAL_DELAY:-0.0}"
APPEND_CHUNK_INTERVAL="${APPEND_CHUNK_INTERVAL:-0.5}"
UPDATE_CHUNK_INTERVAL="${UPDATE_CHUNK_INTERVAL:-0.01}"
SOURCE_FINISH_DELAY="${SOURCE_FINISH_DELAY:-0.0}"

MAX_TOKENS="${MAX_TOKENS:-64}"
TEMPERATURE="${TEMPERATURE:-0}"
SSE_TIMEOUT="${SSE_TIMEOUT:-120}"
HTTP_TIMEOUT="${HTTP_TIMEOUT:-600}"
RUN_PREFIX="${RUN_PREFIX:-ttft}"
FAIL_FAST="${FAIL_FAST:-0}"
EXTRA_REPLAY_ARGS="${EXTRA_REPLAY_ARGS:-}"
extra_replay_args=()
if [[ -n "${EXTRA_REPLAY_ARGS}" ]]; then
  read -r -a extra_replay_args <<< "${EXTRA_REPLAY_ARGS}"
fi

echo "SLURM_JOB_ID=${SLURM_JOB_ID:-}"
echo "SLURM_JOB_NODELIST=${SLURM_JOB_NODELIST:-}"
echo "HOSTNAME=$(hostname)"
echo "REPO_ROOT=${REPO_ROOT}"
echo "DATA_DIR=${DATA_DIR}"
echo "OUTPUT_DIR=${OUTPUT_DIR}"
echo "PROXY_URL=${PROXY_URL}"
echo "MODEL=${MODEL}"
echo "TOKENIZER_MODEL=${TOKENIZER_MODEL}"
echo "TOKENIZER_TRUST_REMOTE_CODE=${TOKENIZER_TRUST_REMOTE_CODE}"
echo "MAX_REQUESTS=${MAX_REQUESTS}"
echo "QPS_LIST=${QPS_LIST}"
echo "ARRIVAL_PROCESS=${ARRIVAL_PROCESS}"
echo "MAX_CONCURRENCY=${MAX_CONCURRENCY}"
echo "APPEND_CHUNK_INTERVAL=${APPEND_CHUNK_INTERVAL}"
echo "UPDATE_CHUNK_INTERVAL=${UPDATE_CHUNK_INTERVAL}"
echo "SOURCE_FINISH_DELAY=${SOURCE_FINISH_DELAY}"

qps_to_run_suffix() {
  local qps="$1"
  echo "${qps//./p}"
}

run_replay() {
  local mode="$1"
  local script_path="$2"
  local qps="$3"
  local qps_suffix

  qps_suffix="$(qps_to_run_suffix "${qps}")"
  run_name="${RUN_PREFIX}-${mode}-qps-${qps_suffix}"

  echo "=== replay mode=${mode} qps=${qps} run_name=${run_name} ==="

  common_args=(
    --data-dir "${DATA_DIR}"
    --max-requests "${MAX_REQUESTS}"
    --proxy-url "${PROXY_URL}"
    --model "${MODEL}"
    --tokenizer-model "${TOKENIZER_MODEL}"
    --qps "${qps}"
    --arrival-process "${ARRIVAL_PROCESS}"
    --random-seed "${RANDOM_SEED}"
    --load-mode "${LOAD_MODE:-auto}" --max-inflight "${MAX_INFLIGHT:-0}"
    --max-concurrency "${MAX_CONCURRENCY}"
    --max-tokens "${MAX_TOKENS}"
    --temperature "${TEMPERATURE}"
    --http-timeout "${HTTP_TIMEOUT}"
    --sse-timeout "${SSE_TIMEOUT}"
    --output-dir "${OUTPUT_DIR}"
    --run-name "${run_name}"
  )
  common_args+=(--input-transport "${INPUT_TRANSPORT:-text}")
  if [[ -n "${REPLAY_CONFIG:-}" ]]; then
    common_args+=(--config "${REPLAY_CONFIG}")
  fi

  if [[ "${TOKENIZER_TRUST_REMOTE_CODE}" == "1" ]]; then
    common_args+=(--tokenizer-trust-remote-code)
  fi

  if [[ "${FAIL_FAST}" == "1" ]]; then
    common_args+=(--fail-fast)
  fi

  if [[ "${mode}" == "stream" || "${mode}" == "merged-update" ]]; then
    "${PYTHON_BIN}" "${script_path}" \
      "${common_args[@]}" \
      --append-initial-delay "${APPEND_INITIAL_DELAY}" \
      --update-initial-delay "${UPDATE_INITIAL_DELAY}" \
      --append-chunk-interval "${APPEND_CHUNK_INTERVAL}" \
      --update-chunk-interval "${UPDATE_CHUNK_INTERVAL}" \
      --source-finish-delay "${SOURCE_FINISH_DELAY}" \
      --ttft-only \
      "${extra_replay_args[@]}"
    return
  fi

  # normal TTFT 也只测 P 节点的首 token，避免 D decode 干扰三种模式比较。
  common_args+=(--ttft-only)

  "${PYTHON_BIN}" "${script_path}" "${common_args[@]}" "${extra_replay_args[@]}"
}

for qps in ${QPS_LIST}; do
  run_replay "stream" \
    "research/scripts/replay/segmented_prefill/replay_prompt_stream.py" \
    "${qps}"
  run_replay "merged-update" \
    "research/scripts/replay/segmented_prefill/replay_merged_update.py" \
    "${qps}"
  run_replay "normal" \
    "research/scripts/replay/segmented_prefill/replay_normal.py" \
    "${qps}"
done
