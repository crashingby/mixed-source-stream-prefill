#!/usr/bin/env bash
#SBATCH --job-name=masked-padding
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=24:00:00
#SBATCH --output=research/scripts/logs/replay-masked-padding-%j.out

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
cd "${REPO_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-${REPO_ROOT}/.venv/bin/python}"
REPLAY_SCRIPT="research/scripts/replay/segmented_prefill/replay_prompt_stream.py"
DATA_DIR="${DATA_DIR:-research/data/mix_source_perf_qwen}"
OUTPUT_DIR="${OUTPUT_DIR:-research/scripts/replay/masked_padding/results}"
PROXY_URL="${PROXY_URL:-http://127.0.0.1:8000}"
MODEL="${MODEL:-qwen2.5-3b}"
TOKENIZER_MODEL="${TOKENIZER_MODEL:-${MODEL_PATH:-}}"
TOKENIZER_TRUST_REMOTE_CODE="${TOKENIZER_TRUST_REMOTE_CODE:-0}"

# 这个标签必须与当前 P/D 的启动配置一致，不能由 replay 改变服务端模式。
MASKED_PADDING="${MASKED_PADDING:?Set MASKED_PADDING to 0 or 1.}"
if [[ "${MASKED_PADDING}" == "1" ]]; then
  PROFILE="masked-padding-enabled"
  EXPECTED_MASKED_PADDING="enabled"
elif [[ "${MASKED_PADDING}" == "0" ]]; then
  PROFILE="masked-padding-disabled"
  EXPECTED_MASKED_PADDING="disabled"
else
  echo "MASKED_PADDING must be 0 or 1, got: ${MASKED_PADDING}" >&2
  exit 2
fi

if [[ -z "${TOKENIZER_MODEL}" ]]; then
  echo "TOKENIZER_MODEL (or MODEL_PATH) must be set." >&2
  exit 2
fi

START_INDEX="${START_INDEX:-0}"
MAX_REQUESTS="${MAX_REQUESTS:-512}"
QPS_LIST="${QPS_LIST:-0.5 1 2 4}"
ARRIVAL_PROCESS="${ARRIVAL_PROCESS:-poisson}"
RANDOM_SEED_BASE="${RANDOM_SEED_BASE:-2026}"
MAX_CONCURRENCY="${MAX_CONCURRENCY:-32}"
NUM_REPEATS="${NUM_REPEATS:-3}"

APPEND_INITIAL_DELAY="${APPEND_INITIAL_DELAY:-0.0}"
UPDATE_INITIAL_DELAY="${UPDATE_INITIAL_DELAY:-0.0}"
APPEND_CHUNK_INTERVAL="${APPEND_CHUNK_INTERVAL:-0.5}"
UPDATE_CHUNK_INTERVAL="${UPDATE_CHUNK_INTERVAL:-0.2}"
SOURCE_FINISH_DELAY="${SOURCE_FINISH_DELAY:-0.0}"
MAX_TOKENS="${MAX_TOKENS:-128}"
TEMPERATURE="${TEMPERATURE:-0}"
SAMPLING_SEED="${SAMPLING_SEED:-2026}"
HTTP_TIMEOUT="${HTTP_TIMEOUT:-600}"
SSE_TIMEOUT="${SSE_TIMEOUT:-300}"

RUN_TTFT_SUITE="${RUN_TTFT_SUITE:-1}"
RUN_FULL_SUITE="${RUN_FULL_SUITE:-1}"
RUN_WARMUP="${RUN_WARMUP:-1}"
WARMUP_REQUESTS="${WARMUP_REQUESTS:-8}"
FAIL_FAST="${FAIL_FAST:-0}"
EXTRA_REPLAY_ARGS="${EXTRA_REPLAY_ARGS:-}"
extra_replay_args=()
if [[ -n "${EXTRA_REPLAY_ARGS}" ]]; then
  read -r -a extra_replay_args <<< "${EXTRA_REPLAY_ARGS}"
fi
RUN_PREFIX="${RUN_PREFIX:-masked-padding}"
JOB_TAG="${SLURM_JOB_ID:-manual}"

for boolean_name in \
  TOKENIZER_TRUST_REMOTE_CODE RUN_TTFT_SUITE RUN_FULL_SUITE RUN_WARMUP FAIL_FAST; do
  boolean_value="${!boolean_name}"
  if [[ "${boolean_value}" != "0" && "${boolean_value}" != "1" ]]; then
    echo "${boolean_name} must be 0 or 1, got: ${boolean_value}" >&2
    exit 2
  fi
done

if ! [[ "${NUM_REPEATS}" =~ ^[1-9][0-9]*$ ]]; then
  echo "NUM_REPEATS must be a positive integer." >&2
  exit 2
fi

mkdir -p "${OUTPUT_DIR}"

echo "SLURM_JOB_ID=${SLURM_JOB_ID:-}"
echo "HOSTNAME=$(hostname)"
echo "PROFILE=${PROFILE}"
echo "EXPECTED_MASKED_PADDING=${EXPECTED_MASKED_PADDING}"
echo "DATA_DIR=${DATA_DIR}"
echo "OUTPUT_DIR=${OUTPUT_DIR}"
echo "PROXY_URL=${PROXY_URL}"
echo "MODEL=${MODEL}"
echo "TOKENIZER_MODEL=${TOKENIZER_MODEL}"
echo "MAX_REQUESTS=${MAX_REQUESTS}"
echo "QPS_LIST=${QPS_LIST}"
echo "MAX_CONCURRENCY=${MAX_CONCURRENCY}"
echo "NUM_REPEATS=${NUM_REPEATS}"
echo "MAX_TOKENS=${MAX_TOKENS}"

qps_suffix() {
  local qps="$1"
  echo "${qps//./p}"
}

run_replay() {
  local phase="$1"
  local qps="$2"
  local repeat_index="$3"
  local random_seed="$4"
  local num_requests="$5"
  local experiment_suite="masked_padding"
  local concurrency="${MAX_CONCURRENCY}"
  local load_mode="${LOAD_MODE:-auto}"
  local max_inflight="${MAX_INFLIGHT:-0}"
  local suffix
  local run_name

  if [[ "${phase}" == "warmup" ]]; then
    experiment_suite="masked_padding_warmup"
    concurrency=1
    load_mode=closed_loop
    max_inflight=0
  fi

  suffix="$(qps_suffix "${qps}")"
  run_name="${RUN_PREFIX}-${PROFILE}-${phase}-qps-${suffix}"
  run_name+="-seed-${random_seed}-rep-${repeat_index}-job-${JOB_TAG}"

  args=(
    --data-dir "${DATA_DIR}"
    --start-index "${START_INDEX}"
    --max-requests "${num_requests}"
    --proxy-url "${PROXY_URL}"
    --model "${MODEL}"
    --tokenizer-model "${TOKENIZER_MODEL}"
    --qps "${qps}"
    --arrival-process "${ARRIVAL_PROCESS}"
    --random-seed "${random_seed}"
    --max-concurrency "${concurrency}"
    --load-mode "${load_mode}"
    --max-inflight "${max_inflight}"
    --append-initial-delay "${APPEND_INITIAL_DELAY}"
    --update-initial-delay "${UPDATE_INITIAL_DELAY}"
    --append-chunk-interval "${APPEND_CHUNK_INTERVAL}"
    --update-chunk-interval "${UPDATE_CHUNK_INTERVAL}"
    --source-finish-delay "${SOURCE_FINISH_DELAY}"
    --max-tokens "${MAX_TOKENS}"
    --temperature "${TEMPERATURE}"
    --seed "${SAMPLING_SEED}"
    --http-timeout "${HTTP_TIMEOUT}"
    --sse-timeout "${SSE_TIMEOUT}"
    --output-dir "${OUTPUT_DIR}"
    --run-name "${run_name}"
    --experiment-suite "${experiment_suite}"
    --experiment-profile "${PROFILE}"
    --repeat-index "${repeat_index}"
    --expected-coalescing disabled
    --expected-masked-padding "${EXPECTED_MASKED_PADDING}"
    --return-token-ids
    --event-progress-log-interval 0
  )
  args+=(--input-transport "${INPUT_TRANSPORT:-text}")
  if [[ -n "${REPLAY_CONFIG:-}" ]]; then
    args+=(--config "${REPLAY_CONFIG}")
  fi

  if [[ "${TOKENIZER_TRUST_REMOTE_CODE}" == "1" ]]; then
    args+=(--tokenizer-trust-remote-code)
  fi
  if [[ "${FAIL_FAST}" == "1" ]]; then
    args+=(--fail-fast)
  fi
  if [[ "${phase}" == "ttft" ]]; then
    args+=(--ttft-only)
  fi

  echo "=== phase=${phase} qps=${qps} repeat=${repeat_index} run=${run_name} ==="
  "${PYTHON_BIN}" "${REPLAY_SCRIPT}" "${args[@]}" "${extra_replay_args[@]}"
}

# 完整路径 warmup 同时触发 P、D 与 NIXL；warmup 结果不会进入正式汇总。
if [[ "${RUN_WARMUP}" == "1" ]]; then
  run_replay "warmup" "0" "0" "${RANDOM_SEED_BASE}" "${WARMUP_REQUESTS}"
fi

for ((repeat_index = 1; repeat_index <= NUM_REPEATS; repeat_index++)); do
  random_seed=$((RANDOM_SEED_BASE + repeat_index - 1))
  for qps in ${QPS_LIST}; do
    if [[ "${RUN_TTFT_SUITE}" == "1" ]]; then
      run_replay "ttft" "${qps}" "${repeat_index}" \
        "${random_seed}" "${MAX_REQUESTS}"
    fi
    if [[ "${RUN_FULL_SUITE}" == "1" ]]; then
      run_replay "full" "${qps}" "${repeat_index}" \
        "${random_seed}" "${MAX_REQUESTS}"
    fi
  done
done
