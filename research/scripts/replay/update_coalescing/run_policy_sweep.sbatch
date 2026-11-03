#!/usr/bin/env bash
#SBATCH --job-name=coalescing-replay
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=12:00:00
#SBATCH --output=research/scripts/logs/replay-coalescing-%j.out

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
cd "${REPO_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-${REPO_ROOT}/.venv/bin/python}"
REPLAY_SCRIPT="${REPO_ROOT}/research/scripts/replay/update_coalescing/replay_update_coalescing.py"
PROFILE_HELPER="${REPO_ROOT}/research/scripts/replay/update_coalescing/coalescing_profiles.sh"

# shellcheck source=/dev/null
source "${PROFILE_HELPER}"

DATA_DIR="${DATA_DIR:-research/data/mix_source}"
OUTPUT_DIR="${OUTPUT_DIR:-research/scripts/replay/update_coalescing/results}"
PROXY_URL="${PROXY_URL:-}"
MODEL="${MODEL:-qwen2.5-3b}"
# 共享回放器用模型 chat template，tokenizer 路径与 REST served name 不同。
TOKENIZER_MODEL="${TOKENIZER_MODEL:-${MODEL_PATH:-}}"
TOKENIZER_TRUST_REMOTE_CODE="${TOKENIZER_TRUST_REMOTE_CODE:-0}"

COALESCING_PROFILE="${COALESCING_PROFILE:-fixed-window}"
P_NODE_CONFIG_JSON="$(build_coalescing_config)"
EXPECTED_COALESCING="$(coalescing_expected_state)"

MAX_REQUESTS="${MAX_REQUESTS:-256}"
QPS_LIST="${QPS_LIST:-1}"
UPDATE_INTERVAL_LIST="${UPDATE_INTERVAL_LIST:-0.2 0.01}"
NUM_REPEATS="${NUM_REPEATS:-3}"
BASE_RANDOM_SEED="${BASE_RANDOM_SEED:-2026}"
ARRIVAL_PROCESS="${ARRIVAL_PROCESS:-poisson}"
MAX_CONCURRENCY="${MAX_CONCURRENCY:-32}"

APPEND_INITIAL_DELAY="${APPEND_INITIAL_DELAY:-0.0}"
UPDATE_INITIAL_DELAY="${UPDATE_INITIAL_DELAY:-0.0}"
APPEND_CHUNK_INTERVAL="${APPEND_CHUNK_INTERVAL:-0.5}"
SOURCE_FINISH_DELAY="${SOURCE_FINISH_DELAY:-0.0}"

MAX_TOKENS="${MAX_TOKENS:-64}"
TEMPERATURE="${TEMPERATURE:-0}"
HTTP_TIMEOUT="${HTTP_TIMEOUT:-600}"
SSE_TIMEOUT="${SSE_TIMEOUT:-120}"
EVENT_PROGRESS_LOG_INTERVAL="${EVENT_PROGRESS_LOG_INTERVAL:-0}"
WARMUP_REQUESTS="${WARMUP_REQUESTS:-0}"
RUN_PREFIX="${RUN_PREFIX:-coalescing}"
FAIL_FAST="${FAIL_FAST:-0}"
FORMAL_FAIL_FAST="${FORMAL_FAIL_FAST:-0}"
EXTRA_REPLAY_ARGS="${EXTRA_REPLAY_ARGS:-}"

if [[ -z "${PROXY_URL}" ]]; then
  echo "PROXY_URL must point to the reachable proxy host, not implicit localhost." >&2
  exit 2
fi
if [[ -z "${TOKENIZER_MODEL}" ]]; then
  echo "TOKENIZER_MODEL (or MODEL_PATH) must be set for chat templates." >&2
  exit 2
fi
if (( NUM_REPEATS <= 0 )); then
  echo "NUM_REPEATS must be positive." >&2
  exit 2
fi

extra_replay_args=()
if [[ -n "${EXTRA_REPLAY_ARGS}" ]]; then
  read -r -a extra_replay_args <<< "${EXTRA_REPLAY_ARGS}"
fi

echo "SLURM_JOB_ID=${SLURM_JOB_ID:-}"
echo "HOSTNAME=$(hostname)"
echo "COALESCING_PROFILE=${COALESCING_PROFILE}"
echo "EXPECTED_COALESCING=${EXPECTED_COALESCING}"
echo "P_NODE_CONFIG_JSON=${P_NODE_CONFIG_JSON:-disabled}"
echo "PROXY_URL=${PROXY_URL}"
echo "TOKENIZER_MODEL=${TOKENIZER_MODEL}"
echo "TOKENIZER_TRUST_REMOTE_CODE=${TOKENIZER_TRUST_REMOTE_CODE}"
echo "QPS_LIST=${QPS_LIST}"
echo "UPDATE_INTERVAL_LIST=${UPDATE_INTERVAL_LIST}"
echo "NUM_REPEATS=${NUM_REPEATS}"

to_run_suffix() {
  local value="$1"
  value="${value//./p}"
  value="${value//-/_}"
  echo "${value}"
}

run_replay() {
  local qps="$1"
  local update_interval="$2"
  local repeat_index="$3"
  local random_seed="$4"
  local max_requests="$5"
  local output_dir="$6"
  local run_name="$7"

  replay_args=(
    --data-dir "${DATA_DIR}"
    --max-requests "${max_requests}"
    --proxy-url "${PROXY_URL}"
    --model "${MODEL}"
    --tokenizer-model "${TOKENIZER_MODEL}"
    --qps "${qps}"
    --arrival-process "${ARRIVAL_PROCESS}"
    --random-seed "${random_seed}"
    --load-mode "${LOAD_MODE:-auto}" --max-inflight "${MAX_INFLIGHT:-0}"
    --max-concurrency "${MAX_CONCURRENCY}"
    --append-initial-delay "${APPEND_INITIAL_DELAY}"
    --update-initial-delay "${UPDATE_INITIAL_DELAY}"
    --append-chunk-interval "${APPEND_CHUNK_INTERVAL}"
    --update-chunk-interval "${update_interval}"
    --source-finish-delay "${SOURCE_FINISH_DELAY}"
    --event-progress-log-interval "${EVENT_PROGRESS_LOG_INTERVAL}"
    --max-tokens "${MAX_TOKENS}"
    --temperature "${TEMPERATURE}"
    --http-timeout "${HTTP_TIMEOUT}"
    --sse-timeout "${SSE_TIMEOUT}"
    --output-dir "${output_dir}"
    --run-name "${run_name}"
    --experiment-profile "${COALESCING_PROFILE}"
    --repeat-index "${repeat_index}"
    --expected-coalescing "${EXPECTED_COALESCING}"
    --ttft-only
  )
  replay_args+=(--input-transport "${INPUT_TRANSPORT:-text}")
  if [[ -n "${REPLAY_CONFIG:-}" ]]; then
    replay_args+=(--config "${REPLAY_CONFIG}")
  fi

  if [[ "${TOKENIZER_TRUST_REMOTE_CODE}" == "1" ]]; then
    replay_args+=(--tokenizer-trust-remote-code)
  fi
  if [[ -n "${P_NODE_CONFIG_JSON}" ]]; then
    replay_args+=(
      --p-node-coalescing-config-json "${P_NODE_CONFIG_JSON}"
    )
  fi
  if [[ "${FAIL_FAST}" == "1" ]]; then
    replay_args+=(--fail-fast)
  fi
  if [[ "${FORMAL_FAIL_FAST}" == "1" ]]; then
    replay_args+=(--formal-fail-fast)
  fi

  echo "=== run ${run_name} ==="
  "${PYTHON_BIN}" "${REPLAY_SCRIPT}" \
    "${replay_args[@]}" \
    "${extra_replay_args[@]}"
}

# Warmup 只使用矩阵中的第一组 source timing，不写入正式结果根目录。
if (( WARMUP_REQUESTS > 0 )); then
  read -r first_update_interval _ <<< "${UPDATE_INTERVAL_LIST}"
  warmup_name="warmup-${COALESCING_PROFILE}-job-${SLURM_JOB_ID:-manual}"
  LOAD_MODE=closed_loop MAX_INFLIGHT=0 MAX_CONCURRENCY=1 run_replay \
    0 \
    "${first_update_interval}" \
    0 \
    "${BASE_RANDOM_SEED}" \
    "${WARMUP_REQUESTS}" \
    "${OUTPUT_DIR}/warmup" \
    "${warmup_name}"
fi

for ((repeat_index = 1; repeat_index <= NUM_REPEATS; repeat_index++)); do
  random_seed=$((BASE_RANDOM_SEED + repeat_index - 1))
  for update_interval in ${UPDATE_INTERVAL_LIST}; do
    update_suffix="$(to_run_suffix "${update_interval}")"
    for qps in ${QPS_LIST}; do
      qps_suffix="$(to_run_suffix "${qps}")"
      run_name="${RUN_PREFIX}-${COALESCING_PROFILE}-ui-${update_suffix}-qps-${qps_suffix}-seed-${random_seed}-rep-${repeat_index}-job-${SLURM_JOB_ID:-manual}"
      run_replay \
        "${qps}" \
        "${update_interval}" \
        "${repeat_index}" \
        "${random_seed}" \
        "${MAX_REQUESTS}" \
        "${OUTPUT_DIR}" \
        "${run_name}"
    done
  done
done
