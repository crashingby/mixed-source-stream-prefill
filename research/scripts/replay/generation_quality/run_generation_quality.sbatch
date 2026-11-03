#!/usr/bin/env bash
#SBATCH --job-name=quality-eval
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=24:00:00
#SBATCH --output=research/scripts/replay/generation_quality/results/quality-%j.out

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
cd "${REPO_ROOT}"
PYTHON_BIN="${PYTHON_BIN:-${REPO_ROOT}/.venv/bin/python}"
DATASET="${DATASET:-triviaqa}"
REPLAY_CONFIG="${REPLAY_CONFIG:-research/scripts/replay/generation_quality/configs/${DATASET}.json}"
OUTPUT_DIR="${OUTPUT_DIR:-research/scripts/replay/generation_quality/results}"
RUN_PREFIX="${RUN_PREFIX:-${DATASET}-quality}"
MODE_ORDER="${MODE_ORDER:-normal segmented sparse}"
JOB_SUFFIX="${SLURM_JOB_ID:-$(date +%Y%m%d-%H%M%S)}"
mkdir -p "${OUTPUT_DIR}" "${REPO_ROOT}/tmp"
export TMPDIR="${REPO_ROOT}/tmp"

# JSON 是共享参数来源；只把显式提供的环境变量追加为 CLI，避免悄悄覆盖配置。
COMMON_ARGS=(--config "${REPLAY_CONFIG}" --output-dir "${OUTPUT_DIR}")
OVERRIDES=(
  PROXY_URL:proxy-url MODEL:model TOKENIZER_MODEL:tokenizer-model
  START_INDEX:start-index MAX_REQUESTS:max-requests MAX_TOKENS:max-tokens
  TEMPERATURE:temperature SEED:seed QPS:qps MAX_CONCURRENCY:max-concurrency
  LOAD_MODE:load-mode MAX_INFLIGHT:max-inflight
  INTER_REQUEST_DELAY:inter-request-delay APPEND_INITIAL_DELAY:append-initial-delay
  UPDATE_INITIAL_DELAY:update-initial-delay APPEND_CHUNK_INTERVAL:append-chunk-interval
  UPDATE_CHUNK_INTERVAL:update-chunk-interval SOURCE_FINISH_DELAY:source-finish-delay
  HTTP_TIMEOUT:http-timeout SSE_TIMEOUT:sse-timeout INPUT_TRANSPORT:input-transport
  PROBE_KEEP_RATIO:probe-keep-ratio
)
for mapping in "${OVERRIDES[@]}"; do
  variable="${mapping%%:*}"
  option="${mapping#*:}"
  if [[ -v "${variable}" ]]; then
    COMMON_ARGS+=("--${option}" "${!variable}")
  fi
done
if [[ -v STOP_AT_NEWLINE ]]; then
  if [[ "${STOP_AT_NEWLINE}" == "1" ]]; then
    COMMON_ARGS+=(--stop-at-newline)
  else
    COMMON_ARGS+=(--no-stop-at-newline)
  fi
fi
if [[ "${TOKENIZER_TRUST_REMOTE_CODE:-0}" == "1" ]]; then
  COMMON_ARGS+=(--tokenizer-trust-remote-code)
fi
if [[ "${FAIL_FAST:-0}" == "1" ]]; then
  COMMON_ARGS+=(--fail-fast)
fi

# 支持只跑 segmented/sparse，也支持加上 normal 来分离两种近似的影响。
read -r -a MODES <<< "${MODE_ORDER}"
declare -A SEEN=()
for mode in "${MODES[@]}"; do
  if [[ "${mode}" != "normal" && "${mode}" != "segmented" && "${mode}" != "sparse" ]]; then
    echo "Unsupported mode: ${mode}" >&2
    exit 2
  fi
  if [[ -v "SEEN[${mode}]" ]]; then
    echo "Duplicate mode: ${mode}" >&2
    exit 2
  fi
  SEEN[${mode}]=1
done
if (( ${#MODES[@]} < 2 )); then
  echo "MODE_ORDER must contain at least two modes." >&2
  exit 2
fi

EVALUATION_ARGS=(--output-dir "${OUTPUT_DIR}/${RUN_PREFIX}-analysis-job-${JOB_SUFFIX}")
echo "SLURM_JOB_ID=${SLURM_JOB_ID:-manual} HOSTNAME=$(hostname)"
echo "REPLAY_CONFIG=${REPLAY_CONFIG} MODE_ORDER=${MODE_ORDER}"
for mode in "${MODES[@]}"; do
  run_name="${RUN_PREFIX}-${mode}-job-${JOB_SUFFIX}"
  echo "Starting quality mode=${mode} run=${run_name}"
  "${PYTHON_BIN}" research/scripts/replay/generation_quality/replay_generation_quality.py \
    "${COMMON_ARGS[@]}" --mode "${mode}" --run-name "${run_name}"
  EVALUATION_ARGS+=(--run "${mode}=${OUTPUT_DIR}/${run_name}")
done
"${PYTHON_BIN}" research/scripts/replay/generation_quality/evaluate_answers.py \
  "${EVALUATION_ARGS[@]}"
