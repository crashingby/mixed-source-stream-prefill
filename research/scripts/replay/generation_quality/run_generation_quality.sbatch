#!/usr/bin/env bash
#SBATCH --job-name=quality-eval
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=24:00:00
#SBATCH --output=research/scripts/logs/replay-generation-quality-%j.out

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
cd "${REPO_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-${REPO_ROOT}/.venv/bin/python}"
DATA_DIR="${DATA_DIR:-research/data/mix_source}"
OUTPUT_DIR="${OUTPUT_DIR:-research/scripts/replay/generation_quality/results}"
PROXY_URL="${PROXY_URL:-http://127.0.0.1:8000}"
MODEL="${MODEL:-qwen2.5-3b}"
TOKENIZER_MODEL="${TOKENIZER_MODEL:-${MODEL_PATH:-}}"
TOKENIZER_TRUST_REMOTE_CODE="${TOKENIZER_TRUST_REMOTE_CODE:-0}"

if [[ -z "${TOKENIZER_MODEL}" ]]; then
  echo "TOKENIZER_MODEL must be a Hugging Face model id or local model path." >&2
  exit 2
fi

START_INDEX="${START_INDEX:-0}"
MAX_REQUESTS="${MAX_REQUESTS:-100}"
MAX_TOKENS="${MAX_TOKENS:-32}"
TEMPERATURE="${TEMPERATURE:-0}"
SEED="${SEED:-2026}"
INTER_REQUEST_DELAY="${INTER_REQUEST_DELAY:-0.2}"

APPEND_INITIAL_DELAY="${APPEND_INITIAL_DELAY:-0.0}"
UPDATE_INITIAL_DELAY="${UPDATE_INITIAL_DELAY:-0.0}"
APPEND_CHUNK_INTERVAL="${APPEND_CHUNK_INTERVAL:-0.05}"
UPDATE_CHUNK_INTERVAL="${UPDATE_CHUNK_INTERVAL:-0.20}"
SOURCE_FINISH_DELAY="${SOURCE_FINISH_DELAY:-0.0}"

HTTP_TIMEOUT="${HTTP_TIMEOUT:-600}"
SSE_TIMEOUT="${SSE_TIMEOUT:-300}"
RUN_PREFIX="${RUN_PREFIX:-squad-quality}"
MODE_ORDER="${MODE_ORDER:-normal segmented}"
STOP_AT_NEWLINE="${STOP_AT_NEWLINE:-1}"
FAIL_FAST="${FAIL_FAST:-0}"

JOB_SUFFIX="${SLURM_JOB_ID:-manual}"
NORMAL_RUN_NAME="${RUN_PREFIX}-normal-job-${JOB_SUFFIX}"
SEGMENTED_RUN_NAME="${RUN_PREFIX}-segmented-job-${JOB_SUFFIX}"
ANALYSIS_DIR="${OUTPUT_DIR}/${RUN_PREFIX}-analysis-job-${JOB_SUFFIX}"

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
echo "START_INDEX=${START_INDEX}"
echo "MAX_REQUESTS=${MAX_REQUESTS}"
echo "MODE_ORDER=${MODE_ORDER}"
echo "MAX_TOKENS=${MAX_TOKENS}"
echo "TEMPERATURE=${TEMPERATURE}"
echo "SEED=${SEED}"
echo "INTER_REQUEST_DELAY=${INTER_REQUEST_DELAY}"
echo "APPEND_CHUNK_INTERVAL=${APPEND_CHUNK_INTERVAL}"
echo "UPDATE_CHUNK_INTERVAL=${UPDATE_CHUNK_INTERVAL}"

run_mode() {
  local mode="$1"
  local run_name

  if [[ "${mode}" == "normal" ]]; then
    run_name="${NORMAL_RUN_NAME}"
  elif [[ "${mode}" == "segmented" ]]; then
    run_name="${SEGMENTED_RUN_NAME}"
  else
    echo "Unsupported MODE_ORDER entry: ${mode}" >&2
    return 2
  fi

  replay_args=(
    --mode "${mode}"
    --data-dir "${DATA_DIR}"
    --start-index "${START_INDEX}"
    --max-requests "${MAX_REQUESTS}"
    --proxy-url "${PROXY_URL}"
    --model "${MODEL}"
    --tokenizer-model "${TOKENIZER_MODEL}"
    --max-tokens "${MAX_TOKENS}"
    --temperature "${TEMPERATURE}"
    --seed "${SEED}"
    --inter-request-delay "${INTER_REQUEST_DELAY}"
    --append-initial-delay "${APPEND_INITIAL_DELAY}"
    --update-initial-delay "${UPDATE_INITIAL_DELAY}"
    --append-chunk-interval "${APPEND_CHUNK_INTERVAL}"
    --update-chunk-interval "${UPDATE_CHUNK_INTERVAL}"
    --source-finish-delay "${SOURCE_FINISH_DELAY}"
    --http-timeout "${HTTP_TIMEOUT}"
    --sse-timeout "${SSE_TIMEOUT}"
    --output-dir "${OUTPUT_DIR}"
    --run-name "${run_name}"
  )
  if [[ "${STOP_AT_NEWLINE}" != "1" ]]; then
    replay_args+=(--no-stop-at-newline)
  fi
  if [[ "${TOKENIZER_TRUST_REMOTE_CODE}" == "1" ]]; then
    replay_args+=(--tokenizer-trust-remote-code)
  fi
  if [[ "${FAIL_FAST}" == "1" ]]; then
    replay_args+=(--fail-fast)
  fi

  echo "=== generation quality mode=${mode} run_name=${run_name} ==="
  "${PYTHON_BIN}" \
    research/scripts/replay/generation_quality/replay_generation_quality.py \
    "${replay_args[@]}"
}

for mode in ${MODE_ORDER}; do
  run_mode "${mode}"
done

if [[ ! -f "${OUTPUT_DIR}/${NORMAL_RUN_NAME}/generations.jsonl" ]]; then
  echo "Normal result is missing; MODE_ORDER must include normal." >&2
  exit 2
fi
if [[ ! -f "${OUTPUT_DIR}/${SEGMENTED_RUN_NAME}/generations.jsonl" ]]; then
  echo "Segmented result is missing; MODE_ORDER must include segmented." >&2
  exit 2
fi

echo "=== official SQuAD EM/F1 evaluation ==="
"${PYTHON_BIN}" \
  research/scripts/replay/generation_quality/evaluate_squad.py \
  --normal-results "${OUTPUT_DIR}/${NORMAL_RUN_NAME}" \
  --segmented-results "${OUTPUT_DIR}/${SEGMENTED_RUN_NAME}" \
  --output-dir "${ANALYSIS_DIR}"

echo "NORMAL_RESULTS=${OUTPUT_DIR}/${NORMAL_RUN_NAME}"
echo "SEGMENTED_RESULTS=${OUTPUT_DIR}/${SEGMENTED_RUN_NAME}"
echo "ANALYSIS_RESULTS=${ANALYSIS_DIR}"
