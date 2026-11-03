#!/usr/bin/env bash
#SBATCH --job-name=mixed-replay
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=research/scripts/replay/mixed_workload/results/mixed-%j.out

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
cd "${REPO_ROOT}"
PYTHON_BIN="${PYTHON_BIN:-${REPO_ROOT}/.venv/bin/python}"
DATASET="${DATASET:-triviaqa}"
REPLAY_CONFIG="${REPLAY_CONFIG:-research/scripts/replay/mixed_workload/configs/${DATASET}.json}"
OUTPUT_DIR="${OUTPUT_DIR:-research/scripts/replay/mixed_workload/results}"
RUN_PREFIX="${RUN_PREFIX:-${DATASET}-mixed}"
JOB_TAG="${SLURM_JOB_ID:-$(date +%Y%m%d-%H%M%S)}"
mkdir -p "${OUTPUT_DIR}" "${REPO_ROOT}/tmp"
export TMPDIR="${REPO_ROOT}/tmp"

# 只申请客户端资源；两组依次复用已启动的同一个 P 和 proxy。
COMMON_ARGS=(--config "${REPLAY_CONFIG}" --output-dir "${OUTPUT_DIR}")
OVERRIDES=(
  PROXY_URL:proxy-url MODEL:model TOKENIZER_MODEL:tokenizer-model
  MAX_REQUESTS:max-requests START_INDEX:start-index STREAM_RATIO:stream-ratio
  MIX_SEED:mix-seed KEEP_RATIO:keep-ratio QPS:qps ARRIVAL_PROCESS:arrival-process
  RANDOM_SEED:random-seed LOAD_MODE:load-mode MAX_CONCURRENCY:max-concurrency
  MAX_INFLIGHT:max-inflight INTER_REQUEST_DELAY:inter-request-delay
  APPEND_INITIAL_DELAY:append-initial-delay UPDATE_INITIAL_DELAY:update-initial-delay
  APPEND_CHUNK_INTERVAL:append-chunk-interval UPDATE_CHUNK_INTERVAL:update-chunk-interval
  SOURCE_FINISH_DELAY:source-finish-delay HTTP_TIMEOUT:http-timeout SSE_TIMEOUT:sse-timeout
  EXPERIMENT_PROFILE:experiment-profile
)
for mapping in "${OVERRIDES[@]}"; do
  variable="${mapping%%:*}"
  option="${mapping#*:}"
  if [[ -v "${variable}" ]]; then
    COMMON_ARGS+=("--${option}" "${!variable}")
  fi
done

MODES=(normal segmented_sparse)
if [[ "${REVERSE_ORDER:-0}" == "1" ]]; then
  MODES=(segmented_sparse normal)
fi
for mode in "${MODES[@]}"; do
  "${PYTHON_BIN}" research/scripts/replay/mixed_workload/replay.py \
    "${COMMON_ARGS[@]}" --stream-mode "${mode}" \
    --run-name "${RUN_PREFIX}-${mode}-${JOB_TAG}"
done
"${PYTHON_BIN}" research/scripts/replay/mixed_workload/compare.py \
  --baseline "${OUTPUT_DIR}/${RUN_PREFIX}-normal-${JOB_TAG}" \
  --optimized "${OUTPUT_DIR}/${RUN_PREFIX}-segmented_sparse-${JOB_TAG}" \
  --output-dir "${OUTPUT_DIR}/${RUN_PREFIX}-comparison-${JOB_TAG}"
