#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#SBATCH --job-name=token-invalidation
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=24:00:00
#SBATCH --output=research/scripts/replay/token_invalidation/results/slurm-%j.out

# 仅运行 CPU 回放客户端；P 和普通 proxy 须已启动。
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
cd "${REPO_ROOT}"
REPO_ROOT="$(pwd)"
REPLAY_DIR="${REPO_ROOT}/research/scripts/replay/token_invalidation"
PYTHON_BIN="${PYTHON_BIN:-${REPO_ROOT}/.venv/bin/python}"

# 三种模式只构建一次公共参数数组，保证数据和时间控制完全一致。
DATASET="${DATASET:-triviaqa}"
PROXY_URL="${PROXY_URL:-http://127.0.0.1:8000}"
MODEL="${MODEL:-qwen2.5-3b}"
TOKENIZER_MODEL="${TOKENIZER_MODEL:-${MODEL_PATH:-${HOME}/Models/Qwen2.5-3B}}"
TOKENIZER_TRUST_REMOTE_CODE="${TOKENIZER_TRUST_REMOTE_CODE:-0}"
INPUT_TRANSPORT="${INPUT_TRANSPORT:-text}"
START_INDEX="${START_INDEX:-0}"
MAX_REQUESTS="${MAX_REQUESTS:-0}"
QPS="${QPS:-0}"
ARRIVAL_PROCESS="${ARRIVAL_PROCESS:-fixed}"
RANDOM_SEED="${RANDOM_SEED:-2026}"
MAX_CONCURRENCY="${MAX_CONCURRENCY:-1}"
APPEND_INITIAL_DELAY="${APPEND_INITIAL_DELAY:-0.0}"
UPDATE_INITIAL_DELAY="${UPDATE_INITIAL_DELAY:-0.0}"
APPEND_CHUNK_INTERVAL="${APPEND_CHUNK_INTERVAL:-0.05}"
UPDATE_CHUNK_INTERVAL="${UPDATE_CHUNK_INTERVAL:-0.02}"
SOURCE_FINISH_DELAY="${SOURCE_FINISH_DELAY:-0.0}"
TEMPERATURE="${TEMPERATURE:-0}"
SAMPLING_SEED="${SAMPLING_SEED:-2026}"
HTTP_TIMEOUT="${HTTP_TIMEOUT:-600}"
SSE_TIMEOUT="${SSE_TIMEOUT:-300}"
EXPECTED_COALESCING="${EXPECTED_COALESCING:-disabled}"
EXPECTED_MASKED_PADDING="${EXPECTED_MASKED_PADDING:-any}"
P_NODE_COALESCING_CONFIG_JSON="${P_NODE_COALESCING_CONFIG_JSON:-}"
REQUEST_PROGRESS_LOG_INTERVAL="${REQUEST_PROGRESS_LOG_INTERVAL:-100}"
EVENT_PROGRESS_LOG_INTERVAL="${EVENT_PROGRESS_LOG_INTERVAL:-0}"
FAIL_FAST="${FAIL_FAST:-0}"
FORMAL_FAIL_FAST="${FORMAL_FAIL_FAST:-0}"
DRY_RUN="${DRY_RUN:-0}"
RUN_PLOTS="${RUN_PLOTS:-1}"
PLOT_X_SCALE="${PLOT_X_SCALE:-linear}"
PLOT_DPI="${PLOT_DPI:-200}"

# 一个作业的三个模式和配对结果放在一起，重复运行须使用不同 RUN_NAME。
OUTPUT_DIR="${OUTPUT_DIR:-${REPLAY_DIR}/results}"
JOB_TAG="${SLURM_JOB_ID:-$(date -u +%Y%m%dT%H%M%S-%N)}"
RUN_NAME="${RUN_NAME:-${DATASET}-job-${JOB_TAG}}"

case "${DATASET}" in
  triviaqa|hotpotqa) ;;
  *) echo "DATASET must be triviaqa or hotpotqa: ${DATASET}" >&2; exit 2 ;;
esac
if [[ ! "${RUN_NAME}" =~ ^[a-zA-Z0-9._-]+$ ||
      "${RUN_NAME}" == "." || "${RUN_NAME}" == ".." ]]; then
  echo "RUN_NAME must be a directory name without slashes: ${RUN_NAME}" >&2
  exit 2
fi
for flag in TOKENIZER_TRUST_REMOTE_CODE FAIL_FAST FORMAL_FAIL_FAST DRY_RUN RUN_PLOTS; do
  if [[ "${!flag}" != "0" && "${!flag}" != "1" ]]; then
    echo "${flag} must be 0 or 1: ${!flag}" >&2
    exit 2
  fi
done
if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "Python environment not found: ${PYTHON_BIN}" >&2
  exit 2
fi

mkdir -p "${OUTPUT_DIR}"
OUTPUT_DIR="$(cd "${OUTPUT_DIR}" && pwd)"
SUITE_DIR="${OUTPUT_DIR}/${RUN_NAME}"
mkdir "${SUITE_DIR}"  # 已存在则停止，避免混入另一轮统计。

shared_args=(
  --dataset "${DATASET}"
  --proxy-url "${PROXY_URL}"
  --model "${MODEL}"
  --tokenizer-model "${TOKENIZER_MODEL}"
  --input-transport "${INPUT_TRANSPORT}"
  --start-index "${START_INDEX}"
  --max-requests "${MAX_REQUESTS}"
  --qps "${QPS}"
  --arrival-process "${ARRIVAL_PROCESS}"
  --random-seed "${RANDOM_SEED}"
  --max-concurrency "${MAX_CONCURRENCY}"
  --append-initial-delay "${APPEND_INITIAL_DELAY}"
  --update-initial-delay "${UPDATE_INITIAL_DELAY}"
  --append-chunk-interval "${APPEND_CHUNK_INTERVAL}"
  --update-chunk-interval "${UPDATE_CHUNK_INTERVAL}"
  --source-finish-delay "${SOURCE_FINISH_DELAY}"
  --temperature "${TEMPERATURE}"
  --seed "${SAMPLING_SEED}"
  --http-timeout "${HTTP_TIMEOUT}"
  --sse-timeout "${SSE_TIMEOUT}"
  --expected-coalescing "${EXPECTED_COALESCING}"
  --expected-masked-padding "${EXPECTED_MASKED_PADDING}"
  --request-progress-log-interval "${REQUEST_PROGRESS_LOG_INTERVAL}"
  --event-progress-log-interval "${EVENT_PROGRESS_LOG_INTERVAL}"
  --output-dir "${SUITE_DIR}"
)

# 可选路径覆盖数据集默认版本，参数以数组传递，保留路径中的空格。
for variable in QUESTIONS_FILE APPEND_DIR PIPELINE_POOLS_DIR; do
  value="${!variable:-}"
  if [[ -n "${value}" ]]; then
    option="${variable,,}"
    shared_args+=("--${option//_/-}" "${value}")
  fi
done
if [[ -n "${P_NODE_COALESCING_CONFIG_JSON}" ]]; then
  shared_args+=(--p-node-coalescing-config-json "${P_NODE_COALESCING_CONFIG_JSON}")
fi
if [[ "${TOKENIZER_TRUST_REMOTE_CODE}" == "1" ]]; then
  shared_args+=(--tokenizer-trust-remote-code)
fi
if [[ "${FAIL_FAST}" == "1" ]]; then
  shared_args+=(--fail-fast)
fi
if [[ "${FORMAL_FAIL_FAST}" == "1" ]]; then
  shared_args+=(--formal-fail-fast)
fi
if [[ "${DRY_RUN}" == "1" ]]; then
  shared_args+=(--dry-run)
fi

printf 'Host: %s\nResults: %s\n' "$(hostname)" "${SUITE_DIR}"
printf 'Shared arguments:'
printf ' %q' "${shared_args[@]}"
printf '\n'

# 顺序执行，三种模式不同时竞争 P 的资源；任一命令失败都会停止作业。
for layout in update_first append_first segmented; do
  printf 'Running layout: %s\n' "${layout}"
  "${PYTHON_BIN}" "${REPLAY_DIR}/replay_${layout}.py" \
    "${shared_args[@]}" --run-name "${layout}"
done

if [[ "${DRY_RUN}" == "1" ]]; then
  printf 'Dry-run previews written to %s; no server statistics to compare.\n' "${SUITE_DIR}"
  exit 0
fi

"${PYTHON_BIN}" "${REPLAY_DIR}/compare_results.py" \
  --update-first-run "${SUITE_DIR}/update_first" \
  --append-first-run "${SUITE_DIR}/append_first" \
  --segmented-run "${SUITE_DIR}/segmented" \
  --output-dir "${SUITE_DIR}/comparison"

if [[ "${RUN_PLOTS}" == "1" ]]; then
  "${PYTHON_BIN}" "${REPLAY_DIR}/plot_token_invalidation.py" \
    --comparison-dir "${SUITE_DIR}/comparison" \
    --x-scale "${PLOT_X_SCALE}" --dpi "${PLOT_DPI}"
fi
printf 'Completed token invalidation suite: %s\n' "${SUITE_DIR}"
