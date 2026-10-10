#!/usr/bin/env bash
#SBATCH --job-name=stable-kv-matrix
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --nodelist=g2c14
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=06:00:00
#SBATCH --output=research/scripts/logs/replay-stable-kv-matrix-%j.out

# 只串行运行回放，不启动额外服务；所有单元共用已启动的唯一一对 P/D。
set -euo pipefail
REPO_ROOT="${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
cd "${REPO_ROOT}"
PYTHON_BIN="${PYTHON_BIN:-${REPO_ROOT}/.venv/bin/python}"
SCRIPT_DIR=research/scripts/replay/stable_kv_push
export PROXY_URL="${PROXY_URL:-http://g2c14:8000}"
export MODEL="${MODEL:-qwen2.5-3b}"
export TOKENIZER_MODEL="${TOKENIZER_MODEL:-${HOME}/Models/Qwen2.5-3B}"
export OUTPUT_DIR="${OUTPUT_DIR:-${SCRIPT_DIR}/results}"
export MAX_REQUESTS="${MAX_REQUESTS:-128}"
export MAX_CONCURRENCY="${MAX_CONCURRENCY:-8}"
export NUM_REPEATS="${NUM_REPEATS:-2}"
export QPS_LIST="${QPS_LIST:-0 1 2}"
export APPEND_INITIAL_DELAY="${APPEND_INITIAL_DELAY:-0.0}"
export UPDATE_INITIAL_DELAY="${UPDATE_INITIAL_DELAY:-0.0}"
export APPEND_CHUNK_INTERVAL="${APPEND_CHUNK_INTERVAL:-0.1}"
export UPDATE_CHUNK_INTERVAL="${UPDATE_CHUNK_INTERVAL:-0.2}"
export SOURCE_FINISH_DELAY="${SOURCE_FINISH_DELAY:-0.0}"
export MAX_TOKENS="${MAX_TOKENS:-64}"
export RANDOM_SEED="${RANDOM_SEED:-2026}"
DATA_ROOT="${DATA_ROOT:-research/data/stable_kv_push_controlled_qwen}"
SCENARIO_DIRS="${SCENARIO_DIRS:-append-0256-update-1792 append-1024-update-1024 append-1792-update-0256}"
WARMUP_REQUESTS="${WARMUP_REQUESTS:-8}"
MATRIX_PREFIX="${RUN_PREFIX:-stable-kv-matrix-job-${SLURM_JOB_ID:-manual}}"
# 每个矩阵独占分析根，不能把之前不匹配的 smoke/QPS 结果混进正式图。
export ANALYSIS_DIR="${OUTPUT_DIR}/analysis/${MATRIX_PREFIX}"
echo "HOSTNAME=$(hostname) MATRIX_PREFIX=${MATRIX_PREFIX} DATA_ROOT=${DATA_ROOT}"
echo "SCENARIO_DIRS=${SCENARIO_DIRS} QPS_LIST=${QPS_LIST} NUM_REPEATS=${NUM_REPEATS}"

for scenario in ${SCENARIO_DIRS}; do
  export DATA_DIR="${DATA_ROOT}/${scenario}"
  if [[ ! -f "${DATA_DIR}/dataset_summary.json" ]]; then
    echo "Missing controlled dataset: ${DATA_DIR}" >&2
    exit 2
  fi
  export RUN_PREFIX="${MATRIX_PREFIX}-${scenario}"
  echo "SCENARIO_START=${scenario}"
  # 首轮并发形状的 Triton 编译只落入独立 warmup，不污染 P90/P95。
  if ((WARMUP_REQUESTS > 0)); then
    for mode in finish-only early; do
      push_flag=--no-early-push
      if [[ "${mode}" == early ]]; then push_flag=--early-push; fi
      "${PYTHON_BIN}" "${SCRIPT_DIR}/replay_stable_kv_push.py" \
        --data-dir "${DATA_DIR}" --output-dir "${OUTPUT_DIR}" \
        --proxy-url "${PROXY_URL}" --model "${MODEL}" \
        --tokenizer-model "${TOKENIZER_MODEL}" \
        --max-requests "${WARMUP_REQUESTS}" --max-concurrency "${MAX_CONCURRENCY}" \
        --qps 4 --arrival-process fixed --random-seed "${RANDOM_SEED}" \
        --seed "${RANDOM_SEED}" --max-tokens "${MAX_TOKENS}" --temperature 0 \
        --append-initial-delay "${APPEND_INITIAL_DELAY}" \
        --update-initial-delay "${UPDATE_INITIAL_DELAY}" \
        --append-chunk-interval "${APPEND_CHUNK_INTERVAL}" \
        --update-chunk-interval "${UPDATE_CHUNK_INTERVAL}" \
        --source-finish-delay "${SOURCE_FINISH_DELAY}" \
        --run-name "${RUN_PREFIX}-warmup-${mode}" "${push_flag}"
    done
  fi
  # bash 直接复用现有 sweep，不提交嵌套 Slurm 作业，也不交叉运行不同负载。
  bash "${SCRIPT_DIR}/run_stable_kv_sweep.sbatch"
  echo "SCENARIO_DONE=${scenario}"
done

"${PYTHON_BIN}" "${SCRIPT_DIR}/summarize_stable_kv_matrix.py" \
  --analysis-dir "${ANALYSIS_DIR}" --output-dir "${ANALYSIS_DIR}/matrix" \
  --percentile p95
echo "MATRIX_DONE=${ANALYSIS_DIR}/matrix"
