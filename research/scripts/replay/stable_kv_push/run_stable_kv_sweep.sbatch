#!/usr/bin/env bash
#SBATCH --job-name=stable-kv-replay
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --nodelist=g2c14
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=12:00:00
#SBATCH --output=research/scripts/logs/replay-stable-kv-%j.out

set -euo pipefail
REPO_ROOT="${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
cd "${REPO_ROOT}"
PYTHON_BIN="${PYTHON_BIN:-${REPO_ROOT}/.venv/bin/python}"
PROXY_URL="${PROXY_URL:-http://g2c14:8000}"
DATA_DIR="${DATA_DIR:-research/data/mix_source_perf_qwen}"
OUTPUT_DIR="${OUTPUT_DIR:-research/scripts/replay/stable_kv_push/results}"
ANALYSIS_DIR="${ANALYSIS_DIR:-${OUTPUT_DIR}/analysis}"
MODEL="${MODEL:-qwen2.5-3b}"
TOKENIZER_MODEL="${TOKENIZER_MODEL:-${HOME}/Models/Qwen2.5-3B}"
MAX_REQUESTS="${MAX_REQUESTS:-128}"
MAX_CONCURRENCY="${MAX_CONCURRENCY:-8}"
QPS_LIST="${QPS_LIST:-0 0.5 1 2}"
NUM_REPEATS="${NUM_REPEATS:-2}"
MAX_TOKENS="${MAX_TOKENS:-64}"
APPEND_CHUNK_INTERVAL="${APPEND_CHUNK_INTERVAL:-0.5}"
UPDATE_CHUNK_INTERVAL="${UPDATE_CHUNK_INTERVAL:-0.2}"
APPEND_INITIAL_DELAY="${APPEND_INITIAL_DELAY:-0.0}"
UPDATE_INITIAL_DELAY="${UPDATE_INITIAL_DELAY:-0.0}"
SOURCE_FINISH_DELAY="${SOURCE_FINISH_DELAY:-0.0}"
RANDOM_SEED="${RANDOM_SEED:-2026}"
RUN_PREFIX="${RUN_PREFIX:-stable-kv-job-${SLURM_JOB_ID:-manual}}"
SCRIPT_DIR=research/scripts/replay/stable_kv_push
replay_input_args=(--input-transport "${INPUT_TRANSPORT:-text}")
if [[ -n "${REPLAY_CONFIG:-}" ]]; then
  replay_input_args+=(--config "${REPLAY_CONFIG}")
fi
echo "HOSTNAME=$(hostname) PROXY_URL=${PROXY_URL} QPS_LIST=${QPS_LIST}"
echo "DATA_DIR=${DATA_DIR} MAX_REQUESTS=${MAX_REQUESTS} NUM_REPEATS=${NUM_REPEATS}"
echo "APPEND_INITIAL_DELAY=${APPEND_INITIAL_DELAY} UPDATE_INITIAL_DELAY=${UPDATE_INITIAL_DELAY}"
echo "APPEND_CHUNK_INTERVAL=${APPEND_CHUNK_INTERVAL} UPDATE_CHUNK_INTERVAL=${UPDATE_CHUNK_INTERVAL}"

# 重复轮次交替 A/B 顺序，降低总是后跑一组带来的 warmup/节点漂移偏差。
for ((repeat=1; repeat<=NUM_REPEATS; repeat++)); do
  modes=(finish-only early)
  if ((repeat % 2 == 0)); then
    modes=(early finish-only)
  fi
  for qps in ${QPS_LIST}; do
    qps_tag="${qps//./p}"
    for mode in "${modes[@]}"; do
      push_flag=--no-early-push
      if [[ "${mode}" == "early" ]]; then
        push_flag=--early-push
      fi
      "${PYTHON_BIN}" "${SCRIPT_DIR}/replay_stable_kv_push.py" \
        "${replay_input_args[@]}" \
        --data-dir "${DATA_DIR}" \
        --output-dir "${OUTPUT_DIR}" \
        --proxy-url "${PROXY_URL}" \
        --model "${MODEL}" \
        --tokenizer-model "${TOKENIZER_MODEL}" \
        --max-requests "${MAX_REQUESTS}" \
        --max-concurrency "${MAX_CONCURRENCY}" \
        --qps "${qps}" \
        --arrival-process poisson \
        --random-seed "${RANDOM_SEED}" \
        --seed "${RANDOM_SEED}" \
        --max-tokens "${MAX_TOKENS}" \
        --temperature 0 \
        --append-initial-delay "${APPEND_INITIAL_DELAY}" \
        --update-initial-delay "${UPDATE_INITIAL_DELAY}" \
        --append-chunk-interval "${APPEND_CHUNK_INTERVAL}" \
        --update-chunk-interval "${UPDATE_CHUNK_INTERVAL}" \
        --source-finish-delay "${SOURCE_FINISH_DELAY}" \
        --run-name "${RUN_PREFIX}-${mode}-qps-${qps_tag}-rep-${repeat}" \
        "${push_flag}"
    done
    early_dir="${OUTPUT_DIR}/${RUN_PREFIX}-early-qps-${qps_tag}-rep-${repeat}"
    finish_dir="${OUTPUT_DIR}/${RUN_PREFIX}-finish-only-qps-${qps_tag}-rep-${repeat}"
    analysis_dir="${ANALYSIS_DIR}/${RUN_PREFIX}-qps-${qps_tag}-rep-${repeat}"
    "${PYTHON_BIN}" "${SCRIPT_DIR}/summarize_stable_kv_push.py" \
      --early-run "${early_dir}" --finish-only-run "${finish_dir}" \
      --output-dir "${analysis_dir}"
    "${PYTHON_BIN}" "${SCRIPT_DIR}/plot_stable_kv_push.py" \
      --early-run "${early_dir}" --finish-only-run "${finish_dir}" \
      --output "${analysis_dir}/comparison.png"
  done
done
