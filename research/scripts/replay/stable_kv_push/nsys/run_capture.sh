#!/usr/bin/env bash
# 复用已启动的唯一 P/D；预热不采集，随后捕获两轮反向顺序的短 A/B。
set -euo pipefail
REPO_ROOT="${REPO_ROOT:-$(pwd)}"
cd "${REPO_ROOT}"
PYTHON_BIN="${REPO_ROOT}/.venv/bin/python"
SCRIPT_DIR=research/scripts/replay/stable_kv_push
OUTPUT_DIR="${OUTPUT_DIR:-${SCRIPT_DIR}/results/nsys-20261004}"
PROXY_URL="${PROXY_URL:-http://g2c14:8000}"
P_URL="${P_URL:-http://g2c9:8100}"
DATA_DIR="${DATA_DIR:-research/data/stable_kv_push_controlled_qwen/append-1792-update-0256}"
TOKENIZER_MODEL="${TOKENIZER_MODEL:-${HOME}/Models/Qwen2.5-3B}"
MAX_REQUESTS="${MAX_REQUESTS:-8}"
QPS="${QPS:-2}"
profile_active=0
cleanup_profile() {
  if ((profile_active)); then
    curl --noproxy '*' --fail --silent --show-error --max-time 60 \
      -X POST "${P_URL}/stop_profile" || true
  fi
}
trap cleanup_profile EXIT

replay() {
  local name="$1" mode="$2" count="$3" qps="$4"
  local flag=--no-early-push
  if [[ "${mode}" == early ]]; then flag=--early-push; fi
  "${PYTHON_BIN}" "${SCRIPT_DIR}/replay_stable_kv_push.py" \
    --data-dir "${DATA_DIR}" --output-dir "${OUTPUT_DIR}" \
    --proxy-url "${PROXY_URL}" --model qwen2.5-3b \
    --tokenizer-model "${TOKENIZER_MODEL}" \
    --max-requests "${count}" --qps "${qps}" --max-concurrency 8 \
    --arrival-process poisson --random-seed 2026 --seed 2026 \
    --append-chunk-interval 0.1 --update-chunk-interval 0.2 \
    --source-finish-delay 0 --max-tokens 64 --temperature 0 \
    --http-timeout 120 --finish-timeout 180 --run-name "${name}" "${flag}"
}

replay warmup-finish finish-only 8 4
replay warmup-early early 8 4
capture_index=0
for repeat in 1 2; do
  modes=(finish-only early)
  if ((repeat == 2)); then modes=(early finish-only); fi
  for mode in "${modes[@]}"; do
    capture_index=$((capture_index + 1))
    name="capture-${capture_index}-${mode}-qps-${QPS}-rep-${repeat}"
    echo "CAPTURE_START=${name}"
    curl --noproxy '*' --fail --silent --show-error --max-time 60 \
      -X POST "${P_URL}/start_profile"
    profile_active=1
    replay "${name}" "${mode}" "${MAX_REQUESTS}" "${QPS}"
    curl --noproxy '*' --fail --silent --show-error --max-time 60 \
      -X POST "${P_URL}/stop_profile"
    profile_active=0
    echo "CAPTURE_DONE=${name}"
  done
  "${PYTHON_BIN}" "${SCRIPT_DIR}/summarize_stable_kv_push.py" \
    --early-run "${OUTPUT_DIR}/capture-$((repeat * 2 - (repeat == 2)))-early-qps-${QPS}-rep-${repeat}" \
    --finish-only-run "${OUTPUT_DIR}/capture-$((repeat * 2 - (repeat == 1)))-finish-only-qps-${QPS}-rep-${repeat}" \
    --output-dir "${OUTPUT_DIR}/comparison-rep-${repeat}"
done
