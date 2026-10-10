#!/usr/bin/env bash
# 诊断专用解释器包装器；仅 P 的 API 主进程进入 nsys，D 和显存检查不采集。
set -euo pipefail
REPO_ROOT="${REPO_ROOT:-$(pwd)}"
REAL_PYTHON="${REPO_ROOT}/.venv/bin/python"
export PATH="${REPO_ROOT}/.venv/bin:${PATH}"

if [[ "${SLURM_PROCID:-}" == "0" && "${1:-}" == "-m" && "${2:-}" == "vllm.entrypoints.openai.api_server" ]]; then
  NSYS_BIN="${NSYS_BIN:-/usr/local/cuda-12.4/bin/nsys}"
  NSYS_OUTPUT_DIR="${NSYS_OUTPUT_DIR:?Set an isolated profiling output directory}"
  mkdir -p "${NSYS_OUTPUT_DIR}"
  echo "NSYS_OUTPUT_DIR=${NSYS_OUTPUT_DIR} CAPTURE=cudaProfilerApi"
  # 预热时不采集；每次 /start_profile 到 /stop_profile 形成独立报告。
  # 不启用全系统采样，也不在 profile stop 时杀掉仍供下一轮使用的 P 服务。
  exec "${NSYS_BIN}" profile \
    --trace=cuda,nvtx \
    --sample=none \
    --cpuctxsw=none \
    --capture-range=cudaProfilerApi \
    --capture-range-end=repeat \
    --kill=none \
    --export=sqlite \
    --output="${NSYS_OUTPUT_DIR}/p-job-${SLURM_JOB_ID}" \
    "${REAL_PYTHON}" "$@" --profiler-config '{"profiler":"cuda"}'
fi

exec "${REAL_PYTHON}" "$@"
