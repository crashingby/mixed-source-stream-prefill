#!/usr/bin/env bash
#SBATCH --job-name=query-probe-perf
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --output=research/scripts/replay/query_probe/results/performance-%j.out

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
cd "${REPO_ROOT}"
mkdir -p "${REPO_ROOT}/tmp"
export TMPDIR="${REPO_ROOT}/tmp"
PYTHON_BIN="${PYTHON_BIN:-${REPO_ROOT}/.venv/bin/python}"
# 时限由提交命令按分区上限指定；脚本不额外设置 1 小时或 24 小时限制。
exec "${PYTHON_BIN}" research/scripts/replay/query_probe/run_performance_matrix.py "$@"
