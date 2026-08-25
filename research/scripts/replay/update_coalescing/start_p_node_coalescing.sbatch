#!/usr/bin/env bash
#SBATCH --job-name=vllm-p-coalescing
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=research/scripts/logs/vllm-p-coalescing-%j.out

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
cd "${REPO_ROOT}"

PROFILE_HELPER="${REPO_ROOT}/research/scripts/replay/update_coalescing/coalescing_profiles.sh"

# shellcheck source=/dev/null
source "${PROFILE_HELPER}"

COALESCING_PROFILE="${COALESCING_PROFILE:-fixed-window}"
PROMPT_STREAM_UPDATE_COALESCING_CONFIG="$(build_coalescing_config)"
export COALESCING_PROFILE
export PROMPT_STREAM_UPDATE_COALESCING_CONFIG

case "${COALESCING_PROFILE}" in
  fixed-window|quiet-debounce|adaptive)
    soft_window_ms="${SOFT_WINDOW_MS:-20}"
    hard_deadline_ms="${HARD_DEADLINE_MS:-80}"
    if ! awk \
      -v soft="${soft_window_ms}" \
      -v hard="${hard_deadline_ms}" \
      'BEGIN { exit !(soft <= hard) }'; then
      echo "soft_window_ms must not exceed hard_deadline_ms: " \
        "soft=${soft_window_ms}, hard=${hard_deadline_ms}." >&2
      exit 2
    fi
    ;;
esac

echo "COALESCING_PROFILE=${COALESCING_PROFILE}"
echo "PROMPT_STREAM_UPDATE_COALESCING_CONFIG=${PROMPT_STREAM_UPDATE_COALESCING_CONFIG:-disabled}"

# 复用正式 P 节点启动脚本；其 #SBATCH 行在当前 allocation 内作为注释忽略。
exec bash "${REPO_ROOT}/research/scripts/server/start_p_node.sbatch"
