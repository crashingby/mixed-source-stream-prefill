#!/usr/bin/env bash
# 在单个 Slurm allocation 内启动恰好一个 P 和一个 D，不分别重复抢 GPU。
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$(pwd)}"
cd "${REPO_ROOT}"
export REPO_ROOT
SERVER_NODE="${SERVER_NODE:-g2c9}"
SERVER_MEMORY="${SERVER_MEMORY:-24G}"
SERVER_CPUS_PER_TASK="${SERVER_CPUS_PER_TASK:-8}"
SERVER_TIME="${SERVER_TIME:-04:00:00}"
mkdir -p research/scripts/logs

# Slurm 为每个 task 绑定一张 GPU；不手写 CUDA_VISIBLE_DEVICES 覆盖调度器。
exec srun \
  --job-name=stable-kv-pd \
  --partition=gpu \
  --nodelist="${SERVER_NODE}" \
  --nodes=1 \
  --ntasks=2 \
  --ntasks-per-node=2 \
  --gpus-per-task=1 \
  --gpu-bind=single:1 \
  --cpus-per-task="${SERVER_CPUS_PER_TASK}" \
  --mem="${SERVER_MEMORY}" \
  --time="${SERVER_TIME}" \
  --kill-on-bad-exit=1 \
  --output="${REPO_ROOT}/research/scripts/logs/stable-kv-pd-%j-rank-%t.out" \
  --error="${REPO_ROOT}/research/scripts/logs/stable-kv-pd-%j-rank-%t.out" \
  bash "${REPO_ROOT}/research/scripts/server/start_stable_kv_node.sh"
