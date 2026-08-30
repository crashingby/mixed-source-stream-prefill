#!/usr/bin/env bash

# 统一生成 P 节点启动 JSON 与 replay 记录标签，避免两处手写配置漂移。

coalescing_expected_state() {
  local profile="${COALESCING_PROFILE:-fixed-window}"
  if [[ "${profile}" == "disabled" ]]; then
    echo "disabled"
    return
  fi
  echo "enabled"
}

build_coalescing_config() {
  local profile="${COALESCING_PROFILE:-fixed-window}"
  local append_gate="${APPEND_GATE:-hold_until_append_finished}"
  local soft_window_ms="${SOFT_WINDOW_MS:-20}"
  local hard_deadline_ms="${HARD_DEADLINE_MS:-80}"
  local ttft_deadline_ms="${TTFT_DEADLINE_MS:-null}"

  case "${profile}" in
    disabled)
      echo ""
      ;;
    immediate-controller)
      echo '{"append_gate":"none","update_policy":"immediate"}'
      ;;
    strict-immediate)
      echo "{\"append_gate\":\"${append_gate}\",\"update_policy\":\"immediate\"}"
      ;;
    fixed-window)
      echo "{\"append_gate\":\"${append_gate}\",\"update_policy\":\"fixed_window\",\"soft_window_ms\":${soft_window_ms},\"hard_deadline_ms\":${hard_deadline_ms},\"ttft_deadline_ms\":${ttft_deadline_ms}}"
      ;;
    quiet-debounce)
      echo "{\"append_gate\":\"${append_gate}\",\"update_policy\":\"quiet_debounce\",\"soft_window_ms\":${soft_window_ms},\"hard_deadline_ms\":${hard_deadline_ms},\"ttft_deadline_ms\":${ttft_deadline_ms}}"
      ;;
    adaptive)
      local interval_alpha="${INTERVAL_EWMA_ALPHA:-0.25}"
      local gain_alpha="${GAIN_EWMA_ALPHA:-0.25}"
      local prior_weight="${COLD_START_PRIOR_WEIGHT:-8}"
      local arrival_prior="${ARRIVAL_RATE_PRIOR_PER_S:-null}"
      local gain_prior="${MERGE_GAIN_PRIOR_TOKENS:-null}"
      local delay_price="${DELAY_PRICE:-1}"
      local load_price_scale="${LOAD_PRICE_SCALE:-0}"
      echo "{\"append_gate\":\"${append_gate}\",\"update_policy\":\"adaptive\",\"soft_window_ms\":${soft_window_ms},\"hard_deadline_ms\":${hard_deadline_ms},\"ttft_deadline_ms\":${ttft_deadline_ms},\"interval_ewma_alpha\":${interval_alpha},\"gain_ewma_alpha\":${gain_alpha},\"cold_start_prior_weight\":${prior_weight},\"arrival_rate_prior_per_s\":${arrival_prior},\"merge_gain_prior_tokens\":${gain_prior},\"delay_price\":${delay_price},\"load_price_scale\":${load_price_scale}}"
      ;;
    full-merge)
      echo "{\"append_gate\":\"${append_gate}\",\"update_policy\":\"full_merge\"}"
      ;;
    custom)
      if [[ -z "${PROMPT_STREAM_UPDATE_COALESCING_CONFIG:-}" ]]; then
        echo "COALESCING_PROFILE=custom requires PROMPT_STREAM_UPDATE_COALESCING_CONFIG." >&2
        return 2
      fi
      echo "${PROMPT_STREAM_UPDATE_COALESCING_CONFIG}"
      ;;
    *)
      echo "Unsupported COALESCING_PROFILE=${profile}." >&2
      return 2
      ;;
  esac
}
