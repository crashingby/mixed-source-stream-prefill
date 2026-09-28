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
  local soft_window_ms="${SOFT_WINDOW_MS:-20}"
  local hard_deadline_ms="${HARD_DEADLINE_MS:-80}"
  local ttft_deadline_ms="${TTFT_DEADLINE_MS:-null}"

  case "${profile}" in
    disabled)
      echo ""
      ;;
    immediate-controller)
      echo '{"update_policy":"immediate"}'
      ;;
    fixed-window)
      echo "{\"update_policy\":\"fixed_window\",\"soft_window_ms\":${soft_window_ms},\"hard_deadline_ms\":${hard_deadline_ms},\"ttft_deadline_ms\":${ttft_deadline_ms}}"
      ;;
    quiet-debounce)
      echo "{\"update_policy\":\"quiet_debounce\",\"soft_window_ms\":${soft_window_ms},\"hard_deadline_ms\":${hard_deadline_ms},\"ttft_deadline_ms\":${ttft_deadline_ms}}"
      ;;
    adaptive)
      # 执行背压策略没有时间窗口，只使用两个资源压力阈值。
      local token_pressure="${ADAPTIVE_TOKEN_PRESSURE_THRESHOLD:-1.0}"
      local slot_pressure="${ADAPTIVE_SLOT_PRESSURE_THRESHOLD:-1.0}"
      echo "{\"update_policy\":\"adaptive\",\"adaptive_token_pressure_threshold\":${token_pressure},\"adaptive_slot_pressure_threshold\":${slot_pressure}}"
      ;;
    full-merge)
      echo "{\"update_policy\":\"full_merge\"}"
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
