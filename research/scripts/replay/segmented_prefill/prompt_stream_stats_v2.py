# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""解析 recovery-aware Prompt Stream stats schema v2。"""

from __future__ import annotations

from typing import Any


class PromptStreamStatsV2Error(ValueError):
    """表示 schema v2 字段或跨字段不变量不成立。"""


def _require_non_negative_int(
    raw_stats: dict[str, Any],
    field_name: str,
) -> int:
    """读取必需的非负整数，拒绝 bool 伪装成整数。"""
    value = raw_stats.get(field_name)
    if type(value) is not int or value < 0:
        raise PromptStreamStatsV2Error(
            f"prompt_stream_stats.{field_name} must be a non-negative int, "
            f"but got {value!r}."
        )
    return value


def _optional_non_negative_int(
    raw_stats: dict[str, Any],
    field_name: str,
) -> int | None:
    """读取可选非负整数；字段缺失和显式 null 都返回 None。"""
    value = raw_stats.get(field_name)
    if value is None:
        return None
    if type(value) is not int or value < 0:
        raise PromptStreamStatsV2Error(
            f"prompt_stream_stats.{field_name} must be null or a "
            f"non-negative int, but got {value!r}."
        )
    return value


def read_metrics_schema_version(raw_stats: dict[str, Any]) -> int:
    """读取 metrics schema；未声明版本的历史响应按 v1 处理。"""
    value = raw_stats.get("metrics_schema_version", 1)
    if type(value) is not int or value not in (1, 2):
        raise PromptStreamStatsV2Error(
            "prompt_stream_stats.metrics_schema_version must be 1 or 2, "
            f"but got {value!r}."
        )
    return value


def apply_prompt_stream_stats_v2(
    metric: Any,
    raw_stats: dict[str, Any],
    *,
    num_sent_update_events: int,
    require_stats: bool,
    coalescing_enabled: bool,
) -> None:
    """校验 schema v2，并把 policy/physical/recovery 指标写入 metric。

    schema v2 将 source event、policy admission 和 physical execution 分开。
    因此这里不再比较 ``scheduled >= admitted``，也不再要求每个 admission
    都实际完成；latest-only recovery 可以显式取消旧 admission。

    Args:
        metric: ``RequestMetric`` 实例。使用 Any 避免与 replay_common 循环导入。
        raw_stats: finish SSE 返回的 ``prompt_stream_stats``。
        num_sent_update_events: 客户端收到 ACK 的 Update 数据事件数。
        require_stats: 是否按正式实验口径要求请求结束时无 outstanding work。
        coalescing_enabled: P 节点报告的 Update coalescer 开关。

    Raises:
        PromptStreamStatsV2Error: 字段类型、计数守恒或 token 派生值不一致。
    """
    metric.metrics_schema_version = 2
    metric.update_events_accepted = _require_non_negative_int(
        raw_stats, "update_events_accepted"
    )
    metric.update_policy_works_admitted = _require_non_negative_int(
        raw_stats, "update_policy_works_admitted"
    )
    metric.update_policy_works_completed = _require_non_negative_int(
        raw_stats, "update_policy_works_completed"
    )
    metric.update_policy_works_cancelled_by_recovery = (
        _require_non_negative_int(
            raw_stats,
            "update_policy_works_cancelled_by_recovery",
        )
    )
    metric.update_policy_works_outstanding = _require_non_negative_int(
        raw_stats, "update_policy_works_outstanding"
    )

    # 旧列继续映射到 policy 层，保证既有 CSV/绘图脚本仍能读取。
    metric.update_works_admitted = metric.update_policy_works_admitted
    metric.update_works_completed = metric.update_policy_works_completed

    metric.eager_update_prefill_tokens = _optional_non_negative_int(
        raw_stats, "eager_update_prefill_tokens"
    )
    metric.policy_admitted_update_prefill_tokens = _require_non_negative_int(
        raw_stats, "policy_admitted_update_prefill_tokens"
    )
    metric.policy_saved_update_prefill_tokens = _optional_non_negative_int(
        raw_stats, "policy_saved_update_prefill_tokens"
    )
    metric.policy_extra_update_prefill_tokens = _optional_non_negative_int(
        raw_stats, "policy_extra_update_prefill_tokens"
    )

    # 兼容列明确映射到 policy 值，不能再用它们推导物理 replay overhead。
    metric.admitted_update_prefill_tokens = (
        metric.policy_admitted_update_prefill_tokens
    )
    metric.saved_update_prefill_tokens = metric.policy_saved_update_prefill_tokens
    metric.extra_update_prefill_tokens = metric.policy_extra_update_prefill_tokens

    metric.update_execution_attempts_started = _require_non_negative_int(
        raw_stats, "update_execution_attempts_started"
    )
    metric.update_execution_attempts_completed = _require_non_negative_int(
        raw_stats, "update_execution_attempts_completed"
    )
    metric.update_execution_attempts_preempted = _require_non_negative_int(
        raw_stats, "update_execution_attempts_preempted"
    )
    metric.update_recovery_works_created = _require_non_negative_int(
        raw_stats, "update_recovery_works_created"
    )
    metric.update_tokens_scheduled_total = _require_non_negative_int(
        raw_stats, "update_tokens_scheduled_total"
    )
    metric.update_tokens_scheduled_for_recovery = _require_non_negative_int(
        raw_stats, "update_tokens_scheduled_for_recovery"
    )
    metric.prefix_append_tokens_scheduled_for_recovery = (
        _require_non_negative_int(
            raw_stats,
            "prefix_append_tokens_scheduled_for_recovery",
        )
    )
    metric.update_tokens_discarded_by_preemption = _require_non_negative_int(
        raw_stats, "update_tokens_discarded_by_preemption"
    )
    metric.prefix_append_tokens_discarded_by_preemption = (
        _require_non_negative_int(
            raw_stats,
            "prefix_append_tokens_discarded_by_preemption",
        )
    )
    metric.question_tokens_discarded_by_preemption = _require_non_negative_int(
        raw_stats, "question_tokens_discarded_by_preemption"
    )

    # 旧 scheduled 列表示所有 origin 的总调度量。
    metric.scheduled_update_prefill_tokens = metric.update_tokens_scheduled_total
    metric.scheduled_replay_overhead_tokens = (
        metric.update_tokens_scheduled_for_recovery
    )

    invalidated = raw_stats.get("update_tokens_invalidated_by_snapshot")
    if invalidated is None:
        invalidated = raw_stats.get("update_tokens_invalidated")
    if type(invalidated) is not int or invalidated < 0:
        raise PromptStreamStatsV2Error(
            "schema v2 requires non-negative "
            "update_tokens_invalidated_by_snapshot or its legacy alias."
        )
    metric.update_tokens_invalidated_by_snapshot = invalidated
    metric.update_tokens_invalidated = invalidated

    # schema v2 保留的 v1 别名必须与 canonical 字段严格一致。
    # 否则旧分析脚本与新脚本会对同一响应得出不同结论。
    alias_pairs = (
        ("update_works_admitted", metric.update_policy_works_admitted),
        ("update_works_completed", metric.update_policy_works_completed),
        (
            "admitted_update_prefill_tokens",
            metric.policy_admitted_update_prefill_tokens,
        ),
        (
            "scheduled_update_prefill_tokens",
            metric.update_tokens_scheduled_total,
        ),
        (
            "saved_update_prefill_tokens",
            metric.policy_saved_update_prefill_tokens,
        ),
        (
            "extra_update_prefill_tokens",
            metric.policy_extra_update_prefill_tokens,
        ),
        ("update_tokens_invalidated", invalidated),
    )
    for alias_name, canonical_value in alias_pairs:
        if alias_name not in raw_stats:
            raise PromptStreamStatsV2Error(
                f"schema v2 is missing compatibility alias {alias_name!r}."
            )
        alias_value = raw_stats[alias_name]
        if type(alias_value) is not type(canonical_value) or (
            alias_value != canonical_value
        ):
            raise PromptStreamStatsV2Error(
                f"Schema-v2 alias {alias_name} is inconsistent with its "
                f"canonical field: alias={alias_value!r}, "
                f"canonical={canonical_value!r}."
            )

    # Pressure counters 在没有对应 Scheduler 事件时为 0；schema v2 必须显式返回。
    pressure_fields = (
        "prompt_stream_running_preemptions",
        "prompt_stream_dormant_evictions",
        "prompt_stream_retained_waiting_evictions",
        "prompt_stream_allocation_failures_running",
        "prompt_stream_allocation_failures_waiting",
        "prompt_stream_prefix_cache_resets",
    )
    for field_name in pressure_fields:
        setattr(
            metric,
            field_name,
            _require_non_negative_int(raw_stats, field_name),
        )

    accepted = metric.update_events_accepted
    admitted = metric.update_policy_works_admitted
    completed = metric.update_policy_works_completed
    cancelled = metric.update_policy_works_cancelled_by_recovery
    outstanding = metric.update_policy_works_outstanding
    if accepted > num_sent_update_events:
        raise PromptStreamStatsV2Error(
            "P node accepted more Update events than the client ACKed: "
            f"accepted={accepted}, sent={num_sent_update_events}."
        )
    if admitted > accepted:
        raise PromptStreamStatsV2Error(
            "Policy admission cannot exceed accepted Update events: "
            f"accepted={accepted}, admitted={admitted}."
        )
    if admitted != completed + cancelled + outstanding:
        raise PromptStreamStatsV2Error(
            "Invalid policy admission conservation: "
            f"admitted={admitted}, completed={completed}, "
            f"cancelled={cancelled}, outstanding={outstanding}."
        )
    if require_stats and outstanding != 0:
        raise PromptStreamStatsV2Error(
            "A successful formal replay cannot retain outstanding Update "
            f"admissions: outstanding={outstanding}."
        )

    attempts_started = metric.update_execution_attempts_started
    attempts_completed = metric.update_execution_attempts_completed
    attempts_preempted = metric.update_execution_attempts_preempted
    if attempts_started != attempts_completed + attempts_preempted:
        raise PromptStreamStatsV2Error(
            "Invalid physical execution conservation: "
            f"started={attempts_started}, completed={attempts_completed}, "
            f"preempted={attempts_preempted}."
        )
    if (
        metric.update_tokens_scheduled_for_recovery
        > metric.update_tokens_scheduled_total
    ):
        raise PromptStreamStatsV2Error(
            "Recovery-scheduled Update tokens cannot exceed total scheduled "
            "Update tokens."
        )

    eager = metric.eager_update_prefill_tokens
    policy_tokens = metric.policy_admitted_update_prefill_tokens
    saved = metric.policy_saved_update_prefill_tokens
    extra = metric.policy_extra_update_prefill_tokens
    counterfactual_values = (eager, saved, extra)
    has_counterfactual = coalescing_enabled and accepted > 0
    if has_counterfactual and any(
        value is None for value in counterfactual_values
    ):
        raise PromptStreamStatsV2Error(
            "Enabled coalescing with Update events requires eager/policy "
            f"saved/extra counters: values={counterfactual_values!r}."
        )
    if not has_counterfactual and any(
        value is not None for value in counterfactual_values
    ):
        raise PromptStreamStatsV2Error(
            "A request without a coalescing counterfactual must return null "
            f"eager/saved/extra counters: values={counterfactual_values!r}."
        )
    if eager is not None and saved is not None and extra is not None:
        if saved != max(0, eager - policy_tokens):
            raise PromptStreamStatsV2Error(
                "policy_saved_update_prefill_tokens is inconsistent."
            )
        if extra != max(0, policy_tokens - eager):
            raise PromptStreamStatsV2Error(
                "policy_extra_update_prefill_tokens is inconsistent."
            )

    metric.absorbed_update_events = max(0, accepted - admitted)
    if accepted > 0:
        metric.coalescing_ratio = metric.absorbed_update_events / accepted
    if eager is not None and eager > 0 and saved is not None:
        metric.token_saving_ratio = saved / eager
    if admitted > 0:
        metric.update_work_completion_ratio = completed / admitted
