# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class PromptStreamStats:
    """保存并校验一次 Prompt Stream 请求的最终 token 统计。

    schema v1 字段继续保持必需，以便旧 P 节点仍能被 simple_test 解析。
    schema v2 将 policy admission、物理执行、recovery 与资源压力拆开；
    这些扩展字段在解析 v1 响应时统一为 ``None``。
    """

    metrics_schema_version: int
    coalescing_enabled: bool
    update_events_accepted: int
    update_works_admitted: int
    update_works_completed: int
    eager_update_prefill_tokens: int | None
    admitted_update_prefill_tokens: int
    scheduled_update_prefill_tokens: int
    saved_update_prefill_tokens: int | None
    extra_update_prefill_tokens: int | None
    update_tokens_invalidated: int
    update_policy_works_admitted: int | None = None
    update_policy_works_completed: int | None = None
    update_policy_works_cancelled_by_recovery: int | None = None
    update_policy_works_outstanding: int | None = None
    policy_admitted_update_prefill_tokens: int | None = None
    policy_saved_update_prefill_tokens: int | None = None
    policy_extra_update_prefill_tokens: int | None = None
    update_tokens_invalidated_by_snapshot: int | None = None
    update_execution_attempts_started: int | None = None
    update_execution_attempts_completed: int | None = None
    update_execution_attempts_preempted: int | None = None
    update_recovery_works_created: int | None = None
    update_tokens_scheduled_total: int | None = None
    update_tokens_scheduled_for_recovery: int | None = None
    prefix_append_tokens_scheduled_for_recovery: int | None = None
    update_tokens_discarded_by_preemption: int | None = None
    prefix_append_tokens_discarded_by_preemption: int | None = None
    question_tokens_discarded_by_preemption: int | None = None
    prompt_stream_running_preemptions: int | None = None
    prompt_stream_dormant_evictions: int | None = None
    prompt_stream_retained_waiting_evictions: int | None = None
    prompt_stream_prefix_cache_resets: int | None = None
    prompt_stream_allocation_failures_running: int | None = None
    prompt_stream_allocation_failures_waiting: int | None = None


def _require_non_negative_int(
    raw_stats: dict[str, Any],
    field_name: str,
) -> int:
    """读取一个必需的非负整数统计字段。

    Args:
        raw_stats: P 节点返回的原始统计对象。
        field_name: 需要读取的字段名。

    Returns:
        已校验的非负整数。

    Raises:
        ValueError: 字段缺失、类型错误或小于零时抛出。
    """
    value = raw_stats.get(field_name)
    # bool 是 int 的子类，因此使用精确类型检查，避免 True 被当作 1。
    if type(value) is not int or value < 0:
        raise ValueError(
            f"prompt_stream_stats.{field_name} must be a non-negative int, "
            f"but got {value!r}."
        )
    return value


def _require_optional_non_negative_int(
    raw_stats: dict[str, Any],
    field_name: str,
) -> int | None:
    """读取一个允许为 None 的非负整数统计字段。

    Args:
        raw_stats: P 节点返回的原始统计对象。
        field_name: 需要读取的字段名。

    Returns:
        已校验的非负整数或 None。

    Raises:
        ValueError: 字段缺失，或字段既不是 None 也不是非负整数时抛出。
    """
    if field_name not in raw_stats:
        raise ValueError(f"prompt_stream_stats is missing {field_name!r}.")
    value = raw_stats[field_name]
    if value is None:
        return None
    if type(value) is not int or value < 0:
        raise ValueError(
            f"prompt_stream_stats.{field_name} must be None or a "
            f"non-negative int, but got {value!r}."
        )
    return value


def _parse_metrics_schema_version(raw_stats: dict[str, Any]) -> int:
    """解析统计版本；没有显式版本的旧响应按 schema v1 处理。"""
    value = raw_stats.get("metrics_schema_version", 1)
    if type(value) is not int or value < 1:
        raise ValueError(
            "prompt_stream_stats.metrics_schema_version must be a positive "
            f"int, but got {value!r}."
        )
    return value


def _require_v2_non_negative_int(
    raw_stats: dict[str, Any],
    field_name: str,
    metrics_schema_version: int,
) -> int | None:
    """只在 schema v2 及以后读取一个必需的非负整数字段。"""
    if metrics_schema_version < 2:
        return None
    return _require_non_negative_int(raw_stats, field_name)


def _require_v2_optional_non_negative_int(
    raw_stats: dict[str, Any],
    field_name: str,
    metrics_schema_version: int,
) -> int | None:
    """只在 schema v2 及以后读取一个允许为 None 的整数字段。"""
    if metrics_schema_version < 2:
        return None
    return _require_optional_non_negative_int(raw_stats, field_name)


def _validate_counterfactual_tokens(stats: PromptStreamStats) -> None:
    """校验 eager 与 policy admitted 推导出的 saved/extra 统计。"""
    if stats.metrics_schema_version >= 2:
        admitted_tokens = stats.policy_admitted_update_prefill_tokens
        saved_tokens = stats.policy_saved_update_prefill_tokens
        extra_tokens = stats.policy_extra_update_prefill_tokens
        assert admitted_tokens is not None
    else:
        admitted_tokens = stats.admitted_update_prefill_tokens
        saved_tokens = stats.saved_update_prefill_tokens
        extra_tokens = stats.extra_update_prefill_tokens

    counterfactual_values = (
        stats.eager_update_prefill_tokens,
        saved_tokens,
        extra_tokens,
    )
    has_update_counterfactual = (
        stats.coalescing_enabled and stats.update_events_accepted > 0
    )
    if has_update_counterfactual and any(
        value is None for value in counterfactual_values
    ):
        raise ValueError(
            "Enabled Update coalescing with accepted Update events must "
            "return eager/saved/extra token statistics, but got "
            f"{counterfactual_values!r}."
        )
    if not has_update_counterfactual and any(
        value is not None for value in counterfactual_values
    ):
        raise ValueError(
            "Requests without an Update coalescing counterfactual must report "
            "eager/saved/extra as None, "
            f"but got {counterfactual_values!r}."
        )

    if any(value is None for value in counterfactual_values):
        return
    eager_tokens = stats.eager_update_prefill_tokens
    assert eager_tokens is not None
    assert saved_tokens is not None
    assert extra_tokens is not None
    expected_saved_tokens = max(0, eager_tokens - admitted_tokens)
    expected_extra_tokens = max(0, admitted_tokens - eager_tokens)
    if saved_tokens != expected_saved_tokens:
        raise ValueError(
            "saved Update prefill tokens are inconsistent with eager and "
            f"policy totals: reported={saved_tokens}, "
            f"expected={expected_saved_tokens}."
        )
    if extra_tokens != expected_extra_tokens:
        raise ValueError(
            "extra Update prefill tokens are inconsistent with eager and "
            f"policy totals: reported={extra_tokens}, "
            f"expected={expected_extra_tokens}."
        )


def _validate_schema_v2(stats: PromptStreamStats) -> None:
    """校验成功 finish 响应中的 schema-v2 lifecycle 守恒式。"""
    admitted = stats.update_policy_works_admitted
    completed = stats.update_policy_works_completed
    cancelled = stats.update_policy_works_cancelled_by_recovery
    outstanding = stats.update_policy_works_outstanding
    attempts_started = stats.update_execution_attempts_started
    attempts_completed = stats.update_execution_attempts_completed
    attempts_preempted = stats.update_execution_attempts_preempted
    scheduled_total = stats.update_tokens_scheduled_total
    scheduled_for_recovery = stats.update_tokens_scheduled_for_recovery
    policy_tokens = stats.policy_admitted_update_prefill_tokens
    invalidated_by_snapshot = stats.update_tokens_invalidated_by_snapshot
    required_values = (
        admitted,
        completed,
        cancelled,
        outstanding,
        attempts_started,
        attempts_completed,
        attempts_preempted,
        scheduled_total,
        scheduled_for_recovery,
        policy_tokens,
        invalidated_by_snapshot,
    )
    assert all(value is not None for value in required_values)
    assert admitted is not None
    assert completed is not None
    assert cancelled is not None
    assert outstanding is not None
    assert attempts_started is not None
    assert attempts_completed is not None
    assert attempts_preempted is not None
    assert scheduled_total is not None
    assert scheduled_for_recovery is not None
    assert policy_tokens is not None
    assert invalidated_by_snapshot is not None

    # v1 aliases 继续存在，但 schema v2 必须保证它们与新口径严格一致。
    alias_pairs = (
        ("update_works_admitted", stats.update_works_admitted, admitted),
        ("update_works_completed", stats.update_works_completed, completed),
        (
            "admitted_update_prefill_tokens",
            stats.admitted_update_prefill_tokens,
            policy_tokens,
        ),
        (
            "scheduled_update_prefill_tokens",
            stats.scheduled_update_prefill_tokens,
            scheduled_total,
        ),
        (
            "saved_update_prefill_tokens",
            stats.saved_update_prefill_tokens,
            stats.policy_saved_update_prefill_tokens,
        ),
        (
            "extra_update_prefill_tokens",
            stats.extra_update_prefill_tokens,
            stats.policy_extra_update_prefill_tokens,
        ),
        (
            "update_tokens_invalidated",
            stats.update_tokens_invalidated,
            invalidated_by_snapshot,
        ),
    )
    for alias_name, alias_value, canonical_value in alias_pairs:
        if alias_value != canonical_value:
            raise ValueError(
                f"Schema-v2 alias {alias_name} is inconsistent with its "
                f"canonical field: alias={alias_value}, "
                f"canonical={canonical_value}."
            )

    if admitted > stats.update_events_accepted:
        raise ValueError(
            "Policy-admitted Update work count cannot exceed accepted events: "
            f"admitted={admitted}, accepted={stats.update_events_accepted}."
        )
    if admitted != completed + cancelled + outstanding:
        raise ValueError(
            "Invalid Update policy lifecycle counters: "
            f"admitted={admitted}, completed={completed}, "
            f"cancelled_by_recovery={cancelled}, outstanding={outstanding}."
        )
    # simple_test 只在成功 finish 后解析，因此不能遗留未终结 admission。
    if outstanding != 0:
        raise ValueError(
            "Successful finish must not leave outstanding Update admissions: "
            f"outstanding={outstanding}."
        )
    if attempts_started != attempts_completed + attempts_preempted:
        raise ValueError(
            "Invalid Update execution attempt counters after successful finish: "
            f"started={attempts_started}, completed={attempts_completed}, "
            f"preempted={attempts_preempted}."
        )
    if scheduled_for_recovery > scheduled_total:
        raise ValueError(
            "Recovery-scheduled Update tokens cannot exceed all scheduled "
            f"Update tokens: recovery={scheduled_for_recovery}, "
            f"total={scheduled_total}."
        )


def parse_prompt_stream_stats(
    finish_response: dict[str, Any],
    *,
    num_sent_update_events: int,
) -> PromptStreamStats:
    """解析并交叉校验 finish 响应中的 Prompt Stream 统计。

    schema v1 保留旧校验路径；schema v2 使用 recovery-safe policy 与物理
    attempt 守恒式。simple_test 只对成功 finish 调用本接口，因此 v2 的
    outstanding admission 必须为零。

    Args:
        finish_response: proxy ``/v1/prompt_stream/finish`` 的 JSON 响应。
        num_sent_update_events: 当前测试实际发送的 Update 事件数量。

    Returns:
        经过类型、范围和生命周期不变量校验的统计对象。

    Raises:
        ValueError: 统计缺失、字段非法或跨字段关系不成立时抛出。
    """
    raw_stats = finish_response.get("prompt_stream_stats")
    if not isinstance(raw_stats, dict):
        raise ValueError(
            "Successful finish response must contain object-valued "
            f"prompt_stream_stats, but got {raw_stats!r}."
        )

    coalescing_enabled = raw_stats.get("coalescing_enabled")
    if type(coalescing_enabled) is not bool:
        raise ValueError(
            "prompt_stream_stats.coalescing_enabled must be bool, "
            f"but got {coalescing_enabled!r}."
        )
    metrics_schema_version = _parse_metrics_schema_version(raw_stats)

    stats = PromptStreamStats(
        metrics_schema_version=metrics_schema_version,
        coalescing_enabled=coalescing_enabled,
        update_events_accepted=_require_non_negative_int(
            raw_stats, "update_events_accepted"
        ),
        update_works_admitted=_require_non_negative_int(
            raw_stats, "update_works_admitted"
        ),
        update_works_completed=_require_non_negative_int(
            raw_stats, "update_works_completed"
        ),
        eager_update_prefill_tokens=_require_optional_non_negative_int(
            raw_stats, "eager_update_prefill_tokens"
        ),
        admitted_update_prefill_tokens=_require_non_negative_int(
            raw_stats, "admitted_update_prefill_tokens"
        ),
        scheduled_update_prefill_tokens=_require_non_negative_int(
            raw_stats, "scheduled_update_prefill_tokens"
        ),
        saved_update_prefill_tokens=_require_optional_non_negative_int(
            raw_stats, "saved_update_prefill_tokens"
        ),
        extra_update_prefill_tokens=_require_optional_non_negative_int(
            raw_stats, "extra_update_prefill_tokens"
        ),
        update_tokens_invalidated=_require_non_negative_int(
            raw_stats, "update_tokens_invalidated"
        ),
        update_policy_works_admitted=_require_v2_non_negative_int(
            raw_stats, "update_policy_works_admitted", metrics_schema_version
        ),
        update_policy_works_completed=_require_v2_non_negative_int(
            raw_stats, "update_policy_works_completed", metrics_schema_version
        ),
        update_policy_works_cancelled_by_recovery=(
            _require_v2_non_negative_int(
                raw_stats,
                "update_policy_works_cancelled_by_recovery",
                metrics_schema_version,
            )
        ),
        update_policy_works_outstanding=_require_v2_non_negative_int(
            raw_stats,
            "update_policy_works_outstanding",
            metrics_schema_version,
        ),
        policy_admitted_update_prefill_tokens=_require_v2_non_negative_int(
            raw_stats,
            "policy_admitted_update_prefill_tokens",
            metrics_schema_version,
        ),
        policy_saved_update_prefill_tokens=(
            _require_v2_optional_non_negative_int(
                raw_stats,
                "policy_saved_update_prefill_tokens",
                metrics_schema_version,
            )
        ),
        policy_extra_update_prefill_tokens=(
            _require_v2_optional_non_negative_int(
                raw_stats,
                "policy_extra_update_prefill_tokens",
                metrics_schema_version,
            )
        ),
        update_tokens_invalidated_by_snapshot=_require_v2_non_negative_int(
            raw_stats,
            "update_tokens_invalidated_by_snapshot",
            metrics_schema_version,
        ),
        update_execution_attempts_started=_require_v2_non_negative_int(
            raw_stats,
            "update_execution_attempts_started",
            metrics_schema_version,
        ),
        update_execution_attempts_completed=_require_v2_non_negative_int(
            raw_stats,
            "update_execution_attempts_completed",
            metrics_schema_version,
        ),
        update_execution_attempts_preempted=_require_v2_non_negative_int(
            raw_stats,
            "update_execution_attempts_preempted",
            metrics_schema_version,
        ),
        update_recovery_works_created=_require_v2_non_negative_int(
            raw_stats,
            "update_recovery_works_created",
            metrics_schema_version,
        ),
        update_tokens_scheduled_total=_require_v2_non_negative_int(
            raw_stats, "update_tokens_scheduled_total", metrics_schema_version
        ),
        update_tokens_scheduled_for_recovery=_require_v2_non_negative_int(
            raw_stats,
            "update_tokens_scheduled_for_recovery",
            metrics_schema_version,
        ),
        prefix_append_tokens_scheduled_for_recovery=(
            _require_v2_non_negative_int(
                raw_stats,
                "prefix_append_tokens_scheduled_for_recovery",
                metrics_schema_version,
            )
        ),
        update_tokens_discarded_by_preemption=_require_v2_non_negative_int(
            raw_stats,
            "update_tokens_discarded_by_preemption",
            metrics_schema_version,
        ),
        prefix_append_tokens_discarded_by_preemption=(
            _require_v2_non_negative_int(
                raw_stats,
                "prefix_append_tokens_discarded_by_preemption",
                metrics_schema_version,
            )
        ),
        question_tokens_discarded_by_preemption=_require_v2_non_negative_int(
            raw_stats,
            "question_tokens_discarded_by_preemption",
            metrics_schema_version,
        ),
        prompt_stream_running_preemptions=_require_v2_non_negative_int(
            raw_stats,
            "prompt_stream_running_preemptions",
            metrics_schema_version,
        ),
        prompt_stream_dormant_evictions=_require_v2_non_negative_int(
            raw_stats,
            "prompt_stream_dormant_evictions",
            metrics_schema_version,
        ),
        prompt_stream_retained_waiting_evictions=_require_v2_non_negative_int(
            raw_stats,
            "prompt_stream_retained_waiting_evictions",
            metrics_schema_version,
        ),
        prompt_stream_prefix_cache_resets=_require_v2_non_negative_int(
            raw_stats,
            "prompt_stream_prefix_cache_resets",
            metrics_schema_version,
        ),
        prompt_stream_allocation_failures_running=_require_v2_non_negative_int(
            raw_stats,
            "prompt_stream_allocation_failures_running",
            metrics_schema_version,
        ),
        prompt_stream_allocation_failures_waiting=_require_v2_non_negative_int(
            raw_stats,
            "prompt_stream_allocation_failures_waiting",
            metrics_schema_version,
        ),
    )

    if stats.update_events_accepted > num_sent_update_events:
        raise ValueError(
            "P node accepted more Update events than the client sent: "
            f"accepted={stats.update_events_accepted}, "
            f"sent={num_sent_update_events}."
        )

    if metrics_schema_version >= 2:
        _validate_schema_v2(stats)
    else:
        # v1 没有 recovery terminal 状态，只能保留无抢占时代的基础检查。
        if stats.update_works_admitted > stats.update_events_accepted:
            raise ValueError(
                "Admitted Update work count cannot exceed accepted events: "
                f"admitted={stats.update_works_admitted}, "
                f"accepted={stats.update_events_accepted}."
            )
        if stats.update_works_completed > stats.update_works_admitted:
            raise ValueError(
                "Completed Update work count cannot exceed admitted work count: "
                f"completed={stats.update_works_completed}, "
                f"admitted={stats.update_works_admitted}."
            )

    _validate_counterfactual_tokens(stats)
    return stats


def print_prompt_stream_stats(stats: PromptStreamStats) -> None:
    """单独打印已经解析和校验的 Prompt Stream 统计。

    Args:
        stats: ``parse_prompt_stream_stats`` 返回的不可变统计对象。
    """
    print("\n=== parsed prompt stream stats ===")
    print(json.dumps(asdict(stats), ensure_ascii=False, indent=2))
