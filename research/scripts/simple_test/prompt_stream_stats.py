# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class PromptStreamStats:
    """保存并校验一次 Prompt Stream 请求的最终 token 统计。

    这些字段来自 P 节点的 Request/Scheduler 状态。simple_test 只负责解析，
    不根据前端文本或事件数量重新估算 token 数。
    """

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
    # bool 是 int 的子类，因此这里使用精确类型检查，避免 True 被当作 1。
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


def parse_prompt_stream_stats(
    finish_response: dict[str, Any],
    *,
    num_sent_update_events: int,
) -> PromptStreamStats:
    """解析并交叉校验 finish 响应中的 Prompt Stream 统计。

    成功的 simple_test 请求缺少统计时会直接失败，避免把未打通的响应链路
    误认为零开销。事件数量只用于检查服务端不可能接受超过客户端实际发送量
    的 Update；合并或幂等处理可以让 accepted 数量更少。

    Args:
        finish_response: proxy /v1/prompt_stream/finish 的 JSON 响应。
        num_sent_update_events: 当前测试实际发送的 Update 事件数量。

    Returns:
        经过类型、范围和基本生命周期不变量校验的统计对象。

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

    stats = PromptStreamStats(
        coalescing_enabled=coalescing_enabled,
        update_events_accepted=_require_non_negative_int(
            raw_stats,
            "update_events_accepted",
        ),
        update_works_admitted=_require_non_negative_int(
            raw_stats,
            "update_works_admitted",
        ),
        update_works_completed=_require_non_negative_int(
            raw_stats,
            "update_works_completed",
        ),
        eager_update_prefill_tokens=_require_optional_non_negative_int(
            raw_stats,
            "eager_update_prefill_tokens",
        ),
        admitted_update_prefill_tokens=_require_non_negative_int(
            raw_stats,
            "admitted_update_prefill_tokens",
        ),
        scheduled_update_prefill_tokens=_require_non_negative_int(
            raw_stats,
            "scheduled_update_prefill_tokens",
        ),
        saved_update_prefill_tokens=_require_optional_non_negative_int(
            raw_stats,
            "saved_update_prefill_tokens",
        ),
        extra_update_prefill_tokens=_require_optional_non_negative_int(
            raw_stats,
            "extra_update_prefill_tokens",
        ),
        update_tokens_invalidated=_require_non_negative_int(
            raw_stats,
            "update_tokens_invalidated",
        ),
    )

    if stats.update_events_accepted > num_sent_update_events:
        raise ValueError(
            "P node accepted more Update events than the client sent: "
            f"accepted={stats.update_events_accepted}, "
            f"sent={num_sent_update_events}."
        )
    if stats.update_works_admitted > stats.update_events_accepted:
        raise ValueError(
            "Admitted Update work count cannot exceed accepted Update events: "
            f"admitted={stats.update_works_admitted}, "
            f"accepted={stats.update_events_accepted}."
        )
    if stats.update_works_completed > stats.update_works_admitted:
        raise ValueError(
            "Completed Update work count cannot exceed admitted work count: "
            f"completed={stats.update_works_completed}, "
            f"admitted={stats.update_works_admitted}."
        )

    counterfactual_values = (
        stats.eager_update_prefill_tokens,
        stats.saved_update_prefill_tokens,
        stats.extra_update_prefill_tokens,
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

    if all(value is not None for value in counterfactual_values):
        eager_tokens = stats.eager_update_prefill_tokens
        saved_tokens = stats.saved_update_prefill_tokens
        extra_tokens = stats.extra_update_prefill_tokens
        assert eager_tokens is not None
        assert saved_tokens is not None
        assert extra_tokens is not None
        expected_saved_tokens = max(
            0,
            eager_tokens - stats.admitted_update_prefill_tokens,
        )
        expected_extra_tokens = max(
            0,
            stats.admitted_update_prefill_tokens - eager_tokens,
        )
        if saved_tokens != expected_saved_tokens:
            raise ValueError(
                "saved_update_prefill_tokens is inconsistent with eager and "
                "admitted totals: "
                f"reported={saved_tokens}, expected={expected_saved_tokens}."
            )
        if extra_tokens != expected_extra_tokens:
            raise ValueError(
                "extra_update_prefill_tokens is inconsistent with eager and "
                "admitted totals: "
                f"reported={extra_tokens}, expected={expected_extra_tokens}."
            )

    return stats


def print_prompt_stream_stats(stats: PromptStreamStats) -> None:
    """单独打印已经解析和校验的 Prompt Stream 统计。

    Args:
        stats: parse_prompt_stream_stats 返回的不可变统计对象。
    """
    print("\n=== parsed prompt stream stats ===")
    print(json.dumps(asdict(stats), ensure_ascii=False, indent=2))
