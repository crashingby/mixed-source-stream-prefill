# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Prompt Stream replay 状态拆分与旧结果兼容测试。"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[2]
_SEGMENTED_REPLAY_DIR = (
    _REPO_ROOT / "research/scripts/replay/segmented_prefill"
)
_COALESCING_REPLAY_DIR = (
    _REPO_ROOT / "research/scripts/replay/update_coalescing"
)
sys.path.insert(0, str(_SEGMENTED_REPLAY_DIR))
sys.path.insert(0, str(_COALESCING_REPLAY_DIR))

from replay_common import (  # noqa: E402
    RequestMetric,
    _summarize_metrics,
    apply_prompt_stream_stats,
)
from summarize_coalescing import _summarize_rows  # noqa: E402


def _make_metric() -> RequestMetric:
    """创建一个已经收到首 token 的最小 stream 指标。"""
    return RequestMetric(
        run_id="run",
        mode="stream",
        sample_id="sample",
        query_id="query",
        sample_status="complete",
        result_status="ok",
        planned_arrival_offset_ms=0.0,
        queue_delay_ms=0.0,
        request_start_wall_time=0.0,
        transport_status="ok",
        first_token_received=True,
        finish_to_first_token_ms=12.5,
        completion_chunks=1,
    )


def _apply_stats(
    metric: RequestMetric,
    raw_stats: dict[str, Any],
    *,
    require_stats: bool = True,
) -> None:
    """以 disabled P 节点配置解析测试统计。"""
    apply_prompt_stream_stats(
        metric,
        raw_stats,
        num_sent_update_events=2,
        require_stats=require_stats,
        expected_coalescing="disabled",
        expected_config_json=None,
    )


def test_stats_error_does_not_remove_first_token_ttft() -> None:
    """schema v2 非法时仍保留已经完成的 transport 与 TTFT。"""
    metric = _make_metric()
    _apply_stats(
        metric,
        {
            "metrics_schema_version": 2,
            "coalescing_enabled": False,
        },
    )

    assert metric.result_status == "ok"
    assert metric.transport_status == "ok"
    assert metric.first_token_received
    assert metric.stats_validation_status == "invalid"
    assert metric.formal_result_status == "invalid"

    summary = _summarize_metrics([metric])["stream"]
    assert summary["num_first_token_success"] == 1
    assert summary["num_stats_valid"] == 0
    assert summary["ttft_ms"]["p50"] == 12.5


def test_schema_v1_recovery_mismatch_is_legacy_incompatible() -> None:
    """旧 lifecycle 无法表达 recovery 时不把它伪装成传输失败。"""
    metric = _make_metric()
    _apply_stats(
        metric,
        {
            "coalescing_enabled": False,
            "update_events_accepted": 2,
            "update_works_admitted": 2,
            "update_works_completed": 1,
            "eager_update_prefill_tokens": None,
            "admitted_update_prefill_tokens": 10,
            "scheduled_update_prefill_tokens": 10,
            "saved_update_prefill_tokens": None,
            "extra_update_prefill_tokens": None,
            "update_tokens_invalidated": 0,
        },
    )

    assert metric.metrics_schema_version == 1
    assert metric.stats_validation_status == "legacy_incompatible"
    assert metric.formal_result_status == "not_evaluated"
    assert metric.result_status == "ok"


def test_old_csv_first_token_proves_transport_success() -> None:
    """旧 stats error 行可恢复 TTFT，并不再低估 transport success。"""
    rows = [
        {
            "result_status": "error",
            "finish_to_first_token_ms": "9.5",
            "request_to_first_token_ms": "12.0",
            "request_start_wall_time": "100.0",
            "request_to_done_ms": "12.0",
            "completion_chunks": "1",
        }
    ]

    summary = _summarize_rows(rows, observation_duration_s=1.0)
    assert summary["num_transport_success"] == 1
    assert summary["num_transport_error"] == 0
    assert summary["num_first_token_success"] == 1
    assert summary["num_stats_valid"] == 0
    assert summary["ttft_p50_ms"] == 9.5


def _valid_recovery_stats_v2() -> dict[str, Any]:
    """构造包含一次抢占取消的合法 schema-v2 统计。"""
    return {
        "metrics_schema_version": 2,
        "coalescing_enabled": False,
        "update_coalescing_config_json": None,
        "update_events_accepted": 2,
        "update_policy_works_admitted": 2,
        "update_policy_works_completed": 1,
        "update_policy_works_cancelled_by_recovery": 1,
        "update_policy_works_outstanding": 0,
        "update_works_admitted": 2,
        "update_works_completed": 1,
        "eager_update_prefill_tokens": None,
        "policy_admitted_update_prefill_tokens": 10,
        "policy_saved_update_prefill_tokens": None,
        "policy_extra_update_prefill_tokens": None,
        "admitted_update_prefill_tokens": 10,
        "scheduled_update_prefill_tokens": 15,
        "saved_update_prefill_tokens": None,
        "extra_update_prefill_tokens": None,
        "update_tokens_invalidated": 2,
        "update_tokens_invalidated_by_snapshot": 2,
        "update_execution_attempts_started": 2,
        "update_execution_attempts_completed": 1,
        "update_execution_attempts_preempted": 1,
        "update_recovery_works_created": 1,
        "update_tokens_scheduled_total": 15,
        "update_tokens_scheduled_for_recovery": 5,
        "prefix_append_tokens_scheduled_for_recovery": 4,
        "update_tokens_discarded_by_preemption": 3,
        "prefix_append_tokens_discarded_by_preemption": 4,
        "question_tokens_discarded_by_preemption": 0,
        "prompt_stream_running_preemptions": 1,
        "prompt_stream_dormant_evictions": 0,
        "prompt_stream_retained_waiting_evictions": 0,
        "prompt_stream_prefix_cache_resets": 0,
        "prompt_stream_allocation_failures_running": 1,
        "prompt_stream_allocation_failures_waiting": 0,
    }


def test_schema_v2_accepts_recovery_aware_conservation() -> None:
    """cancelled admission 与 preempted attempt 可以同时合法存在。"""
    metric = _make_metric()
    _apply_stats(metric, _valid_recovery_stats_v2())

    assert metric.stats_validation_status == "valid"
    assert metric.formal_result_status == "valid"
    assert metric.update_policy_works_cancelled_by_recovery == 1
    assert metric.update_execution_attempts_preempted == 1
    assert metric.update_tokens_scheduled_for_recovery == 5


def test_schema_v2_rejects_inconsistent_compatibility_alias() -> None:
    """canonical 字段与旧别名分歧时不得进入正式聚合。"""
    metric = _make_metric()
    raw_stats = _valid_recovery_stats_v2()
    raw_stats["scheduled_update_prefill_tokens"] = 14
    _apply_stats(metric, raw_stats)

    assert metric.result_status == "ok"
    assert metric.stats_validation_status == "invalid"
    assert metric.formal_result_status == "invalid"
