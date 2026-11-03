# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""实验指标语义、统计守恒与汇总。"""

from __future__ import annotations

import json
import logging
import math
import statistics
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import asdict
from typing import Any

from .load import LOAD_METRIC_FIELDS
from .models import RequestMetric, SseMeasurement
from .stats_v2 import (
    PromptStreamStatsV2Error,
    apply_prompt_stream_stats_v2,
    read_metrics_schema_version,
)
from .transport import LegacyStatsSemanticsIncompatible, ReplayHttpError

logger = logging.getLogger(__name__)


REQUEST_METRIC_FIELDS = (
    *LOAD_METRIC_FIELDS,
    "run_id",
    "mode",
    "sample_id",
    "query_id",
    "sample_status",
    "result_status",
    "error",
    "transport_status",
    "first_token_received",
    "stats_validation_status",
    "stats_validation_error",
    "formal_result_status",
    "request_error_stage",
    "metrics_schema_version",
    "planned_arrival_offset_ms",
    "queue_delay_ms",
    "request_start_wall_time",
    "start_ack_ms",
    "source_dispatch_ms",
    "finish_to_first_token_ms",
    "request_to_first_token_ms",
    "finish_to_done_ms",
    "request_to_done_ms",
    "ttft_origin",
    "num_append_chunks",
    "num_update_snapshots",
    "num_event_acks",
    "append_chars",
    "final_update_chars",
    "normal_prompt_chars",
    "completion_chars",
    "completion_chunks",
    "completion_tokens",
    "completion_token_count_exact",
    "time_to_last_token_ms",
    "mean_time_per_output_token_ms",
    "output_tokens_per_second",
    "masked_padding_expected",
    "masked_padding_observed",
    "masked_padding_validation_status",
    "masked_padding_tokens",
    "coalescing_enabled",
    "actual_p_node_coalescing_config_json",
    "update_events_accepted",
    "update_works_admitted",
    "update_works_completed",
    "update_policy_works_admitted",
    "update_policy_works_completed",
    "update_policy_works_cancelled_by_recovery",
    "update_policy_works_outstanding",
    "eager_update_prefill_tokens",
    "admitted_update_prefill_tokens",
    "scheduled_update_prefill_tokens",
    "saved_update_prefill_tokens",
    "extra_update_prefill_tokens",
    "policy_admitted_update_prefill_tokens",
    "policy_saved_update_prefill_tokens",
    "policy_extra_update_prefill_tokens",
    "update_execution_attempts_started",
    "update_execution_attempts_completed",
    "update_execution_attempts_preempted",
    "update_recovery_works_created",
    "update_tokens_scheduled_total",
    "update_tokens_scheduled_for_recovery",
    "prefix_append_tokens_scheduled_for_recovery",
    "update_tokens_discarded_by_preemption",
    "prefix_append_tokens_discarded_by_preemption",
    "question_tokens_discarded_by_preemption",
    "prompt_stream_running_preemptions",
    "prompt_stream_dormant_evictions",
    "prompt_stream_retained_waiting_evictions",
    "prompt_stream_allocation_failures_running",
    "prompt_stream_allocation_failures_waiting",
    "prompt_stream_prefix_cache_resets",
    "update_tokens_invalidated_by_snapshot",
    "update_tokens_invalidated",
    "absorbed_update_events",
    "coalescing_ratio",
    "token_saving_ratio",
    "update_work_completion_ratio",
    "scheduled_replay_overhead_tokens",
    "mean_update_lcp_ratio",
)


EVENT_METRIC_FIELDS = (
    "run_id",
    "mode",
    "sample_id",
    "query_id",
    "source_id",
    "source_type",
    "event_type",
    "chunk_id",
    "sequence_no",
    "version",
    "scheduled_offset_ms",
    "actual_send_offset_ms",
    "ack_latency_ms",
    "num_event_tokens",
    "update_lcp_tokens",
    "update_lcp_ratio",
    "result_status",
    "error",
)


def require_non_negative_stat(
    raw_stats: dict[str, Any],
    field_name: str,
) -> int:
    """读取服务端返回的必需非负整数统计字段。"""
    value = raw_stats.get(field_name)
    # bool 是 int 的子类，精确类型检查可避免把 True 当作一个 token。
    if type(value) is not int or value < 0:
        raise ReplayHttpError(
            f"prompt_stream_stats.{field_name} must be a non-negative int, "
            f"but got {value!r}."
        )
    return value


def require_optional_non_negative_stat(
    raw_stats: dict[str, Any],
    field_name: str,
) -> int | None:
    """读取允许为 None 的非负整数统计字段。"""
    if field_name not in raw_stats:
        raise ReplayHttpError(f"prompt_stream_stats is missing {field_name!r}.")
    value = raw_stats[field_name]
    if value is None:
        return None
    if type(value) is not int or value < 0:
        raise ReplayHttpError(
            f"prompt_stream_stats.{field_name} must be None or a "
            f"non-negative int, but got {value!r}."
        )
    return value


def apply_prompt_stream_stats_or_raise(
    metric: RequestMetric,
    raw_stats: dict[str, Any] | None,
    *,
    num_sent_update_events: int,
    require_stats: bool,
    expected_coalescing: str,
    expected_config_json: str | None,
) -> None:
    """校验并写入一次成功 finish 返回的 Prompt Stream 统计。

    Args:
        metric: 将被补充服务端原始统计与派生比率的请求指标。
        raw_stats: proxy 首个 SSE payload 中的 ``prompt_stream_stats``。
        num_sent_update_events: 客户端已收到 ACK 的 Update 数据事件数。
        require_stats: 缺少统计时是否将本次实验判为失败。
        expected_coalescing: ``enabled``、``disabled`` 或 ``any``，用于防止
            正式实验误连到配置不匹配的 P 节点。
        expected_config_json: replay 声明的 P 启动配置；逐字段与 P 实际返回的
            规范化配置比较。

    Raises:
        ReplayHttpError: 统计缺失、字段非法或跨字段关系不成立时抛出。
    """
    if raw_stats is None:
        if require_stats:
            raise ReplayHttpError(
                "Successful prompt-stream finish did not return prompt_stream_stats."
            )
        return

    coalescing_enabled = raw_stats.get("coalescing_enabled")
    if type(coalescing_enabled) is not bool:
        raise ReplayHttpError(
            "prompt_stream_stats.coalescing_enabled must be bool, "
            f"but got {coalescing_enabled!r}."
        )
    if expected_coalescing != "any":
        expected_enabled = expected_coalescing == "enabled"
        if coalescing_enabled != expected_enabled:
            raise ReplayHttpError(
                "P node coalescing state does not match the experiment: "
                f"expected={expected_coalescing}, "
                f"actual={coalescing_enabled}."
            )

    raw_actual_config = raw_stats.get("update_coalescing_config_json")
    actual_config: dict[str, Any] | None = None
    if coalescing_enabled:
        if not isinstance(raw_actual_config, str):
            raise ReplayHttpError(
                "An enabled coalescer must return string-valued "
                "update_coalescing_config_json."
            )
        try:
            parsed_actual_config = json.loads(raw_actual_config)
        except json.JSONDecodeError as exc:
            raise ReplayHttpError(
                "P node returned invalid update_coalescing_config_json."
            ) from exc
        if not isinstance(parsed_actual_config, dict):
            raise ReplayHttpError(
                "P-node update_coalescing_config_json must encode an object."
            )
        actual_config = parsed_actual_config
        metric.actual_p_node_coalescing_config_json = json.dumps(
            actual_config,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
    elif raw_actual_config is not None:
        raise ReplayHttpError(
            "A disabled coalescer must return null update_coalescing_config_json."
        )

    if expected_config_json is not None:
        try:
            expected_config = json.loads(expected_config_json)
        except json.JSONDecodeError as exc:
            raise ReplayHttpError(
                "Expected coalescing config is invalid JSON."
            ) from exc
        if not isinstance(expected_config, dict) or actual_config is None:
            raise ReplayHttpError(
                "Expected and actual coalescing configs must both be objects."
            )
        mismatched_fields = {
            field_name: {
                "expected": expected_value,
                "actual": actual_config.get(field_name),
            }
            for field_name, expected_value in expected_config.items()
            if actual_config.get(field_name) != expected_value
        }
        if mismatched_fields:
            raise ReplayHttpError(
                "P-node coalescing config does not match replay metadata: "
                f"{mismatched_fields!r}."
            )

    metric.coalescing_enabled = coalescing_enabled
    if metric.metrics_schema_version == 2:
        apply_prompt_stream_stats_v2(
            metric,
            raw_stats,
            num_sent_update_events=num_sent_update_events,
            require_stats=require_stats,
            coalescing_enabled=coalescing_enabled,
        )
        return

    # 未声明 metrics schema 的历史响应继续按 v1 严格口径解析。
    metric.update_events_accepted = require_non_negative_stat(
        raw_stats, "update_events_accepted"
    )
    metric.update_works_admitted = require_non_negative_stat(
        raw_stats, "update_works_admitted"
    )
    metric.update_works_completed = require_non_negative_stat(
        raw_stats, "update_works_completed"
    )
    metric.eager_update_prefill_tokens = require_optional_non_negative_stat(
        raw_stats, "eager_update_prefill_tokens"
    )
    metric.admitted_update_prefill_tokens = require_non_negative_stat(
        raw_stats, "admitted_update_prefill_tokens"
    )
    metric.scheduled_update_prefill_tokens = require_non_negative_stat(
        raw_stats, "scheduled_update_prefill_tokens"
    )
    metric.saved_update_prefill_tokens = require_optional_non_negative_stat(
        raw_stats, "saved_update_prefill_tokens"
    )
    metric.extra_update_prefill_tokens = require_optional_non_negative_stat(
        raw_stats, "extra_update_prefill_tokens"
    )
    metric.update_tokens_invalidated = require_non_negative_stat(
        raw_stats, "update_tokens_invalidated"
    )

    accepted = metric.update_events_accepted
    admitted = metric.update_works_admitted
    completed = metric.update_works_completed
    assert accepted is not None and admitted is not None and completed is not None
    if accepted > num_sent_update_events:
        raise ReplayHttpError(
            "P node accepted more Update events than the client ACKed: "
            f"accepted={accepted}, sent={num_sent_update_events}."
        )
    if admitted > accepted or completed > admitted:
        raise ReplayHttpError(
            "Invalid Update work lifecycle counters: "
            f"accepted={accepted}, admitted={admitted}, completed={completed}."
        )
    if require_stats and completed != admitted:
        raise LegacyStatsSemanticsIncompatible(
            "A successful formal replay must finish every admitted Update work: "
            f"admitted={admitted}, completed={completed}."
        )

    counterfactual_values = (
        metric.eager_update_prefill_tokens,
        metric.saved_update_prefill_tokens,
        metric.extra_update_prefill_tokens,
    )
    has_counterfactual = coalescing_enabled and accepted > 0
    if has_counterfactual and any(value is None for value in counterfactual_values):
        raise ReplayHttpError(
            "An enabled coalescer with Update events must return integer "
            f"eager/saved/extra values: values={counterfactual_values!r}."
        )
    if not has_counterfactual and any(
        value is not None for value in counterfactual_values
    ):
        raise ReplayHttpError(
            "Requests without a coalescing counterfactual must return null "
            "eager/saved/extra values: "
            f"values={counterfactual_values!r}."
        )

    eager = metric.eager_update_prefill_tokens
    admitted_tokens = metric.admitted_update_prefill_tokens
    scheduled_tokens = metric.scheduled_update_prefill_tokens
    saved = metric.saved_update_prefill_tokens
    extra = metric.extra_update_prefill_tokens
    assert admitted_tokens is not None and scheduled_tokens is not None
    if scheduled_tokens < admitted_tokens:
        raise LegacyStatsSemanticsIncompatible(
            "Scheduled Update tokens cannot be smaller than admitted tokens: "
            f"scheduled={scheduled_tokens}, admitted={admitted_tokens}."
        )
    if eager is not None and saved is not None and extra is not None:
        if saved != max(0, eager - admitted_tokens):
            raise ReplayHttpError("saved_update_prefill_tokens is inconsistent.")
        if extra != max(0, admitted_tokens - eager):
            raise ReplayHttpError("extra_update_prefill_tokens is inconsistent.")

    metric.absorbed_update_events = max(0, accepted - admitted)
    if accepted > 0:
        metric.coalescing_ratio = metric.absorbed_update_events / accepted
    if eager is not None and eager > 0 and saved is not None:
        metric.token_saving_ratio = saved / eager
    if admitted > 0:
        metric.update_work_completion_ratio = completed / admitted
    metric.scheduled_replay_overhead_tokens = max(0, scheduled_tokens - admitted_tokens)


def apply_prompt_stream_stats(
    metric: RequestMetric,
    raw_stats: dict[str, Any] | None,
    *,
    num_sent_update_events: int,
    require_stats: bool,
    expected_coalescing: str,
    expected_config_json: str | None,
) -> None:
    """结构化校验 finish stats，且不覆盖已经成功的首 token 链路。

    schema v2 使用 recovery-aware 守恒式。未声明版本的旧响应仍走 v1
    校验，但标记为 ``legacy``，避免把无法区分抢占的 token 数据纳入新的
    正式结论。所有统计错误都写入 metric，而不会伪装成 HTTP/SSE 错误。

    Args:
        metric: 已记录首 token 的请求指标。
        raw_stats: finish SSE 携带的 Prompt Stream stats。
        num_sent_update_events: 客户端收到 ACK 的 Update 数据事件数量。
        require_stats: 是否要求 stats 存在且通过正式守恒校验。
        expected_coalescing: replay 期望的 P 节点开关状态。
        expected_config_json: replay 期望的 P 节点合并配置。
    """
    if raw_stats is None:
        metric.stats_validation_status = "missing"
        metric.stats_validation_error = (
            "Successful prompt-stream finish did not return prompt_stream_stats."
        )
        metric.formal_result_status = "invalid" if require_stats else "not_evaluated"
        return

    try:
        metric.metrics_schema_version = read_metrics_schema_version(raw_stats)
        apply_prompt_stream_stats_or_raise(
            metric,
            raw_stats,
            num_sent_update_events=num_sent_update_events,
            require_stats=require_stats,
            expected_coalescing=expected_coalescing,
            expected_config_json=expected_config_json,
        )
    except LegacyStatsSemanticsIncompatible as exc:
        metric.stats_validation_status = "legacy_incompatible"
        metric.stats_validation_error = str(exc)
        metric.formal_result_status = "not_evaluated"
        return
    except (ReplayHttpError, PromptStreamStatsV2Error) as exc:
        metric.stats_validation_status = "invalid"
        metric.stats_validation_error = str(exc)
        metric.formal_result_status = "invalid"
        return

    if metric.metrics_schema_version == 2:
        metric.stats_validation_status = "valid"
        metric.formal_result_status = "valid"
    else:
        # v1 无法表达 recovery cancellation/attempt，保留但不纳入 v2 token 聚合。
        metric.stats_validation_status = "legacy"
        metric.formal_result_status = "not_evaluated"


def percentile(values: Sequence[float], percentile: float) -> float | None:
    """计算线性插值百分位数；空序列返回 None。"""
    if not values:
        return None
    ordered_values = sorted(values)
    if len(ordered_values) == 1:
        return ordered_values[0]
    position = (len(ordered_values) - 1) * percentile / 100.0
    lower_index = math.floor(position)
    upper_index = math.ceil(position)
    lower_value = ordered_values[lower_index]
    upper_value = ordered_values[upper_index]
    interpolation = position - lower_index
    return lower_value + (upper_value - lower_value) * interpolation


def summarize_metrics(metrics: Sequence[RequestMetric]) -> dict[str, Any]:
    """汇总各模式成功请求的 TTFT 分布与错误数量。"""
    summary_by_mode: dict[str, dict[str, Any]] = {}
    metrics_by_mode: dict[str, list[RequestMetric]] = defaultdict(list)
    for metric in metrics:
        metrics_by_mode[metric.mode].append(metric)

    for mode, mode_metrics in metrics_by_mode.items():
        successful_ttfts = [
            metric.finish_to_first_token_ms
            for metric in mode_metrics
            if metric.first_token_received
            and metric.finish_to_first_token_ms is not None
        ]
        numeric_ttfts = [float(ttft) for ttft in successful_ttfts]
        numeric_tpots = [
            float(metric.mean_time_per_output_token_ms)
            for metric in mode_metrics
            if metric.completion_token_count_exact
            and metric.mean_time_per_output_token_ms is not None
        ]
        numeric_output_tps = [
            float(metric.output_tokens_per_second)
            for metric in mode_metrics
            if metric.completion_token_count_exact
            and metric.output_tokens_per_second is not None
        ]
        total_exact_output_tokens = sum(
            metric.completion_tokens
            for metric in mode_metrics
            if metric.completion_token_count_exact
        )
        transport_success_metrics = [
            metric for metric in mode_metrics if metric.transport_status == "ok"
        ]
        stats_metrics = [
            metric
            for metric in mode_metrics
            if metric.stats_validation_status == "valid"
        ]
        formal_metrics = [
            metric for metric in mode_metrics if metric.formal_result_status == "valid"
        ]

        def sum_stat(
            field_name: str,
            mode_stats: tuple[RequestMetric, ...] = tuple(stats_metrics),
        ) -> int:
            """对当前模式中通过 schema v2 校验的整数统计求和。"""
            values = [getattr(metric, field_name) for metric in mode_stats]
            return sum(value for value in values if isinstance(value, int))

        total_accepted = sum_stat("update_events_accepted")
        total_admitted = sum_stat("update_works_admitted")
        total_eager = sum_stat("eager_update_prefill_tokens")
        total_saved = sum_stat("saved_update_prefill_tokens")
        lcp_ratios = [
            metric.mean_update_lcp_ratio
            for metric in stats_metrics
            if metric.mean_update_lcp_ratio is not None
        ]
        num_requests = len(mode_metrics)
        num_first_token_success = sum(
            metric.first_token_received for metric in mode_metrics
        )
        summary_by_mode[mode] = {
            "num_requests": num_requests,
            "num_success": num_first_token_success,
            "num_error": num_requests - num_first_token_success,
            "num_with_prompt_stream_stats": len(stats_metrics),
            "num_transport_success": len(transport_success_metrics),
            "num_transport_error": num_requests - len(transport_success_metrics),
            "transport_success_rate": (
                len(transport_success_metrics) / num_requests if num_requests else None
            ),
            "num_first_token_success": num_first_token_success,
            "first_token_success_rate": (
                num_first_token_success / num_requests if num_requests else None
            ),
            "num_stats_valid": len(stats_metrics),
            "stats_valid_rate": (
                len(stats_metrics) / num_requests if num_requests else None
            ),
            "num_formal_valid": len(formal_metrics),
            "formal_valid_rate": (
                len(formal_metrics) / num_requests if num_requests else None
            ),
            "token_stats_coverage": (
                len(stats_metrics) / num_first_token_success
                if num_first_token_success
                else None
            ),
            "ttft_ms": {
                "mean": statistics.fmean(numeric_ttfts) if numeric_ttfts else None,
                "min": min(numeric_ttfts) if numeric_ttfts else None,
                "p50": percentile(numeric_ttfts, 50),
                "p90": percentile(numeric_ttfts, 90),
                "p95": percentile(numeric_ttfts, 95),
                "p99": percentile(numeric_ttfts, 99),
                "max": max(numeric_ttfts) if numeric_ttfts else None,
            },
            "tpot_ms": {
                "mean": statistics.fmean(numeric_tpots) if numeric_tpots else None,
                "p50": percentile(numeric_tpots, 50),
                "p90": percentile(numeric_tpots, 90),
                "p95": percentile(numeric_tpots, 95),
                "p99": percentile(numeric_tpots, 99),
            },
            "output_tokens_per_second": {
                "mean": (
                    statistics.fmean(numeric_output_tps) if numeric_output_tps else None
                ),
                "p50": percentile(numeric_output_tps, 50),
                "p90": percentile(numeric_output_tps, 90),
                "p95": percentile(numeric_output_tps, 95),
                "num_exact_requests": len(numeric_output_tps),
                "total_exact_output_tokens": total_exact_output_tokens,
            },
            "prompt_stream_stats": {
                "update_events_accepted": total_accepted,
                "update_works_admitted": total_admitted,
                "update_works_completed": sum_stat("update_works_completed"),
                "update_policy_works_admitted": sum_stat(
                    "update_policy_works_admitted"
                ),
                "update_policy_works_completed": sum_stat(
                    "update_policy_works_completed"
                ),
                "update_policy_works_cancelled_by_recovery": sum_stat(
                    "update_policy_works_cancelled_by_recovery"
                ),
                "update_policy_works_outstanding": sum_stat(
                    "update_policy_works_outstanding"
                ),
                "update_execution_attempts_started": sum_stat(
                    "update_execution_attempts_started"
                ),
                "update_execution_attempts_completed": sum_stat(
                    "update_execution_attempts_completed"
                ),
                "update_execution_attempts_preempted": sum_stat(
                    "update_execution_attempts_preempted"
                ),
                "update_recovery_works_created": sum_stat(
                    "update_recovery_works_created"
                ),
                "update_tokens_scheduled_total": sum_stat(
                    "update_tokens_scheduled_total"
                ),
                "update_tokens_scheduled_for_recovery": sum_stat(
                    "update_tokens_scheduled_for_recovery"
                ),
                "prefix_append_tokens_scheduled_for_recovery": sum_stat(
                    "prefix_append_tokens_scheduled_for_recovery"
                ),
                "update_tokens_discarded_by_preemption": sum_stat(
                    "update_tokens_discarded_by_preemption"
                ),
                "prefix_append_tokens_discarded_by_preemption": sum_stat(
                    "prefix_append_tokens_discarded_by_preemption"
                ),
                "question_tokens_discarded_by_preemption": sum_stat(
                    "question_tokens_discarded_by_preemption"
                ),
                "prompt_stream_running_preemptions": sum_stat(
                    "prompt_stream_running_preemptions"
                ),
                "prompt_stream_dormant_evictions": sum_stat(
                    "prompt_stream_dormant_evictions"
                ),
                "prompt_stream_retained_waiting_evictions": sum_stat(
                    "prompt_stream_retained_waiting_evictions"
                ),
                "prompt_stream_allocation_failures_running": sum_stat(
                    "prompt_stream_allocation_failures_running"
                ),
                "prompt_stream_allocation_failures_waiting": sum_stat(
                    "prompt_stream_allocation_failures_waiting"
                ),
                "prompt_stream_prefix_cache_resets": sum_stat(
                    "prompt_stream_prefix_cache_resets"
                ),
                "eager_update_prefill_tokens": total_eager,
                "admitted_update_prefill_tokens": sum_stat(
                    "admitted_update_prefill_tokens"
                ),
                "scheduled_update_prefill_tokens": sum_stat(
                    "scheduled_update_prefill_tokens"
                ),
                "saved_update_prefill_tokens": total_saved,
                "extra_update_prefill_tokens": sum_stat("extra_update_prefill_tokens"),
                "update_tokens_invalidated_by_snapshot": sum_stat(
                    "update_tokens_invalidated_by_snapshot"
                ),
                "update_tokens_invalidated": sum_stat("update_tokens_invalidated"),
                "coalescing_ratio": (
                    1.0 - total_admitted / total_accepted
                    if total_accepted > 0
                    else None
                ),
                "token_saving_ratio": (
                    total_saved / total_eager if total_eager > 0 else None
                ),
                "mean_update_lcp_ratio": (
                    statistics.fmean(lcp_ratios) if lcp_ratios else None
                ),
            },
        }
    return summary_by_mode


def request_metric_row(metric: RequestMetric) -> dict[str, Any]:
    """将 RequestMetric 扁平化为 request_metrics.csv 的一行。"""
    row = asdict(metric)
    row.pop("event_metrics", None)
    row.pop("response_payloads", None)
    load = row.pop("load", None) or {}
    row.update({field: load.get(field) for field in LOAD_METRIC_FIELDS})
    return row


def apply_completion_measurement(
    metric: RequestMetric,
    measurement: SseMeasurement,
    start_time: float,
) -> None:
    """把 SSE 输出规模与首 token 后耗时写入指标，缺失 token IDs 不报精确 TPS。"""
    metric.completion_chars = measurement.completion_chars
    metric.completion_chunks = measurement.completion_chunks
    metric.completion_tokens = measurement.completion_tokens
    metric.completion_token_count_exact = measurement.completion_token_count_exact
    if measurement.last_token_time is not None:
        metric.time_to_last_token_ms = (measurement.last_token_time - start_time) * 1000
    if measurement.completion_token_count_exact and measurement.completion_tokens > 1:
        assert measurement.first_token_time is not None
        assert measurement.last_token_time is not None
        duration_ms = (
            measurement.last_token_time - measurement.first_token_time
        ) * 1000
        metric.mean_time_per_output_token_ms = duration_ms / (
            measurement.completion_tokens - 1
        )
        if duration_ms > 0:
            metric.output_tokens_per_second = (
                1000 / metric.mean_time_per_output_token_ms
            )
    metric.response_payloads = list(measurement.payloads)
