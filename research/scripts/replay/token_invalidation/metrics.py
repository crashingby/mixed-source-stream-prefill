# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""只将正式有效的服务端失效统计纳入汇总，失败与缺失值保留独立分母。"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict, fields
from pathlib import Path
from typing import Any

from research.scripts.replay.common.metrics import EVENT_METRIC_FIELDS, percentile
from research.scripts.replay.common.results import write_csv, write_json

from .models import InvalidationMetric


def distribution(values: Sequence[int]) -> dict[str, Any]:
    """输出失效数分布；没有有效记录时返回 null，不把缺失当成零。"""
    return {
        "count": len(values),
        "total": sum(values) if values else None,
        "mean": sum(values) / len(values) if values else None,
        "p50": percentile(values, 50),
        "p95": percentile(values, 95),
        "p99": percentile(values, 99),
        "max": max(values) if values else None,
    }


def summary(metrics: Sequence[InvalidationMetric], num_selected: int) -> dict[str, Any]:
    """分别报告完成率、链路成功率和统计有效率，再汇总真实失效 token。"""
    valid = [
        metric
        for metric in metrics
        if metric.formal_result_status == "valid"
        and metric.snapshot_invalidated_tokens is not None
    ]
    return {
        "num_selected": num_selected,
        "num_completed": len(metrics),
        "num_uncompleted": num_selected - len(metrics),
        "num_transport_ok": sum(metric.transport_status == "ok" for metric in metrics),
        "num_formal_valid": len(valid),
        "num_formal_invalid": sum(
            metric.formal_result_status == "invalid" for metric in metrics
        ),
        "snapshot_invalidated_tokens": distribution(
            [metric.snapshot_invalidated_tokens for metric in valid]
        ),
        "preemption_discarded_tokens": distribution(
            [
                metric.preemption_discarded_tokens
                for metric in valid
                if metric.preemption_discarded_tokens is not None
            ]
        ),
        "zero_snapshot_invalidation_requests": sum(
            metric.snapshot_invalidated_tokens == 0 for metric in valid
        ),
        "potential_replaced_suffix_tokens": distribution(
            [metric.potential_replaced_suffix_tokens for metric in valid]
        ),
    }


def write_outputs(
    directory: Path, metrics: Sequence[InvalidationMetric], num_selected: int
) -> None:
    """保存逐请求、逐事件 CSV 和主指标摘要；JSONL 已在每个请求完成时刷写。"""
    rows = []
    for metric in metrics:
        row = asdict(metric)
        row.pop("event_metrics")
        row.pop("response_payloads")
        rows.append(row)
    names = [
        field.name
        for field in fields(InvalidationMetric)
        if field.name not in ("event_metrics", "response_payloads")
    ]
    write_csv(directory / "request_metrics.csv", names, rows)
    write_csv(
        directory / "event_metrics.csv",
        EVENT_METRIC_FIELDS,
        (asdict(event) for metric in metrics for event in metric.event_metrics),
    )
    write_json(
        directory / "summary.json",
        {
            "schema": "token_invalidation_summary/v1",
            "primary_metric": "snapshot_invalidated_tokens",
            "definition": (
                "服务端已计算的 Update 域中因快照替换失效的 token；"
                "整体基线该域包含全部正文"
            ),
            **summary(metrics, num_selected),
        },
    )
