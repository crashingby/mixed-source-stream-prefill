# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""汇总 Query probe 的端到端 TTFT 与 finish TTFT，保留失败计数。"""

import statistics
from typing import Any

LATENCY_FIELDS = (
    "request_to_first_token_ms",
    "finish_to_first_token_ms",
    "source_dispatch_ms",
    "start_ack_ms",
    "event_lateness_mean_ms",
    "event_lateness_max_ms",
)


def distribution(values: list[float]) -> dict[str, float | int] | None:
    """使用线性插值分位数；空数据没有可报告的延迟。"""
    if not values:
        return None
    ordered = sorted(values)

    def percentile(fraction: float) -> float:
        position = (len(ordered) - 1) * fraction
        lower = int(position)
        upper = min(lower + 1, len(ordered) - 1)
        return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)

    return {
        "count": len(values),
        "mean": statistics.fmean(values),
        "p50": percentile(0.5),
        "p95": percentile(0.95),
        "p99": percentile(0.99),
        "max": ordered[-1],
    }


def summarize(results: list[dict[str, Any]], elapsed: float) -> dict[str, Any]:
    """以成功且通过统计校验的请求计算延迟，并单列失败与吞吐。"""
    valid = [row for row in results if row["status"] == "ok"]
    return {
        "requests": len(results),
        "successful_requests": len(valid),
        "errors": sum(row["status"] == "error" for row in results),
        "elapsed_s": elapsed,
        "successful_requests_per_second": len(valid) / elapsed if elapsed else 0,
        "latency_ms": {
            field: distribution([row[field] for row in valid])
            for field in LATENCY_FIELDS
        },
        "preemptions": sum(
            row["prompt_stream_stats"].get("prompt_stream_running_preemptions", 0)
            for row in valid
        ),
    }
