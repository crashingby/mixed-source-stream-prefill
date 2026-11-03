# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""汇总 Query probe 的端到端 TTFT 与 finish TTFT，保留失败计数。"""

from typing import Any

from research.scripts.replay.common.load import distribution

LATENCY_FIELDS = (
    "queue_delay_ms",
    "dispatch_lag_ms",
    "arrival_to_first_token_ms",
    "sent_to_first_token_ms",
    "arrival_to_done_ms",
    "request_to_first_token_ms",
    "finish_to_first_token_ms",
    "source_dispatch_ms",
    "start_ack_ms",
    "event_lateness_mean_ms",
    "event_lateness_max_ms",
)


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
            field: distribution(
                [row[field] for row in valid if row.get(field) is not None]
            )
            for field in LATENCY_FIELDS
        },
        "preemptions": sum(
            row["prompt_stream_stats"].get("prompt_stream_running_preemptions", 0)
            for row in valid
        ),
    }
