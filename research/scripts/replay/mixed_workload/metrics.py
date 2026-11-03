# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""按普通请求、流式请求和整个混合负载分别统计 P 侧 TTFT。"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from research.scripts.replay.common.dataset import iter_jsonl
from research.scripts.replay.common.load import (
    LOAD_METRIC_FIELDS,
    distribution,
    read_load_summary,
)
from research.scripts.replay.common.results import write_csv, write_json

LATENCY_FIELDS = (
    *LOAD_METRIC_FIELDS,
    "queue_delay_ms",
    "request_to_first_token_ms",
    "finish_to_first_token_ms",
    "start_ack_ms",
    "source_dispatch_ms",
)


def write_summary(directory: Path, config: dict[str, Any]) -> dict[str, Any]:
    """读取已落盘记录，保留失败、取消和未发请求，写入分组摘要与 CSV。

    分组完成吞吐都除以同一轮总时长；首响应延迟只汇总成功且通过模式校验的请求。
    公共 load_summary 保留包括失败在内的实际观测，便于检查幸存者偏差。
    """
    requests = list(iter_jsonl(directory / "requests.jsonl"))
    plan = list(iter_jsonl(directory / "plan.jsonl"))
    load_summary = read_load_summary(directory)
    loads = list(iter_jsonl(directory / "load_requests.jsonl")) if load_summary else []
    elapsed = load_summary["elapsed_s"] if load_summary else None
    groups = {}
    for kind in ("all", "regular", "stream"):
        selected = [row for row in plan if kind == "all" or row["request_kind"] == kind]
        indices = {row["sample_index"] for row in selected}
        rows = [row for row in requests if row["sample_index"] in indices]
        valid = [row for row in rows if row["status"] == "ok"]
        observed = [row for row in loads if row["sample_index"] in indices]
        times = sorted(
            row["sent_offset_ms"]
            for row in observed
            if row["sent_offset_ms"] is not None
        )
        groups[kind] = {
            "selected_requests": len(selected),
            "sent_requests": len(times),
            "successful_requests": len(valid),
            "failed_requests": sum(row["status"] == "error" for row in observed),
            "cancelled_requests": sum(row["status"] == "cancelled" for row in observed),
            "not_sent_requests": len(selected) - len(times),
            "success_fraction_of_selected": len(valid) / len(selected)
            if selected and load_summary
            else None,
            "actual_send_qps": (len(times) - 1) * 1000 / (times[-1] - times[0])
            if len(times) > 1 and times[-1] > times[0]
            else None,
            "successful_requests_per_second": len(valid) / elapsed if elapsed else None,
            "prompt_chars": distribution([row["prompt_chars"] for row in selected]),
            "latency_ms": {
                field: distribution(
                    [row[field] for row in valid if row.get(field) is not None]
                )
                for field in LATENCY_FIELDS
            },
        }
    summary = {
        "schema": "mixed_workload/v1",
        "stream_mode": config["args"]["stream_mode"],
        "keep_ratio": config["args"]["keep_ratio"],
        "requested_stream_ratio": config["args"]["stream_ratio"],
        "actual_stream_ratio": groups["stream"]["selected_requests"] / len(plan)
        if plan
        else None,
        "workload_sha256": config["workload_sha256"],
        "dry_run": config["args"]["dry_run"],
        "ttft_scope": "p_side_prefill_ready",
        "groups": groups,
        "load_summary": load_summary,
    }
    write_json(directory / "summary.json", summary)
    fields = [
        "sample_index",
        "sample_id",
        "query_id",
        "request_kind",
        "stream_mode",
        "status",
        "error",
        "ttft_origin",
        "prompt_chars",
        *LATENCY_FIELDS,
    ]
    write_csv(directory / "requests.csv", fields, requests)
    return summary
