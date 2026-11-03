# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""统一逻辑请求的负载配置、发送时刻和客户端排队统计。"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import time
from collections.abc import Mapping
from contextvars import ContextVar
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx

LOAD_METRIC_FIELDS = (
    "arrival_lag_ms",
    "client_queue_ms",
    "client_preparation_ms",
    "dispatch_lag_ms",
    "arrival_to_first_token_ms",
    "sent_to_first_token_ms",
    "arrival_to_done_ms",
    "sent_to_done_ms",
)


def read_load_summary(directory: Path) -> dict[str, Any] | None:
    """加载同目录的公共负载摘要；dry-run 没有发送统计。"""
    path = directory / "load_summary.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def resolve_load_mode(args: argparse.Namespace) -> str:
    """解析兼容配置；正 QPS 使用独立到达，零 QPS 使用闭环槽位。"""
    qps = getattr(args, "qps", 0.0)
    mode = getattr(args, "load_mode", "auto")
    if mode == "auto":
        mode = "open_loop" if qps > 0 else "closed_loop"
    if not math.isfinite(qps) or qps < 0:
        raise ValueError("qps 必须是有限非负数")
    if mode not in ("closed_loop", "open_loop"):
        raise ValueError(f"未知负载模式：{mode}")
    if (mode == "open_loop") != (qps > 0):
        raise ValueError("open_loop 要求 qps > 0；closed_loop 要求 qps = 0")
    if getattr(args, "max_concurrency", 1) < 1:
        raise ValueError("max_concurrency 必须为正整数")
    cap = getattr(args, "max_inflight", 0)
    if cap < 0 or (mode == "closed_loop" and cap):
        raise ValueError("max_inflight 仅用于 open_loop；0 表示不限制")
    delay = getattr(args, "inter_request_delay", 0.0)
    if not math.isfinite(delay) or delay < 0:
        raise ValueError("inter_request_delay 必须有限非负；仅在 closed_loop 生效")
    return mode


def distribution(values: list[float]) -> dict[str, float | int] | None:
    """汇总有限样本，分位数使用线性插值，空数据保留为 null。"""
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


@dataclass
class LoadSample:
    """记录一个逻辑请求；全部时间来自客户端同一个单调时钟。"""

    index: int
    planned: float | None = None  # 开环为预生成时间；闭环为槽位可用时间。
    released: float | None = None  # 到达协程实际被唤醒的时间。
    admitted: float | None = None  # 获取逻辑请求槽位，开始构造并发送请求。
    sent: float | None = None  # 第一次向 HTTP 客户端提交 start/completions。
    first_response: float | None = None  # P-only 就绪事件或完整生成的 D 首 token。
    response_origin: str | None = None
    completed: float | None = None
    status: str = "pending"

    def row(self, epoch: float) -> dict[str, Any]:
        """把绝对时刻转换为相对毫秒，保留未发送与未返回的缺失值。"""

        def interval(end: float | None, start: float | None) -> float | None:
            # 尚未到计划时间就被取消的请求没有到达后延迟。
            return (
                (end - start) * 1000
                if end is not None and start is not None and end >= start
                else None
            )

        return {
            "sample_index": self.index,
            "status": self.status,
            "planned_arrival_offset_ms": interval(self.planned, epoch),
            "released_offset_ms": interval(self.released, epoch),
            "admitted_offset_ms": interval(self.admitted, epoch),
            "sent_offset_ms": interval(self.sent, epoch),
            "completed_offset_ms": interval(self.completed, epoch),
            "arrival_lag_ms": interval(self.released, self.planned),
            "client_queue_ms": interval(self.admitted, self.released),
            "queue_delay_ms": interval(self.admitted, self.planned),
            "client_preparation_ms": interval(self.sent, self.admitted),
            "dispatch_lag_ms": interval(self.sent, self.planned),
            "arrival_to_first_token_ms": interval(self.first_response, self.planned),
            "sent_to_first_token_ms": interval(self.first_response, self.sent),
            "arrival_to_done_ms": interval(self.completed, self.planned),
            "sent_to_done_ms": interval(self.completed, self.sent),
            "ttft_origin": self.response_origin,
        }


# 每个请求及其 Append/Update 子任务共享同一个记录，HTTP 连接数不充当并发数。
active_sample: ContextVar[LoadSample | None] = ContextVar(
    "replay_load_sample", default=None
)


async def mark_request_sent(request: httpx.Request) -> None:
    """HTTP request hook：只统计首个业务请求，事件和 abort 不增加逻辑 QPS。"""
    sample = active_sample.get()
    if (
        sample is not None
        and sample.sent is None
        and request.url.path.endswith(
            (
                "/completions",
                "/completions_ttft",
                "/prompt_stream/start",
                "/prompt_stream/start_ttft",
            )
        )
    ):
        sample.sent = time.perf_counter()


def mark_first_response(origin: str, timestamp: float | None = None) -> None:
    """记录当前实验的计时终点；完整 PD 的 P 通知不触发此函数。"""
    sample = active_sample.get()
    if sample is not None and sample.first_response is None:
        sample.first_response = time.perf_counter() if timestamp is None else timestamp
        sample.response_origin = origin


class LoadRecorder:
    """持久化统一负载记录，独立于各实验的质量、失效或传输统计。"""

    def __init__(
        self, args: argparse.Namespace, count: int, output_dir: Path | None
    ) -> None:
        self.mode = resolve_load_mode(args)
        self.qps = getattr(args, "qps", 0.0)
        self.arrival_process = getattr(args, "arrival_process", "fixed")
        self.limit = (
            getattr(args, "max_concurrency", 1)
            if self.mode == "closed_loop"
            else getattr(args, "max_inflight", 0)
        )
        self.output_dir = output_dir
        self.epoch = time.perf_counter()
        self.samples = [LoadSample(index) for index in range(count)]
        self.stream = None
        if output_dir is not None:
            output_dir.mkdir(parents=True, exist_ok=True)
            self.stream = (output_dir / "load_requests.jsonl").open(
                "w", encoding="utf-8"
            )

    def record(self, sample: LoadSample, result: Any = None) -> None:
        """完成或取消时立即落盘；实验结果同时携带同一份负载字段。"""

        def value(name: str) -> Any:
            return (
                result.get(name)
                if isinstance(result, Mapping)
                else getattr(result, name, None)
            )

        if result is not None:
            sample.status = (
                value("transport_status")
                or value("result_status")
                or value("status")
                or "ok"
            )
        row = sample.row(self.epoch)
        row.update(sample_id=value("sample_id"), query_id=value("query_id"))
        if isinstance(result, dict):
            result["load"] = row
            result.update({field: row[field] for field in LOAD_METRIC_FIELDS})
        elif result is not None and hasattr(result, "load"):
            result.load = row
        if self.stream is not None:
            self.stream.write(json.dumps(row, ensure_ascii=False) + "\n")
            self.stream.flush()

    @staticmethod
    def _peak(intervals: list[tuple[float, float]]) -> int:
        """计算时间区间最大重叠数，相同时刻先结束再开始。"""
        events = [(start, 1) for start, end in intervals if end > start]
        events += [(end, -1) for start, end in intervals if end > start]
        current = peak = 0
        for _, delta in sorted(events):
            current += delta
            peak = max(peak, current)
        return peak

    def summary(self) -> dict[str, Any]:
        """区分发送窗口速率与包含排空阶段的吞吐，统计排队及全部失败。"""
        end = time.perf_counter()
        elapsed = end - self.epoch
        sent = [s for s in self.samples if s.sent is not None]
        times = sorted(s.sent for s in sent if s.sent is not None)
        completed = [
            s for s in sent if s.completed is not None and s.status != "cancelled"
        ]
        successful = [s for s in completed if s.status == "ok"]
        arrivals = [s.planned for s in self.samples if s.planned is not None]
        last_arrival = max(arrivals) if arrivals else None
        rows = [s.row(self.epoch) for s in self.samples]
        delays = (
            "arrival_lag_ms",
            "client_queue_ms",
            "queue_delay_ms",
            "client_preparation_ms",
            "dispatch_lag_ms",
            "arrival_to_first_token_ms",
            "sent_to_first_token_ms",
            "arrival_to_done_ms",
            "sent_to_done_ms",
        )
        return {
            "schema": "replay_load/v1",
            "load_mode": self.mode,
            "planned_qps": self.qps if self.mode == "open_loop" else None,
            "arrival_process": self.arrival_process
            if self.mode == "open_loop"
            else None,
            "inflight_limit": self.limit or None,
            "selected_requests": len(self.samples),
            "sent_requests": len(sent),
            "completed_requests": len(completed),
            "successful_requests": len(successful),
            "failed_requests": sum(s.status == "error" for s in self.samples),
            "cancelled_requests": sum(s.status == "cancelled" for s in self.samples),
            "not_sent_requests": len(self.samples) - len(sent),
            "elapsed_s": elapsed,
            # N 次发送只有 N-1 个间隔，不能用总运行时间冒充实际发送速率。
            "actual_send_qps": (len(times) - 1) / (times[-1] - times[0])
            if len(times) > 1 and times[-1] > times[0]
            else None,
            "send_window_s": times[-1] - times[0] if len(times) > 1 else None,
            "actual_interarrival_ms": distribution(
                [(right - left) * 1000 for left, right in zip(times, times[1:])]
            ),
            "client_backlog_at_last_arrival": sum(
                s.sent is None or s.sent > last_arrival
                for s in self.samples
                if s.planned is not None
                and s.planned <= last_arrival
                and (s.completed is None or s.completed > last_arrival)
            )
            if self.mode == "open_loop"
            and last_arrival is not None
            and last_arrival <= end
            else None,
            "completed_requests_per_second": len(completed) / elapsed
            if elapsed
            else None,
            "successful_requests_per_second": len(successful) / elapsed
            if elapsed
            else None,
            "drain_time_s": end - times[-1] if times else None,
            "max_inflight": self._peak(
                [
                    (s.admitted, s.completed or end)
                    for s in self.samples
                    if s.admitted is not None
                ]
            ),
            # 到计划时间但尚未提交 HTTP 的请求也算积压，包含事件循环迟发。
            "max_client_backlog": self._peak(
                [
                    (s.planned, s.sent or s.completed or end)
                    for s in self.samples
                    if s.planned is not None and s.planned <= end
                ]
            ),
            "latency_ms": {
                field: distribution(
                    [row[field] for row in rows if row[field] is not None]
                )
                for field in delays
            },
        }

    def close(self) -> None:
        """成功、异常与取消均关闭逐请求日志，并保存部分实验的最终摘要。"""
        if self.stream is not None:
            self.stream.close()
        if self.output_dir is not None:
            (self.output_dir / "load_summary.json").write_text(
                json.dumps(self.summary(), ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
