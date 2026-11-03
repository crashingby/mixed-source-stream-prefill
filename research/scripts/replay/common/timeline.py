# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""请求到达时间和来源事件的计划时钟。"""

from __future__ import annotations

import argparse
import asyncio
import logging
import random
import time

from .config import DEFAULT_EVENT_PROGRESS_LOG_INTERVAL
from .models import ArrivalProcess

logger = logging.getLogger(__name__)


async def sleep_until(target_time: float) -> None:
    """在 monotonic 时钟下等待到指定目标时间，避免累计 sleep 漂移。"""
    remaining_seconds = target_time - time.perf_counter()
    if remaining_seconds > 0:
        await asyncio.sleep(remaining_seconds)


def should_log_event_progress(
    args: argparse.Namespace,
    event_index: int,
) -> bool:
    """判断当前 event 是否需要打印 replay 侧进度日志。

    这个 helper 只控制客户端侧调试日志，不改变请求时序。默认每个
    prompt-stream event 都打印一次 SEND/ACK；实验日志过大时可以把
    ``--event-progress-log-interval`` 设为更大的正数做降采样，或设为
    0 完全关闭。
    """
    log_interval = getattr(
        args,
        "event_progress_log_interval",
        DEFAULT_EVENT_PROGRESS_LOG_INTERVAL,
    )
    return log_interval > 0 and event_index % log_interval == 0


def planned_arrival_offsets(
    num_requests: int,
    qps: float,
    arrival_process: ArrivalProcess,
    random_seed: int,
) -> list[float]:
    """生成一个 mode 内每个请求相对启动点的计划到达时间秒。"""
    if num_requests == 0 or qps == 0:
        return [0.0] * num_requests
    random_generator = random.Random(random_seed)
    offsets = [0.0]
    for _ in range(1, num_requests):
        if arrival_process == "fixed":
            inter_arrival_seconds = 1.0 / qps
        else:
            inter_arrival_seconds = random_generator.expovariate(qps)
        offsets.append(offsets[-1] + inter_arrival_seconds)
    return offsets


def source_timing(
    source_type: str,
    args: argparse.Namespace,
) -> tuple[float, float]:
    """返回一个来源的首 chunk 延迟与后续数据 chunk 间隔。"""
    if source_type == "append":
        return args.append_initial_delay, args.append_chunk_interval
    if source_type == "update":
        return args.update_initial_delay, args.update_chunk_interval
    raise ValueError(f"Unsupported source_type={source_type!r}.")
