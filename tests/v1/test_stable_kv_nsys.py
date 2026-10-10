# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""验证 nsys 离线统计的区间并集和 copy 重叠，不依赖 CUDA 或服务。"""

import sqlite3

import pytest

from research.scripts.replay.stable_kv_push.nsys.summarize_trace import (
    interval_summary,
    slow_launch_summary,
)


def test_gpu_intervals_count_overlap_once() -> None:
    """两个 stream 的重叠区间只能在 GPU 活跃并集中计入一次。"""
    with sqlite3.connect(":memory:") as connection:
        connection.execute(
            "CREATE TABLE CUPTI_ACTIVITY_KIND_KERNEL(start INTEGER,end INTEGER)"
        )
        connection.executemany(
            "INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES (?,?)",
            [(0, 10_000_000), (5_000_000, 15_000_000), (20_000_000, 22_000_000)],
        )
        result = interval_summary(connection, ["CUPTI_ACTIVITY_KIND_KERNEL"])
    assert result["sum_ms"] == 22
    assert result["union_ms"] == 17
    assert result["busy_fraction_within_gpu_span"] == pytest.approx(17 / 22)


def test_slow_launch_copy_overlap_uses_half_open_interval() -> None:
    """只相接但不重叠的 copy 不能算入慢 launch 的同期 copy 数。"""
    with sqlite3.connect(":memory:") as connection:
        connection.execute("CREATE TABLE StringIds(id INTEGER,value TEXT)")
        connection.execute(
            "CREATE TABLE CUPTI_ACTIVITY_KIND_RUNTIME"
            "(start INTEGER,end INTEGER,nameId INTEGER,globalTid INTEGER)"
        )
        connection.executemany(
            "INSERT INTO StringIds VALUES (?,?)",
            [(1, "cuMemcpyAsync"), (2, "cuLaunchKernelEx")],
        )
        connection.executemany(
            "INSERT INTO CUPTI_ACTIVITY_KIND_RUNTIME VALUES (?,?,?,?)",
            [
                (0, 10_000_000, 1, 1),
                (12_000_000, 14_000_000, 1, 1),
                (20_000_000, 30_000_000, 1, 1),
                (10_000_000, 20_000_000, 2, 2),
            ],
        )
        result = slow_launch_summary(connection)
    assert result["count"] == 1
    assert result["count_overlapping_copy"] == 1
    assert result["longest"][0]["overlapping_copy_api_calls"] == 1
