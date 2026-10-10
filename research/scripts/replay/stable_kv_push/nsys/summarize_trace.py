# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""只读汇总 nsys SQLite 的 CUDA 调用与 GPU 活跃区间，不把累积时长当墙钟。"""

from __future__ import annotations

import argparse
import bisect
import json
import sqlite3
from pathlib import Path
from typing import Any

API_TABLES = ("CUPTI_ACTIVITY_KIND_RUNTIME", "CUPTI_ACTIVITY_KIND_DRIVER")
GPU_TABLES = (
    "CUPTI_ACTIVITY_KIND_KERNEL",
    "CUPTI_ACTIVITY_KIND_MEMCPY",
    "CUPTI_ACTIVITY_KIND_MEMSET",
)


def api_summary(connection: sqlite3.Connection, table: str) -> list[dict[str, Any]]:
    """按 CUDA API 和线程汇总，避免混淆主计算线程与后台 writer。

    Args:
        connection: 已按只读模式打开的 nsys SQLite 数据库。
        table: API_TABLES 中的 runtime 或 driver 表名。

    Returns:
        按累计时长排序的调用统计；所有时长统一转换为毫秒。
    """
    if table not in API_TABLES:
        raise ValueError(f"Not a supported API table: {table}")
    # 表名来自固定白名单；字符串和调用线程通过官方导出的 ID 关联。
    rows = connection.execute(
        f"SELECT s.value, a.globalTid, count(*), sum(a.end-a.start), "
        f"avg(a.end-a.start), max(a.end-a.start) FROM {table} a "
        "JOIN StringIds s ON s.id=a.nameId GROUP BY a.nameId, a.globalTid "
        "ORDER BY sum(a.end-a.start) DESC"
    )
    return [
        {
            "name": name,
            "global_tid": global_tid,
            "count": count,
            "total_ms": total_ns / 1e6,
            "mean_ms": mean_ns / 1e6,
            "max_ms": max_ns / 1e6,
        }
        for name, global_tid, count, total_ns, mean_ns, max_ns in rows
    ]


def interval_summary(
    connection: sqlite3.Connection, tables: list[str]
) -> dict[str, Any]:
    """计算跨 stream 的 GPU 活跃区间并集，避免重叠 kernel 重复计时。"""
    if not tables:
        return {"count": 0, "sum_ms": 0.0, "union_ms": 0.0}
    if any(table not in GPU_TABLES for table in tables):
        raise ValueError("Unexpected GPU activity table")
    query = " UNION ALL ".join(f"SELECT start,end FROM {table}" for table in tables)
    intervals = connection.execute(f"SELECT * FROM ({query}) ORDER BY start")
    count = 0
    total_ns = 0
    union_ns = 0
    first_start: int | None = None
    last_end = 0
    covered_end = 0  # 前面全部区间已经覆盖到的位置。
    for start, end in intervals:
        count += 1
        total_ns += end - start
        if first_start is None:
            first_start = start
        union_ns += max(0, end - max(start, covered_end))
        covered_end = max(covered_end, end)
        last_end = max(last_end, end)
    span_ns = last_end - first_start if first_start is not None else 0
    return {
        "count": count,
        "sum_ms": total_ns / 1e6,
        "union_ms": union_ns / 1e6,
        "first_to_last_ms": span_ns / 1e6,
        "busy_fraction_within_gpu_span": union_ns / span_ns if span_ns else None,
    }


def summarize(path: Path) -> dict[str, Any]:
    """读取单个 trace；缺少真实 CUDA 数据时报错，不返回虚假的全零统计。"""
    # mode=ro 防止 SQLite 在文件名写错时悄悄创建空数据库。
    with sqlite3.connect(f"{path.resolve().as_uri()}?mode=ro", uri=True) as connection:
        tables = {
            row[0]
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            )
        }
        available_apis = [table for table in API_TABLES if table in tables]
        available_gpu = [table for table in GPU_TABLES if table in tables]
        if not available_apis or not available_gpu:
            raise ValueError(f"No usable CUDA trace tables in {path}")
        apis = {table: api_summary(connection, table) for table in available_apis}
        synchronization = [
            {"table": table, **row}
            for table, rows in apis.items()
            for row in rows
            if "Synchronize" in row["name"]
        ]
        return {
            "sqlite": str(path),
            "cuda_api_by_thread": apis,
            "synchronization": synchronization,
            "gpu_activity": interval_summary(connection, available_gpu),
            "gpu_kernels": interval_summary(
                connection,
                ["CUPTI_ACTIVITY_KIND_KERNEL"]
                if "CUPTI_ACTIVITY_KIND_KERNEL" in tables
                else [],
            ),
            "copies": copy_summary(connection)
            if "CUPTI_ACTIVITY_KIND_MEMCPY" in tables
            else [],
            "slow_launches": slow_launch_summary(connection),
            "limitations": (
                "API totals can overlap across threads; GPU totals can overlap "
                "across streams. Neither total is request critical-path time. "
                "Profiling perturbs execution; use unprofiled replay for latency."
            ),
        }


def copy_summary(connection: sqlite3.Connection) -> list[dict[str, Any]]:
    """按真实 copy 类型和字节粒度统计，区分 host staging 与大块直传。"""
    rows = connection.execute(
        "SELECT e.label,m.bytes,count(*),sum(m.end-m.start) "
        "FROM CUPTI_ACTIVITY_KIND_MEMCPY m "
        "JOIN ENUM_CUDA_MEMCPY_OPER e ON e.id=m.copyKind "
        "GROUP BY m.copyKind,m.bytes ORDER BY count(*) DESC"
    )
    return [
        {
            "kind": kind,
            "bytes_per_copy": size,
            "count": count,
            "total_bytes": size * count,
            "gpu_total_ms": total_ns / 1e6,
        }
        for kind, size, count, total_ns in rows
    ]


def slow_launch_summary(connection: sqlite3.Connection) -> dict[str, Any]:
    """核对长时间 CUDA launch 与后台 copy API 的重叠，不推导因果结论。

    只检查 CPU 端提交调用，不把 API 时长当作 kernel GPU 执行时长。
    同期 copy 只是相关证据；未做关闭 host staging 等控制变量实验。
    """
    available = {
        row[0]
        for row in connection.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        )
    }
    # 不同 nsys 版本把 driver API 放入独立 DRIVER 表或同一个 RUNTIME 表。
    api_union = " UNION ALL ".join(
        f"SELECT start,end,nameId,globalTid FROM {table}"
        for table in API_TABLES
        if table in available
    )
    if not api_union:
        raise ValueError("No CUDA API records to correlate")
    copy_ranges = list(
        connection.execute(
            f"SELECT a.start,a.end FROM ({api_union}) a "
            "JOIN StringIds s ON s.id=a.nameId "
            "WHERE s.value LIKE 'cuMemcpyAsync%' ORDER BY a.start"
        )
    )
    copy_starts = [start for start, _ in copy_ranges]
    copy_ends = sorted(end for _, end in copy_ranges)
    launches = connection.execute(
        f"SELECT s.value,a.start,a.end,a.globalTid FROM ({api_union}) a "
        "JOIN StringIds s ON s.id=a.nameId "
        "WHERE (s.value LIKE '%LaunchKernel%' OR s.value='cuLaunchKernelEx') "
        "AND a.end-a.start >= 5000000 ORDER BY a.end-a.start DESC"
    )
    records: list[dict[str, Any]] = []
    for name, start, end, tid in launches:
        # 半开区间 [start,end)：开始早于 end，且结束晚于 start 的 copy。
        num_overlaps = bisect.bisect_left(copy_starts, end) - bisect.bisect_right(
            copy_ends, start
        )
        records.append(
            {
                "name": name,
                "start_ns": start,
                "end_ns": end,
                "duration_ms": (end - start) / 1e6,
                "global_tid": tid,
                "overlapping_copy_api_calls": num_overlaps,
            }
        )
    return {
        "threshold_ms": 5,
        "count": len(records),
        "count_overlapping_copy": sum(
            row["overlapping_copy_api_calls"] > 0 for row in records
        ),
        "total_ms": sum(row["duration_ms"] for row in records),
        "longest": records[:20],
    }


def main() -> None:
    """写出隔离的诊断 JSON，不修改原始 SQLite 或正式实验数据。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sqlite", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    summary = summarize(args.sqlite)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary["synchronization"], ensure_ascii=False, indent=2))
    print(json.dumps(summary["gpu_activity"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
