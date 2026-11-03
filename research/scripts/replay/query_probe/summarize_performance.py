# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""按相同样本配对 dense/sparse，输出 TTFT 对比表与图。"""

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

from research.scripts.replay.query_probe.performance_stats import distribution


def read_requests(path: Path) -> dict[str, dict[str, Any]]:
    """只加载已完成的逐请求记录，保留失败供配对计数。"""
    return {
        row["sample_id"]: row
        for line in path.read_text().splitlines()
        if (row := json.loads(line))
    }


def compare(root: Path) -> list[dict[str, Any]]:
    """按闭环并发或开环 QPS 配对，要求数据、负载和事件计划一致。"""
    rows = []
    for dense_dir in sorted(root.glob("dense-*")):
        sparse_dir = root / dense_dir.name.replace("dense-", "sparse-", 1)
        if (
            not (dense_dir / "summary.json").exists()
            or not (sparse_dir / "summary.json").exists()
        ):
            continue
        summaries = {
            mode: json.loads((directory / "summary.json").read_text())
            for mode, directory in (("dense", dense_dir), ("sparse", sparse_dir))
        }
        left_load = summaries["dense"].get("load_summary") or {}
        right_load = summaries["sparse"].get("load_summary") or {}
        for key in ("load_mode", "planned_qps", "inflight_limit", "arrival_process"):
            if left_load.get(key) != right_load.get(key):
                raise ValueError(f"负载配置不一致：{key}")
        dense = read_requests(dense_dir / "requests.jsonl")
        sparse = read_requests(sparse_dir / "requests.jsonl")
        if dense.keys() != sparse.keys():
            raise ValueError(f"两边样本集合不同：{dense_dir.name}")
        paired = []
        for sample_id, left in dense.items():
            right = sparse[sample_id]
            for field in ("prompt_sha256", "events_sha256"):
                if left[field] != right[field]:
                    raise ValueError(f"{sample_id} 的 {field} 不一致")
            if left_load.get("load_mode") == "open_loop" and not math.isclose(
                left["load"]["planned_arrival_offset_ms"],
                right["load"]["planned_arrival_offset_ms"],
                abs_tol=0.01,
            ):
                raise ValueError(f"{sample_id} 的请求到达计划不一致")
            if left["status"] == right["status"] == "ok":
                paired.append((left, right))
        mode = left_load.get("load_mode", "closed_loop")
        qps = left_load.get("planned_qps")
        concurrency = (
            int(dense_dir.name.split("-c")[1]) if mode == "closed_loop" else None
        )
        metrics = (
            (
                "arrival_to_first_token_ms",
                "sent_to_first_token_ms",
                "finish_to_first_token_ms",
            )
            if mode == "open_loop"
            else ("request_to_first_token_ms", "finish_to_first_token_ms")
        )
        for metric in metrics:
            left_stats = distribution([left[metric] for left, _ in paired])
            right_stats = distribution([right[metric] for _, right in paired])
            row: dict[str, Any] = {
                "load_mode": mode,
                "qps": qps,
                "concurrency": concurrency,
                "inflight_limit": left_load.get("inflight_limit"),
                "dense_actual_send_qps": left_load.get("actual_send_qps"),
                "sparse_actual_send_qps": right_load.get("actual_send_qps"),
                "dense_throughput": left_load.get("successful_requests_per_second"),
                "sparse_throughput": right_load.get("successful_requests_per_second"),
                "dense_max_client_backlog": left_load.get("max_client_backlog"),
                "sparse_max_client_backlog": right_load.get("max_client_backlog"),
                "metric": metric,
                "attempted_per_mode": len(dense),
                "paired_successful": len(paired),
                "dense_errors": sum(r["status"] != "ok" for r in dense.values()),
                "sparse_errors": sum(r["status"] != "ok" for r in sparse.values()),
            }
            for variant, stats in (("dense", left_stats), ("sparse", right_stats)):
                for key in ("mean", "p50", "p95", "p99"):
                    row[f"{variant}_{key}_ms"] = stats[key] if stats else None
            row["mean_speedup"] = (
                left_stats["mean"] / right_stats["mean"]
                if left_stats and right_stats and right_stats["mean"] > 0
                else None
            )
            rows.append(row)
    output = root / "comparison"
    output.mkdir(exist_ok=True)
    (output / "comparison.json").write_text(json.dumps(rows, indent=2))
    if rows:
        with (output / "comparison.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        plot(rows, output)
    return rows


def plot(rows: list[dict[str, Any]], output: Path) -> None:
    """绘制闭环并发或开环 QPS 曲线；开环包含客户端等待的完整延迟。"""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    for mode in sorted({row["load_mode"] for row in rows}):
        is_open = mode == "open_loop"
        x_key = "qps" if is_open else "concurrency"
        metrics = (
            (
                ("arrival_to_first_token_ms", "Planned arrival to P ready"),
                ("sent_to_first_token_ms", "HTTP send to P ready"),
            )
            if is_open
            else (
                ("request_to_first_token_ms", "Request start to P ready"),
                ("finish_to_first_token_ms", "Finish to P ready"),
            )
        )
        figure, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
        for axis, (metric, title) in zip(axes, metrics):
            selected = sorted(
                (
                    r
                    for r in rows
                    if r["load_mode"] == mode
                    and r["metric"] == metric
                    and r["paired_successful"]
                ),
                key=lambda r: r[x_key],
            )
            for variant, color in (("dense", "tab:blue"), ("sparse", "tab:orange")):
                for stat, style in (("mean", "-o"), ("p95", "--x")):
                    axis.plot(
                        [r[x_key] for r in selected],
                        [r[f"{variant}_{stat}_ms"] / 1000 for r in selected],
                        style,
                        color=color,
                        label=f"{variant} {stat}",
                    )
            axis.set(
                title=title,
                xlabel="Offered request QPS"
                if is_open
                else "Concurrent logical requests",
                ylabel="Latency (s)",
            )
            axis.grid(alpha=0.3)
            axis.legend()
        filename = "qps_ttft_comparison.png" if is_open else "ttft_comparison.png"
        figure.savefig(output / filename, dpi=180)
        plt.close(figure)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results_dir", type=Path)
    args = parser.parse_args()
    print(json.dumps(compare(args.results_dir), indent=2))
