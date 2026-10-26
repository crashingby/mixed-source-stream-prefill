# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""按相同样本配对 dense/sparse，输出 TTFT 对比表与图。"""

import argparse
import csv
import json
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
    """按并发度配对；内容或事件计划不一致时停止生成误导性对比。"""
    rows = []
    for dense_dir in sorted(root.glob("dense-c*")):
        sparse_dir = root / dense_dir.name.replace("dense-", "sparse-", 1)
        if (
            not (dense_dir / "summary.json").exists()
            or not (sparse_dir / "summary.json").exists()
        ):
            continue
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
            if left["status"] == right["status"] == "ok":
                paired.append((left, right))
        concurrency = int(dense_dir.name.split("-c")[1])
        for metric in ("request_to_first_token_ms", "finish_to_first_token_ms"):
            left_stats = distribution([left[metric] for left, _ in paired])
            right_stats = distribution([right[metric] for _, right in paired])
            row: dict[str, Any] = {
                "concurrency": concurrency,
                "metric": metric,
                "attempted_per_mode": len(dense),
                "paired_successful": len(paired),
                "dense_errors": sum(r["status"] != "ok" for r in dense.values()),
                "sparse_errors": sum(r["status"] != "ok" for r in sparse.values()),
            }
            for mode, stats in (("dense", left_stats), ("sparse", right_stats)):
                for key in ("mean", "p50", "p95", "p99"):
                    row[f"{mode}_{key}_ms"] = stats[key] if stats else None
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
    """绘制均值与 P95；横轴为闭环在途请求数，纵轴为秒。"""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    for axis, metric, title in zip(
        axes,
        ("request_to_first_token_ms", "finish_to_first_token_ms"),
        ("Request start to first token", "Finish to first token"),
    ):
        selected = sorted(
            (r for r in rows if r["metric"] == metric and r["paired_successful"]),
            key=lambda r: r["concurrency"],
        )
        for mode, color in (("dense", "tab:blue"), ("sparse", "tab:orange")):
            for stat, style in (("mean", "-o"), ("p95", "--x")):
                axis.plot(
                    [r["concurrency"] for r in selected],
                    [r[f"{mode}_{stat}_ms"] / 1000 for r in selected],
                    style,
                    color=color,
                    label=f"{mode} {stat}",
                )
        axis.set(title=title, xlabel="Concurrent requests", ylabel="Latency (s)")
        axis.grid(alpha=0.3)
        axis.legend()
    figure.savefig(output / "ttft_comparison.png", dpi=180)
    plt.close(figure)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results_dir", type=Path)
    args = parser.parse_args()
    print(json.dumps(compare(args.results_dir), indent=2))
