# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""绘制 masked-padding 开关的 TTFT、TPOT 与吞吐对比图。"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt

PROFILE_LABELS = {
    "masked-padding-disabled": "Padding participates in attention",
    "masked-padding-enabled": "Masked padding",
}
PROFILE_COLORS = {
    "masked-padding-disabled": "#D62728",
    "masked-padding-enabled": "#1F77B4",
}


def _parse_args() -> argparse.Namespace:
    """解析汇总 CSV 与图片输出目录。"""
    parser = argparse.ArgumentParser(
        description="Plot masked-padding enabled/disabled performance."
    )
    parser.add_argument(
        "--run-summary-csv",
        type=Path,
        required=True,
        help="run_summary.csv produced by summarize_masked_padding.py.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory for generated PNG figures.",
    )
    return parser.parse_args()


def _load_rows(path: Path) -> list[dict[str, str]]:
    """读取逐轮汇总 CSV。"""
    with path.open(encoding="utf-8", newline="") as csv_file:
        return list(csv.DictReader(csv_file))


def _group_mean(
    rows: list[dict[str, str]],
    phase: str,
    field_name: str,
) -> dict[str, tuple[list[float], list[float]]]:
    """按 profile/QPS 聚合一个指标的重复实验均值。"""
    grouped: dict[tuple[str, float], list[float]] = defaultdict(list)
    for row in rows:
        raw_value = row.get(field_name)
        if row.get("phase") != phase or raw_value in (None, ""):
            continue
        grouped[(row["profile"], float(row["qps"]))].append(float(raw_value))

    series: dict[str, tuple[list[float], list[float]]] = {}
    for profile in PROFILE_LABELS:
        qps_values = sorted(
            qps for candidate_profile, qps in grouped if candidate_profile == profile
        )
        means = [
            sum(grouped[(profile, qps)]) / len(grouped[(profile, qps)])
            for qps in qps_values
        ]
        series[profile] = (qps_values, means)
    return series


def _plot_metric(
    axis: plt.Axes,
    rows: list[dict[str, str]],
    phase: str,
    field_name: str,
    title: str,
    y_label: str,
) -> None:
    """在一个子图中绘制两种 profile 的 QPS 曲线。"""
    for profile, (qps_values, metric_values) in _group_mean(
        rows, phase, field_name
    ).items():
        if not qps_values:
            continue
        axis.plot(
            qps_values,
            metric_values,
            marker="o",
            linewidth=2,
            color=PROFILE_COLORS[profile],
            label=PROFILE_LABELS[profile],
        )
    axis.set_title(title)
    axis.set_xlabel("Offered load (requests/s)")
    axis.set_ylabel(y_label)
    axis.grid(alpha=0.25, linestyle="--")


def _plot_ttft_only(rows: list[dict[str, str]], output_dir: Path) -> None:
    """绘制 P-only TTFT 曲线。"""
    figure, axis = plt.subplots(figsize=(7.2, 4.8), constrained_layout=True)
    _plot_metric(
        axis,
        rows,
        "ttft",
        "ttft_p95_ms",
        "P-only TTFT (p95)",
        "TTFT (ms)",
    )
    axis.legend(frameon=True)
    figure.savefig(output_dir / "masked_padding_ttft_only.png", dpi=180)
    plt.close(figure)


def _plot_full_generation(rows: list[dict[str, str]], output_dir: Path) -> None:
    """绘制完整 P→D 的 TTFT、TPOT 与 aggregate TPS。"""
    figure, axes = plt.subplots(1, 3, figsize=(15.6, 4.8), constrained_layout=True)
    _plot_metric(
        axes[0],
        rows,
        "full",
        "ttft_p95_ms",
        "Full P→D TTFT (p95)",
        "TTFT (ms)",
    )
    _plot_metric(
        axes[1],
        rows,
        "full",
        "tpot_p50_ms",
        "Per-request TPOT (p50)",
        "TPOT (ms/token)",
    )
    _plot_metric(
        axes[2],
        rows,
        "full",
        "aggregate_output_tps",
        "Aggregate output throughput",
        "Output tokens/s",
    )
    axes[0].legend(frameon=True)
    figure.savefig(output_dir / "masked_padding_full_generation.png", dpi=180)
    plt.close(figure)


def main() -> None:
    """读取汇总结果并生成两张对比图。"""
    args = _parse_args()
    rows = _load_rows(args.run_summary_csv)
    if not rows:
        raise ValueError(f"No rows found in {args.run_summary_csv}.")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    _plot_ttft_only(rows, args.output_dir)
    _plot_full_generation(rows, args.output_dir)
    print(f"Wrote masked-padding figures to {args.output_dir}")


if __name__ == "__main__":
    main()
