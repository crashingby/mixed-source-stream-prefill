# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""绘制三种布局在共同有效请求上的失效 token 与节省量，保存 PNG 和 PDF。"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")  # Slurm 节点无显示设备时也能直接写文件。
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.ticker import PercentFormatter

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

from research.scripts.replay.common.dataset import load_json
from research.scripts.replay.token_invalidation.compare_results import (
    DEFAULT_COMPARISON_DIR,
    LAYOUTS,
)
from research.scripts.replay.token_invalidation.metrics import distribution

LABELS = {
    "update_first": "Update-first",
    "append_first": "Append-first",
    "segmented": "Segmented",
}
COLORS = {
    "update_first": "#d45b35",
    "append_first": "#2864b4",
    "segmented": "#339568",
}


def load_comparison(directory: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """读取配对 CSV 与摘要，确认样本数和计数一致，避免混用不同轮次的文件。

    Args:
        directory: compare_results.py 生成的结果目录。

    Returns:
        共同有效请求的整数计数，以及配对汇总。

    Raises:
        ValueError: 数据为空、计数非法或 CSV 与摘要不一致。
    """
    summary = load_json(directory / "comparison_summary.json")
    if summary.get("schema") != "token_invalidation_comparison/v1":
        raise ValueError(f"不是 token 失效配对结果：{directory}")
    columns = (*LAYOUTS, "saved_vs_update_first", "saved_vs_append_first")
    rows: list[dict[str, Any]] = []
    sample_ids: set[str] = set()
    with (directory / "paired_requests.csv").open(
        encoding="utf-8", newline=""
    ) as stream:
        reader = csv.DictReader(stream)
        if not {"sample_id", *columns}.issubset(reader.fieldnames or []):
            raise ValueError("配对 CSV 缺少样本 ID 或计数列")
        for record in reader:
            sample_id = record["sample_id"]
            if not sample_id or sample_id in sample_ids:
                raise ValueError(f"配对 CSV 的样本 ID 为空或重复：{sample_id}")
            try:
                row = {
                    "sample_id": sample_id,
                    **{key: int(record[key]) for key in columns},
                }
            except (TypeError, ValueError) as exc:
                raise ValueError(f"失效与节省量必须为整数：{sample_id}") from exc
            if any(row[layout] < 0 for layout in LAYOUTS):
                raise ValueError(f"失效 token 数不能为负：{sample_id}")
            for baseline in LAYOUTS[:2]:
                if row[f"saved_vs_{baseline}"] != row[baseline] - row["segmented"]:
                    raise ValueError(f"节省量与失效计数不符：{sample_id}:{baseline}")
            sample_ids.add(sample_id)
            rows.append(row)
    if not rows:
        raise ValueError("没有三种模式共同有效的请求，无法绘图")
    if summary.get("num_paired_valid") != len(rows):
        raise ValueError("CSV 样本数与配对摘要不一致")
    for layout in LAYOUTS:
        if distribution([row[layout] for row in rows]) != summary.get(
            "paired_snapshot_invalidated_tokens", {}
        ).get(layout):
            raise ValueError(f"CSV 计数与配对摘要不一致：{layout}")
    for baseline in LAYOUTS[:2]:
        if distribution([row[f"saved_vs_{baseline}"] for row in rows]) != summary.get(
            "saved_tokens", {}
        ).get(baseline):
            raise ValueError(f"CSV 节省量与配对摘要不一致：{baseline}")
    return rows, summary


def empirical_cdf(values: list[int]) -> tuple[list[int], list[float]]:
    """返回 P(value <= x) 的阶梯点；保留有效零值、重复值和负节省量。"""
    counts = Counter(values)
    ordered = sorted(counts)
    x_values = [ordered[0], *ordered]
    y_values = [0.0]
    cumulative = 0
    for value in ordered:
        cumulative += counts[value]
        y_values.append(cumulative / len(values))
    return x_values, y_values


def build_figure(
    rows: list[dict[str, Any]], summary: dict[str, Any], x_scale: str
) -> Figure:
    """生成总量、均值/P95、失效 CDF 与绝对节省量 CDF 的四联图。

    节省量采用基线减分段；负值保留。三个模式共用相同的配对分母。
    调整 Matplotlib 样式并创建 Figure，由调用方保存和关闭。
    """
    plt.rcParams.update(
        {
            "font.size": 11,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "savefig.bbox": "tight",
        }
    )
    figure, axes = plt.subplots(2, 2, figsize=(13, 9), constrained_layout=True)
    totals_axis, quantiles_axis, cdf_axis, saved_axis = axes.flat
    counts = summary["paired_snapshot_invalidated_tokens"]
    indices = range(len(LAYOUTS))
    totals = [counts[layout]["total"] for layout in LAYOUTS]
    bars = totals_axis.bar(
        indices, totals, color=[COLORS[layout] for layout in LAYOUTS]
    )
    totals_axis.bar_label(bars, labels=[f"{value:,}" for value in totals], padding=3)
    totals_axis.set(
        title="Total invalidated tokens",
        ylabel="Tokens (paired requests)",
        xticks=list(indices),
        xticklabels=[LABELS[layout] for layout in LAYOUTS],
        ylim=(0, max(1, max(totals) * 1.2)),
    )
    for offset, metric, label in ((-0.2, "mean", "Mean"), (0.2, "p95", "P95")):
        quantiles_axis.bar(
            [index + offset for index in indices],
            [counts[layout][metric] for layout in LAYOUTS],
            width=0.38,
            label=label,
            color=[COLORS[layout] for layout in LAYOUTS],
            alpha=0.5 if metric == "mean" else 1,
        )
    quantiles_axis.set(
        title="Invalidation per request",
        ylabel="Tokens / request",
        xticks=list(indices),
        xticklabels=[LABELS[layout] for layout in LAYOUTS],
    )
    quantiles_axis.legend(frameon=False)
    invalidation_right = max(1, max(counts[layout]["max"] for layout in LAYOUTS) * 1.05)
    for layout in LAYOUTS:
        x_values, y_values = empirical_cdf([row[layout] for row in rows])
        # 曲线覆盖共同横轴区间，最大观测值之后仍保持 100%。
        cdf_axis.step(
            [0, *x_values, invalidation_right],
            [0, *y_values, 1],
            where="post",
            color=COLORS[layout],
            label=LABELS[layout],
        )
    savings = [row[f"saved_vs_{baseline}"] for row in rows for baseline in LAYOUTS[:2]]
    margin = max(1, (max(savings) - min(savings)) * 0.05)
    saved_bounds = (min(0, min(savings)) - margin, max(0, max(savings)) + margin)
    for baseline in LAYOUTS[:2]:
        x_values, y_values = empirical_cdf(
            [row[f"saved_vs_{baseline}"] for row in rows]
        )
        # 比例从共同有效样本总量计算；基线为零时不画成 0% 节省。
        total = counts[baseline]["total"]
        saved = sum(row[f"saved_vs_{baseline}"] for row in rows)
        ratio_label = f"{saved / total:.1%}" if total else "undefined"
        saved_axis.step(
            [saved_bounds[0], *x_values, saved_bounds[1]],
            [0, *y_values, 1],
            where="post",
            color=COLORS[baseline],
            label=f"vs {LABELS[baseline]} (total saving: {ratio_label})",
        )
    cdf_axis.set(
        title="Invalidation distribution",
        xlabel="Invalidated tokens",
        xlim=(0, invalidation_right),
    )
    saved_axis.set(
        title="Tokens saved by segmentation",
        xlabel="Baseline - segmented (tokens)",
        xlim=saved_bounds,
    )
    saved_axis.axvline(0, color="#666666", linestyle="--", linewidth=1)
    for axis in (cdf_axis, saved_axis):
        axis.set(ylabel="Paired requests <= x", ylim=(0, 1.02))
        axis.yaxis.set_major_formatter(PercentFormatter(xmax=1))
        if x_scale == "symlog":
            axis.set_xscale("symlog", linthresh=1)
        axis.legend(frameon=False, fontsize=9)
    for axis in axes.flat:
        axis.grid(axis="y", linestyle="--", alpha=0.25)
        axis.set_axisbelow(True)
    figure.suptitle(
        f"Token invalidation: paired n={len(rows):,} / "
        f"{summary['num_selected']:,} selected; "
        f"{summary['num_excluded']:,} excluded"
    )
    return figure


def main() -> None:
    """读取配对结果，默认在其 plots 子目录写出 PNG 与 PDF。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--comparison-dir",
        type=Path,
        default=DEFAULT_COMPARISON_DIR,
        help="默认读取本实验 results/comparison",
    )
    parser.add_argument("--output-dir", type=Path, help="默认使用 comparison-dir/plots")
    parser.add_argument("--dpi", type=int, default=200)
    parser.add_argument(
        "--x-scale",
        choices=("linear", "symlog"),
        default="linear",
        help="CDF 横轴；symlog 同时支持零失效和负节省量",
    )
    args = parser.parse_args()
    if args.dpi <= 0:
        parser.error("--dpi 必须为正整数")
    rows, summary = load_comparison(args.comparison_dir)
    figure = build_figure(rows, summary, args.x_scale)
    output_dir = args.output_dir or args.comparison_dir / "plots"
    output_dir.mkdir(parents=True, exist_ok=True)
    try:
        for suffix in ("png", "pdf"):
            path = output_dir / f"token_invalidation_comparison.{suffix}"
            figure.savefig(path, dpi=args.dpi)
            print(f"Wrote figure to {path}")
    finally:
        plt.close(figure)


if __name__ == "__main__":
    main()
