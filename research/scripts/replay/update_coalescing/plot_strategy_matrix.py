# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""按 Update interval 和 QPS 绘制策略 TTFT 与 token saving 矩阵。"""

# ruff: noqa: I001

from __future__ import annotations

import argparse
import math
from collections.abc import Sequence
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.colors import LogNorm, Normalize
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle
from matplotlib.ticker import PercentFormatter

from plot_coalescing_tradeoff import (
    DEFAULT_SUMMARY_CSV,
    SUPPORTED_OUTPUT_SUFFIXES,
    ConfigurationPoint,
    TtftPercentile,
    filter_points,
    load_configuration_points,
)


DEFAULT_OUTPUT = Path(
    "research/scripts/replay/update_coalescing/results/analysis/"
    "coalescing_strategy_matrix.png"
)
DEFAULT_PROFILE_ORDER = (
    "disabled",
    "immediate-controller",
    "strict-immediate",
    "fixed-window",
    "quiet-debounce",
    "full-merge",
)


def parse_args() -> argparse.Namespace:
    """解析策略矩阵绘图参数。"""
    parser = argparse.ArgumentParser(
        description=(
            "Plot strategy-by-(Update interval, QPS) matrices for TTFT and "
            "Update Prefill token saving."
        )
    )
    parser.add_argument(
        "--summary-csv",
        type=Path,
        default=DEFAULT_SUMMARY_CSV,
        help="configuration_summary.csv produced by summarize_coalescing.py.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Output .png or .pdf path; the suffix selects the file format.",
    )
    parser.add_argument(
        "--qps",
        action="append",
        type=float,
        default=None,
        help="Optional QPS value to retain. Repeat to retain several values.",
    )
    parser.add_argument(
        "--update-chunk-interval",
        action="append",
        type=float,
        default=None,
        help=(
            "Optional Update interval in seconds to retain. Repeat to retain "
            "several intervals."
        ),
    )
    parser.add_argument(
        "--append-chunk-interval",
        action="append",
        type=float,
        default=None,
        help=(
            "Optional Append interval in seconds to retain. Repeat to retain "
            "several intervals."
        ),
    )
    parser.add_argument(
        "--profile",
        action="append",
        default=None,
        help="Optional experiment_profile to retain. Repeat as needed.",
    )
    parser.add_argument(
        "--workload-id",
        action="append",
        default=None,
        help="Optional workload_id to retain. Repeat as needed.",
    )
    parser.add_argument(
        "--min-success-rate",
        type=float,
        default=0.95,
        help=(
            "Cells below this success rate are marked unreliable and excluded "
            "from winner selection. Defaults to 0.95."
        ),
    )
    parser.add_argument(
        "--ttft-percentile",
        choices=("p50", "p90", "p95", "p99"),
        default="p95",
        help="TTFT percentile displayed by the first matrix; defaults to p95.",
    )
    parser.add_argument(
        "--baseline-profile",
        default="disabled",
        help="Profile used for per-cell TTFT relative changes.",
    )
    parser.add_argument(
        "--title",
        default="Update Coalescing Strategy Matrix",
        help="Figure title.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=220,
        help="Output DPI for PNG; accepted but not material for vector PDF.",
    )
    return parser.parse_args()


def _format_number(value: float) -> str:
    """把数值压缩成适合热力图单元格的文本。"""
    if abs(value) >= 1000:
        return f"{value / 1000:.2f}k"
    if abs(value) >= 100:
        return f"{value:.0f}"
    return f"{value:.1f}"


def _profile_sort_key(profile: str) -> tuple[int, str]:
    """优先使用实验文档中的策略顺序，未知策略排在末尾。"""
    try:
        return (DEFAULT_PROFILE_ORDER.index(profile), profile)
    except ValueError:
        return (len(DEFAULT_PROFILE_ORDER), profile)


def _is_reliable(point: ConfigurationPoint, min_success_rate: float) -> bool:
    """判断配置点能否进入正式性能与 winner 比较。"""
    return (
        point.success_rate is not None
        and point.success_rate >= min_success_rate
    )


def _build_point_index(
    points: Sequence[ConfigurationPoint],
) -> dict[tuple[str, float, float], ConfigurationPoint]:
    """建立策略、Update interval、QPS 到唯一配置点的索引。

    同一坐标出现多个 workload 或 P 配置时，矩阵无法给出不歧义的单元格，
    因此要求调用方先通过命令行过滤到唯一实验切片。
    """
    point_index: dict[tuple[str, float, float], ConfigurationPoint] = {}
    for point in points:
        if point.update_interval_s is None:
            raise ValueError(
                "Strategy matrix requires update_chunk_interval_s for every "
                f"point: profile={point.profile!r}, qps={point.qps}."
            )
        coordinate = (point.profile, point.update_interval_s, point.qps)
        previous_point = point_index.get(coordinate)
        if previous_point is not None:
            raise ValueError(
                "Multiple points map to one strategy-matrix cell. Filter by "
                "--workload-id or provide a single P configuration: "
                f"profile={point.profile!r}, "
                f"update_interval={point.update_interval_s}, qps={point.qps}."
            )
        point_index[coordinate] = point
    return point_index


def _get_color_normalization(
    values: Sequence[float],
    *,
    use_log_scale: bool,
) -> Normalize:
    """为有效单元格构造稳定的颜色归一化对象。"""
    if not values:
        return Normalize(vmin=0.0, vmax=1.0)
    min_value = min(values)
    max_value = max(values)
    if math.isclose(min_value, max_value):
        padding = max(abs(min_value) * 0.05, 1.0)
        return Normalize(vmin=max(0.0, min_value - padding), vmax=max_value + padding)
    if use_log_scale and min_value > 0:
        return LogNorm(vmin=min_value, vmax=max_value)
    return Normalize(vmin=min_value, vmax=max_value)


def _add_interval_headers(
    axis: Axes,
    update_intervals: Sequence[float],
    num_qps: int,
) -> None:
    """在矩阵顶部标出 Update interval 分组并添加分隔线。"""
    for interval_index, update_interval in enumerate(update_intervals):
        first_column = interval_index * num_qps
        last_column = first_column + num_qps - 1
        group_center = (first_column + last_column) / 2
        axis.text(
            group_center,
            1.04,
            f"Update interval = {update_interval:g} s",
            transform=axis.get_xaxis_transform(),
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
        )
        if interval_index > 0:
            axis.axvline(first_column - 0.5, color="white", linewidth=3.0)


def _add_winner_border(axis: Axes, row_index: int, column_index: int) -> None:
    """用醒目边框标记同一 workload 坐标下的最优可靠策略。"""
    axis.add_patch(
        Rectangle(
            (column_index - 0.48, row_index - 0.48),
            0.96,
            0.96,
            fill=False,
            edgecolor="#f59e0b",
            linewidth=2.8,
        )
    )


def _draw_ttft_matrix(
    axis: Axes,
    profiles: Sequence[str],
    columns: Sequence[tuple[float, float]],
    point_index: dict[tuple[str, float, float], ConfigurationPoint],
    *,
    percentile: TtftPercentile,
    baseline_profile: str,
    min_success_rate: float,
) -> None:
    """绘制绝对 TTFT、相对 baseline 变化和每列最优策略。"""
    matrix = np.full((len(profiles), len(columns)), np.nan)
    valid_values: list[float] = []
    annotations: dict[tuple[int, int], str] = {}

    for column_index, (update_interval, qps) in enumerate(columns):
        baseline_point = point_index.get((baseline_profile, update_interval, qps))
        baseline_ttft = None
        if baseline_point is not None and _is_reliable(
            baseline_point, min_success_rate
        ):
            baseline_ttft = baseline_point.ttft_ms(percentile)

        reliable_column_values: list[tuple[int, float]] = []
        for row_index, profile in enumerate(profiles):
            point = point_index.get((profile, update_interval, qps))
            if point is None:
                annotations[(row_index, column_index)] = "missing"
                continue
            if not _is_reliable(point, min_success_rate):
                success_percent = (
                    point.success_rate * 100
                    if point.success_rate is not None
                    else math.nan
                )
                annotations[(row_index, column_index)] = (
                    f"{success_percent:.1f}% success\nunreliable"
                )
                continue
            ttft_ms = point.ttft_ms(percentile)
            if ttft_ms is None:
                annotations[(row_index, column_index)] = "TTFT N/A"
                continue

            matrix[row_index, column_index] = ttft_ms
            valid_values.append(ttft_ms)
            reliable_column_values.append((row_index, ttft_ms))
            annotation = f"{_format_number(ttft_ms)} ms"
            if profile == baseline_profile:
                annotation += "\nbaseline"
            elif baseline_ttft is not None and baseline_ttft > 0:
                relative_change = (ttft_ms / baseline_ttft - 1.0) * 100
                annotation += f"\n{relative_change:+.1f}% vs base"
            annotations[(row_index, column_index)] = annotation

        if reliable_column_values:
            best_ttft = min(value for _, value in reliable_column_values)
            for row_index, value in reliable_column_values:
                if math.isclose(value, best_ttft, rel_tol=1e-9, abs_tol=1e-9):
                    _add_winner_border(axis, row_index, column_index)

    color_map = plt.get_cmap("RdYlGn_r").copy()
    color_map.set_bad("#e5e7eb")
    image = axis.imshow(
        matrix,
        aspect="auto",
        cmap=color_map,
        norm=_get_color_normalization(valid_values, use_log_scale=True),
    )
    for (row_index, column_index), annotation in annotations.items():
        axis.text(
            column_index,
            row_index,
            annotation,
            ha="center",
            va="center",
            fontsize=7.2,
            color="#111827",
        )
    axis.set_title(f"{percentile.upper()} TTFT (lower is better)", pad=32)
    color_bar = axis.figure.colorbar(image, ax=axis, pad=0.012)
    color_bar.set_label("TTFT (ms)")


def _draw_token_saving_matrix(
    axis: Axes,
    profiles: Sequence[str],
    columns: Sequence[tuple[float, float]],
    point_index: dict[tuple[str, float, float], ConfigurationPoint],
    *,
    min_success_rate: float,
) -> None:
    """绘制 token saving ratio、每请求节省量和每列最优策略。"""
    matrix = np.full((len(profiles), len(columns)), np.nan)
    valid_values: list[float] = []
    annotations: dict[tuple[int, int], str] = {}

    for column_index, (update_interval, qps) in enumerate(columns):
        reliable_column_values: list[tuple[int, float]] = []
        for row_index, profile in enumerate(profiles):
            point = point_index.get((profile, update_interval, qps))
            if point is None:
                annotations[(row_index, column_index)] = "missing"
                continue
            if not _is_reliable(point, min_success_rate):
                success_percent = (
                    point.success_rate * 100
                    if point.success_rate is not None
                    else math.nan
                )
                annotations[(row_index, column_index)] = (
                    f"{success_percent:.1f}% success\nunreliable"
                )
                continue
            if point.token_saving_ratio is None:
                annotations[(row_index, column_index)] = "N/A"
                continue

            saving_ratio = point.token_saving_ratio
            matrix[row_index, column_index] = saving_ratio
            valid_values.append(saving_ratio)
            reliable_column_values.append((row_index, saving_ratio))
            annotation = f"{saving_ratio * 100:.1f}%"
            if point.mean_saved_tokens is not None:
                annotation += (
                    f"\n{_format_number(point.mean_saved_tokens)} tok/req"
                )
            annotations[(row_index, column_index)] = annotation

        if reliable_column_values:
            best_saving = max(value for _, value in reliable_column_values)
            for row_index, value in reliable_column_values:
                if math.isclose(value, best_saving, rel_tol=1e-9, abs_tol=1e-9):
                    _add_winner_border(axis, row_index, column_index)

    color_map = plt.get_cmap("YlGn").copy()
    color_map.set_bad("#e5e7eb")
    max_value = max(valid_values, default=1.0)
    image = axis.imshow(
        matrix,
        aspect="auto",
        cmap=color_map,
        norm=Normalize(vmin=0.0, vmax=max(max_value, 0.01)),
    )
    for (row_index, column_index), annotation in annotations.items():
        axis.text(
            column_index,
            row_index,
            annotation,
            ha="center",
            va="center",
            fontsize=7.2,
            color="#111827",
        )
    axis.set_title("Update Prefill token saving (higher is better)", pad=32)
    color_bar = axis.figure.colorbar(image, ax=axis, pad=0.012)
    color_bar.ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0))
    color_bar.set_label("Token saving ratio")


def _configure_matrix_axis(
    axis: Axes,
    profiles: Sequence[str],
    update_intervals: Sequence[float],
    qps_values: Sequence[float],
) -> None:
    """设置矩阵坐标标签、interval 分组和单元格网格。"""
    columns = len(update_intervals) * len(qps_values)
    axis.set_xticks(range(columns))
    axis.set_xticklabels(
        [f"QPS {qps:g}" for _ in update_intervals for qps in qps_values],
        rotation=35,
        ha="right",
    )
    axis.set_yticks(range(len(profiles)))
    axis.set_yticklabels(profiles)
    axis.set_xticks(np.arange(-0.5, columns, 1), minor=True)
    axis.set_yticks(np.arange(-0.5, len(profiles), 1), minor=True)
    axis.grid(which="minor", color="white", linewidth=1.0)
    axis.tick_params(which="minor", bottom=False, left=False)
    _add_interval_headers(axis, update_intervals, len(qps_values))


def plot_strategy_matrix(
    points: Sequence[ConfigurationPoint],
    output_path: Path,
    *,
    percentile: TtftPercentile,
    baseline_profile: str,
    min_success_rate: float,
    title: str,
    dpi: int,
) -> None:
    """把选中的配置点绘制为 TTFT 和 token saving 双矩阵。

    灰色单元格表示数据缺失、不适用或成功率低于阈值。橙色边框只在可靠
    配置中标记每个 `(Update interval, QPS)` 列的最优策略。

    Args:
        points: 经过 workload 和实验参数过滤的配置点。
        output_path: 输出 PNG 或 PDF 文件。
        percentile: TTFT 矩阵使用的百分位。
        baseline_profile: TTFT 相对变化使用的基线策略。
        min_success_rate: 进入性能比较所需的最低成功率。
        title: 整张图的标题。
        dpi: PNG 输出分辨率。
    """
    if not points:
        raise ValueError("No configuration points remain after filtering.")
    if not 0.0 <= min_success_rate <= 1.0:
        raise ValueError("--min-success-rate must be between 0 and 1.")

    workload_ids = {point.workload_id for point in points}
    if len(workload_ids) != 1:
        raise ValueError(
            "Strategy matrix requires exactly one workload_id. Use "
            f"--workload-id to select one of: {sorted(workload_ids)}."
        )

    profiles = sorted({point.profile for point in points}, key=_profile_sort_key)
    update_intervals = sorted(
        {
            point.update_interval_s
            for point in points
            if point.update_interval_s is not None
        },
        reverse=True,
    )
    qps_values = sorted({point.qps for point in points})
    columns = [
        (update_interval, qps)
        for update_interval in update_intervals
        for qps in qps_values
    ]
    point_index = _build_point_index(points)

    figure_width = max(14.0, len(columns) * 1.35)
    figure_height = max(9.0, len(profiles) * 1.25 + 3.5)
    figure, axes = plt.subplots(
        nrows=2,
        ncols=1,
        figsize=(figure_width, figure_height),
        constrained_layout=True,
    )
    assert isinstance(figure, Figure)
    ttft_axis, saving_axis = axes

    _draw_ttft_matrix(
        ttft_axis,
        profiles,
        columns,
        point_index,
        percentile=percentile,
        baseline_profile=baseline_profile,
        min_success_rate=min_success_rate,
    )
    _draw_token_saving_matrix(
        saving_axis,
        profiles,
        columns,
        point_index,
        min_success_rate=min_success_rate,
    )
    for axis in axes:
        _configure_matrix_axis(
            axis,
            profiles,
            update_intervals,
            qps_values,
        )

    figure.suptitle(title, fontsize=16, fontweight="bold")
    figure.text(
        0.5,
        -0.015,
        "Orange border = best reliable strategy in the column; gray = N/A, "
        f"missing, or success rate < {min_success_rate:.0%}.",
        ha="center",
        fontsize=9,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    """加载、过滤并绘制策略矩阵。"""
    args = parse_args()
    output_suffix = args.output.suffix.lower()
    if output_suffix not in SUPPORTED_OUTPUT_SUFFIXES:
        raise ValueError(
            "--output must end in one of: "
            f"{', '.join(sorted(SUPPORTED_OUTPUT_SUFFIXES))}."
        )

    points = load_configuration_points(args.summary_csv)
    selected_points = filter_points(
        points,
        qps_values=args.qps,
        append_intervals=args.append_chunk_interval,
        update_intervals=args.update_chunk_interval,
        profiles=args.profile,
        workload_ids=args.workload_id,
    )
    plot_strategy_matrix(
        selected_points,
        args.output,
        percentile=args.ttft_percentile,
        baseline_profile=args.baseline_profile,
        min_success_rate=args.min_success_rate,
        title=args.title,
        dpi=args.dpi,
    )
    print(f"Wrote strategy matrix to {args.output}.")


if __name__ == "__main__":
    main()
