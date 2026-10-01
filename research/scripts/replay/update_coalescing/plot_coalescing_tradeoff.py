# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""绘制 Update 合并的延迟、计算节省、实际失效和 Pareto 权衡四联图。"""

# ruff: noqa: I001

from __future__ import annotations

import argparse
import csv
import math
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.ticker import PercentFormatter


DEFAULT_SUMMARY_CSV = Path(
    "research/scripts/replay/update_coalescing/results/analysis/"
    "configuration_summary.csv"
)
DEFAULT_OUTPUT = Path(
    "research/scripts/replay/update_coalescing/results/analysis/"
    "coalescing_tradeoff.png"
)
TtftPercentile = Literal["p50", "p90", "p95", "p99"]
SUPPORTED_OUTPUT_SUFFIXES = frozenset((".png", ".pdf"))


@dataclass(frozen=True)
class ConfigurationPoint:
    """configuration_summary.csv 中的一个实验点。"""

    profile: str
    workload_id: str
    qps: float
    append_interval_s: float | None
    update_interval_s: float | None
    coalescing_config_json: str
    update_policy: str
    soft_window_ms: float | None
    hard_deadline_ms: float | None
    num_requests: int
    num_runs: int
    success_rate: float | None
    token_stats_coverage: float | None
    formal_valid_rate: float | None
    ttft_p50_ms: float | None
    ttft_p90_ms: float | None
    ttft_p95_ms: float | None
    ttft_p99_ms: float | None
    mean_saved_tokens: float | None
    mean_invalidated_tokens: float | None
    token_saving_ratio: float | None
    # 压力参数只影响 load_aware；旧结果缺少字段时仍可按原配置绘图。
    load_aware_token_pressure_threshold: float | None = None
    load_aware_slot_pressure_threshold: float | None = None

    @property
    def series_key(self) -> tuple[str, float | None, str, str]:
        """返回跨 QPS 连线时使用的稳定配置与 workload 标识。"""
        return (
            self.profile,
            self.update_interval_s,
            self.workload_id,
            self.coalescing_config_json,
        )

    @property
    def pareto_slice_key(self) -> tuple[str, float, float | None]:
        """返回允许互相比较 Pareto 支配关系的 workload 切片。"""
        return (self.workload_id, self.qps, self.update_interval_s)

    def ttft_ms(self, percentile: TtftPercentile) -> float | None:
        """返回指定百分位的 TTFT。"""
        return getattr(self, f"ttft_{percentile}_ms")


@dataclass(frozen=True)
class PlotSeries:
    """同一 profile、Update interval 和 P 配置的 QPS 序列。"""

    label: str
    points: tuple[ConfigurationPoint, ...]


def parse_args() -> argparse.Namespace:
    """解析四联图命令行参数。"""
    parser = argparse.ArgumentParser(
        description=(
            "Plot TTFT, saved tokens, invalidated tokens, and the "
            "token-saving-vs-P95-TTFT Pareto trade-off from "
            "configuration_summary.csv."
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
            "Reject selected configuration points below this success rate. "
            "Set to 0 only when deliberately inspecting overload failures."
        ),
    )
    parser.add_argument(
        "--min-stats-coverage",
        type=float,
        default=0.95,
        help=(
            "Reject selected points whose recovery-safe token statistics cover "
            "less than this fraction of first-token successes. Defaults to 0.95."
        ),
    )
    parser.add_argument(
        "--ttft-percentile",
        choices=("p50", "p90", "p95", "p99"),
        default="p95",
        help="TTFT percentile used by the first panel; defaults to p95.",
    )
    parser.add_argument(
        "--title",
        default="Update Coalescing Trade-off",
        help="Figure title.",
    )
    parser.add_argument(
        "--annotate-pareto",
        action="store_true",
        help="Annotate Pareto-panel points with profile and QPS.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=220,
        help="Output DPI for PNG; accepted but not material for vector PDF.",
    )
    return parser.parse_args()


def _optional_float(value: str | None, *, context: str) -> float | None:
    """解析可选浮点 CSV 字段，并拒绝非有限数值。"""
    if value is None or not value.strip():
        return None
    try:
        numeric_value = float(value)
    except ValueError as exc:
        raise ValueError(f"Invalid float for {context}: {value!r}.") from exc
    if not math.isfinite(numeric_value):
        raise ValueError(f"Non-finite float for {context}: {value!r}.")
    return numeric_value


def _required_float(value: str | None, *, context: str) -> float:
    """解析必需浮点 CSV 字段。"""
    numeric_value = _optional_float(value, context=context)
    if numeric_value is None:
        raise ValueError(f"Missing required float for {context}.")
    return numeric_value


def _required_int(value: str | None, *, context: str) -> int:
    """解析必需整数 CSV 字段。"""
    numeric_value = _required_float(value, context=context)
    integer_value = int(numeric_value)
    if integer_value != numeric_value:
        raise ValueError(f"Expected integer for {context}: {value!r}.")
    return integer_value


def _require_columns(
    field_names: Sequence[str] | None,
    required_fields: Iterable[str],
    *,
    summary_path: Path,
) -> None:
    """验证 configuration summary 是否包含绘图所需字段。"""
    if field_names is None:
        raise ValueError(f"Missing CSV header in {summary_path}.")
    missing_fields = sorted(set(required_fields) - set(field_names))
    if missing_fields:
        raise ValueError(
            f"{summary_path} is missing required columns: "
            f"{', '.join(missing_fields)}."
        )


def _point_from_row(
    row: Mapping[str, str],
    *,
    row_index: int,
) -> ConfigurationPoint:
    """把一行 configuration summary 转换为强类型实验点。"""
    context_prefix = f"configuration row {row_index}"
    update_policy = row.get("update_policy", "unknown") or "unknown"
    mean_saved_tokens = _optional_float(
        row.get("mean_saved_update_prefill_tokens_per_counterfactual_request"),
        context=f"{context_prefix}:mean_saved_tokens",
    )
    mean_invalidated_tokens = _optional_float(
        row.get("mean_update_tokens_invalidated_per_stats_request"),
        context=f"{context_prefix}:mean_invalidated_tokens",
    )
    token_saving_ratio = _optional_float(
        row.get("token_saving_ratio"),
        context=f"{context_prefix}:token_saving_ratio",
    )

    return ConfigurationPoint(
        profile=row.get("experiment_profile", "unknown") or "unknown",
        workload_id=row.get("workload_id", "unknown") or "unknown",
        qps=_required_float(row.get("qps"), context=f"{context_prefix}:qps"),
        append_interval_s=_optional_float(
            row.get("append_chunk_interval_s"),
            context=f"{context_prefix}:append_chunk_interval_s",
        ),
        update_interval_s=_optional_float(
            row.get("update_chunk_interval_s"),
            context=f"{context_prefix}:update_chunk_interval_s",
        ),
        coalescing_config_json=(
            row.get("coalescing_config_json", "null") or "null"
        ),
        update_policy=update_policy,
        soft_window_ms=_optional_float(
            row.get("soft_window_ms"),
            context=f"{context_prefix}:soft_window_ms",
        ),
        hard_deadline_ms=_optional_float(
            row.get("hard_deadline_ms"),
            context=f"{context_prefix}:hard_deadline_ms",
        ),
        load_aware_token_pressure_threshold=_optional_float(
            row.get("load_aware_token_pressure_threshold"),
            context=f"{context_prefix}:load_aware_token_pressure_threshold",
        ),
        load_aware_slot_pressure_threshold=_optional_float(
            row.get("load_aware_slot_pressure_threshold"),
            context=f"{context_prefix}:load_aware_slot_pressure_threshold",
        ),
        num_requests=_required_int(
            row.get("num_requests"),
            context=f"{context_prefix}:num_requests",
        ),
        num_runs=_required_int(
            row.get("num_runs"),
            context=f"{context_prefix}:num_runs",
        ),
        success_rate=_optional_float(
            row.get("success_rate"),
            context=f"{context_prefix}:success_rate",
        ),
        token_stats_coverage=_optional_float(
            row.get("token_stats_coverage"),
            context=f"{context_prefix}:token_stats_coverage",
        ),
        formal_valid_rate=_optional_float(
            row.get("formal_valid_rate"),
            context=f"{context_prefix}:formal_valid_rate",
        ),
        ttft_p50_ms=_optional_float(
            row.get("ttft_p50_ms"), context=f"{context_prefix}:ttft_p50_ms"
        ),
        ttft_p90_ms=_optional_float(
            row.get("ttft_p90_ms"), context=f"{context_prefix}:ttft_p90_ms"
        ),
        ttft_p95_ms=_optional_float(
            row.get("ttft_p95_ms"), context=f"{context_prefix}:ttft_p95_ms"
        ),
        ttft_p99_ms=_optional_float(
            row.get("ttft_p99_ms"), context=f"{context_prefix}:ttft_p99_ms"
        ),
        mean_saved_tokens=mean_saved_tokens,
        mean_invalidated_tokens=mean_invalidated_tokens,
        token_saving_ratio=token_saving_ratio,
    )


def load_configuration_points(summary_path: Path) -> list[ConfigurationPoint]:
    """读取 configuration_summary.csv 的全部配置点。"""
    required_fields = (
        "experiment_profile",
        "workload_id",
        "qps",
        "append_chunk_interval_s",
        "update_chunk_interval_s",
        "coalescing_config_json",
        "update_policy",
        "num_requests",
        "num_runs",
        "success_rate",
        "token_stats_coverage",
        "formal_valid_rate",
        "ttft_p50_ms",
        "ttft_p90_ms",
        "ttft_p95_ms",
        "ttft_p99_ms",
        "mean_saved_update_prefill_tokens_per_counterfactual_request",
        "mean_update_tokens_invalidated_per_stats_request",
        "token_saving_ratio",
    )
    try:
        with summary_path.open(encoding="utf-8", newline="") as csv_file:
            reader = csv.DictReader(csv_file)
            _require_columns(
                reader.fieldnames,
                required_fields,
                summary_path=summary_path,
            )
            return [
                _point_from_row(row, row_index=row_index)
                for row_index, row in enumerate(reader, start=2)
            ]
    except OSError as exc:
        raise ValueError(f"Cannot read configuration summary {summary_path}: {exc}")


def _matches_any(value: float, expected_values: Sequence[float]) -> bool:
    """使用稳定容差匹配命令行传入的浮点过滤值。"""
    return any(
        math.isclose(value, expected, rel_tol=1e-9, abs_tol=1e-12)
        for expected in expected_values
    )


def filter_points(
    points: Sequence[ConfigurationPoint],
    *,
    qps_values: Sequence[float] | None,
    append_intervals: Sequence[float] | None,
    update_intervals: Sequence[float] | None,
    profiles: Sequence[str] | None,
    workload_ids: Sequence[str] | None,
) -> list[ConfigurationPoint]:
    """按 workload、QPS、source interval 和 profile 过滤实验点。"""
    selected_points: list[ConfigurationPoint] = []
    for point in points:
        if qps_values and not _matches_any(point.qps, qps_values):
            continue
        if append_intervals:
            if point.append_interval_s is None or not _matches_any(
                point.append_interval_s,
                append_intervals,
            ):
                continue
        if update_intervals:
            if point.update_interval_s is None or not _matches_any(
                point.update_interval_s,
                update_intervals,
            ):
                continue
        if profiles and point.profile not in profiles:
            continue
        if workload_ids and point.workload_id not in workload_ids:
            continue
        selected_points.append(point)
    return selected_points


def _format_interval(interval_s: float | None) -> str:
    """返回适合图例的 Update interval 文本。"""
    return "unknown" if interval_s is None else f"{interval_s:g}s"


def _format_config_suffix(point: ConfigurationPoint) -> str:
    """在 profile 标签不唯一时生成简短的策略参数后缀。"""
    fields = [point.update_policy]
    if point.update_policy == "load_aware":
        # 新策略不使用 window/deadline，避免把配置默认值画成实际计时规则。
        token_threshold = point.load_aware_token_pressure_threshold
        slot_threshold = point.load_aware_slot_pressure_threshold
        if token_threshold is not None:
            fields.append(f"tokens={token_threshold:g}")
        if slot_threshold is not None:
            fields.append(f"slots={slot_threshold:g}")
        return ", ".join(fields)
    if point.soft_window_ms is not None:
        fields.append(f"w={point.soft_window_ms:g}ms")
    if point.hard_deadline_ms is not None:
        fields.append(f"h={point.hard_deadline_ms:g}ms")
    return ", ".join(fields)


def build_plot_series(
    points: Sequence[ConfigurationPoint],
) -> list[PlotSeries]:
    """按稳定 P 配置与 workload 分组，并生成无歧义图例。"""
    points_by_key: dict[
        tuple[str, float | None, str, str], list[ConfigurationPoint]
    ] = defaultdict(list)
    for point in points:
        points_by_key[point.series_key].append(point)

    profile_interval_counts = Counter(
        (profile, interval) for profile, interval, _, _ in points_by_key
    )
    num_intervals = len({point.update_interval_s for point in points})
    num_workloads = len({point.workload_id for point in points})
    series_list: list[PlotSeries] = []
    for series_key, grouped_points in sorted(
        points_by_key.items(),
        key=lambda item: tuple(str(value) for value in item[0]),
    ):
        profile, interval_s, workload_id, _ = series_key
        label = profile
        if num_intervals > 1:
            label = f"{label}, update interval={_format_interval(interval_s)}"
        if num_workloads > 1:
            label = f"{label}, workload={workload_id[:8]}"
        if profile_interval_counts[(profile, interval_s)] > 1:
            label = f"{label}, {_format_config_suffix(grouped_points[0])}"
        series_list.append(
            PlotSeries(
                label=label,
                points=tuple(sorted(grouped_points, key=lambda point: point.qps)),
            )
        )
    return series_list


def _pareto_frontier(
    points: Sequence[tuple[float, float]],
) -> list[tuple[float, float]]:
    """返回最大化 token saving、最小化 P95 TTFT 的非支配前沿。"""
    frontier: list[tuple[float, float]] = []
    for candidate_x, candidate_y in points:
        is_dominated = any(
            other_x >= candidate_x
            and other_y <= candidate_y
            and (other_x > candidate_x or other_y < candidate_y)
            for other_x, other_y in points
        )
        if not is_dominated:
            frontier.append((candidate_x, candidate_y))
    return sorted(set(frontier), key=lambda point: (point[0], point[1]))


def _configure_axes(axes: Sequence[Axes], percentile: TtftPercentile) -> None:
    """统一四幅子图的标签、网格和可读性设置。"""
    ttft_axis, saved_axis, invalidated_axis, pareto_axis = axes
    ttft_axis.set_title(f"TTFT {percentile.upper()} vs QPS")
    ttft_axis.set_xlabel("Offered load (QPS)")
    ttft_axis.set_ylabel(f"TTFT {percentile.upper()} (ms)")

    saved_axis.set_title("Saved Update Prefill Tokens vs QPS")
    saved_axis.set_xlabel("Offered load (QPS)")
    saved_axis.set_ylabel(
        "Mean saved tokens per counterfactual request"
    )

    invalidated_axis.set_title("Invalidated Update KV Tokens vs QPS")
    invalidated_axis.set_xlabel("Offered load (QPS)")
    invalidated_axis.set_ylabel("Mean invalidated tokens per stats request")

    pareto_axis.set_title("Token Saving vs P95 TTFT")
    pareto_axis.set_xlabel("Token saving ratio")
    pareto_axis.set_ylabel("TTFT P95 (ms)")
    pareto_axis.xaxis.set_major_formatter(PercentFormatter(xmax=100.0))

    for axis in axes:
        axis.grid(True, linestyle="--", linewidth=0.7, alpha=0.35)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)


def plot_tradeoff(
    points: Sequence[ConfigurationPoint],
    *,
    percentile: TtftPercentile,
    title: str,
    annotate_pareto: bool,
) -> Figure:
    """构造 Update 合并四联图并返回 matplotlib Figure。"""
    series_list = build_plot_series(points)
    if not series_list:
        raise ValueError("No configuration points remain for plotting.")

    figure, axes_array = plt.subplots(2, 2, figsize=(14.5, 10.2))
    axes = list(axes_array.flat)
    _configure_axes(axes, percentile)
    color_map = plt.get_cmap("tab20")
    pareto_points_by_slice: dict[
        tuple[str, float, float | None], list[tuple[float, float]]
    ] = defaultdict(list)

    for series_index, series in enumerate(series_list):
        color = color_map(series_index % color_map.N)
        ttft_points = [
            (point.qps, point.ttft_ms(percentile))
            for point in series.points
            if point.ttft_ms(percentile) is not None
        ]
        if ttft_points:
            axes[0].plot(
                [point[0] for point in ttft_points],
                [float(point[1]) for point in ttft_points],
                marker="o",
                linewidth=2.0,
                markersize=5.5,
                color=color,
                label=series.label,
            )

        saved_points = [
            (point.qps, point.mean_saved_tokens)
            for point in series.points
            if point.mean_saved_tokens is not None
        ]
        if saved_points:
            axes[1].plot(
                [point[0] for point in saved_points],
                [float(point[1]) for point in saved_points],
                marker="s",
                linewidth=2.0,
                markersize=5.2,
                color=color,
                label=series.label,
            )

        invalidated_points = [
            (point.qps, point.mean_invalidated_tokens)
            for point in series.points
            if point.mean_invalidated_tokens is not None
        ]
        if invalidated_points:
            axes[2].plot(
                [point[0] for point in invalidated_points],
                [float(point[1]) for point in invalidated_points],
                marker="^",
                linewidth=2.0,
                markersize=5.4,
                color=color,
                label=series.label,
            )

        for point in series.points:
            if point.token_saving_ratio is None or point.ttft_p95_ms is None:
                continue
            saving_percent = point.token_saving_ratio * 100.0
            pareto_coordinate = (saving_percent, point.ttft_p95_ms)
            pareto_points_by_slice[point.pareto_slice_key].append(
                pareto_coordinate
            )
            axes[3].scatter(
                saving_percent,
                point.ttft_p95_ms,
                s=48,
                color=color,
                edgecolor="white",
                linewidth=0.7,
                alpha=0.92,
            )
            if annotate_pareto:
                axes[3].annotate(
                    f"{point.profile}\nQPS={point.qps:g}",
                    pareto_coordinate,
                    xytext=(5, 5),
                    textcoords="offset points",
                    fontsize=7.5,
                    alpha=0.85,
                )

    frontier: list[tuple[float, float]] = []
    if len(pareto_points_by_slice) == 1:
        frontier = _pareto_frontier(next(iter(pareto_points_by_slice.values())))
        if frontier:
            axes[3].plot(
                [point[0] for point in frontier],
                [point[1] for point in frontier],
                color="#202020",
                linestyle="--",
                linewidth=1.5,
                alpha=0.8,
                label="Pareto frontier",
            )
    elif pareto_points_by_slice:
        axes[3].set_title(
            "Token Saving vs P95 TTFT (filter to one workload/QPS/interval "
            "for a frontier)"
        )

    # 只在第一幅图放完整策略图例，避免四幅图重复占用画布。
    if axes[0].get_legend_handles_labels()[0]:
        axes[0].legend(fontsize=8, frameon=True, framealpha=0.92)
    if frontier:
        axes[3].legend(fontsize=8, frameon=True, framealpha=0.92)
    coverage_values = [
        point.token_stats_coverage
        for point in points
        if point.token_stats_coverage is not None
    ]
    formal_values = [
        point.formal_valid_rate
        for point in points
        if point.formal_valid_rate is not None
    ]
    coverage_label = (
        f"{min(coverage_values):.1%}-{max(coverage_values):.1%}"
        if coverage_values
        else "unknown"
    )
    formal_label = (
        f"{min(formal_values):.1%}-{max(formal_values):.1%}"
        if formal_values
        else "unknown"
    )
    figure.suptitle(title, fontsize=15, fontweight="bold")
    figure.text(
        0.5,
        0.01,
        f"Stats coverage: {coverage_label}; formal-valid rate: {formal_label}",
        ha="center",
        fontsize=9,
    )
    figure.tight_layout(rect=(0.0, 0.04, 1.0, 0.95))
    return figure


def _validate_args(args: argparse.Namespace) -> None:
    """在读取 CSV 前检查输出格式和数值过滤条件。"""
    if args.output.suffix.lower() not in SUPPORTED_OUTPUT_SUFFIXES:
        raise ValueError("--output must end with .png or .pdf.")
    if args.dpi <= 0:
        raise ValueError("--dpi must be positive.")
    if not 0.0 <= args.min_success_rate <= 1.0:
        raise ValueError(
            "--min-success-rate must be between 0 and 1."
        )
    if not 0.0 <= args.min_stats_coverage <= 1.0:
        raise ValueError(
            "--min-stats-coverage must be between 0 and 1."
        )
    if args.qps and any(qps < 0 for qps in args.qps):
        raise ValueError("--qps filter values must be non-negative.")
    if args.append_chunk_interval and any(
        interval < 0 for interval in args.append_chunk_interval
    ):
        raise ValueError(
            "--append-chunk-interval filter values must be non-negative."
        )
    if args.update_chunk_interval and any(
        interval < 0 for interval in args.update_chunk_interval
    ):
        raise ValueError(
            "--update-chunk-interval filter values must be non-negative."
        )


def main() -> None:
    """加载、过滤并保存 Update 合并四联图。"""
    args = parse_args()
    _validate_args(args)
    all_points = load_configuration_points(args.summary_csv)
    selected_points = filter_points(
        all_points,
        qps_values=args.qps,
        append_intervals=args.append_chunk_interval,
        update_intervals=args.update_chunk_interval,
        profiles=args.profile,
        workload_ids=args.workload_id,
    )
    if not selected_points:
        raise ValueError("No configuration points remain after applying filters.")
    low_success_points = [
        point
        for point in selected_points
        if point.success_rate is None
        or point.success_rate < args.min_success_rate
    ]
    if low_success_points:
        descriptions = ", ".join(
            f"{point.profile}@qps={point.qps:g}:"
            f"success={point.success_rate!r},"
            f"coverage={point.token_stats_coverage!r},"
            f"formal={point.formal_valid_rate!r}"
            for point in low_success_points[:8]
        )
        raise ValueError(
            "Selected points do not meet --min-success-rate="
            f"{args.min_success_rate:g}: {descriptions}."
        )
    low_coverage_points = [
        point
        for point in selected_points
        if point.token_stats_coverage is None
        or point.token_stats_coverage < args.min_stats_coverage
    ]
    if low_coverage_points:
        descriptions = ", ".join(
            f"{point.profile}@qps={point.qps:g}:"
            f"coverage={point.token_stats_coverage!r},"
            f"formal={point.formal_valid_rate!r}"
            for point in low_coverage_points[:8]
        )
        raise ValueError(
            "Selected points do not meet --min-stats-coverage="
            f"{args.min_stats_coverage:g}: {descriptions}."
        )

    figure = plot_tradeoff(
        selected_points,
        percentile=args.ttft_percentile,
        title=args.title,
        annotate_pareto=args.annotate_pareto,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(
        args.output,
        dpi=args.dpi,
        bbox_inches="tight",
        facecolor="white",
    )
    plt.close(figure)
    print(
        f"Saved {len(selected_points)} configuration points to {args.output}."
    )


if __name__ == "__main__":
    main()
