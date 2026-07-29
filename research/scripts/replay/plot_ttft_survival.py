# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""绘制流式 prefill 与普通模式的 TTFT survival curve。"""

# ruff: noqa: I001

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter


DEFAULT_OUTPUT = Path("research/scripts/replay/results/ttft_survival.png")
DEFAULT_STREAM_TTFT_COLUMN = "finish_to_first_token_ms"
DEFAULT_NORMAL_TTFT_COLUMN = "request_to_first_token_ms"


@dataclass(frozen=True)
class TtftSeries:
    """一条 survival curve 所需的 TTFT 数据。"""

    label: str
    result_dir: Path
    ttft_column: str
    ttft_ms: list[float]
    num_rows: int
    num_success_rows: int


def parse_args() -> argparse.Namespace:
    """解析画图脚本参数。"""
    parser = argparse.ArgumentParser(
        description=(
            "Plot percent(TTFT > x) curves for prompt-stream and normal "
            "mixed-source replay results."
        )
    )
    parser.add_argument(
        "--stream-result-dir",
        type=Path,
        required=True,
        help="Replay result directory produced by replay_prompt_stream.py.",
    )
    parser.add_argument(
        "--normal-result-dir",
        type=Path,
        required=True,
        help="Replay result directory produced by replay_normal.py.",
    )
    parser.add_argument(
        "--stream-label",
        type=str,
        default="Streaming prefill",
        help="Legend label for the prompt-stream curve.",
    )
    parser.add_argument(
        "--normal-label",
        type=str,
        default="Normal prefill",
        help="Legend label for the second curve.",
    )
    parser.add_argument(
        "--stream-ttft-column",
        type=str,
        default=DEFAULT_STREAM_TTFT_COLUMN,
        help="TTFT column for the prompt-stream curve.",
    )
    parser.add_argument(
        "--normal-ttft-column",
        type=str,
        default=DEFAULT_NORMAL_TTFT_COLUMN,
        help=(
            "TTFT column for the second curve. Use "
            "finish_to_first_token_ms when comparing merged_update."
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Output figure path. The extension controls the format.",
    )
    parser.add_argument(
        "--title",
        type=str,
        default="TTFT Survival Curve",
        help="Figure title.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=220,
        help="Output DPI for raster formats such as PNG.",
    )
    return parser.parse_args()


def _metrics_path(result_dir: Path) -> Path:
    """返回一个 replay 结果目录内的 request_metrics.csv 路径。"""
    metrics_path = result_dir / "request_metrics.csv"
    if not metrics_path.exists():
        raise FileNotFoundError(f"Missing request metrics file: {metrics_path}")
    return metrics_path


def _parse_float(value: str, *, column_name: str, row_index: int) -> float | None:
    """把 CSV 字段解析成 float，空值返回 None。"""
    normalized = value.strip()
    if not normalized:
        return None
    try:
        return float(normalized)
    except ValueError as exc:
        raise ValueError(
            f"Invalid float in column {column_name!r} at row {row_index}: {value!r}"
        ) from exc


def load_ttft_series(
    result_dir: Path,
    label: str,
    ttft_column: str,
) -> TtftSeries:
    """从 request_metrics.csv 加载成功请求的 TTFT。

    Args:
        result_dir: replay 输出目录。
        label: 图例标签。
        ttft_column: 当前模式使用的 TTFT 列名。

    Returns:
        包含有效 TTFT 样本和计数信息的数据序列。
    """
    ttft_values: list[float] = []
    num_rows = 0
    num_success_rows = 0
    with _metrics_path(result_dir).open(encoding="utf-8", newline="") as csv_file:
        reader = csv.DictReader(csv_file)
        if reader.fieldnames is None or ttft_column not in reader.fieldnames:
            raise ValueError(
                f"{_metrics_path(result_dir)} does not contain column {ttft_column!r}."
            )

        for row_index, row in enumerate(reader, start=2):
            num_rows += 1
            if row.get("result_status") != "ok":
                continue
            num_success_rows += 1
            ttft_ms = _parse_float(
                row.get(ttft_column, ""),
                column_name=ttft_column,
                row_index=row_index,
            )
            if ttft_ms is not None:
                ttft_values.append(ttft_ms)

    if not ttft_values:
        raise ValueError(f"No valid TTFT values found in {result_dir}.")
    return TtftSeries(
        label=label,
        result_dir=result_dir,
        ttft_column=ttft_column,
        ttft_ms=sorted(ttft_values),
        num_rows=num_rows,
        num_success_rows=num_success_rows,
    )


def survival_curve(ttft_ms: list[float]) -> tuple[list[float], list[float]]:
    """计算 `percent(TTFT > x)` 的阶梯曲线坐标。"""
    if not ttft_ms:
        raise ValueError("TTFT series must not be empty.")

    num_values = len(ttft_ms)
    x_values = [0.0]
    y_values = [100.0]
    for index, value in enumerate(ttft_ms):
        x_values.append(value)
        y_values.append((num_values - index - 1) * 100.0 / num_values)
    x_values.append(max(ttft_ms[-1] * 1.02, ttft_ms[-1] + 1.0))
    y_values.append(0.0)
    return x_values, y_values


def _percentile(sorted_values: list[float], percentile: float) -> float:
    """计算线性插值百分位数，用于日志摘要。"""
    if not sorted_values:
        raise ValueError("Cannot compute percentile of empty values.")
    if len(sorted_values) == 1:
        return sorted_values[0]
    position = (len(sorted_values) - 1) * percentile / 100.0
    lower_index = int(position)
    upper_index = min(lower_index + 1, len(sorted_values) - 1)
    fraction = position - lower_index
    lower_value = sorted_values[lower_index]
    upper_value = sorted_values[upper_index]
    return lower_value + (upper_value - lower_value) * fraction


def summarize_series(series: TtftSeries) -> dict[str, float | int | str]:
    """生成一条曲线的简短数值摘要。"""
    values = series.ttft_ms
    return {
        "label": series.label,
        "result_dir": str(series.result_dir),
        "ttft_column": series.ttft_column,
        "num_rows": series.num_rows,
        "num_success_rows": series.num_success_rows,
        "num_valid_ttft": len(values),
        "mean_ms": sum(values) / len(values),
        "min_ms": values[0],
        "p50_ms": _percentile(values, 50),
        "p90_ms": _percentile(values, 90),
        "p95_ms": _percentile(values, 95),
        "max_ms": values[-1],
    }


def _configure_matplotlib() -> None:
    """设置更适合论文草图的 matplotlib 风格。"""
    plt.rcParams.update(
        {
            "figure.figsize": (7.4, 4.8),
            "font.size": 11,
            "axes.titlesize": 14,
            "axes.labelsize": 12,
            "legend.fontsize": 10,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.32,
            "grid.linestyle": "--",
            "grid.linewidth": 0.7,
            "savefig.bbox": "tight",
        }
    )


def _legend_label(series: TtftSeries) -> str:
    """生成包含样本数和 p50 的曲线图例。"""
    p50_ms = _percentile(series.ttft_ms, 50)
    return f"{series.label} (n={len(series.ttft_ms)}, p50={p50_ms:.1f} ms)"


def plot_survival_curves(
    stream_series: TtftSeries,
    normal_series: TtftSeries,
    output_path: Path,
    title: str,
    dpi: int,
) -> None:
    """绘制并保存两条 TTFT survival curve。"""
    _configure_matplotlib()
    fig, ax = plt.subplots()
    for series, color in (
        (stream_series, "#1f77b4"),
        (normal_series, "#d62728"),
    ):
        x_values, y_values = survival_curve(series.ttft_ms)
        ax.step(
            x_values,
            y_values,
            where="post",
            linewidth=2.25,
            color=color,
            label=_legend_label(series),
        )

    max_ttft = max(stream_series.ttft_ms[-1], normal_series.ttft_ms[-1])
    ax.set_xlim(left=0, right=max(max_ttft * 1.06, max_ttft + 1.0))
    ax.set_ylim(bottom=0, top=102)
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=100, decimals=0))
    ax.set_title(title)
    ax.set_xlabel("TTFT x (ms)")
    ax.set_ylabel("Requests with TTFT > x")
    ax.legend(loc="upper right", frameon=True, fancybox=False, framealpha=0.92)
    fig.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)


def main() -> None:
    """加载两个 replay 结果目录并输出图像和 JSON 摘要。"""
    args = parse_args()
    stream_series = load_ttft_series(
        args.stream_result_dir,
        args.stream_label,
        args.stream_ttft_column,
    )
    normal_series = load_ttft_series(
        args.normal_result_dir,
        args.normal_label,
        args.normal_ttft_column,
    )
    plot_survival_curves(
        stream_series=stream_series,
        normal_series=normal_series,
        output_path=args.output,
        title=args.title,
        dpi=args.dpi,
    )

    summary_path = args.output.with_suffix(".summary.json")
    summary = {
        "output": str(args.output),
        "survival_definition": "percent(TTFT > x)",
        "series": [
            summarize_series(stream_series),
            summarize_series(normal_series),
        ],
    }
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote figure to {args.output}")
    print(f"Wrote summary to {summary_path}")


if __name__ == "__main__":
    main()
