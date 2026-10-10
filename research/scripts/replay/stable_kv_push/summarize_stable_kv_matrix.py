# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""按稳定知识占比与 QPS 汇总逐轮 A/B，避免把重复轮次池化成尾部分位数。"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

MODES = ("early_push", "finish_only")  # 同一链路的唯一实验变量。
PERCENTILES = ("p90", "p95", "p99")  # P99 保留用于探索，不作为默认主图。
GROUP_FIELDS = (
    "dataset_sha256",
    "model",
    "tokenizer_model",
    "chat_template_sha256",
    "qps",
    "max_concurrency",
    "arrival_process",
    "random_seed",
    "temperature",
    "seed",
    "max_tokens",
    "append_initial_delay",
    "update_initial_delay",
    "append_chunk_interval",
    "update_chunk_interval",
    "source_finish_delay",
)


def _finite(value: Any) -> float | None:
    """排除缺失、布尔值和非有限统计，缺失不等价于零。"""
    return (
        float(value) if type(value) in (int, float) and math.isfinite(value) else None
    )


def _digest(value: Any) -> str:
    """给可比实验配置生成短标识，不依赖目录命名。"""
    payload = json.dumps(value, sort_keys=True, ensure_ascii=False).encode()
    return hashlib.sha256(payload).hexdigest()[:12]


def load_comparisons(
    analysis_dirs: list[Path],
) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    """递归读取正式配对汇总；旧数据无占比时记录排除理由。"""
    paths = {
        path.resolve()
        for directory in analysis_dirs
        for path in directory.rglob("comparison.json")
    }
    if not paths:
        raise ValueError(f"No comparison.json found under {analysis_dirs}.")
    comparisons: list[dict[str, Any]] = []  # 保持每轮独立的样本分母。
    skipped: list[dict[str, str]] = []  # 不悄悄将不可比实验并入占比曲线。
    for path in sorted(paths):
        comparison = json.loads(path.read_text(encoding="utf-8"))
        scenario = comparison.get("scenario")
        if not isinstance(scenario, dict):
            skipped.append({"path": str(path), "reason": "Missing scenario metadata"})
            continue
        fraction = _finite(scenario.get("append_fraction"))
        if fraction is None or not 0 <= fraction <= 1:
            raise ValueError(f"Invalid append_fraction in {path}: {scenario}")
        if not isinstance(comparison.get("controlled_settings"), dict):
            raise ValueError(f"Missing controlled_settings in {path}.")
        for mode in MODES:
            if mode not in comparison.get("modes", {}):
                raise ValueError(f"Missing mode={mode} in {path}.")
        comparison["comparison_path"] = str(path)
        comparisons.append(comparison)
    return comparisons, skipped


def aggregate_comparisons(
    comparisons: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """保留逐轮数值，并对每轮分位数取中位数及最小/最大范围。

    不混合各轮请求的延迟样本。成功率可以累计成功数量和总请求数，
    但 P90/P95/P99 始终是逐轮分位数的汇总，不能称为 pooled percentile。
    """
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    per_run: list[dict[str, Any]] = []  # 逐轮 CSV 保留失败数量和分位数样本量。
    for comparison in comparisons:
        settings = comparison["controlled_settings"]
        controls = {key: settings.get(key) for key in GROUP_FIELDS}
        scenario = comparison["scenario"]
        group_id = _digest({"controls": controls, "scenario": scenario})
        # 绘图上下文排除占比和 QPS，其他计算/到达参数必须一致。
        context = {
            key: value
            for key, value in controls.items()
            if key not in ("qps", "dataset_sha256")
        }
        context["scenario_controls"] = {
            key: value
            for key, value in scenario.items()
            if key not in ("append_tokens", "update_tokens", "append_fraction")
        }
        context_id = _digest(context)
        comparison["group_id"] = group_id
        comparison["context_id"] = context_id
        grouped[group_id].append(comparison)
        for mode in MODES:
            mode_stats = comparison["modes"][mode]
            for metric, values in mode_stats["paired_metrics"].items():
                per_run.append(
                    {
                        "group_id": group_id,
                        "context_id": context_id,
                        "comparison_path": comparison.get("comparison_path"),
                        "append_fraction": scenario["append_fraction"],
                        "append_tokens": scenario["append_tokens"],
                        "update_tokens": scenario["update_tokens"],
                        "qps": comparison["qps"],
                        "mode": mode,
                        "metric": metric,
                        "num_selected": comparison["num_selected"],
                        "num_formal_pairs": comparison["num_formal_pairs"],
                        "num_transport_success": mode_stats["num_transport_success"],
                        "num_formal": mode_stats["num_formal"],
                        "success_rate": mode_stats["success_rate"],
                        "formal_rate": mode_stats.get("formal_rate"),
                        "n": values["n"],
                        "mean": values.get("mean"),
                        **{name: values.get(name) for name in PERCENTILES},
                    }
                )
    aggregate: list[dict[str, Any]] = []
    for group_id, runs in sorted(grouped.items()):
        exemplar = runs[0]
        scenario = exemplar["scenario"]
        num_selected = sum(run["num_selected"] for run in runs)
        for mode in MODES:
            success = sum(run["modes"][mode]["num_transport_success"] for run in runs)
            formal = sum(run["modes"][mode]["num_formal"] for run in runs)
            metric_names = sorted(
                {name for run in runs for name in run["modes"][mode]["paired_metrics"]}
            )
            for metric in metric_names:
                metric_stats = [
                    run["modes"][mode]["paired_metrics"].get(metric, {}) for run in runs
                ]
                counts = [int(stats.get("n", 0)) for stats in metric_stats]
                row: dict[str, Any] = {
                    "group_id": group_id,
                    "context_id": exemplar["context_id"],
                    "append_fraction": scenario["append_fraction"],
                    "append_tokens": scenario["append_tokens"],
                    "update_tokens": scenario["update_tokens"],
                    "qps": exemplar["qps"],
                    "mode": mode,
                    "metric": metric,
                    "num_runs": len(runs),
                    "num_selected": num_selected,
                    "num_formal_pairs": sum(run["num_formal_pairs"] for run in runs),
                    "num_transport_success": success,
                    "num_formal": formal,
                    "success_rate": success / max(1, num_selected),
                    "formal_rate": formal / max(1, num_selected),
                    "n_min": min(counts),
                    "n_max": max(counts),
                }
                for name in ("mean", *PERCENTILES):
                    values = [
                        value
                        for stats in metric_stats
                        if (value := _finite(stats.get(name))) is not None
                    ]
                    row[f"{name}_num_runs"] = len(values)
                    row[f"{name}_median"] = (
                        statistics.median(values) if values else None
                    )
                    row[f"{name}_min"] = min(values) if values else None
                    row[f"{name}_max"] = max(values) if values else None
                aggregate.append(row)
    return per_run, aggregate


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    """写入矩阵结果；空输入仍创建空文件，排除原因另存 JSON。"""
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def paired_tail_changes(comparisons: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """报告每轮两组尾分位数之差，不将其误称为逐请求差值的分位数。"""
    rows: list[dict[str, Any]] = []
    for comparison in comparisons:
        early = comparison["modes"]["early_push"]["paired_metrics"]
        finish = comparison["modes"]["finish_only"]["paired_metrics"]
        for metric in (
            "p_finish_to_d_first_token_ms",
            "finish_to_d_first_client_ms",
            "finish_to_done_ms",
            "final_transfer_ms",
            "finish_to_p_first_client_ms",
        ):
            for percentile in PERCENTILES:
                early_value = _finite(early.get(metric, {}).get(percentile))
                finish_value = _finite(finish.get(metric, {}).get(percentile))
                difference = (
                    early_value - finish_value
                    if early_value is not None and finish_value is not None
                    else None
                )
                rows.append(
                    {
                        "group_id": comparison["group_id"],
                        "context_id": comparison["context_id"],
                        "comparison_path": comparison.get("comparison_path"),
                        "append_fraction": comparison["scenario"]["append_fraction"],
                        "qps": comparison["qps"],
                        "num_selected": comparison["num_selected"],
                        "num_formal_pairs": comparison["num_formal_pairs"],
                        "metric": metric,
                        "percentile": percentile,
                        "early_ms": early_value,
                        "finish_only_ms": finish_value,
                        "early_minus_finish_ms": difference,
                        # 正百分比表示提前传输更快，零基线不能用于比值。
                        "reduction_percent": (
                            -100 * difference / finish_value
                            if difference is not None
                            and finish_value is not None
                            and finish_value > 0
                            else None
                        ),
                    }
                )
    return rows


def plot_matrix(
    rows: list[dict[str, Any]], output_dir: Path, percentile: str
) -> list[str]:
    """按共同实验配置画占比曲线；颜色表示 QPS，线型表示传输模式。"""
    import matplotlib

    matplotlib.use("Agg")  # 无图形桌面的 Slurm 节点也能写图。
    import matplotlib.pyplot as plt

    contexts: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        contexts[row["context_id"]].append(row)
    panels = (
        ("p_finish_to_d_first_token_ms", percentile, "P finish to first D token (ms)"),
        (
            "finish_to_d_first_client_ms",
            percentile,
            "Client finish to D continuation token (ms)",
        ),
        ("final_transfer_ms", percentile, "Final NIXL transfer (ms)"),
        ("final_transferred_bytes", "mean", "Final transferred KV (MiB)"),
        (
            "completed_byte_fraction_before_p_finish",
            "mean",
            "KV completed before P finish (%)",
        ),
        ("finish_to_p_first_client_ms", percentile, "Finish to P first token (ms)"),
        ("success_rate", "rate", "Transport success (%)"),
        ("finish_to_done_ms", percentile, "Client finish to full response (ms)"),
    )
    plt.rcParams.update(
        {"font.size": 10, "axes.spines.top": False, "axes.spines.right": False}
    )
    outputs: list[str] = []
    for context_id, context_rows in sorted(contexts.items()):
        # 整体续写等待与完整响应必须和阶段收益一起显示，避免掩盖 P 的退化。
        figure, axes = plt.subplots(4, 2, figsize=(13, 15), constrained_layout=True)
        qps_values = sorted({row["qps"] for row in context_rows})
        colors = plt.get_cmap("tab10")
        for axis, (metric, stat, title) in zip(axes.flat, panels):
            for qps_index, qps in enumerate(qps_values):
                for mode in MODES:
                    candidates = sorted(
                        (
                            row
                            for row in context_rows
                            if row["qps"] == qps
                            and row["mode"] == mode
                            and row["metric"]
                            == (
                                "p_finish_to_d_first_token_ms"
                                if stat == "rate"
                                else metric
                            )
                        ),
                        key=lambda row: row["append_fraction"],
                    )
                    # 不同样本集合不能伪装成同一条占比曲线。
                    fractions = [row["append_fraction"] for row in candidates]
                    if len(fractions) != len(set(fractions)):
                        raise ValueError(
                            f"Ambiguous datasets in context={context_id}, qps={qps}."
                        )
                    center_key = "success_rate" if stat == "rate" else f"{stat}_median"
                    points = [
                        row
                        for row in candidates
                        if _finite(row[center_key]) is not None
                    ]
                    if not points:
                        continue
                    scale = 1 / 2**20 if metric.endswith("_bytes") else 1.0
                    if stat == "rate" or "fraction" in metric:
                        scale = 100.0
                    x = [100 * row["append_fraction"] for row in points]
                    y = [row[center_key] * scale for row in points]
                    color = colors(qps_index % 10)
                    line_mode = "early" if mode == "early_push" else "finish-only"
                    axis.plot(
                        x,
                        y,
                        color=color,
                        marker="o" if mode == "early_push" else "s",
                        linestyle="-" if mode == "early_push" else "--",
                        label=f"QPS={qps:g}, {line_mode}",
                    )
                    if stat != "rate":
                        lower = [row[f"{stat}_min"] * scale for row in points]
                        upper = [row[f"{stat}_max"] * scale for row in points]
                        axis.fill_between(x, lower, upper, color=color, alpha=0.1)
            axis.set(xlabel="Append fraction of final knowledge (%)", ylabel=title)
            axis.grid(linestyle="--", alpha=0.25)
            axis.set_axisbelow(True)
            if stat == "rate" or "fraction" in metric:
                axis.set_ylim(-1, 101)
            if stat == percentile:
                axis.set_title(percentile.upper())
        axes[0, 0].legend(fontsize=8, frameon=False)
        figure.suptitle(
            f"Stable KV push, context={context_id}\n"
            "Lines: median of per-run estimates; shading: repeat min-max, not CI"
        )
        output = output_dir / f"matrix-{context_id}-{percentile}.png"
        figure.savefig(output, dpi=180)
        plt.close(figure)
        outputs.append(str(output))
    return outputs


def main() -> None:
    """保存逐轮 CSV、重复轮次汇总 CSV、配置清单及默认 P95 图。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis-dir", action="append", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--percentile", choices=PERCENTILES, default="p95")
    args = parser.parse_args()
    comparisons, skipped = load_comparisons(args.analysis_dir)
    per_run, aggregate = aggregate_comparisons(comparisons)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(args.output_dir / "per_run.csv", per_run)
    _write_csv(args.output_dir / "matrix_summary.csv", aggregate)
    _write_csv(
        args.output_dir / "paired_tail_changes.csv", paired_tail_changes(comparisons)
    )
    figures = plot_matrix(aggregate, args.output_dir, args.percentile)
    manifest = {
        "num_comparisons": len(comparisons),
        "percentile": args.percentile,
        "pool_request_latencies": False,
        "repeat_band": "min-max, not confidence interval",
        "comparisons": comparisons,
        "skipped": skipped,
        "figures": figures,
    }
    (args.output_dir / "matrix_summary.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Summarized {len(comparisons)} pairs; skipped={len(skipped)}; {figures}")


if __name__ == "__main__":
    main()
