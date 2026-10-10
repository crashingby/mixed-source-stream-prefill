# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""绘制 P/D 交接时延与 finish 后传输量的配对对照图。"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from summarize_stable_kv_push import compare_runs, load_run  # noqa: E402


def main() -> None:
    """复用配对校验，仅绘制两边都完成的同一批请求，并标注成功率。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--early-run", type=Path, required=True)
    parser.add_argument("--finish-only-run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--percentile", choices=("p90", "p95", "p99"), default="p95")
    args = parser.parse_args()
    comparison = compare_runs(args.early_run, args.finish_only_run)
    paired = comparison["paired_sample_ids"]
    if not paired:
        raise ValueError("No formally valid paired requests to plot.")
    _, early = load_run(args.early_run)
    _, finish = load_run(args.finish_only_run)
    series = (
        ("Early stable push", early, "#2864b4"),
        ("Finish-only push", finish, "#d45b35"),
    )
    plt.rcParams.update(
        {
            "font.size": 11,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.dpi": 160,
        }
    )
    figure, axes = plt.subplots(1, 3, figsize=(15, 4.5), constrained_layout=True)
    for label, rows, color in series:
        durations = sorted(
            float(rows[key]["p_finish_to_d_first_token_ms"]) for key in paired
        )
        survival = [
            100 * (len(durations) - index - 1) / len(durations)
            for index in range(len(durations))
        ]
        mode = "early_push" if rows is early else "finish_only"
        quantiles = comparison["modes"][mode]["paired_metrics"][
            "p_finish_to_d_first_token_ms"
        ]
        # 尾部分位数直接写入图例，避免只看重叠曲线无法判断差异。
        tail_label = (
            f"{label}\nP90={quantiles['p90']:.1f}, P95={quantiles['p95']:.1f} ms"
        )
        axes[0].step(
            [0.0, *durations],
            [100.0, *survival],
            where="post",
            color=color,
            label=tail_label,
        )
    axes[0].set(
        xlabel="P finish to first D token (ms)",
        ylabel="Requests > x (%)",
        ylim=(0, 101),
    )
    axes[0].legend(frameon=False)
    labels = [entry[0] for entry in series]
    for index, (label, rows, color) in enumerate(series):
        early_mb = (
            sum(rows[key]["early_transferred_bytes"] for key in paired)
            / len(paired)
            / 2**20
        )
        final_mb = (
            sum(rows[key]["final_transferred_bytes"] for key in paired)
            / len(paired)
            / 2**20
        )
        axes[1].bar(
            index,
            early_mb,
            color=color,
            alpha=0.35,
            label="Early ranges (light)" if index == 0 else None,
        )
        axes[1].bar(
            index,
            final_mb,
            bottom=early_mb,
            color=color,
            label="Final ranges (solid)" if index == 0 else None,
        )
        mode = "early_push" if rows is early else "finish_only"
        # 时间指标展示用户选定的尾部分位数，而不是由均值遮蔽尾部等待。
        transfer_tail = comparison["modes"][mode]["paired_metrics"][
            "final_transfer_ms"
        ][args.percentile]
        axes[2].bar(index, transfer_tail, color=color)
    axes[1].set(xticks=[0, 1], xticklabels=labels, ylabel="Mean transferred KV (MiB)")
    axes[1].legend(frameon=False, loc="lower right", fontsize=8)
    axes[2].set(
        xticks=[0, 1],
        xticklabels=labels,
        ylabel=f"{args.percentile.upper()} final NIXL transfer time (ms)",
    )
    for axis in axes:
        axis.grid(axis="y", linestyle="--", alpha=0.25)
        axis.set_axisbelow(True)
    rates = comparison["modes"]
    figure.suptitle(
        f"Stable KV push, QPS={comparison['qps']}, "
        f"paired n={len(paired)} / {comparison['num_selected']}\n"
        f"Transport success: early={rates['early_push']['success_rate']:.1%}, "
        f"finish-only={rates['finish_only']['success_rate']:.1%}"
        + ("; small-sample tail estimate" if len(paired) < 100 else "")
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output)
    plt.close(figure)
    print(f"Wrote figure to {args.output}")


if __name__ == "__main__":
    main()
