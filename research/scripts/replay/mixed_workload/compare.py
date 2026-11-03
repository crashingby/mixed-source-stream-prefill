# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""配对相同混合负载，分别报告普通请求和流式请求的 TTFT 变化。"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

from research.scripts.replay.common.dataset import iter_jsonl, load_json
from research.scripts.replay.common.load import distribution
from research.scripts.replay.common.results import write_csv, write_json


def compare(baseline: Path, optimized: Path, output_dir: Path) -> dict[str, Any]:
    """验证输入与负载指纹，按同一 sample_index 配对，写入 CSV 和 JSON。"""
    left = load_json(baseline / "summary.json")
    right = load_json(optimized / "summary.json")
    if left["dry_run"] or right["dry_run"]:
        raise ValueError("dry-run 不包含性能测量")
    if left["stream_mode"] != "normal" or right["stream_mode"] != "segmented_sparse":
        raise ValueError("baseline 必须为 normal，optimized 必须为 segmented_sparse")
    if left["workload_sha256"] != right["workload_sha256"]:
        raise ValueError("样本、类型分配、最终文本、事件计划或负载参数不一致")
    inputs = [
        list(iter_jsonl(path / "requests.jsonl")) for path in (baseline, optimized)
    ]
    successful = [
        {row["sample_index"]: row for row in records if row["status"] == "ok"}
        for records in inputs
    ]
    shared = sorted(successful[0].keys() & successful[1].keys())
    pairs = [(successful[0][index], successful[1][index]) for index in shared]
    rows = []
    for kind in ("regular", "stream", "all"):
        selected = [
            (a, b) for a, b in pairs if kind == "all" or a["request_kind"] == kind
        ]
        for metric in (
            "arrival_to_first_token_ms",
            "sent_to_first_token_ms",
            "finish_to_first_token_ms",
            "dispatch_lag_ms",
        ):
            observed = [
                (a[metric], b[metric])
                for a, b in selected
                if a.get(metric) is not None and b.get(metric) is not None
            ]
            dense = distribution([a for a, _ in observed])
            sparse = distribution([b for _, b in observed])
            row: dict[str, Any] = {
                "request_kind": kind,
                "metric": metric,
                "paired_requests": len(observed),
            }
            for label, stats in (("baseline", dense), ("optimized", sparse)):
                for stat in ("mean", "p50", "p95", "p99", "max"):
                    row[f"{label}_{stat}_ms"] = stats[stat] if stats else None
            row["mean_speedup"] = (
                dense["mean"] / sparse["mean"]
                if dense and sparse and sparse["mean"] > 0
                else None
            )
            row["mean_reduction_fraction"] = (
                1 - sparse["mean"] / dense["mean"]
                if dense and sparse and dense["mean"] > 0
                else None
            )
            rows.append(row)
    report = {
        "schema": "mixed_workload_comparison/v1",
        "ttft_scope": "p_side_prefill_ready",
        "baseline": str(baseline.resolve()),
        "optimized": str(optimized.resolve()),
        "workload_sha256": left["workload_sha256"],
        "optimized_keep_ratio": right["keep_ratio"],
        "baseline_groups": left["groups"],
        "optimized_groups": right["groups"],
        "baseline_load": left["load_summary"],
        "optimized_load": right["load_summary"],
        "paired_latency": rows,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "comparison.json", report)
    write_csv(output_dir / "comparison.csv", list(rows[0]), rows)
    return report


def main() -> None:
    """手动汇总两组运行；输出目录默认位于本实验 results 下。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--optimized", type=Path, required=True)
    parser.add_argument(
        "--output-dir", type=Path, default=Path(__file__).parent / "results/comparison"
    )
    args = parser.parse_args()
    compare(args.baseline, args.optimized, args.output_dir)
    print(args.output_dir)


if __name__ == "__main__":
    main()
