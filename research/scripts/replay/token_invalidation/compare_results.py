# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""离线配对比较三个失效回放结果，不连接服务或重新分词。"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

from research.scripts.replay.common.dataset import iter_jsonl, load_json
from research.scripts.replay.common.results import write_csv, write_json
from research.scripts.replay.token_invalidation.metrics import distribution

LAYOUTS = ("update_first", "append_first", "segmented")
DEFAULT_COMPARISON_DIR = Path(__file__).resolve().parent / "results" / "comparison"
CONTROL_FIELDS = (
    "model",
    "tokenizer_model",
    "input_transport",
    "temperature",
    "seed",
    "qps",
    "arrival_process",
    "random_seed",
    "max_concurrency",
    "append_initial_delay",
    "update_initial_delay",
    "append_chunk_interval",
    "update_chunk_interval",
    "source_finish_delay",
    "system_message",
    "question_template",
)


def read_results(
    directory: Path, layout: str
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    """加载配置与增量日志；拒绝重复 ID，避免续跑误拼接后重复计数。"""
    config = load_json(directory / "config.json")
    if (
        config.get("schema") != "token_invalidation_replay/v1"
        or config["args"].get("layout") != layout
    ):
        raise ValueError(f"结果目录不是预期布局 {layout}：{directory}")
    if config["args"].get("dry_run"):
        raise ValueError("dry-run 没有服务端失效统计，不能用于配对比较")
    rows: dict[str, dict[str, Any]] = {}
    selected = {row["sample_id"]: row for row in config["selected_samples"]}
    for row in iter_jsonl(directory / "request_metrics.jsonl"):
        sample_id = row["sample_id"]
        if sample_id in rows or sample_id not in selected or row.get("mode") != layout:
            raise ValueError(f"重复或不属于本次选择的结果：{directory}:{sample_id}")
        for key in ("input_sha256", "plan_sha256", "evidence_sha256"):
            if row.get(key) != selected[sample_id].get(key):
                raise ValueError(f"结果与配置中的样本指纹不符：{sample_id}:{key}")
        rows[sample_id] = row
    return config, rows


def is_valid(row: dict[str, Any] | None) -> bool:
    """区分正式有效的零失效与缺失、链路失败或统计非法。"""
    return (
        row is not None
        and row.get("transport_status") == "ok"
        and row.get("formal_result_status") == "valid"
        and type(row.get("snapshot_invalidated_tokens")) is int
        and row["snapshot_invalidated_tokens"] >= 0
    )


def compare_runs(
    directories: Mapping[str, Path],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """只在共同有效请求上计算三边分布和分段管理相对基线的节省量。"""
    loaded = {layout: read_results(directories[layout], layout) for layout in LAYOUTS}
    reference = loaded["segmented"][0]
    for layout in LAYOUTS:
        config = loaded[layout][0]
        if config.get("selected_sha256") != reference.get("selected_sha256"):
            raise ValueError("三种布局必须选择同一批原始样本和同一事件计划")
        if config.get("selected_samples") != reference.get("selected_samples"):
            raise ValueError("样本顺序或输入指纹不同，不能进行公平配对")
        if config.get("chat_template_sha256") != reference.get("chat_template_sha256"):
            raise ValueError("三个结果使用的模型模板不同")
        for key in CONTROL_FIELDS:
            if config["args"].get(key) != reference["args"].get(key):
                raise ValueError(f"控制参数不同：{layout}:{key}")
    records = {layout: loaded[layout][1] for layout in LAYOUTS}
    paired: list[dict[str, Any]] = []
    for selected in reference["selected_samples"]:
        sample_id = selected["sample_id"]
        trio = {layout: records[layout].get(sample_id) for layout in LAYOUTS}
        if not all(is_valid(row) for row in trio.values()):
            continue
        # 配置中的预期开关只是声明；有效配对还要核对 P 实际报告的开关。
        for key in (
            "coalescing_enabled",
            "actual_p_node_coalescing_config_json",
            "masked_padding_observed",
        ):
            if any(
                trio[layout].get(key) != trio["segmented"].get(key)
                for layout in LAYOUTS
            ):
                raise ValueError(f"实际 P 设置不同：{sample_id}:{key}")
        if trio["append_first"].get("logical_prompt_sha256") != trio["segmented"].get(
            "logical_prompt_sha256"
        ):
            raise ValueError(f"Append-first 与分段模式的最终文本不一致：{sample_id}")
        row = {
            "sample_id": sample_id,
            **{
                layout: trio[layout]["snapshot_invalidated_tokens"]
                for layout in LAYOUTS
            },
        }
        row["saved_vs_update_first"] = row["update_first"] - row["segmented"]
        row["saved_vs_append_first"] = row["append_first"] - row["segmented"]
        paired.append(row)
    result = {
        "schema": "token_invalidation_comparison/v1",
        "num_selected": len(reference["selected_samples"]),
        "num_paired_valid": len(paired),
        "num_excluded": len(reference["selected_samples"]) - len(paired),
        "runs": {
            layout: {
                "num_completed": len(records[layout]),
                "num_formal_valid": sum(
                    is_valid(row) for row in records[layout].values()
                ),
            }
            for layout in LAYOUTS
        },
        "paired_snapshot_invalidated_tokens": {
            layout: distribution([row[layout] for row in paired]) for layout in LAYOUTS
        },
        "saved_tokens": {
            baseline: distribution([row[f"saved_vs_{baseline}"] for row in paired])
            for baseline in ("update_first", "append_first")
        },
        "saving_ratio_from_paired_totals": {},
    }
    for baseline in ("update_first", "append_first"):
        total = sum(row[baseline] for row in paired)
        saved = sum(row[f"saved_vs_{baseline}"] for row in paired)
        # 基线没有失效时比例未定义；节省量可为负，不截断真实回退。
        result["saving_ratio_from_paired_totals"][baseline] = (
            saved / total if total else None
        )
    return paired, result


def main() -> None:
    """读取三个已完成或中断的运行目录，写出共同有效请求的配对统计。"""
    parser = argparse.ArgumentParser(description=__doc__)
    for layout in LAYOUTS:
        parser.add_argument(
            f"--{layout.replace('_', '-')}-run", type=Path, required=True
        )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_COMPARISON_DIR,
        help="默认保存到本实验 results/comparison",
    )
    args = parser.parse_args()
    directories = {layout: getattr(args, f"{layout}_run") for layout in LAYOUTS}
    paired, result = compare_runs(directories)
    write_csv(
        args.output_dir / "paired_requests.csv",
        ["sample_id", *LAYOUTS, "saved_vs_update_first", "saved_vs_append_first"],
        paired,
    )
    write_json(args.output_dir / "comparison_summary.json", result)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
