# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""汇总同一 NIXL Push 链路的提前发送 / finish 后发送配对实验。"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from pathlib import Path
from typing import Any

METRICS = (
    "p_finish_to_d_first_token_ms",
    "d_request_to_first_token_ms",
    "finish_to_d_first_client_ms",
    "finish_to_p_first_client_ms",
    "finish_to_done_ms",
    "finalize_wait_ms",
    "final_transfer_ms",
    "early_transferred_bytes",
    "final_transferred_bytes",
    "total_transferred_bytes",
    "early_byte_fraction",
    "completed_bytes_before_p_finish",
    "completed_blocks_before_p_finish",
    "received_blocks_at_finalize",
    "completed_byte_fraction_before_p_finish",
    "decode_tpot_ms",
    "decode_tps",
    "queue_delay_ms",
)


def percentile(values: list[float], probability: float) -> float | None:
    """用相邻次序统计量线性插值，避免小样本 P95 的固定分桶。"""
    if not values:
        return None
    ordered = sorted(values)
    rank = probability * (len(ordered) - 1)
    lower = math.floor(rank)
    upper = math.ceil(rank)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (rank - lower)


def load_run(directory: Path) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    """读取配置与逐请求数据；重复 sample id 会破坏配对，明确拒绝。"""
    config = json.loads((directory / "config.json").read_text(encoding="utf-8"))
    rows: dict[str, dict[str, Any]] = {}
    with (directory / "requests.jsonl").open(encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            sample_id = row["sample_id"]
            if sample_id in rows:
                raise ValueError(f"Duplicate sample_id={sample_id} in {directory}")
            rows[sample_id] = row
    return config, rows


def _is_formal(row: dict[str, Any]) -> bool:
    """正式 D 交接结果同时要求完整成功、有效服务端统计与实际 D 输出。"""
    return (
        row.get("transport_status") == "ok"
        and row.get("metrics_status") == "valid"
        and row.get("decode_observed") is True
    )


def metric_value(row: dict[str, Any], name: str) -> float | None:
    """读取有限数值；提前完成比例由实际完成水位计算，不冒充分类比例。"""
    if name == "completed_byte_fraction_before_p_finish":
        completed = row.get("completed_bytes_before_p_finish")
        total = row.get("total_transferred_bytes")
        # 没有完成水位时不能把“已发布的提前字节”替代成已完成字节。
        if type(completed) not in (int, float) or type(total) not in (int, float):
            return None
        value = completed / total if total > 0 else None
    else:
        value = row.get(name)
    return (
        float(value) if type(value) in (int, float) and math.isfinite(value) else None
    )


def _scenario_metadata(config: dict[str, Any]) -> dict[str, Any] | None:
    """优先读取回放保存的元数据，旧回放可从仍存在的数据目录补充。"""
    metadata = config.get("controlled_dataset_metadata")
    if not isinstance(metadata, dict) and config.get("data_dir"):
        summary_path = Path(config["data_dir"]) / "dataset_summary.json"
        # 数据不存在并不影响旧 A/B 对照，只是不加入占比曲线。
        if summary_path.is_file():
            metadata = json.loads(summary_path.read_text(encoding="utf-8"))
    scenario = metadata.get("scenario") if isinstance(metadata, dict) else None
    if scenario is not None and not isinstance(scenario, dict):
        raise ValueError("Controlled dataset scenario must be a JSON object.")
    return scenario


def compare_runs(early_dir: Path, finish_dir: Path) -> dict[str, Any]:
    """校验受控变量，并报告全量成功率及相同样本上的正式配对统计。"""
    early_config, early_rows = load_run(early_dir)
    finish_config, finish_rows = load_run(finish_dir)
    if (
        early_config.get("early_push") is not True
        or finish_config.get("early_push") is not False
    ):
        raise ValueError(
            "--early-run must enable early_push and --finish-only-run disable it."
        )
    controlled_fields = (
        "dataset_sha256",
        "sample_ids",
        "model",
        "tokenizer_model",
        "chat_template_sha256",
        "qps",
        "arrival_process",
        "random_seed",
        "max_concurrency",
        "seed",
        "temperature",
        "max_tokens",
        "append_initial_delay",
        "update_initial_delay",
        "append_chunk_interval",
        "update_chunk_interval",
        "source_finish_delay",
        "controlled_dataset_metadata",
    )
    mismatch = {
        key: [early_config.get(key), finish_config.get(key)]
        for key in controlled_fields
        if early_config.get(key) != finish_config.get(key)
    }
    if mismatch:
        raise ValueError(f"Unmatched experiment settings: {mismatch}")
    expected = set(early_config["sample_ids"])
    if set(early_rows) != expected or set(finish_rows) != expected:
        raise ValueError("Run is incomplete: some selected requests have no final row.")
    paired_ids = sorted(
        sample_id
        for sample_id in expected
        if _is_formal(early_rows[sample_id]) and _is_formal(finish_rows[sample_id])
    )
    output: dict[str, Any] = {
        "num_selected": len(expected),
        "num_formal_pairs": len(paired_ids),
        "paired_sample_ids": paired_ids,
        "excluded_pair_sample_ids": sorted(expected - set(paired_ids)),
        "qps": early_config["qps"],
        "early_run": str(early_dir.resolve()),
        "finish_only_run": str(finish_dir.resolve()),
        "controlled_settings": {
            key: early_config.get(key) for key in controlled_fields
        },
        "scenario": _scenario_metadata(early_config),
        "percentile_note": (
            "P90/P95 are primary. Each percentile uses formally valid pairs only; "
            "P99 from fewer than 100 pairs is an exploratory tail estimate."
        ),
        "modes": {},
        "paired_changes": {},
    }
    for label, rows in (("early_push", early_rows), ("finish_only", finish_rows)):
        mode: dict[str, Any] = {
            "num_transport_success": sum(
                row.get("transport_status") == "ok" for row in rows.values()
            ),
            "num_metrics_valid": sum(
                row.get("metrics_status") == "valid" for row in rows.values()
            ),
            "num_with_decode": sum(
                row.get("decode_observed") is True for row in rows.values()
            ),
            "num_formal": sum(_is_formal(row) for row in rows.values()),
            "paired_metrics": {},
        }
        mode["success_rate"] = mode["num_transport_success"] / max(1, len(expected))
        mode["formal_rate"] = mode["num_formal"] / max(1, len(expected))
        for name in METRICS:
            values = [
                value
                for sample_id in paired_ids
                if (value := metric_value(rows[sample_id], name)) is not None
            ]
            mode["paired_metrics"][name] = {
                "n": len(values),
                "mean": statistics.mean(values) if values else None,
                "p50": percentile(values, 0.5),
                "p90": percentile(values, 0.90),
                "p95": percentile(values, 0.95),
                "p99": percentile(values, 0.99),
            }
        output["modes"][label] = mode
    # 改变量统一为 early - finish-only；负的时间差表示提前传输改善交接。
    for name in METRICS:
        changes = [
            early_value - finish_value
            for key in paired_ids
            if (early_value := metric_value(early_rows[key], name)) is not None
            and (finish_value := metric_value(finish_rows[key], name)) is not None
        ]
        output["paired_changes"][name] = {
            "n": len(changes),
            "mean_early_minus_finish": statistics.mean(changes) if changes else None,
            "p50_early_minus_finish": percentile(changes, 0.5),
            "p90_early_minus_finish": percentile(changes, 0.90),
            "p95_early_minus_finish": percentile(changes, 0.95),
            "p99_early_minus_finish": percentile(changes, 0.99),
        }
    exact_pairs = [
        key
        for key in paired_ids
        if early_rows[key].get("completion_token_count_exact")
        and finish_rows[key].get("completion_token_count_exact")
    ]
    output["output_comparison"] = {
        "num_exact_token_pairs": len(exact_pairs),
        "num_identical_token_sequences": sum(
            early_rows[key]["prefill_token_ids"] + early_rows[key]["decode_token_ids"]
            == finish_rows[key]["prefill_token_ids"]
            + finish_rows[key]["decode_token_ids"]
            for key in exact_pairs
        ),
    }
    return output


def main() -> None:
    """写入配对 JSON 和紧凑 CSV，所有失败比例保留在汇总中。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--early-run", type=Path, required=True)
    parser.add_argument("--finish-only-run", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    result = compare_runs(args.early_run, args.finish_only_run)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "comparison.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    with (args.output_dir / "metrics.csv").open(
        "w", encoding="utf-8", newline=""
    ) as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=("mode", "metric", "n", "mean", "p50", "p90", "p95", "p99"),
        )
        writer.writeheader()
        for mode, stats in result["modes"].items():
            for metric, values in stats["paired_metrics"].items():
                writer.writerow({"mode": mode, "metric": metric, **values})
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
