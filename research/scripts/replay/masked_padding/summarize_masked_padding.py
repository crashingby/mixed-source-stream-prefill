# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""汇总 masked-padding 开关的配对性能实验。"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from collections import defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

PROFILES = ("masked-padding-disabled", "masked-padding-enabled")
RUN_SUMMARY_FIELDS = (
    "run_id",
    "profile",
    "phase",
    "qps",
    "repeat_index",
    "random_seed",
    "sample_set_sha256",
    "num_requests",
    "num_success",
    "success_rate",
    "num_mask_valid",
    "num_exact_token_requests",
    "ttft_mean_ms",
    "ttft_p50_ms",
    "ttft_p95_ms",
    "tpot_mean_ms",
    "tpot_p50_ms",
    "tpot_p95_ms",
    "output_tps_mean",
    "output_tps_p50",
    "output_tps_p95",
    "aggregate_output_tps",
    "completed_requests_per_second",
    "total_exact_output_tokens",
    "mean_padding_tokens",
)
PAIRED_REQUEST_FIELDS = (
    "phase",
    "qps",
    "repeat_index",
    "random_seed",
    "sample_id",
    "query_id",
    "padding_tokens",
    "disabled_completion_tokens",
    "enabled_completion_tokens",
    "disabled_ttft_ms",
    "enabled_ttft_ms",
    "ttft_change_percent",
    "disabled_tpot_ms",
    "enabled_tpot_ms",
    "tpot_change_percent",
    "disabled_output_tps",
    "enabled_output_tps",
    "output_tps_change_percent",
)
PADDING_BUCKET_FIELDS = (
    "phase",
    "qps",
    "padding_bucket",
    "num_paired_requests",
    "mean_ttft_change_percent",
    "mean_tpot_change_percent",
    "mean_output_tps_change_percent",
)
COMPARISON_FIELDS = (
    "phase",
    "qps",
    "num_disabled_runs",
    "num_enabled_runs",
    "disabled_success_rate",
    "enabled_success_rate",
    "disabled_ttft_p50_ms",
    "enabled_ttft_p50_ms",
    "ttft_p50_change_percent",
    "disabled_ttft_p95_ms",
    "enabled_ttft_p95_ms",
    "ttft_p95_change_percent",
    "disabled_tpot_p50_ms",
    "enabled_tpot_p50_ms",
    "tpot_p50_change_percent",
    "disabled_output_tps_mean",
    "enabled_output_tps_mean",
    "output_tps_change_percent",
    "disabled_aggregate_output_tps",
    "enabled_aggregate_output_tps",
    "aggregate_output_tps_change_percent",
)


@dataclass(frozen=True)
class RunSummary:
    """一个结果目录汇总出的单次实验指标。"""

    run_id: str
    profile: str
    phase: str
    qps: float
    repeat_index: int
    random_seed: int
    sample_set_sha256: str
    num_requests: int
    num_success: int
    success_rate: float
    num_mask_valid: int
    num_exact_token_requests: int
    ttft_mean_ms: float | None
    ttft_p50_ms: float | None
    ttft_p95_ms: float | None
    tpot_mean_ms: float | None
    tpot_p50_ms: float | None
    tpot_p95_ms: float | None
    output_tps_mean: float | None
    output_tps_p50: float | None
    output_tps_p95: float | None
    aggregate_output_tps: float | None
    completed_requests_per_second: float | None
    total_exact_output_tokens: int
    mean_padding_tokens: float | None


def _parse_args() -> argparse.Namespace:
    """解析结果根目录与输出参数。"""
    parser = argparse.ArgumentParser(
        description="Summarize paired PROMPT_STREAM_MASKED_PADDING=0/1 runs."
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("research/scripts/replay/masked_padding/results"),
        help="Root containing replay run directories.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Analysis output directory; defaults to RESULTS_DIR/analysis.",
    )
    parser.add_argument(
        "--allow-unpaired",
        action="store_true",
        help="Keep groups whose enabled/disabled sample hashes do not match.",
    )
    return parser.parse_args()


def _read_json(path: Path) -> dict[str, Any]:
    """读取一个 JSON 对象，并为错误路径补充上下文。"""
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Cannot read {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object.")
    return value


def _read_csv(path: Path) -> list[dict[str, str]]:
    """读取逐请求 CSV。"""
    try:
        with path.open(encoding="utf-8", newline="") as csv_file:
            return list(csv.DictReader(csv_file))
    except OSError as exc:
        raise ValueError(f"Cannot read {path}: {exc}") from exc


def _optional_float(value: str | None) -> float | None:
    """把 CSV 可选数值转换为有限浮点数。"""
    if value in (None, ""):
        return None
    parsed = float(value)
    return parsed if math.isfinite(parsed) else None


def _as_bool(value: str | None) -> bool:
    """解析 CSV 中由 csv.DictWriter 输出的布尔值。"""
    return value == "True"


def _percentile(values: Sequence[float], percentile: float) -> float | None:
    """对非空数列计算线性插值百分位数。"""
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * percentile / 100
    lower_index = math.floor(position)
    upper_index = math.ceil(position)
    ratio = position - lower_index
    return ordered[lower_index] + (
        ordered[upper_index] - ordered[lower_index]
    ) * ratio


def _mean(values: Sequence[float]) -> float | None:
    """计算均值；空数列返回 None。"""
    return statistics.fmean(values) if values else None


def _phase_from_args(args: dict[str, Any]) -> str:
    """从 replay 配置确定 P-only 或完整 P→D 阶段。"""
    return "ttft" if args.get("ttft_only") is True else "full"


def _summarize_run(run_dir: Path) -> RunSummary | None:
    """读取并汇总一个正式 masked-padding replay 目录。"""
    config = _read_json(run_dir / "config.json")
    args = config.get("args")
    if not isinstance(args, dict) or args.get("experiment_suite") != "masked_padding":
        return None

    profile = str(args.get("experiment_profile"))
    if profile not in PROFILES:
        raise ValueError(f"Unexpected profile in {run_dir}: {profile!r}.")
    rows = _read_csv(run_dir / "request_metrics.csv")
    successful_rows = [
        row
        for row in rows
        if row.get("transport_status") == "ok"
        and row.get("first_token_received") == "True"
    ]
    valid_rows = [
        row
        for row in successful_rows
        if row.get("masked_padding_validation_status") == "valid"
    ]
    exact_rows = [
        row for row in valid_rows if _as_bool(row.get("completion_token_count_exact"))
    ]

    ttfts = [
        value
        for row in valid_rows
        if (value := _optional_float(row.get("finish_to_first_token_ms")))
        is not None
    ]
    tpots = [
        value
        for row in exact_rows
        if (value := _optional_float(row.get("mean_time_per_output_token_ms")))
        is not None
    ]
    output_tps_values = [
        value
        for row in exact_rows
        if (value := _optional_float(row.get("output_tokens_per_second")))
        is not None
    ]
    padding_lengths = [
        value
        for row in valid_rows
        if (value := _optional_float(row.get("masked_padding_tokens")))
        is not None
    ]
    total_output_tokens = sum(
        int(row.get("completion_tokens") or 0) for row in exact_rows
    )

    observation_duration_s: float | None = None
    if valid_rows:
        request_starts = [float(row["request_start_wall_time"]) for row in valid_rows]
        request_ends = [
            float(row["request_start_wall_time"])
            + float(row["request_to_done_ms"]) / 1000
            for row in valid_rows
            if row.get("request_to_done_ms") not in (None, "")
        ]
        if request_ends:
            observation_duration_s = max(request_ends) - min(request_starts)

    aggregate_output_tps = None
    completed_requests_per_second = None
    if observation_duration_s is not None and observation_duration_s > 0:
        aggregate_output_tps = total_output_tokens / observation_duration_s
        completed_requests_per_second = len(valid_rows) / observation_duration_s

    return RunSummary(
        run_id=run_dir.name,
        profile=profile,
        phase=_phase_from_args(args),
        qps=float(args["qps"]),
        repeat_index=int(args["repeat_index"]),
        random_seed=int(args["random_seed"]),
        sample_set_sha256=str(config["selected_sample_set_sha256"]),
        num_requests=len(rows),
        num_success=len(valid_rows),
        success_rate=len(valid_rows) / len(rows) if rows else 0.0,
        num_mask_valid=len(valid_rows),
        num_exact_token_requests=len(exact_rows),
        ttft_mean_ms=_mean(ttfts),
        ttft_p50_ms=_percentile(ttfts, 50),
        ttft_p95_ms=_percentile(ttfts, 95),
        tpot_mean_ms=_mean(tpots),
        tpot_p50_ms=_percentile(tpots, 50),
        tpot_p95_ms=_percentile(tpots, 95),
        output_tps_mean=_mean(output_tps_values),
        output_tps_p50=_percentile(output_tps_values, 50),
        output_tps_p95=_percentile(output_tps_values, 95),
        aggregate_output_tps=aggregate_output_tps,
        completed_requests_per_second=completed_requests_per_second,
        total_exact_output_tokens=total_output_tokens,
        mean_padding_tokens=_mean(padding_lengths),
    )


def _load_valid_request_rows(run_dir: Path) -> list[dict[str, Any]]:
    """读取正式 run 中通过传输与模式校验的逐请求结果。"""
    config = _read_json(run_dir / "config.json")
    args = config.get("args")
    if not isinstance(args, dict) or args.get("experiment_suite") != "masked_padding":
        return []
    profile = str(args.get("experiment_profile"))
    phase = _phase_from_args(args)
    records: list[dict[str, Any]] = []
    for row in _read_csv(run_dir / "request_metrics.csv"):
        if (
            row.get("transport_status") != "ok"
            or row.get("first_token_received") != "True"
            or row.get("masked_padding_validation_status") != "valid"
        ):
            continue
        records.append(
            {
                "profile": profile,
                "phase": phase,
                "qps": float(args["qps"]),
                "repeat_index": int(args["repeat_index"]),
                "random_seed": int(args["random_seed"]),
                "sample_id": row["sample_id"],
                "query_id": row["query_id"],
                "completion_tokens": int(row.get("completion_tokens") or 0),
                "padding_tokens": _optional_float(
                    row.get("masked_padding_tokens")
                ),
                "ttft_ms": _optional_float(
                    row.get("finish_to_first_token_ms")
                ),
                "tpot_ms": _optional_float(
                    row.get("mean_time_per_output_token_ms")
                ),
                "output_tps": _optional_float(
                    row.get("output_tokens_per_second")
                ),
            }
        )
    return records


def _build_paired_request_rows(
    records: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    """按 sample 对齐 enabled/disabled，并计算请求级百分比变化。"""
    grouped: dict[tuple[Any, ...], dict[str, dict[str, Any]]] = defaultdict(dict)
    for record in records:
        key = (
            record["phase"],
            record["qps"],
            record["repeat_index"],
            record["random_seed"],
            record["sample_id"],
        )
        grouped[key][record["profile"]] = record

    paired_rows: list[dict[str, Any]] = []
    for key, profile_records in sorted(grouped.items()):
        if set(profile_records) != set(PROFILES):
            continue
        disabled = profile_records[PROFILES[0]]
        enabled = profile_records[PROFILES[1]]
        if disabled["query_id"] != enabled["query_id"]:
            raise ValueError(f"Paired sample has different query ids: key={key!r}.")
        paired_rows.append(
            {
                "phase": disabled["phase"],
                "qps": disabled["qps"],
                "repeat_index": disabled["repeat_index"],
                "random_seed": disabled["random_seed"],
                "sample_id": disabled["sample_id"],
                "query_id": disabled["query_id"],
                # 关闭组不导出语义 manifest；同一样本的 gap 取开启组。
                "padding_tokens": enabled["padding_tokens"],
                "disabled_completion_tokens": disabled["completion_tokens"],
                "enabled_completion_tokens": enabled["completion_tokens"],
                "disabled_ttft_ms": disabled["ttft_ms"],
                "enabled_ttft_ms": enabled["ttft_ms"],
                "ttft_change_percent": _change_percent(
                    disabled["ttft_ms"], enabled["ttft_ms"]
                ),
                "disabled_tpot_ms": disabled["tpot_ms"],
                "enabled_tpot_ms": enabled["tpot_ms"],
                "tpot_change_percent": _change_percent(
                    disabled["tpot_ms"], enabled["tpot_ms"]
                ),
                "disabled_output_tps": disabled["output_tps"],
                "enabled_output_tps": enabled["output_tps"],
                "output_tps_change_percent": _change_percent(
                    disabled["output_tps"], enabled["output_tps"]
                ),
            }
        )
    return paired_rows


def _padding_bucket(padding_tokens: float | None) -> str:
    """把 block 尾部空隙划分为便于观察趋势的区间。"""
    if padding_tokens is None:
        return "unknown"
    if padding_tokens == 0:
        return "0"
    if padding_tokens <= 4:
        return "1-4"
    if padding_tokens <= 8:
        return "5-8"
    if padding_tokens <= 12:
        return "9-12"
    return "13+"


def _build_padding_bucket_rows(
    paired_rows: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    """按物理 gap 大小汇总请求级性能变化。"""
    grouped: dict[tuple[str, float, str], list[dict[str, Any]]] = defaultdict(list)
    for row in paired_rows:
        key = (row["phase"], row["qps"], _padding_bucket(row["padding_tokens"]))
        grouped[key].append(row)

    output_rows: list[dict[str, Any]] = []
    for (phase, qps, bucket), rows in sorted(grouped.items()):
        def mean_change(field_name: str) -> float | None:
            values = [
                float(row[field_name])
                for row in rows
                if row[field_name] is not None
            ]
            return _mean(values)

        output_rows.append(
            {
                "phase": phase,
                "qps": qps,
                "padding_bucket": bucket,
                "num_paired_requests": len(rows),
                "mean_ttft_change_percent": mean_change(
                    "ttft_change_percent"
                ),
                "mean_tpot_change_percent": mean_change(
                    "tpot_change_percent"
                ),
                "mean_output_tps_change_percent": mean_change(
                    "output_tps_change_percent"
                ),
            }
        )
    return output_rows


def _average_field(summaries: Sequence[RunSummary], field_name: str) -> float | None:
    """对重复实验的一个可选数值字段取均值。"""
    values = [getattr(summary, field_name) for summary in summaries]
    numeric_values = [float(value) for value in values if value is not None]
    return _mean(numeric_values)


def _change_percent(
    baseline: float | None,
    enabled: float | None,
) -> float | None:
    """计算 enabled 相对 disabled 的有符号百分比变化。"""
    if baseline is None or enabled is None or baseline == 0:
        return None
    return (enabled - baseline) / baseline * 100


def _validate_pairing(
    summaries: Sequence[RunSummary],
    allow_unpaired: bool,
) -> None:
    """验证每个 phase/QPS/repeat 的两种模式使用同一批样本。"""
    groups: dict[tuple[str, float, int, int], list[RunSummary]] = defaultdict(list)
    for summary in summaries:
        key = (
            summary.phase,
            summary.qps,
            summary.repeat_index,
            summary.random_seed,
        )
        groups[key].append(summary)

    errors: list[str] = []
    for key, group in groups.items():
        profiles = {summary.profile for summary in group}
        hashes = {summary.sample_set_sha256 for summary in group}
        if (
            len(group) != len(PROFILES)
            or profiles != set(PROFILES)
            or len(hashes) != 1
        ):
            errors.append(
                f"group={key!r}, num_runs={len(group)}, "
                f"profiles={sorted(profiles)!r}, "
                f"sample_hashes={sorted(hashes)!r}"
            )
    if errors and not allow_unpaired:
        raise ValueError(
            "Enabled/disabled runs are not paired. Re-run missing profiles or "
            "pass --allow-unpaired for exploratory analysis:\n"
            + "\n".join(errors)
        )


def _build_comparisons(summaries: Sequence[RunSummary]) -> list[dict[str, Any]]:
    """按 phase/QPS 聚合重复实验并计算开关相对变化。"""
    grouped: dict[tuple[str, float, str], list[RunSummary]] = defaultdict(list)
    for summary in summaries:
        grouped[(summary.phase, summary.qps, summary.profile)].append(summary)

    comparisons: list[dict[str, Any]] = []
    keys = sorted({(summary.phase, summary.qps) for summary in summaries})
    for phase, qps in keys:
        disabled = grouped.get((phase, qps, PROFILES[0]), [])
        enabled = grouped.get((phase, qps, PROFILES[1]), [])

        def average(group: Sequence[RunSummary], field_name: str) -> float | None:
            return _average_field(group, field_name)

        disabled_ttft_p50 = average(disabled, "ttft_p50_ms")
        enabled_ttft_p50 = average(enabled, "ttft_p50_ms")
        disabled_ttft_p95 = average(disabled, "ttft_p95_ms")
        enabled_ttft_p95 = average(enabled, "ttft_p95_ms")
        disabled_tpot_p50 = average(disabled, "tpot_p50_ms")
        enabled_tpot_p50 = average(enabled, "tpot_p50_ms")
        disabled_output_tps = average(disabled, "output_tps_mean")
        enabled_output_tps = average(enabled, "output_tps_mean")
        disabled_aggregate_tps = average(disabled, "aggregate_output_tps")
        enabled_aggregate_tps = average(enabled, "aggregate_output_tps")
        comparisons.append(
            {
                "phase": phase,
                "qps": qps,
                "num_disabled_runs": len(disabled),
                "num_enabled_runs": len(enabled),
                "disabled_success_rate": average(disabled, "success_rate"),
                "enabled_success_rate": average(enabled, "success_rate"),
                "disabled_ttft_p50_ms": disabled_ttft_p50,
                "enabled_ttft_p50_ms": enabled_ttft_p50,
                "ttft_p50_change_percent": _change_percent(
                    disabled_ttft_p50, enabled_ttft_p50
                ),
                "disabled_ttft_p95_ms": disabled_ttft_p95,
                "enabled_ttft_p95_ms": enabled_ttft_p95,
                "ttft_p95_change_percent": _change_percent(
                    disabled_ttft_p95, enabled_ttft_p95
                ),
                "disabled_tpot_p50_ms": disabled_tpot_p50,
                "enabled_tpot_p50_ms": enabled_tpot_p50,
                "tpot_p50_change_percent": _change_percent(
                    disabled_tpot_p50, enabled_tpot_p50
                ),
                "disabled_output_tps_mean": disabled_output_tps,
                "enabled_output_tps_mean": enabled_output_tps,
                "output_tps_change_percent": _change_percent(
                    disabled_output_tps, enabled_output_tps
                ),
                "disabled_aggregate_output_tps": disabled_aggregate_tps,
                "enabled_aggregate_output_tps": enabled_aggregate_tps,
                "aggregate_output_tps_change_percent": _change_percent(
                    disabled_aggregate_tps, enabled_aggregate_tps
                ),
            }
        )
    return comparisons


def _write_csv(
    path: Path,
    field_names: Sequence[str],
    rows: Iterable[dict[str, Any]],
) -> None:
    """写入字段顺序稳定的 CSV。"""
    with path.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=field_names)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    """扫描正式结果，校验配对并输出逐轮与开关对比汇总。"""
    args = _parse_args()
    output_dir = args.output_dir or args.results_dir / "analysis"
    summaries: list[RunSummary] = []
    request_records: list[dict[str, Any]] = []
    for config_path in sorted(args.results_dir.glob("*/config.json")):
        summary = _summarize_run(config_path.parent)
        if summary is not None:
            summaries.append(summary)
            request_records.extend(_load_valid_request_rows(config_path.parent))
    if not summaries:
        raise ValueError(f"No formal masked-padding runs found in {args.results_dir}.")

    _validate_pairing(summaries, args.allow_unpaired)
    comparisons = _build_comparisons(summaries)
    paired_request_rows = _build_paired_request_rows(request_records)
    padding_bucket_rows = _build_padding_bucket_rows(paired_request_rows)
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(
        output_dir / "run_summary.csv",
        RUN_SUMMARY_FIELDS,
        (asdict(summary) for summary in summaries),
    )
    _write_csv(
        output_dir / "comparison.csv",
        COMPARISON_FIELDS,
        comparisons,
    )
    _write_csv(
        output_dir / "paired_request_deltas.csv",
        PAIRED_REQUEST_FIELDS,
        paired_request_rows,
    )
    _write_csv(
        output_dir / "padding_bucket_summary.csv",
        PADDING_BUCKET_FIELDS,
        padding_bucket_rows,
    )
    (output_dir / "summary.json").write_text(
        json.dumps(
            {
                "num_runs": len(summaries),
                "profiles": list(PROFILES),
                "comparisons": comparisons,
                "num_paired_requests": len(paired_request_rows),
                "padding_bucket_summary": padding_bucket_rows,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"Wrote masked-padding analysis to {output_dir}")


if __name__ == "__main__":
    main()
