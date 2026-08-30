# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""汇总 Update 合并实验的逐请求指标与配置级结果。"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import math
import re
import statistics
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

DEFAULT_INPUT_ROOT = Path(
    "research/scripts/replay/update_coalescing/results"
)
DEFAULT_OUTPUT_DIR = DEFAULT_INPUT_ROOT / "analysis"
CONFIG_FILE_NAME = "config.json"
REQUEST_METRICS_FILE_NAME = "request_metrics.csv"

# 这些计数器由 P 节点在 finish 响应中返回，汇总时只使用成功请求的值。
SERVER_COUNTER_FIELDS = (
    "update_events_accepted",
    "update_works_admitted",
    "update_works_completed",
    "update_policy_works_admitted",
    "update_policy_works_completed",
    "update_policy_works_cancelled_by_recovery",
    "update_policy_works_outstanding",
    "eager_update_prefill_tokens",
    "admitted_update_prefill_tokens",
    "scheduled_update_prefill_tokens",
    "saved_update_prefill_tokens",
    "extra_update_prefill_tokens",
    "policy_admitted_update_prefill_tokens",
    "policy_saved_update_prefill_tokens",
    "policy_extra_update_prefill_tokens",
    "update_execution_attempts_started",
    "update_execution_attempts_completed",
    "update_execution_attempts_preempted",
    "update_recovery_works_created",
    "update_tokens_scheduled_total",
    "update_tokens_scheduled_for_recovery",
    "prefix_append_tokens_scheduled_for_recovery",
    "update_tokens_discarded_by_preemption",
    "prefix_append_tokens_discarded_by_preemption",
    "question_tokens_discarded_by_preemption",
    "prompt_stream_running_preemptions",
    "prompt_stream_dormant_evictions",
    "prompt_stream_retained_waiting_evictions",
    "prompt_stream_allocation_failures_running",
    "prompt_stream_allocation_failures_waiting",
    "prompt_stream_prefix_cache_resets",
    "update_tokens_invalidated_by_snapshot",
    "update_tokens_invalidated",
)

RUN_METADATA_FIELDS = (
    "source_run_dir",
    "run_id",
    "experiment_suite",
    "experiment_profile",
    "repeat_index",
    "workload_id",
    "selected_sample_set_sha256",
    "qps",
    "arrival_process",
    "model",
    "max_concurrency",
    "append_initial_delay_s",
    "update_initial_delay_s",
    "append_chunk_interval_s",
    "update_chunk_interval_s",
    "source_finish_delay_s",
    "ttft_only",
    "max_tokens",
    "temperature",
    "expected_coalescing",
    "coalescing_config_json",
    "update_policy",
    "append_gate",
    "soft_window_ms",
    "hard_deadline_ms",
    "ttft_deadline_ms",
)

SUMMARY_METRIC_FIELDS = (
    "num_requests",
    "num_success",
    "num_error",
    "num_valid_ttft",
    "num_stats_requests",
    "num_transport_success",
    "num_transport_error",
    "num_first_token_success",
    "num_stats_valid",
    "num_formal_valid",
    "success_rate",
    "transport_success_rate",
    "first_token_success_rate",
    "stats_valid_rate",
    "formal_valid_rate",
    "token_stats_coverage",
    "observation_duration_s",
    "completed_requests_per_s",
    "ttft_mean_ms",
    "ttft_p50_ms",
    "ttft_p90_ms",
    "ttft_p95_ms",
    "ttft_p99_ms",
    *(f"total_{field_name}" for field_name in SERVER_COUNTER_FIELDS),
    "coalescing_ratio",
    "token_saving_ratio",
    "mean_saved_update_prefill_tokens_per_counterfactual_request",
    "mean_update_tokens_invalidated_per_stats_request",
)

RUN_SUMMARY_FIELDS = RUN_METADATA_FIELDS + SUMMARY_METRIC_FIELDS

CONFIGURATION_SUMMARY_FIELDS = (
    "experiment_suite",
    "experiment_profile",
    "workload_id",
    "selected_sample_set_sha256",
    "qps",
    "update_chunk_interval_s",
    "coalescing_config_json",
    "update_policy",
    "append_gate",
    "soft_window_ms",
    "hard_deadline_ms",
    "ttft_deadline_ms",
    "arrival_process",
    "model",
    "max_concurrency",
    "append_initial_delay_s",
    "update_initial_delay_s",
    "append_chunk_interval_s",
    "source_finish_delay_s",
    "ttft_only",
    "max_tokens",
    "temperature",
    "expected_coalescing",
    "num_runs",
    "num_repeat_indices",
    "run_ids",
) + SUMMARY_METRIC_FIELDS

COMBINED_METADATA_FIELDS = (
    "analysis_run_dir",
    "analysis_experiment_suite",
    "analysis_experiment_profile",
    "analysis_repeat_index",
    "analysis_workload_id",
    "analysis_selected_sample_set_sha256",
    "analysis_qps",
    "analysis_update_chunk_interval_s",
    "analysis_coalescing_config_json",
    "analysis_update_policy",
    "analysis_append_gate",
)

_WARMUP_PATTERN = re.compile(r"(^|[-_])warmup($|[-_])", re.IGNORECASE)


@dataclass(frozen=True)
class RunMetadata:
    """一个 replay run 的配置与 workload 维度。"""

    source_run_dir: str
    run_id: str
    experiment_suite: str
    experiment_profile: str
    repeat_index: int
    workload_id: str
    selected_sample_set_sha256: str
    qps: float
    arrival_process: str
    model: str
    max_concurrency: int
    append_initial_delay_s: float | None
    update_initial_delay_s: float | None
    append_chunk_interval_s: float | None
    update_chunk_interval_s: float | None
    source_finish_delay_s: float | None
    ttft_only: bool
    max_tokens: int | None
    temperature: float | None
    expected_coalescing: str
    coalescing_config_json: str
    update_policy: str
    append_gate: str
    soft_window_ms: float | None
    hard_deadline_ms: float | None
    ttft_deadline_ms: float | None

    def as_dict(self) -> dict[str, Any]:
        """返回可直接写入 CSV/JSON 的元数据字典。"""
        return {
            field_name: getattr(self, field_name)
            for field_name in RUN_METADATA_FIELDS
        }


@dataclass(frozen=True)
class LoadedRun:
    """一个已读取且通过基本格式验证的 replay run。"""

    run_dir: Path
    metadata: RunMetadata
    request_rows: tuple[dict[str, str], ...]
    request_field_names: tuple[str, ...]


def parse_args() -> argparse.Namespace:
    """解析 Update 合并结果汇总参数。"""
    parser = argparse.ArgumentParser(
        description=(
            "Aggregate update-coalescing replay runs into run-level, "
            "configuration-level, and combined request metrics."
        )
    )
    parser.add_argument(
        "--input-root",
        action="append",
        type=Path,
        default=None,
        help=(
            "Root searched recursively for run directories. Repeat this "
            "option for several result roots. Defaults to the update-"
            "coalescing results directory."
        ),
    )
    parser.add_argument(
        "--run-dir",
        action="append",
        type=Path,
        default=None,
        help=(
            "Explicit run directory containing config.json and "
            "request_metrics.csv. Repeat this option as needed."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory receiving the four aggregate output files.",
    )
    parser.add_argument(
        "--include-warmup",
        action="store_true",
        help="Include runs whose profile, run name, or directory marks warmup.",
    )
    parser.add_argument(
        "--log-level",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
        default="INFO",
        help="Logging verbosity.",
    )
    return parser.parse_args()


def _load_json_object(path: Path) -> dict[str, Any]:
    """读取 JSON 对象，并把格式问题转成带路径的异常。"""
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Cannot read JSON object {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}.")
    return value


def _load_request_rows(
    metrics_path: Path,
) -> tuple[tuple[dict[str, str], ...], tuple[str, ...]]:
    """读取 request_metrics.csv，并保留其原始字段顺序。"""
    try:
        with metrics_path.open(encoding="utf-8", newline="") as csv_file:
            reader = csv.DictReader(csv_file)
            if reader.fieldnames is None:
                raise ValueError(f"Missing CSV header in {metrics_path}.")
            rows = tuple(dict(row) for row in reader)
            return rows, tuple(reader.fieldnames)
    except OSError as exc:
        raise ValueError(f"Cannot read request metrics {metrics_path}: {exc}") from exc


def _optional_float(value: Any, *, context: str) -> float | None:
    """解析可选浮点值；空字符串和 None 保持为空。"""
    if value is None or value == "":
        return None
    try:
        numeric_value = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid float for {context}: {value!r}.") from exc
    if not math.isfinite(numeric_value):
        raise ValueError(f"Non-finite float for {context}: {value!r}.")
    return numeric_value


def _optional_int(value: Any, *, context: str) -> int | None:
    """解析可选整数，并拒绝带小数部分的统计值。"""
    numeric_value = _optional_float(value, context=context)
    if numeric_value is None:
        return None
    integer_value = int(numeric_value)
    if integer_value != numeric_value:
        raise ValueError(f"Expected an integer for {context}: {value!r}.")
    return integer_value


def _required_float(value: Any, *, context: str) -> float:
    """解析必需浮点配置字段。"""
    numeric_value = _optional_float(value, context=context)
    if numeric_value is None:
        raise ValueError(f"Missing required float for {context}.")
    return numeric_value


def _required_int(value: Any, *, context: str) -> int:
    """解析必需整数配置字段。"""
    integer_value = _optional_int(value, context=context)
    if integer_value is None:
        raise ValueError(f"Missing required integer for {context}.")
    return integer_value


def _parse_coalescing_config(
    raw_config: Any,
    *,
    context: str,
) -> tuple[dict[str, Any] | None, str]:
    """解析 replay 记录的 P 节点合并 JSON，并产生稳定规范串。"""
    if raw_config is None or raw_config == "":
        return None, "null"
    if isinstance(raw_config, str):
        try:
            parsed_config = json.loads(raw_config)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"Invalid coalescing JSON for {context}: {raw_config!r}."
            ) from exc
    else:
        parsed_config = raw_config
    if not isinstance(parsed_config, dict):
        raise ValueError(f"Coalescing config for {context} must be an object.")
    canonical_json = json.dumps(
        parsed_config,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return parsed_config, canonical_json


def _required_bool(value: Any, *, context: str) -> bool:
    """解析 config.json 中的必需布尔字段。"""
    if type(value) is not bool:
        raise ValueError(f"Expected a boolean for {context}: {value!r}.")
    return value


def _required_csv_bool(value: Any, *, context: str) -> bool:
    """解析 csv.DictReader 返回的布尔文本。"""
    if value in (True, "True", "true", "1"):
        return True
    if value in (False, "False", "false", "0"):
        return False
    raise ValueError(f"Expected a CSV boolean for {context}: {value!r}.")


def _row_first_token_received(
    row: Mapping[str, str],
    *,
    context: str,
) -> bool:
    """读取首 token 状态，并兼容没有显式状态列的历史 CSV。"""
    raw_status = row.get("first_token_received")
    if raw_status not in (None, ""):
        return _required_csv_bool(
            raw_status,
            context=f"{context}:first_token_received",
        )

    # 旧 replay 可能在 stats 后置校验失败后把 result_status 改成 error，
    # 但已写入的 TTFT 和 completion_chunks 足以证明首 token 已经到达。
    ttft_ms = _optional_float(
        row.get("finish_to_first_token_ms"),
        context=f"{context}:finish_to_first_token_ms",
    )
    completion_chunks = _optional_int(
        row.get("completion_chunks"),
        context=f"{context}:completion_chunks",
    )
    return ttft_ms is not None and (completion_chunks or 0) > 0


def _row_transport_success(
    row: Mapping[str, str],
    *,
    context: str,
) -> bool:
    """返回网络/SSE 链路是否成功；旧 CSV 回退到 result_status。"""
    transport_status = row.get("transport_status")
    if transport_status not in (None, ""):
        return transport_status == "ok"
    if row.get("result_status") == "ok":
        return True
    # 旧 replay 会在首 token 后因 lifecycle 校验失败覆盖 result_status。
    return _row_first_token_received(row, context=context)


def _row_stats_valid(row: Mapping[str, str]) -> bool:
    """仅接受通过 recovery-aware schema v2 校验的统计。"""
    return row.get("stats_validation_status") == "valid"


def _row_formal_valid(row: Mapping[str, str]) -> bool:
    """返回请求是否满足首 token、配置和 schema v2 的正式实验条件。"""
    return row.get("formal_result_status") == "valid"


def _canonical_sha256(value: Any) -> str:
    """返回一个 JSON 可表达对象的稳定 SHA-256。"""
    canonical_json = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(canonical_json.encode("utf-8")).hexdigest()


def _selected_sample_set_sha256(
    config: Mapping[str, Any],
    *,
    context: str,
) -> str:
    """读取新格式内容指纹，并为旧 run 生成 ID manifest 回退指纹。"""
    recorded_hash = config.get("selected_sample_set_sha256")
    if isinstance(recorded_hash, str) and re.fullmatch(
        r"[0-9a-f]{64}", recorded_hash
    ):
        return recorded_hash
    selected_samples = config.get("selected_samples")
    if not isinstance(selected_samples, list):
        raise ValueError(
            f"Missing selected sample manifest/hash for {context}."
        )
    return _canonical_sha256(selected_samples)


def _validate_formal_prompt_stream_stats(
    rows: Sequence[Mapping[str, str]],
    *,
    expected_coalescing: str,
    expected_config: Mapping[str, Any] | None,
    context: str,
) -> str | None:
    """校验正式 run，并返回成功请求报告的唯一实际 P 配置。"""
    required_counter_fields = (
        "update_events_accepted",
        "update_policy_works_admitted",
        "update_policy_works_completed",
        "update_policy_works_cancelled_by_recovery",
        "update_policy_works_outstanding",
        "policy_admitted_update_prefill_tokens",
        "update_execution_attempts_started",
        "update_execution_attempts_completed",
        "update_execution_attempts_preempted",
        "update_tokens_scheduled_total",
        "update_tokens_scheduled_for_recovery",
        "update_tokens_invalidated_by_snapshot",
    )
    optional_counterfactual_fields = (
        "eager_update_prefill_tokens",
        "policy_saved_update_prefill_tokens",
        "policy_extra_update_prefill_tokens",
    )
    actual_config_jsons: set[str] = set()
    for row_index, row in enumerate(rows, start=2):
        if not _row_stats_valid(row):
            continue
        row_context = f"{context}:request row {row_index}"
        enabled = _required_csv_bool(
            row.get("coalescing_enabled"),
            context=f"{row_context}:coalescing_enabled",
        )
        if expected_coalescing in ("enabled", "disabled"):
            expected_enabled = expected_coalescing == "enabled"
            if enabled != expected_enabled:
                raise ValueError(
                    f"{row_context} reports coalescing_enabled={enabled}, "
                    f"expected {expected_coalescing}."
                )
        raw_actual_config = row.get("actual_p_node_coalescing_config_json")
        if enabled:
            if not raw_actual_config:
                raise ValueError(
                    f"Missing actual P-node coalescing config in {row_context}."
                )
            try:
                actual_config = json.loads(raw_actual_config)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid actual P-node config in {row_context}."
                ) from exc
            if not isinstance(actual_config, dict) or expected_config is None:
                raise ValueError(
                    f"Expected/actual P-node configs must be objects in "
                    f"{row_context}."
                )
            actual_config_jsons.add(
                json.dumps(
                    actual_config,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                )
            )
            mismatches = {
                name: (value, actual_config.get(name))
                for name, value in expected_config.items()
                if actual_config.get(name) != value
            }
            if mismatches:
                raise ValueError(
                    f"P-node config mismatch in {row_context}: "
                    f"{mismatches!r}."
                )
        elif raw_actual_config:
            raise ValueError(
                f"Disabled request returned a P-node config in {row_context}."
            )
        counters = {
            field_name: _required_int(
                row.get(field_name),
                context=f"{row_context}:{field_name}",
            )
            for field_name in required_counter_fields
        }
        if any(value < 0 for value in counters.values()):
            raise ValueError(f"Negative Prompt Stream counter in {row_context}.")
        accepted = counters["update_events_accepted"]
        admitted = counters["update_policy_works_admitted"]
        completed = counters["update_policy_works_completed"]
        cancelled = counters["update_policy_works_cancelled_by_recovery"]
        outstanding = counters["update_policy_works_outstanding"]
        attempts_started = counters["update_execution_attempts_started"]
        attempts_completed = counters["update_execution_attempts_completed"]
        attempts_preempted = counters["update_execution_attempts_preempted"]
        policy_is_valid = (
            admitted <= accepted
            and admitted == completed + cancelled + outstanding
            and outstanding == 0
        )
        execution_is_valid = (
            attempts_started == attempts_completed + attempts_preempted
        )
        if not policy_is_valid or not execution_is_valid:
            raise ValueError(
                f"Invalid recovery-aware lifecycle counters in {row_context}: "
                f"accepted={accepted}, admitted={admitted}, "
                f"completed={completed}, cancelled={cancelled}, "
                f"outstanding={outstanding}, attempts_started={attempts_started}, "
                f"attempts_completed={attempts_completed}, "
                f"attempts_preempted={attempts_preempted}."
            )
        optional_values = [
            _optional_int(
                row.get(field_name),
                context=f"{row_context}:{field_name}",
            )
            for field_name in optional_counterfactual_fields
        ]
        has_counterfactual = enabled and accepted > 0
        if has_counterfactual != all(
            value is not None for value in optional_values
        ):
            raise ValueError(
                f"Invalid eager/saved/extra nullability in {row_context}: "
                f"values={optional_values!r}."
            )
        if not has_counterfactual and any(
            value is not None for value in optional_values
        ):
            raise ValueError(
                f"Unexpected eager/saved/extra values in {row_context}: "
                f"values={optional_values!r}."
            )
    if len(actual_config_jsons) > 1:
        raise ValueError(
            f"Run {context} was served by multiple P-node configs: "
            f"{sorted(actual_config_jsons)!r}."
        )
    return next(iter(actual_config_jsons), None)


def _is_warmup_run(
    run_dir: Path,
    config: Mapping[str, Any],
) -> bool:
    """判断一个 run 是否属于不应进入正式聚合的 warmup。"""
    args = config.get("args")
    if not isinstance(args, Mapping):
        args = {}

    # 显式布尔标记优先，便于未来 replay 增加正式 warmup 字段。
    for field_name in ("warmup", "is_warmup"):
        if args.get(field_name) is True or config.get(field_name) is True:
            return True

    # 兼容当前通过 profile/run-name/目录名表达 warmup 的批处理脚本。
    candidate_labels = (
        run_dir.name,
        str(args.get("experiment_profile", "")),
        str(args.get("run_name", "")),
    )
    return any(_WARMUP_PATTERN.search(label) for label in candidate_labels)


def _discover_run_dirs(
    input_roots: Sequence[Path],
    explicit_run_dirs: Sequence[Path],
) -> list[Path]:
    """递归发现同时包含配置和请求指标的 run 目录。"""
    discovered_paths: dict[Path, Path] = {}
    for input_root in input_roots:
        if not input_root.is_dir():
            raise FileNotFoundError(f"Input root does not exist: {input_root}")
        for config_path in input_root.rglob(CONFIG_FILE_NAME):
            run_dir = config_path.parent
            if not (run_dir / REQUEST_METRICS_FILE_NAME).is_file():
                logger.debug("Skipping directory without request metrics: %s", run_dir)
                continue
            discovered_paths[run_dir.resolve()] = run_dir

    for run_dir in explicit_run_dirs:
        config_path = run_dir / CONFIG_FILE_NAME
        metrics_path = run_dir / REQUEST_METRICS_FILE_NAME
        if not config_path.is_file() or not metrics_path.is_file():
            raise FileNotFoundError(
                f"Explicit run directory {run_dir} must contain "
                f"{CONFIG_FILE_NAME} and {REQUEST_METRICS_FILE_NAME}."
            )
        discovered_paths[run_dir.resolve()] = run_dir

    return sorted(discovered_paths.values(), key=lambda path: path.as_posix())


def _load_run(run_dir: Path, config: dict[str, Any]) -> LoadedRun:
    """加载一个 run，并冻结可复现的 workload 身份。"""
    args = config.get("args")
    if not isinstance(args, Mapping):
        raise ValueError(f"{run_dir / CONFIG_FILE_NAME} requires object-valued args.")
    rows, field_names = _load_request_rows(
        run_dir / REQUEST_METRICS_FILE_NAME
    )
    recorded_run_id = next(
        (row.get("run_id", "") for row in rows if row.get("run_id")),
        "",
    )
    run_id = str(args.get("run_name") or recorded_run_id or run_dir.name)
    profile = str(args.get("experiment_profile") or "unknown")
    suite = str(args.get("experiment_suite") or "unknown")
    expected_coalescing = str(args.get("expected_coalescing") or "any")
    coalescing_config, canonical_json = _parse_coalescing_config(
        args.get("p_node_coalescing_config_json"),
        context=run_dir.as_posix(),
    )
    if expected_coalescing == "enabled" and coalescing_config is None:
        raise ValueError(
            f"Enabled run {run_dir} is missing its P-node config JSON."
        )
    if expected_coalescing == "disabled" and coalescing_config is not None:
        raise ValueError(
            f"Disabled run {run_dir} unexpectedly declares a P-node config."
        )
    if suite == "update_coalescing" and not args.get("dry_run", False):
        actual_config_json = _validate_formal_prompt_stream_stats(
            rows,
            expected_coalescing=expected_coalescing,
            expected_config=coalescing_config,
            context=run_dir.as_posix(),
        )
        if actual_config_json is not None:
            parsed_actual_config = json.loads(actual_config_json)
            assert isinstance(parsed_actual_config, dict)
            coalescing_config = parsed_actual_config
            canonical_json = actual_config_json
    update_policy = (
        str(coalescing_config.get("update_policy", "unknown"))
        if coalescing_config is not None
        else "disabled"
    )
    append_gate = (
        str(coalescing_config.get("append_gate", "none"))
        if coalescing_config is not None
        else "none"
    )

    def config_float(field_name: str) -> float | None:
        """读取当前合并 JSON 中的可选浮点参数。"""
        if coalescing_config is None:
            return None
        return _optional_float(
            coalescing_config.get(field_name),
            context=f"{run_dir}:{field_name}",
        )

    sample_set_sha256 = _selected_sample_set_sha256(
        config,
        context=run_dir.as_posix(),
    )
    arrival_process = str(args.get("arrival_process") or "unknown")
    model = str(args.get("model") or "unknown")
    max_concurrency = _required_int(
        args.get("max_concurrency"), context=f"{run_dir}:max_concurrency"
    )
    append_initial_delay_s = _optional_float(
        args.get("append_initial_delay"),
        context=f"{run_dir}:append_initial_delay",
    )
    update_initial_delay_s = _optional_float(
        args.get("update_initial_delay"),
        context=f"{run_dir}:update_initial_delay",
    )
    append_chunk_interval_s = _optional_float(
        args.get("append_chunk_interval"),
        context=f"{run_dir}:append_chunk_interval",
    )
    update_chunk_interval_s = _optional_float(
        args.get("update_chunk_interval"),
        context=f"{run_dir}:update_chunk_interval",
    )
    source_finish_delay_s = _optional_float(
        args.get("source_finish_delay"),
        context=f"{run_dir}:source_finish_delay",
    )
    ttft_only = _required_bool(
        args.get("ttft_only"), context=f"{run_dir}:ttft_only"
    )
    max_tokens = _optional_int(
        args.get("max_tokens"), context=f"{run_dir}:max_tokens"
    )
    temperature = _optional_float(
        args.get("temperature"), context=f"{run_dir}:temperature"
    )
    workload_descriptor = {
        "experiment_suite": suite,
        "selected_sample_set_sha256": sample_set_sha256,
        "arrival_process": arrival_process,
        "model": model,
        "max_concurrency": max_concurrency,
        "append_initial_delay_s": append_initial_delay_s,
        "update_initial_delay_s": update_initial_delay_s,
        "append_chunk_interval_s": append_chunk_interval_s,
        "source_finish_delay_s": source_finish_delay_s,
        "ttft_only": ttft_only,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "sampling_seed": args.get("seed"),
    }
    workload_id = _canonical_sha256(workload_descriptor)[:16]
    metadata = RunMetadata(
        source_run_dir=run_dir.as_posix(),
        run_id=run_id,
        experiment_suite=suite,
        experiment_profile=profile,
        repeat_index=_required_int(
            args.get("repeat_index", 0), context=f"{run_dir}:repeat_index"
        ),
        workload_id=workload_id,
        selected_sample_set_sha256=sample_set_sha256,
        qps=_required_float(args.get("qps"), context=f"{run_dir}:qps"),
        arrival_process=arrival_process,
        model=model,
        max_concurrency=max_concurrency,
        append_initial_delay_s=append_initial_delay_s,
        update_initial_delay_s=update_initial_delay_s,
        append_chunk_interval_s=append_chunk_interval_s,
        update_chunk_interval_s=update_chunk_interval_s,
        source_finish_delay_s=source_finish_delay_s,
        ttft_only=ttft_only,
        max_tokens=max_tokens,
        temperature=temperature,
        expected_coalescing=expected_coalescing,
        coalescing_config_json=canonical_json,
        update_policy=update_policy,
        append_gate=append_gate,
        soft_window_ms=config_float("soft_window_ms"),
        hard_deadline_ms=config_float("hard_deadline_ms"),
        ttft_deadline_ms=config_float("ttft_deadline_ms"),
    )
    return LoadedRun(
        run_dir=run_dir,
        metadata=metadata,
        request_rows=rows,
        request_field_names=field_names,
    )


def _percentile(values: Sequence[float], percentile: float) -> float | None:
    """计算线性插值百分位数；空序列返回 None。"""
    if not values:
        return None
    ordered_values = sorted(values)
    if len(ordered_values) == 1:
        return ordered_values[0]
    position = (len(ordered_values) - 1) * percentile / 100.0
    lower_index = math.floor(position)
    upper_index = math.ceil(position)
    interpolation = position - lower_index
    return ordered_values[lower_index] + (
        ordered_values[upper_index] - ordered_values[lower_index]
    ) * interpolation


def _observation_duration_s(rows: Sequence[Mapping[str, str]]) -> float | None:
    """按最早请求开始到最晚请求完成估计一个 run 的观测时长。"""
    start_times: list[float] = []
    end_times: list[float] = []
    for row_index, row in enumerate(rows, start=2):
        start_time = _optional_float(
            row.get("request_start_wall_time"),
            context=f"request row {row_index}:request_start_wall_time",
        )
        duration_ms = _optional_float(
            row.get("request_to_done_ms") or row.get("request_to_first_token_ms"),
            context=f"request row {row_index}:request duration",
        )
        if start_time is None or duration_ms is None:
            continue
        start_times.append(start_time)
        end_times.append(start_time + duration_ms / 1000.0)
    if not start_times or not end_times:
        return None
    duration_s = max(end_times) - min(start_times)
    return max(0.0, duration_s)


def _sum_server_counter(
    rows: Sequence[Mapping[str, str]],
    field_name: str,
) -> tuple[int | None, int]:
    """对成功请求的一个服务端计数器求和，并返回有效值数量。"""
    values: list[int] = []
    for row_index, row in enumerate(rows, start=2):
        if not _row_stats_valid(row):
            continue
        value = _optional_int(
            row.get(field_name),
            context=f"request row {row_index}:{field_name}",
        )
        if value is not None:
            values.append(value)
    return (sum(values), len(values)) if values else (None, 0)


def _summarize_rows(
    rows: Sequence[Mapping[str, str]],
    *,
    observation_duration_s: float | None,
) -> dict[str, Any]:
    """按统一口径汇总一组逐请求记录。"""
    transport_successful_rows = [
        row
        for row_index, row in enumerate(rows, start=2)
        if _row_transport_success(row, context=f"request row {row_index}")
    ]
    stats_valid_rows = [row for row in rows if _row_stats_valid(row)]
    formal_valid_rows = [row for row in rows if _row_formal_valid(row)]
    first_token_rows: list[Mapping[str, str]] = []
    ttft_values: list[float] = []
    for row_index, row in enumerate(rows, start=2):
        row_context = f"request row {row_index}"
        if not _row_first_token_received(row, context=row_context):
            continue
        first_token_rows.append(row)
        ttft_ms = _optional_float(
            row.get("finish_to_first_token_ms"),
            context=f"{row_context}:finish_to_first_token_ms",
        )
        if ttft_ms is not None:
            ttft_values.append(ttft_ms)

    counter_totals: dict[str, int | None] = {}
    counter_counts: dict[str, int] = {}
    for field_name in SERVER_COUNTER_FIELDS:
        total, num_values = _sum_server_counter(rows, field_name)
        counter_totals[field_name] = total
        counter_counts[field_name] = num_values

    # token 聚合只使用通过 schema v2 recovery-aware 校验的请求。
    num_stats_requests = len(stats_valid_rows)
    total_accepted = counter_totals["update_events_accepted"]
    total_admitted = counter_totals["update_policy_works_admitted"]
    total_eager = counter_totals["eager_update_prefill_tokens"]
    total_saved = counter_totals["policy_saved_update_prefill_tokens"]
    total_invalidated = counter_totals["update_tokens_invalidated_by_snapshot"]

    coalescing_ratio = None
    if total_accepted is not None and total_admitted is not None:
        if total_accepted > 0:
            coalescing_ratio = 1.0 - total_admitted / total_accepted

    token_saving_ratio = None
    if total_eager is not None and total_saved is not None and total_eager > 0:
        token_saving_ratio = total_saved / total_eager

    throughput = None
    if observation_duration_s is not None and observation_duration_s > 0:
        throughput = len(first_token_rows) / observation_duration_s

    num_requests = len(rows)
    num_first_token_success = len(first_token_rows)
    return {
        "num_requests": num_requests,
        # 旧 success_rate 继续表示可用于 TTFT 的首 token 成功率。
        "num_success": num_first_token_success,
        "num_error": num_requests - num_first_token_success,
        "num_valid_ttft": len(ttft_values),
        "num_stats_requests": num_stats_requests,
        "num_transport_success": len(transport_successful_rows),
        "num_transport_error": num_requests - len(transport_successful_rows),
        "num_first_token_success": num_first_token_success,
        "num_stats_valid": len(stats_valid_rows),
        "num_formal_valid": len(formal_valid_rows),
        "success_rate": (
            num_first_token_success / num_requests if num_requests else None
        ),
        "transport_success_rate": (
            len(transport_successful_rows) / num_requests if num_requests else None
        ),
        "first_token_success_rate": (
            num_first_token_success / num_requests if num_requests else None
        ),
        "stats_valid_rate": (
            len(stats_valid_rows) / num_requests if num_requests else None
        ),
        "formal_valid_rate": (
            len(formal_valid_rows) / num_requests if num_requests else None
        ),
        "token_stats_coverage": (
            len(stats_valid_rows) / num_first_token_success
            if num_first_token_success
            else None
        ),
        "observation_duration_s": observation_duration_s,
        "completed_requests_per_s": throughput,
        "ttft_mean_ms": statistics.fmean(ttft_values) if ttft_values else None,
        "ttft_p50_ms": _percentile(ttft_values, 50),
        "ttft_p90_ms": _percentile(ttft_values, 90),
        "ttft_p95_ms": _percentile(ttft_values, 95),
        "ttft_p99_ms": _percentile(ttft_values, 99),
        **{
            f"total_{field_name}": counter_totals[field_name]
            for field_name in SERVER_COUNTER_FIELDS
        },
        "coalescing_ratio": coalescing_ratio,
        "token_saving_ratio": token_saving_ratio,
        "mean_saved_update_prefill_tokens_per_counterfactual_request": (
            total_saved / counter_counts["policy_saved_update_prefill_tokens"]
            if total_saved is not None
            and counter_counts["policy_saved_update_prefill_tokens"] > 0
            else None
        ),
        "mean_update_tokens_invalidated_per_stats_request": (
            total_invalidated
            / counter_counts["update_tokens_invalidated_by_snapshot"]
            if total_invalidated is not None
            and counter_counts["update_tokens_invalidated_by_snapshot"] > 0
            else None
        ),
    }


def _configuration_key(metadata: RunMetadata) -> tuple[Any, ...]:
    """返回配置级聚合键，不把 repeat index 和 arrival seed 纳入。"""
    return (
        metadata.experiment_suite,
        metadata.experiment_profile,
        metadata.workload_id,
        metadata.qps,
        metadata.update_chunk_interval_s,
        metadata.coalescing_config_json,
    )


def _common_value(values: Iterable[Any]) -> Any:
    """返回一组相同配置值；不一致时显式标记为 mixed。"""
    unique_values = {value for value in values}
    if len(unique_values) == 1:
        return next(iter(unique_values))
    return "mixed"


def _run_summary(loaded_run: LoadedRun) -> dict[str, Any]:
    """生成单个 run 的完整汇总行。"""
    duration_s = _observation_duration_s(loaded_run.request_rows)
    return {
        **loaded_run.metadata.as_dict(),
        **_summarize_rows(
            loaded_run.request_rows,
            observation_duration_s=duration_s,
        ),
    }


def _configuration_summaries(
    loaded_runs: Sequence[LoadedRun],
    run_summaries: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """把相同 P 配置与 workload 的重复 run 合并。"""
    runs_by_key: dict[tuple[Any, ...], list[LoadedRun]] = defaultdict(list)
    summaries_by_run_dir = {
        str(summary["source_run_dir"]): summary for summary in run_summaries
    }
    for loaded_run in loaded_runs:
        runs_by_key[_configuration_key(loaded_run.metadata)].append(loaded_run)
    configuration_rows: list[dict[str, Any]] = []
    for _, grouped_runs in sorted(
        runs_by_key.items(),
        key=lambda item: tuple(str(value) for value in item[0]),
    ):
        metadata_rows = [loaded_run.metadata for loaded_run in grouped_runs]
        request_rows = [
            row for loaded_run in grouped_runs for row in loaded_run.request_rows
        ]
        run_durations = [
            summaries_by_run_dir[loaded_run.metadata.source_run_dir][
                "observation_duration_s"
            ]
            for loaded_run in grouped_runs
        ]
        numeric_durations = [
            float(duration) for duration in run_durations if duration is not None
        ]
        total_duration_s = sum(numeric_durations) if numeric_durations else None
        metadata = metadata_rows[0]
        configuration_rows.append(
            {
                "experiment_suite": metadata.experiment_suite,
                "experiment_profile": metadata.experiment_profile,
                "workload_id": metadata.workload_id,
                "selected_sample_set_sha256": metadata.selected_sample_set_sha256,
                "qps": metadata.qps,
                "update_chunk_interval_s": metadata.update_chunk_interval_s,
                "coalescing_config_json": metadata.coalescing_config_json,
                "update_policy": metadata.update_policy,
                "append_gate": metadata.append_gate,
                "soft_window_ms": metadata.soft_window_ms,
                "hard_deadline_ms": metadata.hard_deadline_ms,
                "ttft_deadline_ms": metadata.ttft_deadline_ms,
                "arrival_process": metadata.arrival_process,
                "model": metadata.model,
                "max_concurrency": metadata.max_concurrency,
                "append_initial_delay_s": metadata.append_initial_delay_s,
                "update_initial_delay_s": metadata.update_initial_delay_s,
                "append_chunk_interval_s": metadata.append_chunk_interval_s,
                "source_finish_delay_s": metadata.source_finish_delay_s,
                "ttft_only": metadata.ttft_only,
                "max_tokens": metadata.max_tokens,
                "temperature": metadata.temperature,
                "expected_coalescing": metadata.expected_coalescing,
                "num_runs": len(grouped_runs),
                "num_repeat_indices": len(
                    {item.repeat_index for item in metadata_rows}
                ),
                "run_ids": ";".join(
                    sorted(item.run_id for item in metadata_rows)
                ),
                **_summarize_rows(
                    request_rows, observation_duration_s=total_duration_s
                ),
            }
        )
    return configuration_rows


def _combined_request_rows(
    loaded_runs: Sequence[LoadedRun],
) -> tuple[list[dict[str, Any]], list[str]]:
    """给所有逐请求记录附加配置元数据，并生成稳定字段并集。"""
    original_fields: list[str] = []
    seen_fields: set[str] = set()
    combined_rows: list[dict[str, Any]] = []
    for loaded_run in loaded_runs:
        for field_name in loaded_run.request_field_names:
            if field_name not in seen_fields:
                seen_fields.add(field_name)
                original_fields.append(field_name)
        metadata = loaded_run.metadata
        for request_row in loaded_run.request_rows:
            combined_rows.append(
                {
                    "analysis_run_dir": metadata.source_run_dir,
                    "analysis_experiment_suite": metadata.experiment_suite,
                    "analysis_experiment_profile": metadata.experiment_profile,
                    "analysis_repeat_index": metadata.repeat_index,
                    "analysis_workload_id": metadata.workload_id,
                    "analysis_selected_sample_set_sha256": (
                        metadata.selected_sample_set_sha256
                    ),
                    "analysis_qps": metadata.qps,
                    "analysis_update_chunk_interval_s": (
                        metadata.update_chunk_interval_s
                    ),
                    "analysis_coalescing_config_json": (
                        metadata.coalescing_config_json
                    ),
                    "analysis_update_policy": metadata.update_policy,
                    "analysis_append_gate": metadata.append_gate,
                    **request_row,
                }
            )
    return combined_rows, [*COMBINED_METADATA_FIELDS, *original_fields]


def _write_csv(
    path: Path,
    field_names: Sequence[str],
    rows: Iterable[Mapping[str, Any]],
) -> None:
    """以 UTF-8 和固定列顺序写入 CSV。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=field_names,
            extrasaction="ignore",
        )
        writer.writeheader()
        writer.writerows(rows)


def _write_json(path: Path, value: Any) -> None:
    """写入格式稳定的 UTF-8 JSON。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        f"{json.dumps(value, ensure_ascii=False, indent=2)}\n",
        encoding="utf-8",
    )


def summarize(args: argparse.Namespace) -> dict[str, Any]:
    """发现、加载并汇总所有指定的 replay run。"""
    explicit_run_dirs = args.run_dir or []
    input_roots = args.input_root or (
        [] if explicit_run_dirs else [DEFAULT_INPUT_ROOT]
    )
    candidate_dirs = _discover_run_dirs(input_roots, explicit_run_dirs)
    if not candidate_dirs:
        raise ValueError("No replay run directories were discovered.")

    loaded_runs: list[LoadedRun] = []
    skipped_warmup_dirs: list[str] = []
    for run_dir in candidate_dirs:
        config = _load_json_object(run_dir / CONFIG_FILE_NAME)
        if not args.include_warmup and _is_warmup_run(run_dir, config):
            skipped_warmup_dirs.append(run_dir.as_posix())
            logger.info("Skipping warmup run: %s", run_dir)
            continue
        loaded_runs.append(_load_run(run_dir, config))
    if not loaded_runs:
        raise ValueError("No non-warmup replay runs remain after filtering.")

    run_summaries = [_run_summary(loaded_run) for loaded_run in loaded_runs]
    configuration_summaries = _configuration_summaries(
        loaded_runs,
        run_summaries,
    )
    combined_rows, combined_fields = _combined_request_rows(loaded_runs)

    output_dir = args.output_dir
    _write_csv(output_dir / "run_summary.csv", RUN_SUMMARY_FIELDS, run_summaries)
    _write_csv(
        output_dir / "configuration_summary.csv",
        CONFIGURATION_SUMMARY_FIELDS,
        configuration_summaries,
    )
    _write_csv(
        output_dir / "combined_request_metrics.csv",
        combined_fields,
        combined_rows,
    )
    summary_payload = {
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "input_roots": [path.as_posix() for path in input_roots],
        "explicit_run_dirs": [path.as_posix() for path in explicit_run_dirs],
        "include_warmup": bool(args.include_warmup),
        "num_candidate_runs": len(candidate_dirs),
        "num_included_runs": len(loaded_runs),
        "num_skipped_warmup_runs": len(skipped_warmup_dirs),
        "skipped_warmup_dirs": skipped_warmup_dirs,
        "num_configuration_points": len(configuration_summaries),
        "num_combined_request_rows": len(combined_rows),
        "run_summary": run_summaries,
        "configuration_summary": configuration_summaries,
    }
    _write_json(output_dir / "summary.json", summary_payload)
    return summary_payload


def main() -> None:
    """运行命令行汇总并打印输出规模。"""
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
    summary_payload = summarize(args)
    logger.info(
        "Wrote %d runs, %d configurations, and %d request rows to %s.",
        summary_payload["num_included_runs"],
        summary_payload["num_configuration_points"],
        summary_payload["num_combined_request_rows"],
        args.output_dir,
    )


if __name__ == "__main__":
    main()
