# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""mixed-source REST 回放的共享实现。

三个对外入口 replay_prompt_stream.py、replay_merged_update.py 和
replay_normal.py 共用本模块的数据解析、SSE TTFT 计时和结果写入能力。
前两个使用 prompt-stream API，最后一个把完整逻辑 prompt 交给同一个 proxy
的普通 /v1/completions 路径。
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import hashlib
import json
import logging
import math
import random
import statistics
import time
from collections import defaultdict
from collections.abc import Awaitable, Callable, Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

import httpx
from transformers import AutoTokenizer, PreTrainedTokenizerBase

from prompt_stream_stats_v2 import (
    PromptStreamStatsV2Error,
    apply_prompt_stream_stats_v2,
    read_metrics_schema_version,
)

logger = logging.getLogger(__name__)

MIX_SOURCE_SCHEMA_VERSION = "prompt_stream_mixed/v2"
DEFAULT_DATA_DIR = Path("research/data/mix_source")
DEFAULT_OUTPUT_DIR = Path(
    "research/scripts/replay/segmented_prefill/results"
)
DEFAULT_PROXY_URL = "http://127.0.0.1:8000"

ReplayMode = Literal["stream", "normal", "merged_update"]
ArrivalProcess = Literal["fixed", "poisson"]
STREAM_TIMING_MODES = frozenset(("stream", "merged_update"))
DEFAULT_EVENT_PROGRESS_LOG_INTERVAL = 1
CHAT_TEMPLATE_CONTENT_MARKER = "VLLM_PROMPT_STREAM_CONTENT_7CDA3F6B"

REQUEST_METRIC_FIELDS = (
    "run_id",
    "mode",
    "sample_id",
    "query_id",
    "sample_status",
    "result_status",
    "error",
    "transport_status",
    "first_token_received",
    "stats_validation_status",
    "stats_validation_error",
    "formal_result_status",
    "request_error_stage",
    "metrics_schema_version",
    "planned_arrival_offset_ms",
    "queue_delay_ms",
    "request_start_wall_time",
    "start_ack_ms",
    "source_dispatch_ms",
    "finish_to_first_token_ms",
    "request_to_first_token_ms",
    "finish_to_done_ms",
    "request_to_done_ms",
    "ttft_origin",
    "num_append_chunks",
    "num_update_snapshots",
    "num_event_acks",
    "append_chars",
    "final_update_chars",
    "normal_prompt_chars",
    "completion_chars",
    "completion_chunks",
    "coalescing_enabled",
    "actual_p_node_coalescing_config_json",
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
    "absorbed_update_events",
    "coalescing_ratio",
    "token_saving_ratio",
    "update_work_completion_ratio",
    "scheduled_replay_overhead_tokens",
    "mean_update_lcp_ratio",
)

EVENT_METRIC_FIELDS = (
    "run_id",
    "mode",
    "sample_id",
    "query_id",
    "source_id",
    "source_type",
    "event_type",
    "chunk_id",
    "sequence_no",
    "version",
    "scheduled_offset_ms",
    "actual_send_offset_ms",
    "ack_latency_ms",
    "num_event_tokens",
    "update_lcp_tokens",
    "update_lcp_ratio",
    "result_status",
    "error",
)


@dataclass(frozen=True)
class SourceEventStream:
    """一个来源的有序 prompt-stream RESTFUL 事件。

    Args:
        source_id: 数据集内稳定的来源 id。
        source_type: ``append`` 或 ``update``。
        event_records: 按 source 内 ``sequence_no`` 排列的完整事件记录。
    """

    source_id: str
    source_type: Literal["append", "update"]
    event_records: tuple[dict[str, Any], ...]


@dataclass(frozen=True)
class SampleReplayInput:
    """回放一个 mixed-source 样本所需的已验证状态。"""

    sample_id: str
    query_id: str
    status: str
    start_request: dict[str, Any]
    source_streams: tuple[SourceEventStream, ...]
    finish_event: dict[str, Any]
    raw_sample: dict[str, Any]


@dataclass(frozen=True)
class RenderedReplayPrompt:
    """模型 chat template 渲染后的文本和可分段 token 序列。"""

    start_token_ids: tuple[int, ...]
    finish_text: str
    finish_token_ids: tuple[int, ...]
    logical_prompt_token_ids: tuple[int, ...]
    logical_prompt_text: str
    append_chars: int
    final_update_chars: int


@dataclass(frozen=True)
class ChatTemplateRenderer:
    """使用目标模型 tokenizer 构造可分段的 chat prompt。"""

    tokenizer: PreTrainedTokenizerBase
    tokenizer_model: str
    chat_template_sha256: str

    @classmethod
    def from_pretrained(
        cls,
        tokenizer_model: str,
        trust_remote_code: bool,
    ) -> "ChatTemplateRenderer":
        """加载 tokenizer，并固定本轮实验使用的 chat template。"""
        tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_model,
            trust_remote_code=trust_remote_code,
        )
        try:
            chat_template = tokenizer.get_chat_template()
        except ValueError as exc:
            raise ValueError(
                f"Tokenizer {tokenizer_model!r} has no usable chat template."
            ) from exc
        if not isinstance(chat_template, str) or not chat_template:
            raise ValueError(
                f"Tokenizer {tokenizer_model!r} returned an invalid chat template."
            )
        return cls(
            tokenizer=tokenizer,
            tokenizer_model=tokenizer_model,
            chat_template_sha256=hashlib.sha256(
                chat_template.encode("utf-8")
            ).hexdigest(),
        )

    def encode_content(self, text: str) -> tuple[int, ...]:
        """编码一个动态 prompt 分段，不重复添加 BOS/EOS。"""
        return tuple(
            int(token_id)
            for token_id in self.tokenizer.encode(text, add_special_tokens=False)
        )

    def render_envelope(self, system_message: str) -> tuple[str, str]:
        """提取 user 内容前后的模型模板文本。"""
        rendered = self.tokenizer.apply_chat_template(
            [
                {"role": "system", "content": system_message},
                {"role": "user", "content": CHAT_TEMPLATE_CONTENT_MARKER},
            ],
            tokenize=False,
            add_generation_prompt=True,
        )
        if not isinstance(rendered, str):
            raise ValueError("apply_chat_template(tokenize=False) must return text.")
        if rendered.count(CHAT_TEMPLATE_CONTENT_MARKER) != 1:
            raise ValueError(
                "Chat template must preserve the prompt-stream content marker "
                "exactly once."
            )
        prefix, suffix = rendered.split(CHAT_TEMPLATE_CONTENT_MARKER, maxsplit=1)
        return prefix, suffix

    def render_chat_prompt(self, system_message: str, user_content: str) -> str:
        """渲染完整 chat prompt，用于验证分段拼接边界。"""
        rendered = self.tokenizer.apply_chat_template(
            [
                {"role": "system", "content": system_message},
                {"role": "user", "content": user_content},
            ],
            tokenize=False,
            add_generation_prompt=True,
        )
        if not isinstance(rendered, str):
            raise ValueError("apply_chat_template(tokenize=False) must return text.")
        return rendered


@dataclass
class EventMetric:
    """一次 prompt-stream event REST 调用的发送与 ack 指标。"""

    run_id: str
    mode: str
    sample_id: str
    query_id: str
    source_id: str
    source_type: str
    event_type: str
    chunk_id: str
    sequence_no: int
    version: int
    scheduled_offset_ms: float
    actual_send_offset_ms: float
    ack_latency_ms: float | None
    result_status: str
    error: str | None = None
    num_event_tokens: int | None = None
    update_lcp_tokens: int | None = None
    update_lcp_ratio: float | None = None


@dataclass
class RequestMetric:
    """一个样本的一次 stream 或 normal 回放结果。

    ``finish_to_first_token_ms`` 是本实验的主指标。prompt-stream 类模式从
    finish HTTP 请求提交到 proxy SSE 的首 token；normal 模式从普通 PD 请求
    提交到 P 节点产生的首个 SSE token。
    """

    run_id: str
    mode: str
    sample_id: str
    query_id: str
    sample_status: str
    result_status: str
    planned_arrival_offset_ms: float
    queue_delay_ms: float
    request_start_wall_time: float
    # result_status 保持旧 CSV 兼容；以下字段拆分链路成功与统计可信度。
    transport_status: str = "pending"
    first_token_received: bool = False
    stats_validation_status: str = "not_applicable"
    stats_validation_error: str | None = None
    formal_result_status: str = "not_evaluated"
    request_error_stage: str | None = None
    metrics_schema_version: int | None = None
    start_ack_ms: float | None = None
    source_dispatch_ms: float | None = None
    finish_to_first_token_ms: float | None = None
    request_to_first_token_ms: float | None = None
    finish_to_done_ms: float | None = None
    request_to_done_ms: float | None = None
    ttft_origin: str | None = None
    num_append_chunks: int = 0
    num_update_snapshots: int = 0
    num_event_acks: int = 0
    append_chars: int = 0
    final_update_chars: int = 0
    normal_prompt_chars: int = 0
    completion_chars: int = 0
    completion_chunks: int = 0
    coalescing_enabled: bool | None = None
    actual_p_node_coalescing_config_json: str | None = None
    update_events_accepted: int | None = None
    update_works_admitted: int | None = None
    update_works_completed: int | None = None
    update_policy_works_admitted: int | None = None
    update_policy_works_completed: int | None = None
    update_policy_works_cancelled_by_recovery: int | None = None
    update_policy_works_outstanding: int | None = None
    eager_update_prefill_tokens: int | None = None
    admitted_update_prefill_tokens: int | None = None
    scheduled_update_prefill_tokens: int | None = None
    saved_update_prefill_tokens: int | None = None
    extra_update_prefill_tokens: int | None = None
    policy_admitted_update_prefill_tokens: int | None = None
    policy_saved_update_prefill_tokens: int | None = None
    policy_extra_update_prefill_tokens: int | None = None
    update_execution_attempts_started: int | None = None
    update_execution_attempts_completed: int | None = None
    update_execution_attempts_preempted: int | None = None
    update_recovery_works_created: int | None = None
    update_tokens_scheduled_total: int | None = None
    update_tokens_scheduled_for_recovery: int | None = None
    prefix_append_tokens_scheduled_for_recovery: int | None = None
    update_tokens_discarded_by_preemption: int | None = None
    prefix_append_tokens_discarded_by_preemption: int | None = None
    question_tokens_discarded_by_preemption: int | None = None
    prompt_stream_running_preemptions: int | None = None
    prompt_stream_dormant_evictions: int | None = None
    prompt_stream_retained_waiting_evictions: int | None = None
    prompt_stream_allocation_failures_running: int | None = None
    prompt_stream_allocation_failures_waiting: int | None = None
    prompt_stream_prefix_cache_resets: int | None = None
    update_tokens_invalidated_by_snapshot: int | None = None
    update_tokens_invalidated: int | None = None
    absorbed_update_events: int | None = None
    coalescing_ratio: float | None = None
    token_saving_ratio: float | None = None
    update_work_completion_ratio: float | None = None
    scheduled_replay_overhead_tokens: int | None = None
    mean_update_lcp_ratio: float | None = None
    error: str | None = None
    event_metrics: list[EventMetric] = field(default_factory=list)
    response_payloads: list[dict[str, Any]] = field(default_factory=list)


@dataclass(frozen=True)
class SseMeasurement:
    """一次 OpenAI completions SSE 响应的首 token 与结束状态。"""

    first_token_time: float | None
    first_token_origin: str | None
    completed_time: float
    completion_chars: int
    completion_chunks: int
    payloads: tuple[dict[str, Any], ...]
    prompt_stream_stats: dict[str, Any] | None


ReplaySample = Callable[
    [
        httpx.AsyncClient,
        SampleReplayInput,
        RenderedReplayPrompt,
        ChatTemplateRenderer,
        argparse.Namespace,
        str,
        float,
        float,
    ],
    Awaitable[RequestMetric],
]


class ReplayHttpError(RuntimeError):
    """包装 HTTP/SSE 上游错误，并保留有助定位的响应正文。"""


class LegacyStatsSemanticsIncompatible(ReplayHttpError):
    """表示 schema v1 无法表达抢占后的 recovery lifecycle。"""


def _require_non_negative_stat(
    raw_stats: dict[str, Any],
    field_name: str,
) -> int:
    """读取服务端返回的必需非负整数统计字段。"""
    value = raw_stats.get(field_name)
    # bool 是 int 的子类，精确类型检查可避免把 True 当作一个 token。
    if type(value) is not int or value < 0:
        raise ReplayHttpError(
            f"prompt_stream_stats.{field_name} must be a non-negative int, "
            f"but got {value!r}."
        )
    return value


def _require_optional_non_negative_stat(
    raw_stats: dict[str, Any],
    field_name: str,
) -> int | None:
    """读取允许为 None 的非负整数统计字段。"""
    if field_name not in raw_stats:
        raise ReplayHttpError(f"prompt_stream_stats is missing {field_name!r}.")
    value = raw_stats[field_name]
    if value is None:
        return None
    if type(value) is not int or value < 0:
        raise ReplayHttpError(
            f"prompt_stream_stats.{field_name} must be None or a "
            f"non-negative int, but got {value!r}."
        )
    return value


def _apply_prompt_stream_stats_or_raise(
    metric: RequestMetric,
    raw_stats: dict[str, Any] | None,
    *,
    num_sent_update_events: int,
    require_stats: bool,
    expected_coalescing: str,
    expected_config_json: str | None,
) -> None:
    """校验并写入一次成功 finish 返回的 Prompt Stream 统计。

    Args:
        metric: 将被补充服务端原始统计与派生比率的请求指标。
        raw_stats: proxy 首个 SSE payload 中的 ``prompt_stream_stats``。
        num_sent_update_events: 客户端已收到 ACK 的 Update 数据事件数。
        require_stats: 缺少统计时是否将本次实验判为失败。
        expected_coalescing: ``enabled``、``disabled`` 或 ``any``，用于防止
            正式实验误连到配置不匹配的 P 节点。
        expected_config_json: replay 声明的 P 启动配置；逐字段与 P 实际返回的
            规范化配置比较。

    Raises:
        ReplayHttpError: 统计缺失、字段非法或跨字段关系不成立时抛出。
    """
    if raw_stats is None:
        if require_stats:
            raise ReplayHttpError(
                "Successful prompt-stream finish did not return "
                "prompt_stream_stats."
            )
        return

    coalescing_enabled = raw_stats.get("coalescing_enabled")
    if type(coalescing_enabled) is not bool:
        raise ReplayHttpError(
            "prompt_stream_stats.coalescing_enabled must be bool, "
            f"but got {coalescing_enabled!r}."
        )
    if expected_coalescing != "any":
        expected_enabled = expected_coalescing == "enabled"
        if coalescing_enabled != expected_enabled:
            raise ReplayHttpError(
                "P node coalescing state does not match the experiment: "
                f"expected={expected_coalescing}, "
                f"actual={coalescing_enabled}."
            )

    raw_actual_config = raw_stats.get("update_coalescing_config_json")
    actual_config: dict[str, Any] | None = None
    if coalescing_enabled:
        if not isinstance(raw_actual_config, str):
            raise ReplayHttpError(
                "An enabled coalescer must return string-valued "
                "update_coalescing_config_json."
            )
        try:
            parsed_actual_config = json.loads(raw_actual_config)
        except json.JSONDecodeError as exc:
            raise ReplayHttpError(
                "P node returned invalid update_coalescing_config_json."
            ) from exc
        if not isinstance(parsed_actual_config, dict):
            raise ReplayHttpError(
                "P-node update_coalescing_config_json must encode an object."
            )
        actual_config = parsed_actual_config
        metric.actual_p_node_coalescing_config_json = json.dumps(
            actual_config,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
    elif raw_actual_config is not None:
        raise ReplayHttpError(
            "A disabled coalescer must return null "
            "update_coalescing_config_json."
        )

    if expected_config_json is not None:
        try:
            expected_config = json.loads(expected_config_json)
        except json.JSONDecodeError as exc:
            raise ReplayHttpError(
                "Expected coalescing config is invalid JSON."
            ) from exc
        if not isinstance(expected_config, dict) or actual_config is None:
            raise ReplayHttpError(
                "Expected and actual coalescing configs must both be objects."
            )
        mismatched_fields = {
            field_name: {
                "expected": expected_value,
                "actual": actual_config.get(field_name),
            }
            for field_name, expected_value in expected_config.items()
            if actual_config.get(field_name) != expected_value
        }
        if mismatched_fields:
            raise ReplayHttpError(
                "P-node coalescing config does not match replay metadata: "
                f"{mismatched_fields!r}."
            )

    metric.coalescing_enabled = coalescing_enabled
    if metric.metrics_schema_version == 2:
        apply_prompt_stream_stats_v2(
            metric,
            raw_stats,
            num_sent_update_events=num_sent_update_events,
            require_stats=require_stats,
            coalescing_enabled=coalescing_enabled,
        )
        return

    # 未声明 metrics schema 的历史响应继续按 v1 严格口径解析。
    metric.update_events_accepted = _require_non_negative_stat(
        raw_stats, "update_events_accepted"
    )
    metric.update_works_admitted = _require_non_negative_stat(
        raw_stats, "update_works_admitted"
    )
    metric.update_works_completed = _require_non_negative_stat(
        raw_stats, "update_works_completed"
    )
    metric.eager_update_prefill_tokens = _require_optional_non_negative_stat(
        raw_stats, "eager_update_prefill_tokens"
    )
    metric.admitted_update_prefill_tokens = _require_non_negative_stat(
        raw_stats, "admitted_update_prefill_tokens"
    )
    metric.scheduled_update_prefill_tokens = _require_non_negative_stat(
        raw_stats, "scheduled_update_prefill_tokens"
    )
    metric.saved_update_prefill_tokens = _require_optional_non_negative_stat(
        raw_stats, "saved_update_prefill_tokens"
    )
    metric.extra_update_prefill_tokens = _require_optional_non_negative_stat(
        raw_stats, "extra_update_prefill_tokens"
    )
    metric.update_tokens_invalidated = _require_non_negative_stat(
        raw_stats, "update_tokens_invalidated"
    )

    accepted = metric.update_events_accepted
    admitted = metric.update_works_admitted
    completed = metric.update_works_completed
    assert accepted is not None and admitted is not None and completed is not None
    if accepted > num_sent_update_events:
        raise ReplayHttpError(
            "P node accepted more Update events than the client ACKed: "
            f"accepted={accepted}, sent={num_sent_update_events}."
        )
    if admitted > accepted or completed > admitted:
        raise ReplayHttpError(
            "Invalid Update work lifecycle counters: "
            f"accepted={accepted}, admitted={admitted}, completed={completed}."
        )
    if require_stats and completed != admitted:
        raise LegacyStatsSemanticsIncompatible(
            "A successful formal replay must finish every admitted Update work: "
            f"admitted={admitted}, completed={completed}."
        )

    counterfactual_values = (
        metric.eager_update_prefill_tokens,
        metric.saved_update_prefill_tokens,
        metric.extra_update_prefill_tokens,
    )
    has_counterfactual = coalescing_enabled and accepted > 0
    if has_counterfactual and any(
        value is None for value in counterfactual_values
    ):
        raise ReplayHttpError(
            "An enabled coalescer with Update events must return integer "
            f"eager/saved/extra values: values={counterfactual_values!r}."
        )
    if not has_counterfactual and any(
        value is not None for value in counterfactual_values
    ):
        raise ReplayHttpError(
            "Requests without a coalescing counterfactual must return null "
            "eager/saved/extra values: "
            f"values={counterfactual_values!r}."
        )

    eager = metric.eager_update_prefill_tokens
    admitted_tokens = metric.admitted_update_prefill_tokens
    scheduled_tokens = metric.scheduled_update_prefill_tokens
    saved = metric.saved_update_prefill_tokens
    extra = metric.extra_update_prefill_tokens
    assert admitted_tokens is not None and scheduled_tokens is not None
    if scheduled_tokens < admitted_tokens:
        raise LegacyStatsSemanticsIncompatible(
            "Scheduled Update tokens cannot be smaller than admitted tokens: "
            f"scheduled={scheduled_tokens}, admitted={admitted_tokens}."
        )
    if eager is not None and saved is not None and extra is not None:
        if saved != max(0, eager - admitted_tokens):
            raise ReplayHttpError("saved_update_prefill_tokens is inconsistent.")
        if extra != max(0, admitted_tokens - eager):
            raise ReplayHttpError("extra_update_prefill_tokens is inconsistent.")

    metric.absorbed_update_events = max(0, accepted - admitted)
    if accepted > 0:
        metric.coalescing_ratio = metric.absorbed_update_events / accepted
    if eager is not None and eager > 0 and saved is not None:
        metric.token_saving_ratio = saved / eager
    if admitted > 0:
        metric.update_work_completion_ratio = completed / admitted
    metric.scheduled_replay_overhead_tokens = max(
        0, scheduled_tokens - admitted_tokens
    )


def apply_prompt_stream_stats(
    metric: RequestMetric,
    raw_stats: dict[str, Any] | None,
    *,
    num_sent_update_events: int,
    require_stats: bool,
    expected_coalescing: str,
    expected_config_json: str | None,
) -> None:
    """结构化校验 finish stats，且不覆盖已经成功的首 token 链路。

    schema v2 使用 recovery-aware 守恒式。未声明版本的旧响应仍走 v1
    校验，但标记为 ``legacy``，避免把无法区分抢占的 token 数据纳入新的
    正式结论。所有统计错误都写入 metric，而不会伪装成 HTTP/SSE 错误。

    Args:
        metric: 已记录首 token 的请求指标。
        raw_stats: finish SSE 携带的 Prompt Stream stats。
        num_sent_update_events: 客户端收到 ACK 的 Update 数据事件数量。
        require_stats: 是否要求 stats 存在且通过正式守恒校验。
        expected_coalescing: replay 期望的 P 节点开关状态。
        expected_config_json: replay 期望的 P 节点合并配置。
    """
    if raw_stats is None:
        metric.stats_validation_status = "missing"
        metric.stats_validation_error = (
            "Successful prompt-stream finish did not return "
            "prompt_stream_stats."
        )
        metric.formal_result_status = (
            "invalid" if require_stats else "not_evaluated"
        )
        return

    try:
        metric.metrics_schema_version = read_metrics_schema_version(raw_stats)
        _apply_prompt_stream_stats_or_raise(
            metric,
            raw_stats,
            num_sent_update_events=num_sent_update_events,
            require_stats=require_stats,
            expected_coalescing=expected_coalescing,
            expected_config_json=expected_config_json,
        )
    except LegacyStatsSemanticsIncompatible as exc:
        metric.stats_validation_status = "legacy_incompatible"
        metric.stats_validation_error = str(exc)
        metric.formal_result_status = "not_evaluated"
        return
    except (ReplayHttpError, PromptStreamStatsV2Error) as exc:
        metric.stats_validation_status = "invalid"
        metric.stats_validation_error = str(exc)
        metric.formal_result_status = "invalid"
        return

    if metric.metrics_schema_version == 2:
        metric.stats_validation_status = "valid"
        metric.formal_result_status = "valid"
    else:
        # v1 无法表达 recovery cancellation/attempt，保留但不纳入 v2 token 聚合。
        metric.stats_validation_status = "legacy"
        metric.formal_result_status = "not_evaluated"


def parse_args(mode: ReplayMode) -> argparse.Namespace:
    """解析一个指定模式的 mixed-source REST 回放参数。

    Args:
        mode: 回放模式，决定暴露的 CLI 参数及请求路径。

    Returns:
        命令行参数命名空间。
    """
    parser = argparse.ArgumentParser(
        description=(
            f"Replay mixed-source samples through the selected REST mode: {mode}."
        )
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DEFAULT_DATA_DIR,
        help="Directory containing requests/*.json generated by mix_source.",
    )
    parser.add_argument(
        "--query-id",
        action="append",
        dest="query_ids",
        default=None,
        help="Query id to replay. Repeat to select several samples.",
    )
    parser.add_argument(
        "--start-index",
        type=int,
        default=0,
        help="Start position in numeric query-id order when --query-id is absent.",
    )
    parser.add_argument(
        "--max-requests",
        type=int,
        default=1,
        help="Maximum selected requests; 0 means all selected requests.",
    )
    parser.add_argument(
        "--include-partial",
        action="store_true",
        help="Include samples whose data-generation status is partial.",
    )
    parser.add_argument(
        "--proxy-url",
        type=str,
        default=DEFAULT_PROXY_URL,
        help="Proxy base URL; may end in /v1 or omit it.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="qwen2.5-3b",
        help="Served model name sent through the proxy.",
    )
    parser.add_argument(
        "--tokenizer-model",
        type=str,
        required=True,
        help=(
            "Tokenizer model used to apply the served model chat template and "
            "encode every prompt segment."
        ),
    )
    parser.add_argument(
        "--tokenizer-trust-remote-code",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Allow remote tokenizer code while loading --tokenizer-model.",
    )
    parser.add_argument(
        "--qps",
        type=float,
        default=1.0,
        help=(
            "Logical request start rate. Set 0 for closed-loop sequential execution."
        ),
    )
    parser.add_argument(
        "--arrival-process",
        choices=("fixed", "poisson"),
        default="fixed",
        help="Inter-request arrival process when --qps is positive.",
    )
    parser.add_argument(
        "--max-concurrency",
        type=int,
        default=1,
        help="Maximum in-flight logical requests within one replay mode.",
    )
    if mode in STREAM_TIMING_MODES:
        parser.add_argument(
            "--append-initial-delay",
            type=float,
            default=0.0,
            help="Seconds from start ack to the first append chunk send.",
        )
        parser.add_argument(
            "--update-initial-delay",
            type=float,
            default=0.0,
            help="Seconds from start ack to the first update snapshot send.",
        )
        parser.add_argument(
            "--append-chunk-interval",
            type=float,
            default=0.0,
            help="Target seconds between append data chunks in the same source.",
        )
        parser.add_argument(
            "--update-chunk-interval",
            type=float,
            default=0.0,
            help="Target seconds between update snapshots in the same source.",
        )
        parser.add_argument(
            "--source-finish-delay",
            type=float,
            default=0.0,
            help="Seconds after a source's last data event before source_finished.",
        )
        parser.add_argument(
            "--event-progress-log-interval",
            type=int,
            default=DEFAULT_EVENT_PROGRESS_LOG_INTERVAL,
            help=(
                "Log SEND/ACK progress for every N prompt-stream events. "
                "Set 0 to disable detailed event progress logs."
            ),
        )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=None,
        help="Override each sample start.max_tokens for this replay.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=None,
        help="Override each sample start.temperature for this replay.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional vLLM sampling seed forwarded through the proxy.",
    )
    parser.add_argument(
        "--http-timeout",
        type=float,
        default=600.0,
        help="Per HTTP request timeout in seconds.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory under which this replay creates a timestamped run folder.",
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default=None,
        help="Optional stable run folder suffix; defaults to timestamp + mode.",
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=0,
        help="Seed for reproducible Poisson inter-arrival draws only.",
    )
    parser.add_argument(
        "--experiment-suite",
        type=str,
        default="segmented_prefill",
        help="Research suite label recorded in config.json.",
    )
    parser.add_argument(
        "--experiment-profile",
        type=str,
        default="manual",
        help="Stable P-node configuration label recorded with every run.",
    )
    parser.add_argument(
        "--repeat-index",
        type=int,
        default=0,
        help="Repeat label recorded in config.json; it does not change timing.",
    )
    parser.add_argument(
        "--expected-coalescing",
        choices=("any", "enabled", "disabled"),
        default="any",
        help="Fail when the P-node response reports a different enable state.",
    )
    parser.add_argument(
        "--p-node-coalescing-config-json",
        type=str,
        default=None,
        help=(
            "Expected P-node startup JSON recorded in config.json and "
            "checked against the actual config returned at finish. It cannot "
            "change the Engine policy."
        ),
    )
    parser.add_argument(
        "--require-prompt-stream-stats",
        action="store_true",
        help=(
            "Mark a result formally invalid when a successful prompt-stream "
            "request omits engine statistics; its TTFT remains usable."
        ),
    )
    parser.add_argument(
        "--save-responses",
        action="store_true",
        help="Save parsed SSE response payloads per request for debugging.",
    )
    parser.add_argument(
        "--ttft-only",
        action="store_true",
        help=(
            "Stop consuming each SSE response immediately after the first "
            "completion token. This records TTFT only and avoids waiting for "
            "the full decode stream."
        ),
    )
    parser.add_argument(
        "--sse-timeout",
        type=float,
        default=None,
        help=(
            "Optional maximum seconds spent consuming one successful SSE "
            "response. The regular --http-timeout still controls connection "
            "and request-level timeouts."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate and summarize selected data without issuing HTTP requests.",
    )
    parser.add_argument(
        "--fail-fast",
        action="store_true",
        help="Stop after the first request error instead of recording and continuing.",
    )
    parser.add_argument(
        "--formal-fail-fast",
        action="store_true",
        help=("Stop after the first formally invalid result. Disabled by default."),
    )
    parser.add_argument(
        "--log-level",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
        default="INFO",
        help="Logging verbosity.",
    )
    args = parser.parse_args()
    args.replay_mode = mode
    return args


def _utc_timestamp() -> str:
    """返回用于输出目录名称的 UTC 时间戳。"""
    return datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")


def _json_default(value: Any) -> Any:
    """把 argparse Path 等少量对象转换为 JSON 可序列化值。"""
    if isinstance(value, Path):
        return value.as_posix()
    raise TypeError(f"Unsupported JSON value: {type(value)!r}.")


def _write_json(path: Path, value: Any) -> None:
    """写入格式稳定的 UTF-8 JSON 文件。

    Args:
        path: 输出文件路径。
        value: 可 JSON 序列化的数据。
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        f"{json.dumps(value, ensure_ascii=False, indent=2, default=_json_default)}\n",
        encoding="utf-8",
    )


def _write_csv(
    path: Path, field_names: Sequence[str], rows: Iterable[dict[str, Any]]
) -> None:
    """以固定列顺序写入 CSV 指标文件。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=field_names, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _numeric_query_sort_key(query_id: str) -> tuple[int, int | str]:
    """把数字 query id 按数值排序，其它 id 回退到字符串排序。"""
    if query_id.isdigit():
        return (0, int(query_id))
    return (1, query_id)


def _normalize_v1_url(base_url: str, endpoint: str) -> str:
    """把服务 base URL 和 `/v1/...` endpoint 规范化地拼接。

    Args:
        base_url: 允许带或不带尾部 `/v1` 的服务地址。
        endpoint: 以 `/v1/` 开头的 OpenAI 或 prompt-stream 路径。

    Returns:
        可直接传给 httpx 的完整 URL。
    """
    normalized_base_url = base_url.rstrip("/")
    normalized_endpoint = endpoint if endpoint.startswith("/") else f"/{endpoint}"
    if normalized_base_url.endswith("/v1") and normalized_endpoint.startswith("/v1/"):
        normalized_endpoint = normalized_endpoint[3:]
    return f"{normalized_base_url}{normalized_endpoint}"


def _prompt_stream_finish_endpoint(args: argparse.Namespace) -> str:
    """返回当前 prompt-stream replay 应使用的 finish 路径。

    TTFT-only 实验只需要 P 节点 finish 后的首 token。此时走 proxy 的
    `/v1/prompt_stream/finish_ttft`，避免 proxy 再启动 D 节点 decode 并触发
    不必要的 KV transfer。完整生成或非 TTFT-only 调试仍走原 `/finish`。
    """
    if getattr(args, "ttft_only", False):
        return "/v1/prompt_stream/finish_ttft"
    return "/v1/prompt_stream/finish"


def _prompt_stream_start_endpoint(args: argparse.Namespace) -> str:
    """返回当前 prompt-stream replay 应使用的 start 路径。

    TTFT-only 必须从会话创建时就不携带 ``kv_transfer_params``；若仍使用
    普通 ``/start``，P 会错误地把该请求登记为未来由 D 拉取 KV 的 producer。

    Args:
        args: 回放命令行参数。

    Returns:
        proxy 的完整 P/D 或严格 P-only start endpoint。
    """
    if getattr(args, "ttft_only", False):
        return "/v1/prompt_stream/start_ttft"
    return "/v1/prompt_stream/start"


def _load_json(path: Path) -> dict[str, Any]:
    """读取单个样本 JSON，并将格式错误转成易定位的 ValueError。"""
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Cannot read sample {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"Sample {path} must contain a JSON object.")
    return value


def _extract_source_streams(sample: dict[str, Any]) -> tuple[SourceEventStream, ...]:
    """从 v2 flat events 恢复按来源分组的有序事件流。

    Args:
        sample: 一个 ``prompt_stream_mixed/v2`` 样本对象。

    Returns:
        依照样本 `sources` 声明排列的来源事件流。

    Raises:
        ValueError: source/event 布局不满足当前数据格式不变量。
    """
    source_definitions = sample.get("sources")
    event_records = sample.get("events")
    if not isinstance(source_definitions, list) or not isinstance(event_records, list):
        raise ValueError("Sample requires list-valued sources and events.")

    records_by_source: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in event_records:
        event = record.get("event") if isinstance(record, dict) else None
        if not isinstance(event, dict):
            raise ValueError("Every sample event record must contain an event object.")
        source_id = event.get("source_id")
        if not isinstance(source_id, str):
            raise ValueError("Every sample event requires string source_id.")
        records_by_source[source_id].append(record)

    source_streams: list[SourceEventStream] = []
    for source_definition in source_definitions:
        if not isinstance(source_definition, dict):
            raise ValueError("Sample source definition must be an object.")
        source_id = source_definition.get("source_id")
        source_type = source_definition.get("source_type")
        if not isinstance(source_id, str) or source_type not in ("append", "update"):
            raise ValueError(
                f"Invalid sample source definition: {source_definition!r}."
            )
        source_records = records_by_source.pop(source_id, [])
        source_records.sort(key=lambda record: int(record["event"]["sequence_no"]))
        expected_sequence_numbers = list(range(len(source_records)))
        actual_sequence_numbers = [
            int(record["event"]["sequence_no"]) for record in source_records
        ]
        if actual_sequence_numbers != expected_sequence_numbers:
            raise ValueError(
                f"Source {source_id} has non-contiguous sequence numbers "
                f"{actual_sequence_numbers}."
            )
        if (
            not source_records
            or source_records[-1]["event"].get("event_type") != "source_finished"
        ):
            raise ValueError(f"Source {source_id} must end with source_finished.")
        source_streams.append(
            SourceEventStream(
                source_id=source_id,
                source_type=source_type,
                event_records=tuple(source_records),
            )
        )

    if records_by_source:
        raise ValueError(
            f"Events refer to undeclared sources: {sorted(records_by_source)}."
        )
    return tuple(source_streams)


def _parse_sample(path: Path) -> SampleReplayInput:
    """读取并校验一个 mixed-source v2 样本。"""
    sample = _load_json(path)
    if sample.get("schema_version") != MIX_SOURCE_SCHEMA_VERSION:
        raise ValueError(
            f"Sample {path} has schema={sample.get('schema_version')!r}; "
            f"expected {MIX_SOURCE_SCHEMA_VERSION!r}."
        )
    if sample.get("start", {}).get("api") != "completions":
        raise ValueError(f"Sample {path} does not use the completions API.")
    finish = sample.get("finish")
    if not isinstance(finish, dict) or not isinstance(finish.get("event"), dict):
        raise ValueError(f"Sample {path} has no valid finish event.")
    event_order = sample.get("event_order", {})
    if event_order.get("across_sources") != "unspecified":
        raise ValueError(
            f"Sample {path} incorrectly fixes an across-source event order."
        )

    sample_id = sample.get("sample_id")
    query_id = sample.get("query_id")
    if not isinstance(sample_id, str) or not isinstance(query_id, str):
        raise ValueError(f"Sample {path} requires string sample_id and query_id.")
    start_request = sample.get("start")
    if not isinstance(start_request, dict) or not isinstance(
        start_request.get("prompt"), str
    ):
        raise ValueError(f"Sample {path} has no text start.prompt.")

    return SampleReplayInput(
        sample_id=sample_id,
        query_id=query_id,
        status=str(sample.get("status", "unknown")),
        start_request=dict(start_request),
        source_streams=_extract_source_streams(sample),
        finish_event=dict(finish["event"]),
        raw_sample=sample,
    )


def _select_samples(args: argparse.Namespace) -> list[SampleReplayInput]:
    """扫描已落盘 request 文件并按 CLI 条件选择可回放样本。

    不依赖 manifest，因为生成器可能仍在写入请求文件、尚未重建最终 manifest。
    """
    request_directory = args.data_dir / "requests"
    request_paths = sorted(request_directory.glob("*.json"))
    if not request_paths:
        raise FileNotFoundError(f"No request JSON files found in {request_directory}.")

    samples: list[SampleReplayInput] = []
    for request_path in request_paths:
        try:
            sample = _parse_sample(request_path)
        except ValueError as exc:
            logger.warning("Skipping invalid sample %s: %s", request_path, exc)
            continue
        if sample.status != "complete" and not args.include_partial:
            continue
        samples.append(sample)

    samples.sort(key=lambda sample: _numeric_query_sort_key(sample.query_id))
    if args.query_ids:
        requested_query_ids = set(args.query_ids)
        samples = [
            sample for sample in samples if sample.query_id in requested_query_ids
        ]
        found_query_ids = {sample.query_id for sample in samples}
        missing_query_ids = sorted(requested_query_ids - found_query_ids)
        if missing_query_ids:
            raise ValueError(
                f"Requested query ids were not selectable: {missing_query_ids}."
            )
    else:
        if args.start_index < 0:
            raise ValueError("--start-index must be non-negative.")
        samples = samples[args.start_index :]

    if not samples:
        raise ValueError("No eligible mixed-source samples selected.")
    return samples


def _get_data_events(source_stream: SourceEventStream) -> list[dict[str, Any]]:
    """返回一个来源的 append/update 数据事件，排除 source_finished。"""
    return [
        record
        for record in source_stream.event_records
        if record["event"].get("event_type") in ("append", "update")
    ]


def _render_sample_prompt(
    sample: SampleReplayInput,
    renderer: ChatTemplateRenderer,
) -> RenderedReplayPrompt:
    """用目标模型 chat template 构造可分段且可复现的逻辑 prompt。

    其中 ``start_request.prompt`` 是 system message；Append、最终 Update
    快照和 finish 文本共同构成同一个 user message。动态部分均以
    ``add_special_tokens=False`` 独立编码，因此 REST 传输不会重复插入 BOS、
    EOS 或 assistant header。
    """
    append_text_parts: list[str] = []
    final_update_text_parts: list[str] = []
    for source_stream in sample.source_streams:
        data_records = _get_data_events(source_stream)
        if source_stream.source_type == "append":
            append_text_parts.extend(
                str(record["event"].get("text") or "")
                for record in data_records
            )
        elif data_records:
            final_update_text_parts.append(
                str(data_records[-1]["event"].get("text") or "")
            )

    system_message = str(sample.start_request["prompt"]).strip()
    append_text = "".join(append_text_parts)
    final_update_text = "".join(final_update_text_parts)
    finish_text = str(sample.finish_event.get("text") or "")
    template_prefix, template_suffix = renderer.render_envelope(system_message)
    user_content = f"{append_text}{final_update_text}{finish_text}"
    logical_prompt_text = f"{template_prefix}{user_content}{template_suffix}"
    expected_prompt_text = renderer.render_chat_prompt(system_message, user_content)
    if logical_prompt_text != expected_prompt_text:
        raise ValueError(
            "Chat template envelope does not reproduce the complete prompt; "
            "cannot safely replay segmented token ids."
        )

    start_token_ids = renderer.encode_content(template_prefix)
    finish_token_ids = (
        *renderer.encode_content(finish_text),
        *renderer.encode_content(template_suffix),
    )
    logical_prompt_token_ids = (
        *start_token_ids,
        *renderer.encode_content(append_text),
        *renderer.encode_content(final_update_text),
        *finish_token_ids,
    )
    return RenderedReplayPrompt(
        start_token_ids=start_token_ids,
        finish_text=finish_text,
        finish_token_ids=finish_token_ids,
        logical_prompt_token_ids=logical_prompt_token_ids,
        logical_prompt_text=logical_prompt_text,
        append_chars=len(append_text),
        final_update_chars=len(final_update_text),
    )


def _override_sampling_params(
    request: dict[str, Any],
    args: argparse.Namespace,
) -> None:
    """把 CLI 覆盖项写入一个 start 或 normal 请求体。"""
    if args.max_tokens is not None:
        request["max_tokens"] = args.max_tokens
    if args.temperature is not None:
        request["temperature"] = args.temperature
    if args.seed is not None:
        request["seed"] = args.seed


async def _sleep_until(target_time: float) -> None:
    """在 monotonic 时钟下等待到指定目标时间，避免累计 sleep 漂移。"""
    remaining_seconds = target_time - time.perf_counter()
    if remaining_seconds > 0:
        await asyncio.sleep(remaining_seconds)


def should_log_event_progress(
    args: argparse.Namespace,
    event_index: int,
) -> bool:
    """判断当前 event 是否需要打印 replay 侧进度日志。

    这个 helper 只控制客户端侧调试日志，不改变请求时序。默认每个
    prompt-stream event 都打印一次 SEND/ACK；实验日志过大时可以把
    ``--event-progress-log-interval`` 设为更大的正数做降采样，或设为
    0 完全关闭。
    """
    log_interval = getattr(
        args,
        "event_progress_log_interval",
        DEFAULT_EVENT_PROGRESS_LOG_INTERVAL,
    )
    return log_interval > 0 and event_index % log_interval == 0


async def _post_json(
    client: httpx.AsyncClient,
    url: str,
    payload: dict[str, Any],
) -> dict[str, Any]:
    """发送 JSON POST，并在错误时保留可读的上游正文。"""
    try:
        response = await client.post(url, json=payload)
    except httpx.RequestError as exc:
        raise ReplayHttpError(f"HTTP request failed for {url}: {exc}") from exc
    if response.is_error:
        raise ReplayHttpError(
            f"HTTP {response.status_code} from {url}: {response.text}"
        )
    try:
        response_json = response.json()
    except ValueError as exc:
        raise ReplayHttpError(f"Non-JSON response from {url}: {response.text}") from exc
    if not isinstance(response_json, dict):
        raise ReplayHttpError(f"JSON response from {url} must be an object.")
    return response_json


def _request_limit(args: argparse.Namespace) -> int | None:
    """返回本次应保留的样本上限。

    显式 ``--query-id`` 延续原有语义：回放所有指定 id，不受
    ``--max-requests`` 默认值影响。批量回放时，0 表示不设上限。
    """
    if args.query_ids or args.max_requests == 0:
        return None
    return args.max_requests


def _limit_selected_samples(
    samples: Sequence[SampleReplayInput],
    args: argparse.Namespace,
) -> list[SampleReplayInput]:
    """按本地选择参数保留候选样本，不提前查看完整请求长度。"""
    request_limit = _request_limit(args)
    selected_samples = list(samples)
    if request_limit is not None:
        selected_samples = selected_samples[:request_limit]
    return selected_samples


def _is_first_token_choice(choice: dict[str, Any]) -> bool:
    """判断一个 OpenAI SSE choice 是否实际代表一个生成 token。"""
    if "token_ids" in choice and isinstance(choice["token_ids"], list):
        return True
    if "text" in choice:
        # 仅有 final finish_reason 的空 completion chunk 不应伪装成首 token。
        return choice.get("finish_reason") is None or bool(choice.get("text"))
    delta = choice.get("delta")
    if isinstance(delta, dict) and "content" in delta:
        return delta.get("content") is not None
    return False


def _chunk_text(choice: dict[str, Any]) -> str:
    """从 completions/chat-completions 兼容 choice 提取文本。"""
    text = choice.get("text")
    if isinstance(text, str):
        return text
    delta = choice.get("delta")
    if isinstance(delta, dict) and isinstance(delta.get("content"), str):
        return delta["content"]
    return ""


async def _consume_sse_response(
    response: httpx.Response,
    first_token_start_time: float,
    save_payloads: bool,
    ttft_only: bool,
) -> SseMeasurement:
    """读取一个 OpenAI SSE 响应并定位第一个生成 token。

    Args:
        response: 已打开且 HTTP status 成功的流式 HTTP 响应。
        first_token_start_time: TTFT 的起点；stream 模式为 finish 发送时刻。
        save_payloads: 是否把每个 JSON payload 保存到结果目录。
        ttft_only: 首 token 出现后是否立即停止消费 SSE body。

    Returns:
        首 token、结束时间和输出规模统计。

    Raises:
        ReplayHttpError: 上游返回非 SSE、错误 payload 或无有效首 token。
    """
    content_type = response.headers.get("content-type", "")
    if "text/event-stream" not in content_type:
        body = (await response.aread()).decode("utf-8", errors="replace")
        raise ReplayHttpError(
            "Expected text/event-stream for TTFT measurement, got "
            f"{content_type!r}: {body}"
        )

    first_token_time: float | None = None
    first_token_origin: str | None = None
    completion_chars = 0
    completion_chunks = 0
    payloads: list[dict[str, Any]] = []
    prompt_stream_stats: dict[str, Any] | None = None
    async for line in response.aiter_lines():
        if not line.startswith("data:"):
            continue
        data = line.removeprefix("data:").strip()
        if not data:
            continue
        if data == "[DONE]":
            break
        try:
            payload = json.loads(data)
        except json.JSONDecodeError as exc:
            raise ReplayHttpError(f"Invalid SSE JSON payload: {data!r}") from exc
        if not isinstance(payload, dict):
            raise ReplayHttpError(f"SSE payload must be an object: {payload!r}.")
        if save_payloads:
            payloads.append(payload)
        if "error" in payload:
            raise ReplayHttpError(f"Upstream SSE error: {payload['error']!r}")
        raw_stats = payload.get("prompt_stream_stats")
        if raw_stats is not None:
            if not isinstance(raw_stats, dict):
                raise ReplayHttpError(
                    "SSE prompt_stream_stats must be an object, "
                    f"but got {type(raw_stats).__name__}."
                )
            if prompt_stream_stats is not None and prompt_stream_stats != raw_stats:
                raise ReplayHttpError(
                    "One SSE response returned inconsistent prompt_stream_stats."
                )
            # 独立保存统计，避免 --ttft-only 且未保存 payload 时丢失首帧字段。
            prompt_stream_stats = dict(raw_stats)

        choices = payload.get("choices")
        if not isinstance(choices, list):
            continue
        for choice in choices:
            if not isinstance(choice, dict):
                continue
            if _is_first_token_choice(choice):
                if first_token_time is None:
                    first_token_time = time.perf_counter()
                    first_token_origin = (
                        "prefill" if "token_ids" in choice else "decode"
                    )
                completion_chunks += 1
                completion_chars += len(_chunk_text(choice))
                if ttft_only:
                    completed_time = time.perf_counter()
                    return SseMeasurement(
                        first_token_time=first_token_time,
                        first_token_origin=first_token_origin,
                        completed_time=completed_time,
                        completion_chars=completion_chars,
                        completion_chunks=completion_chunks,
                        payloads=tuple(payloads),
                        prompt_stream_stats=prompt_stream_stats,
                    )

    completed_time = time.perf_counter()
    if first_token_time is None:
        raise ReplayHttpError("SSE response ended before emitting a completion token.")
    if first_token_time < first_token_start_time:
        raise AssertionError("First token time cannot precede TTFT start time.")
    return SseMeasurement(
        first_token_time=first_token_time,
        first_token_origin=first_token_origin,
        completed_time=completed_time,
        completion_chars=completion_chars,
        completion_chunks=completion_chunks,
        payloads=tuple(payloads),
        prompt_stream_stats=prompt_stream_stats,
    )


async def consume_sse_response(
    response: httpx.Response,
    first_token_start_time: float,
    save_payloads: bool,
    ttft_only: bool,
    timeout_seconds: float | None,
) -> SseMeasurement:
    """带可选整体超时的 SSE 消费包装。"""
    try:
        if timeout_seconds is None:
            return await _consume_sse_response(
                response,
                first_token_start_time,
                save_payloads,
                ttft_only,
            )
        async with asyncio.timeout(timeout_seconds):
            return await _consume_sse_response(
                response,
                first_token_start_time,
                save_payloads,
                ttft_only,
            )
    except TimeoutError as exc:
        raise ReplayHttpError(
            f"Timed out after {timeout_seconds:.3f}s while consuming SSE response."
        ) from exc


def _planned_arrival_offsets(
    num_requests: int,
    qps: float,
    arrival_process: ArrivalProcess,
    random_seed: int,
) -> list[float]:
    """生成一个 mode 内每个请求相对启动点的计划到达时间秒。"""
    if qps == 0:
        return [0.0] * num_requests
    random_generator = random.Random(random_seed)
    offsets = [0.0]
    for _ in range(1, num_requests):
        if arrival_process == "fixed":
            inter_arrival_seconds = 1.0 / qps
        else:
            inter_arrival_seconds = random_generator.expovariate(qps)
        offsets.append(offsets[-1] + inter_arrival_seconds)
    return offsets


async def _replay_requests(
    samples: Sequence[SampleReplayInput],
    rendered_prompts: Mapping[str, RenderedReplayPrompt],
    renderer: ChatTemplateRenderer,
    args: argparse.Namespace,
    run_id: str,
    replay_sample: ReplaySample,
) -> list[RequestMetric]:
    """依据 QPS 和并发限制回放一个脚本的全部样本。

    Args:
        samples: 已验证的 mixed-source 样本。
        rendered_prompts: 每个 sample 已预渲染的 chat-template prompt。
        renderer: 目标模型 tokenizer 与固定 chat template。
        args: 当前脚本的 CLI 参数。
        run_id: 本次实验的稳定标识。
        replay_sample: 当前模式负责单请求 REST 链路的异步函数。

    Returns:
        与输入样本顺序一致的请求指标。
    """
    semaphore = asyncio.Semaphore(args.max_concurrency)
    planned_offsets = _planned_arrival_offsets(
        len(samples),
        args.qps,
        args.arrival_process,
        args.random_seed,
    )
    mode_start_time = time.perf_counter()

    async with httpx.AsyncClient(timeout=args.http_timeout) as client:

        async def run_one(
            sample: SampleReplayInput,
            planned_offset: float,
        ) -> RequestMetric:
            await _sleep_until(mode_start_time + planned_offset)
            semaphore_wait_start_time = time.perf_counter()
            async with semaphore:
                queue_delay_ms = (
                    time.perf_counter() - semaphore_wait_start_time
                ) * 1000
                rendered_prompt = rendered_prompts[sample.sample_id]
                return await replay_sample(
                    client,
                    sample,
                    rendered_prompt,
                    renderer,
                    args,
                    run_id,
                    planned_offset * 1000,
                    queue_delay_ms,
                )

        def fail_fast_error(result: RequestMetric) -> str | None:
            """返回当前请求触发的 fail-fast 错误；未触发时返回 None。"""
            if args.fail_fast and result.result_status != "ok":
                return result.error or "Replay request failed."
            if (
                args.formal_fail_fast
                and result.formal_result_status == "invalid"
            ):
                return (
                    result.stats_validation_error
                    or "Replay result failed formal validation."
                )
            return None

        if args.qps == 0:
            results: list[RequestMetric] = []
            for sample in samples:
                result = await run_one(sample, 0.0)
                results.append(result)
                if error := fail_fast_error(result):
                    raise ReplayHttpError(error)
            return results

        if not (args.fail_fast or args.formal_fail_fast):
            tasks = [
                asyncio.create_task(run_one(sample, planned_offset))
                for sample, planned_offset in zip(samples, planned_offsets)
            ]
            return await asyncio.gather(*tasks)

        async def run_indexed(
            index: int,
            sample: SampleReplayInput,
            planned_offset: float,
        ) -> tuple[int, RequestMetric]:
            return index, await run_one(sample, planned_offset)

        tasks = [
            asyncio.create_task(run_indexed(index, sample, planned_offset))
            for index, (sample, planned_offset) in enumerate(
                zip(samples, planned_offsets)
            )
        ]
        ordered_results: list[RequestMetric | None] = [None] * len(tasks)
        try:
            for completed_task in asyncio.as_completed(tasks):
                index, result = await completed_task
                ordered_results[index] = result
                if error := fail_fast_error(result):
                    raise ReplayHttpError(error)
        except BaseException:
            for task in tasks:
                if not task.done():
                    task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            raise

        assert all(result is not None for result in ordered_results)
        return [result for result in ordered_results if result is not None]


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
    lower_value = ordered_values[lower_index]
    upper_value = ordered_values[upper_index]
    interpolation = position - lower_index
    return lower_value + (upper_value - lower_value) * interpolation


def _summarize_metrics(metrics: Sequence[RequestMetric]) -> dict[str, Any]:
    """汇总各模式成功请求的 TTFT 分布与错误数量。"""
    summary_by_mode: dict[str, dict[str, Any]] = {}
    metrics_by_mode: dict[str, list[RequestMetric]] = defaultdict(list)
    for metric in metrics:
        metrics_by_mode[metric.mode].append(metric)

    for mode, mode_metrics in metrics_by_mode.items():
        successful_ttfts = [
            metric.finish_to_first_token_ms
            for metric in mode_metrics
            if metric.first_token_received
            and metric.finish_to_first_token_ms is not None
        ]
        numeric_ttfts = [float(ttft) for ttft in successful_ttfts]
        transport_success_metrics = [
            metric
            for metric in mode_metrics
            if metric.transport_status == "ok"
        ]
        stats_metrics = [
            metric
            for metric in mode_metrics
            if metric.stats_validation_status == "valid"
        ]
        formal_metrics = [
            metric
            for metric in mode_metrics
            if metric.formal_result_status == "valid"
        ]

        def sum_stat(field_name: str) -> int:
            """对当前模式中通过 schema v2 校验的整数统计求和。"""
            values = [getattr(metric, field_name) for metric in stats_metrics]
            return sum(value for value in values if isinstance(value, int))

        total_accepted = sum_stat("update_events_accepted")
        total_admitted = sum_stat("update_works_admitted")
        total_eager = sum_stat("eager_update_prefill_tokens")
        total_saved = sum_stat("saved_update_prefill_tokens")
        lcp_ratios = [
            metric.mean_update_lcp_ratio
            for metric in stats_metrics
            if metric.mean_update_lcp_ratio is not None
        ]
        num_requests = len(mode_metrics)
        num_first_token_success = sum(
            metric.first_token_received for metric in mode_metrics
        )
        summary_by_mode[mode] = {
            "num_requests": num_requests,
            "num_success": num_first_token_success,
            "num_error": num_requests - num_first_token_success,
            "num_with_prompt_stream_stats": len(stats_metrics),
            "num_transport_success": len(transport_success_metrics),
            "num_transport_error": num_requests - len(transport_success_metrics),
            "transport_success_rate": (
                len(transport_success_metrics) / num_requests
                if num_requests
                else None
            ),
            "num_first_token_success": num_first_token_success,
            "first_token_success_rate": (
                num_first_token_success / num_requests if num_requests else None
            ),
            "num_stats_valid": len(stats_metrics),
            "stats_valid_rate": (
                len(stats_metrics) / num_requests if num_requests else None
            ),
            "num_formal_valid": len(formal_metrics),
            "formal_valid_rate": (
                len(formal_metrics) / num_requests if num_requests else None
            ),
            "token_stats_coverage": (
                len(stats_metrics) / num_first_token_success
                if num_first_token_success
                else None
            ),
            "ttft_ms": {
                "mean": statistics.fmean(numeric_ttfts) if numeric_ttfts else None,
                "min": min(numeric_ttfts) if numeric_ttfts else None,
                "p50": _percentile(numeric_ttfts, 50),
                "p90": _percentile(numeric_ttfts, 90),
                "p95": _percentile(numeric_ttfts, 95),
                "p99": _percentile(numeric_ttfts, 99),
                "max": max(numeric_ttfts) if numeric_ttfts else None,
            },
            "prompt_stream_stats": {
                "update_events_accepted": total_accepted,
                "update_works_admitted": total_admitted,
                "update_works_completed": sum_stat("update_works_completed"),
                "update_policy_works_admitted": sum_stat(
                    "update_policy_works_admitted"
                ),
                "update_policy_works_completed": sum_stat(
                    "update_policy_works_completed"
                ),
                "update_policy_works_cancelled_by_recovery": sum_stat(
                    "update_policy_works_cancelled_by_recovery"
                ),
                "update_policy_works_outstanding": sum_stat(
                    "update_policy_works_outstanding"
                ),
                "update_execution_attempts_started": sum_stat(
                    "update_execution_attempts_started"
                ),
                "update_execution_attempts_completed": sum_stat(
                    "update_execution_attempts_completed"
                ),
                "update_execution_attempts_preempted": sum_stat(
                    "update_execution_attempts_preempted"
                ),
                "update_recovery_works_created": sum_stat(
                    "update_recovery_works_created"
                ),
                "update_tokens_scheduled_total": sum_stat(
                    "update_tokens_scheduled_total"
                ),
                "update_tokens_scheduled_for_recovery": sum_stat(
                    "update_tokens_scheduled_for_recovery"
                ),
                "prefix_append_tokens_scheduled_for_recovery": sum_stat(
                    "prefix_append_tokens_scheduled_for_recovery"
                ),
                "update_tokens_discarded_by_preemption": sum_stat(
                    "update_tokens_discarded_by_preemption"
                ),
                "prefix_append_tokens_discarded_by_preemption": sum_stat(
                    "prefix_append_tokens_discarded_by_preemption"
                ),
                "question_tokens_discarded_by_preemption": sum_stat(
                    "question_tokens_discarded_by_preemption"
                ),
                "prompt_stream_running_preemptions": sum_stat(
                    "prompt_stream_running_preemptions"
                ),
                "prompt_stream_dormant_evictions": sum_stat(
                    "prompt_stream_dormant_evictions"
                ),
                "prompt_stream_retained_waiting_evictions": sum_stat(
                    "prompt_stream_retained_waiting_evictions"
                ),
                "prompt_stream_allocation_failures_running": sum_stat(
                    "prompt_stream_allocation_failures_running"
                ),
                "prompt_stream_allocation_failures_waiting": sum_stat(
                    "prompt_stream_allocation_failures_waiting"
                ),
                "prompt_stream_prefix_cache_resets": sum_stat(
                    "prompt_stream_prefix_cache_resets"
                ),
                "eager_update_prefill_tokens": total_eager,
                "admitted_update_prefill_tokens": sum_stat(
                    "admitted_update_prefill_tokens"
                ),
                "scheduled_update_prefill_tokens": sum_stat(
                    "scheduled_update_prefill_tokens"
                ),
                "saved_update_prefill_tokens": total_saved,
                "extra_update_prefill_tokens": sum_stat(
                    "extra_update_prefill_tokens"
                ),
                "update_tokens_invalidated_by_snapshot": sum_stat(
                    "update_tokens_invalidated_by_snapshot"
                ),
                "update_tokens_invalidated": sum_stat(
                    "update_tokens_invalidated"
                ),
                "coalescing_ratio": (
                    1.0 - total_admitted / total_accepted
                    if total_accepted > 0
                    else None
                ),
                "token_saving_ratio": (
                    total_saved / total_eager if total_eager > 0 else None
                ),
                "mean_update_lcp_ratio": (
                    statistics.fmean(lcp_ratios) if lcp_ratios else None
                ),
            },
        }
    return summary_by_mode


def _request_metric_row(metric: RequestMetric) -> dict[str, Any]:
    """将 RequestMetric 扁平化为 request_metrics.csv 的一行。"""
    row = asdict(metric)
    row.pop("event_metrics", None)
    row.pop("response_payloads", None)
    return row


def _write_run_outputs(
    run_directory: Path,
    args: argparse.Namespace,
    renderer: ChatTemplateRenderer,
    samples: Sequence[SampleReplayInput],
    metrics: Sequence[RequestMetric],
) -> None:
    """写入可复现实验的配置、逐请求指标、逐事件指标和摘要。"""
    request_rows = [_request_metric_row(metric) for metric in metrics]
    event_rows = [
        asdict(event_metric)
        for metric in metrics
        for event_metric in metric.event_metrics
    ]
    selected_sample_manifest = [
        {
            "sample_id": sample.sample_id,
            "query_id": sample.query_id,
            "status": sample.status,
            "content_sha256": hashlib.sha256(
                json.dumps(
                    sample.raw_sample,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode("utf-8")
            ).hexdigest(),
        }
        for sample in samples
    ]
    selected_sample_set_sha256 = hashlib.sha256(
        json.dumps(
            selected_sample_manifest,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    _write_csv(
        run_directory / "request_metrics.csv", REQUEST_METRIC_FIELDS, request_rows
    )
    _write_csv(run_directory / "event_metrics.csv", EVENT_METRIC_FIELDS, event_rows)
    _write_json(
        run_directory / "config.json",
        {
            "args": vars(args),
            "prompt_format": {
                "name": "model_chat_template_system_user/v1",
                "tokenizer_model": renderer.tokenizer_model,
                "chat_template_sha256": renderer.chat_template_sha256,
                "transport": "pretokenized_segments",
            },
            "num_selected_samples": len(samples),
            "selected_sample_set_sha256": selected_sample_set_sha256,
            "selected_samples": selected_sample_manifest,
        },
    )
    _write_json(
        run_directory / "summary.json",
        {
            "replay_mode": args.replay_mode,
            "ttft_definition": (
                "Milliseconds from POST /v1/prompt_stream/finish to the first "
                "proxy SSE completion token."
                if args.replay_mode in STREAM_TIMING_MODES
                else (
                    "Milliseconds from POST /v1/completions_ttft to the first "
                    "P-node completion token."
                    if args.ttft_only
                    else (
                        "Milliseconds from POST /v1/completions to the first "
                        "P-node completion token in the P-to-D path."
                    )
                )
            ),
            "summary_by_mode": _summarize_metrics(metrics),
        },
    )
    if args.save_responses:
        response_directory = run_directory / "responses"
        for metric in metrics:
            if not metric.response_payloads:
                continue
            _write_json(
                response_directory / f"{metric.mode}-{metric.sample_id}.json",
                metric.response_payloads,
            )


def _validate_args(args: argparse.Namespace) -> None:
    """在读取数据或发 HTTP 前检查数值与模式参数。"""
    if not args.tokenizer_model.strip():
        raise ValueError("--tokenizer-model must not be empty.")
    if args.start_index < 0:
        raise ValueError("--start-index must be non-negative.")
    if args.max_requests < 0:
        raise ValueError("--max-requests must be non-negative.")
    if args.qps < 0:
        raise ValueError("--qps must be non-negative.")
    if args.max_concurrency <= 0:
        raise ValueError("--max-concurrency must be positive.")
    if args.http_timeout <= 0:
        raise ValueError("--http-timeout must be positive.")
    if args.sse_timeout is not None and args.sse_timeout <= 0:
        raise ValueError("--sse-timeout must be positive when specified.")
    if args.max_tokens is not None and args.max_tokens <= 0:
        raise ValueError("--max-tokens must be positive when specified.")
    if args.repeat_index < 0:
        raise ValueError("--repeat-index must be non-negative.")
    if args.p_node_coalescing_config_json is not None:
        try:
            coalescing_config = json.loads(args.p_node_coalescing_config_json)
        except json.JSONDecodeError as exc:
            raise ValueError(
                "--p-node-coalescing-config-json must be valid JSON."
            ) from exc
        if not isinstance(coalescing_config, dict):
            raise ValueError(
                "--p-node-coalescing-config-json must contain a JSON object."
            )
    if args.experiment_suite == "update_coalescing":
        if args.expected_coalescing == "any":
            raise ValueError(
                "Update coalescing experiments must declare --expected-"
                "coalescing enabled or disabled."
            )
        has_expected_config = args.p_node_coalescing_config_json is not None
        if args.expected_coalescing == "enabled" and not has_expected_config:
            raise ValueError(
                "Enabled Update coalescing experiments require --p-node-"
                "coalescing-config-json."
            )
        if args.expected_coalescing == "disabled" and has_expected_config:
            raise ValueError(
                "Disabled Update coalescing experiments must not declare a "
                "P-node coalescing config."
            )
    timing_field_names = (
        (
            "append_initial_delay",
            "update_initial_delay",
            "append_chunk_interval",
            "update_chunk_interval",
            "source_finish_delay",
        )
        if args.replay_mode in STREAM_TIMING_MODES
        else ()
    )
    for field_name in timing_field_names:
        if getattr(args, field_name) < 0:
            raise ValueError(f"--{field_name.replace('_', '-')} must be non-negative.")
    if (
        args.replay_mode in STREAM_TIMING_MODES
        and args.event_progress_log_interval < 0
    ):
        raise ValueError("--event-progress-log-interval must be non-negative.")


def _make_run_directory(args: argparse.Namespace) -> tuple[str, Path]:
    """创建本次回放的独立结果目录。"""
    run_id = args.run_name or f"{_utc_timestamp()}-{args.replay_mode}"
    run_directory = args.output_dir / run_id
    if run_directory.exists():
        raise FileExistsError(
            f"Replay output directory already exists: {run_directory}. "
            "Choose a new --run-name."
        )
    run_directory.mkdir(parents=True)
    return run_id, run_directory


async def run(
    args: argparse.Namespace,
    replay_sample: ReplaySample,
) -> Path:
    """选择样本、按模式回放，并写出完整实验指标。

    Args:
        args: 已解析的命令行参数。
        replay_sample: 当前脚本的单请求回放函数。

    Returns:
        本次结果目录。
    """
    _validate_args(args)
    candidate_samples = _select_samples(args)
    samples = _limit_selected_samples(candidate_samples, args)
    renderer = ChatTemplateRenderer.from_pretrained(
        args.tokenizer_model,
        args.tokenizer_trust_remote_code,
    )
    rendered_prompts = {
        sample.sample_id: _render_sample_prompt(sample, renderer)
        for sample in samples
    }
    if len(rendered_prompts) != len(samples):
        raise ValueError("Selected samples must have unique sample_id values.")
    run_id, run_directory = _make_run_directory(args)
    logger.info(
        "Selected %d samples; mode=%s qps=%.3f max_concurrency=%d.",
        len(samples),
        args.replay_mode,
        args.qps,
        args.max_concurrency,
    )

    if args.dry_run:
        dry_metrics: list[RequestMetric] = []
        for sample in samples:
            rendered_prompt = rendered_prompts[sample.sample_id]
            dry_metrics.append(
                RequestMetric(
                    run_id=run_id,
                    mode="dry_run",
                    sample_id=sample.sample_id,
                    query_id=sample.query_id,
                    sample_status=sample.status,
                    result_status="ok",
                    planned_arrival_offset_ms=0.0,
                    queue_delay_ms=0.0,
                    request_start_wall_time=time.time(),
                    num_append_chunks=sum(
                        len(_get_data_events(source_stream))
                        for source_stream in sample.source_streams
                        if source_stream.source_type == "append"
                    ),
                    num_update_snapshots=sum(
                        len(_get_data_events(source_stream))
                        for source_stream in sample.source_streams
                        if source_stream.source_type == "update"
                    ),
                    append_chars=rendered_prompt.append_chars,
                    final_update_chars=rendered_prompt.final_update_chars,
                    normal_prompt_chars=len(rendered_prompt.logical_prompt_text),
                )
            )
        _write_run_outputs(run_directory, args, renderer, samples, dry_metrics)
        return run_directory

    metrics = await _replay_requests(
        samples,
        rendered_prompts,
        renderer,
        args,
        run_id,
        replay_sample,
    )
    _write_run_outputs(run_directory, args, renderer, samples, metrics)
    return run_directory


def main(
    mode: ReplayMode,
    replay_sample: ReplaySample,
) -> None:
    """运行一个指定模式的回放 CLI。

    Args:
        mode: 回放模式，用于选择 CLI 参数及结果标签。
        replay_sample: 由对应入口提供的单请求 REST 回放函数。
    """
    args = parse_args(mode)
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
    run_directory = asyncio.run(run(args, replay_sample))
    logger.info("Replay results written to %s.", run_directory)
