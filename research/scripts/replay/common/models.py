# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""回放样本、请求和指标的数据结构。"""

from __future__ import annotations

import argparse
import logging
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal

import httpx

if TYPE_CHECKING:
    from .prompt import ChatTemplateRenderer

logger = logging.getLogger(__name__)


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
    question: str = ""
    evaluation: dict[str, Any] = field(default_factory=dict)


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
    start_text: str = ""
    input_transport: str = "text"


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
    finish HTTP 请求提交开始计时；normal 模式从完整请求提交开始计时。
    P-only 的终点是 prefill_ready，完整生成的终点是 D 首个回答 token。
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
    load: dict[str, Any] | None = None  # 统一负载时钟；与服务端 work/token 统计独立。
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
    completion_tokens: int = 0
    completion_token_count_exact: bool = False
    time_to_last_token_ms: float | None = None
    mean_time_per_output_token_ms: float | None = None
    output_tokens_per_second: float | None = None
    masked_padding_expected: str = "any"
    masked_padding_observed: bool | None = None
    masked_padding_validation_status: str = "not_evaluated"
    masked_padding_tokens: int | None = None
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
    completion_tokens: int
    completion_token_count_exact: bool
    last_token_time: float | None
    payloads: tuple[dict[str, Any], ...]
    prompt_stream_stats: dict[str, Any] | None
    masked_padding_manifest: dict[str, int] | None
    prefill_ready_time: float | None = None  # P 控制事件，与回答首 token 独立。

    @property
    def ttft_time(self) -> float:
        """返回本次计时终点；P-only 使用就绪事件，完整生成使用回答首 token。"""
        measured = (
            self.first_token_time
            if self.first_token_time is not None
            else self.prefill_ready_time
        )
        assert measured is not None
        return measured

    @property
    def ttft_origin(self) -> str | None:
        """显式区分 P 完成通知和实际回答来源。"""
        return (
            self.first_token_origin
            if self.first_token_time is not None
            else "prefill_ready"
        )


ReplaySample = Callable[
    [
        httpx.AsyncClient,
        SampleReplayInput,
        RenderedReplayPrompt,
        "ChatTemplateRenderer",
        argparse.Namespace,
        str,
        float,
        float,
    ],
    Awaitable[RequestMetric],
]


ReplayMode = Literal["stream", "normal", "merged_update"]


ArrivalProcess = Literal["fixed", "poisson"]
