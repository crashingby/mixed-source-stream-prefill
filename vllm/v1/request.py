# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import enum
import json
import math
import time
from collections import deque
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field, fields
from typing import TYPE_CHECKING, Any

import torch

from vllm.multimodal.inputs import MultiModalFeatureSpec
from vllm.pooling_params import PoolingParams
from vllm.sampling_params import SamplingParams
from vllm.utils import length_from_prompt_token_ids_or_embeds
from vllm.v1.engine import (
    EngineCoreEvent,
    EngineCoreEventType,
    EngineCoreRequest,
    FinishReason,
)
from vllm.v1.engine.prompt_stream import (
    PromptSourceType,
    PromptStreamEvent,
    PromptStreamEventType,
    parse_prompt_stream_padding_range,
)
from vllm.v1.metrics.stats import PrefillStats
from vllm.v1.structured_output.request import StructuredOutputRequest
from vllm.v1.utils import ConstantList

if TYPE_CHECKING:
    from vllm.config.prompt_stream import (
        PromptStreamUpdateCoalescingConfig,
    )
    from vllm.lora.request import LoRARequest
    from vllm.v1.core.kv_cache_utils import BlockHash


class PromptStreamKVSegment(enum.Enum):
    """流式 prompt 的两个物理 KV 域。

    `PREFIX_APPEND` 保存通用模板和所有 append chunk。它只会单调增长，
    因此可以使用原始 request_id 持有紧凑 block table。

    `UPDATE` 保存 update 快照和最终用户问题。update 快照可能失效重算，
    所以它必须和 PREFIX_APPEND 隔离，避免释放 update 后缀时误伤稳定前缀。
    """

    PREFIX_APPEND = "prefix_append"
    UPDATE = "update"


class PromptStreamExecutionPhase(enum.Enum):
    """流式 prompt 当前需要执行的 prefill 阶段。"""

    PREFIX_APPEND = "prefix_append"
    UPDATE = "update"
    QUESTION = "question"


class PromptStreamExecutionOrigin(enum.Enum):
    """标记 Prompt Stream work 的物理执行来源。"""

    POLICY = "policy"
    RECOVERY = "recovery"


class PromptStreamRecoveryReason(enum.Enum):
    """标记 Prompt Stream 丢失物理 KV 后进入恢复的原因。"""

    RUNNING_PREEMPTION = "running_preemption"
    DORMANT_EVICTION = "dormant_eviction"
    RETAINED_WAITING_EVICTION = "retained_waiting_eviction"
    PREFIX_CACHE_RESET = "prefix_cache_reset"
    UNKNOWN = "unknown"


class UpdateAdmissionStatus(enum.Enum):
    """记录一次 Update policy admission 的稳定生命周期状态。"""

    PENDING = "pending"
    ACTIVE = "active"
    COMPLETED = "completed"
    CANCELLED_BY_RECOVERY = "cancelled_by_recovery"


class AppendGatePhase(enum.Enum):
    """表示 Append gate 当前允许 Update admission 的阶段。"""

    # Append 尚未结束，严格 gate 会阻止 Update work 物化。
    WAITING_FOR_APPEND = "waiting_for_append"
    # Append source 已封闭，严格 gate 已经打开。
    APPEND_FINISHED = "append_finished"
    # Update 先结束并触发近似早放行。
    EARLY_RELEASE = "early_release"


@dataclass
class AppendUpdateGateState:
    """保存单请求 Append gate、source barrier 和 Append epoch。"""

    # gate 的可读阶段，便于 scheduler 日志区分严格与近似路径。
    phase: AppendGatePhase = AppendGatePhase.WAITING_FOR_APPEND
    # 第一版只允许一个真实 Append source；start 合成事件不计入该限制。
    append_source_id: str | None = None
    # 第一版只允许一个 Update source。
    update_source_id: str | None = None
    # 两个来源的逻辑封闭状态。
    append_source_finished: bool = False
    update_source_finished: bool = False
    # 每次非空 Append 事件递增，表示已接受的逻辑 Append 上下文。
    append_epoch: int = 0
    # 已完整物化到 PREFIX_APPEND KV 的 Append epoch。
    committed_append_epoch: int = 0
    # Append source 封闭时冻结的最终 epoch。
    final_append_epoch: int | None = None
    # UPDATE KV 当前实际绑定的 Append epoch。
    update_committed_append_epoch: int | None = None


@dataclass
class UpdateCoalescingState:
    """保存唯一尚未 admission 的 Update 合并窗口。"""

    # 启动级配置由 Scheduler 在 Request 入队后注入。
    config: "PromptStreamUpdateCoalescingConfig"
    # Request 进入 EngineCore 时对应的 monotonic 时间，用于 TTFT deadline。
    request_start_time: float
    # 窗口内最新且唯一可覆盖的完整 Update 快照。
    latest_pending_token_ids: tuple[int, ...] | None = None
    # 最近一次已 admission 的完整快照，即逻辑执行链尾部。
    last_admitted_token_ids: tuple[int, ...] = ()
    # 最近一次已 admission work 绑定的 Append epoch。
    last_admitted_append_epoch: int | None = None
    # 当前窗口第一次 Update 到达的 monotonic 时间。
    first_pending_time: float | None = None
    # 当前窗口最近一次 Update 到达的 monotonic 时间。
    last_update_time: float | None = None
    # 最近一次合法 Update 的跨窗口到达时间。
    last_observed_update_time: float | None = None
    # 最近一次合法 Update 到达时绑定的 Append epoch。跨 epoch 的 eager
    # 反事实不能复用旧 Update LCP，必须按从零 Prefill 计数。
    last_observed_update_append_epoch: int | None = None
    # interval 与 gain EWMA 各自拥有独立的跨窗口观测数。
    interval_observation_count: int = 0
    gain_observation_count: int = 0
    # policy 计算出的软、硬与 TTFT 绝对 deadline。
    soft_deadline: float | None = None
    hard_deadline: float | None = None
    ttft_deadline: float | None = None
    # Adaptive 每次新 Update 只允许 Scheduler 按当时负载重算
    # 一次绝对 soft deadline。recovery 不能重新打开该开关。
    adaptive_deadline_needs_refresh: bool = False
    # policy 已要求 flush、但 admission 暂时被 gate 阻挡。
    flush_ready: bool = False
    # sticky ready 的首次触发原因，仅用于日志与实验统计。
    flush_ready_reason: str | None = None
    # deadline heap 使用的惰性失效代数。
    generation: int = 0
    # 当前窗口累计的 Update 事件与假想 eager token 成本。
    pending_event_count: int = 0
    hypothetical_eager_tokens: int = 0
    # 到达间隔与覆盖收益的请求级 EWMA。
    interval_ewma: float | None = None
    merge_gain_ewma: float | None = None
    # 统计口径中的最近一次逻辑 admission。它不随 preemption 回退，避免
    # 恢复物理 KV 时把同一目标重复计为算法创建的 work。
    stats_last_admitted_token_ids: tuple[int, ...] = ()
    stats_last_admitted_append_epoch: int | None = None


@dataclass
class UpdateAdmissionRecord:
    """保存一次 Update policy admission 的稳定身份与终态。"""

    admission_id: int
    target_token_ids: tuple[int, ...]
    required_append_epoch: int
    policy_prefill_tokens: int
    status: UpdateAdmissionStatus = UpdateAdmissionStatus.PENDING


@dataclass(frozen=True)
class PromptStreamRecoveryDelta:
    """描述一次 latest-only recovery 对请求物理状态的影响。"""

    reason: PromptStreamRecoveryReason
    cancelled_update_admission_ids: tuple[int, ...]
    carried_update_admission_id: int | None
    prefix_append_tokens_discarded: int
    update_tokens_discarded: int
    question_tokens_discarded: int


@dataclass
class PromptStreamWorkItem:
    """表示同一长期请求内一项可竞争的流式 prefill 工作。

    固定模板完成 bootstrap 后，Append 和 Update 事件都会产生一个 work item，
    并按进入 Request 的顺序进入同一个队列。这样二者不再由阶段选择函数的
    固定优先级决定。

    Update work 在 flush 时冻结执行基准、目标快照、source-local LCP 和
    所需 Append epoch。进入 FIFO 后不再被后续 Update 合并、替换或取消。
    Question 也作为显式 work 排在最终 Update work 后面。
    """

    # 本 work item 要写入的物理 KV 域。
    phase: PromptStreamExecutionPhase
    # Append/Question work 完成时 PREFIX_APPEND 应达到的逻辑长度（含模板）。
    prefix_append_logical_len: int = 0
    # Append work 完成时应提交的 Append epoch。
    target_append_epoch: int = 0
    # Update work 在 flush 时冻结的完整目标快照。
    update_token_ids: tuple[int, ...] | None = None
    # Update work 在 flush 时冻结的逻辑执行基准。
    execution_base_token_ids: tuple[int, ...] | None = None
    # 执行基准 work 当时绑定的 Append epoch。
    execution_base_append_epoch: int | None = None
    # 当前 work 读取 PREFIX_APPEND KV 所要求的物理 epoch。
    required_append_epoch: int = 0
    # 目标快照相对 execution base 的 source-local LCP；epoch 不同时为 0。
    update_common_prefix_len: int = 0
    # Update/Question 被选中时安装的 PREFIX_APPEND 物理 token 视图。
    prefix_append_token_ids: tuple[int, ...] = ()
    # Question work 在 PROMPT_FINISHED 时冻结的 suffix。
    question_token_ids: tuple[int, ...] | None = None
    # 当前 work item 已完成的 segment-local token 数。
    num_computed_tokens: int = 0
    # Update policy admission 的稳定 ID；纯物理 recovery work 使用 None。
    update_admission_id: int | None = None
    # work 真正成为 active 时分配的物理执行 attempt ID。
    execution_attempt_id: int | None = None
    # 区分 policy work 与抢占后重建物理 KV 的 recovery work。
    execution_origin: PromptStreamExecutionOrigin = (
        PromptStreamExecutionOrigin.POLICY
    )


@dataclass(frozen=True)
class PromptStreamExecutionView:
    """描述 scheduler 和 worker 本轮看到的物理 prompt 视图。

    `token_ids` 已包含 `PREFIX_APPEND` 末尾的 block 对齐 padding。worker
    必须同时使用这里的 token 序列、`num_computed_tokens` 和按
    `required_segments` 拼接的 block table，三者不能独立更新。

    这里需要区分两个概念：
    - `required_segments` 表示本轮 attention 需要读哪些 KV blocks。
    - `write_segment` 表示本轮新 token 的 KV 应写入哪个物理 segment。

    例如 update 阶段会读取 `PREFIX_APPEND + UPDATE`，但只向 `UPDATE`
    segment 继续申请/写入新 blocks。
    """

    phase: PromptStreamExecutionPhase
    write_segment: PromptStreamKVSegment
    required_segments: tuple[PromptStreamKVSegment, ...]
    token_ids: tuple[int, ...]
    num_computed_tokens: int
    prefix_append_physical_len: int
    segment_offset: int
    # 当前 work 冻结的物理 padding 区间；Append 增长后不能沿用旧视图。
    padding_range: tuple[int, int] = (0, 0)


@dataclass
class StreamingUpdate:
    """Lightweight data for streaming session continuation.

    Contains only the fields needed to update an existing streaming session
    with new input data.
    """

    mm_features: list[MultiModalFeatureSpec] | None
    prompt_token_ids: list[int] | None
    max_tokens: int
    arrival_time: float
    sampling_params: SamplingParams | None

    @classmethod
    def from_request(cls, request: "Request") -> "StreamingUpdate | None":
        if not request.resumable:
            return None
        return cls(
            mm_features=request.mm_features,
            prompt_token_ids=request.prompt_token_ids,
            max_tokens=request.max_tokens,
            arrival_time=request.arrival_time,
            sampling_params=request.sampling_params,
        )


@dataclass
class PromptStreamMutation:
    """描述一次流式 prompt 事件对 Request prompt 的影响。"""

    # 幂等重复 chunk 等没有实际修改 prompt 的事件。
    applied: bool
    old_num_prompt_tokens: int
    new_num_prompt_tokens: int
    # 仅用于统计和调试的逻辑 prompt 公共前缀长度。
    common_prefix_len: int
    # 每次 prompt 真实变化都递增，后面防止 scheduler 用旧状态 decode。
    prompt_revision: int

    @classmethod
    def noop(
        cls,
        num_prompt_tokens: int,
        prompt_revision: int,
    ) -> "PromptStreamMutation":
        """构造一个没有改变 prompt 的 mutation。"""
        return cls(
            applied=False,
            old_num_prompt_tokens=num_prompt_tokens,
            new_num_prompt_tokens=num_prompt_tokens,
            common_prefix_len=num_prompt_tokens,
            prompt_revision=prompt_revision,
        )


@dataclass
class PromptSourceState:
    """保存单个 prompt source 的流式状态。"""

    # 来源 ID。
    source_id: str
    # 来源语义，创建后不应该被更改
    source_type: PromptSourceType
    # 该来源当前有效 token。
    token_ids: list[int]
    # 该来源当前有效版本。
    version: int
    # 下一条期望收到的序号。
    next_sequence_no: int
    # 已经应用过的 chunk，用于去重。
    seen_chunk_ids: set[str]
    # 为后续乱序协议预留；当前实现对 sequence gap 直接 fail closed。
    pending_events: dict[int, PromptStreamEvent]
    # 该来源是否结束。
    finished: bool
    # 该来源最近一次事件到达时间。
    last_arrival_time: float
    # SOURCE_FINISHED 的幂等信息。
    finished_version: int | None = None
    finished_sequence_no: int | None = None
    finished_chunk_id: str | None = None


@dataclass(frozen=True)
class PromptStreamTokenStats:
    """保存单个 Prompt Stream 请求的只读 token 统计快照。

    所有字段均为 JSON 可表达的标量。调用方可以直接使用 ``to_dict()``
    构造跨进程或 REST 响应载荷；快照本身不会随 Request 后续状态变化。
    """

    metrics_schema_version: int
    coalescing_enabled: bool
    update_coalescing_config_json: str | None
    update_events_accepted: int
    update_policy_works_admitted: int
    update_policy_works_completed: int
    update_policy_works_cancelled_by_recovery: int
    update_policy_works_outstanding: int
    # v1 aliases 保持旧客户端兼容；语义映射到 policy 层。
    update_works_admitted: int
    update_works_completed: int
    eager_update_prefill_tokens: int | None
    policy_admitted_update_prefill_tokens: int
    policy_saved_update_prefill_tokens: int | None
    policy_extra_update_prefill_tokens: int | None
    admitted_update_prefill_tokens: int
    scheduled_update_prefill_tokens: int
    saved_update_prefill_tokens: int | None
    extra_update_prefill_tokens: int | None
    update_tokens_invalidated: int
    update_tokens_invalidated_by_snapshot: int
    update_execution_attempts_started: int
    update_execution_attempts_completed: int
    update_execution_attempts_preempted: int
    update_recovery_works_created: int
    update_tokens_scheduled_total: int
    update_tokens_scheduled_for_recovery: int
    prefix_append_tokens_scheduled_for_recovery: int
    update_tokens_discarded_by_preemption: int
    prefix_append_tokens_discarded_by_preemption: int
    question_tokens_discarded_by_preemption: int
    prompt_stream_running_preemptions: int
    prompt_stream_dormant_evictions: int
    prompt_stream_retained_waiting_evictions: int
    prompt_stream_prefix_cache_resets: int
    prompt_stream_allocation_failures_running: int
    prompt_stream_allocation_failures_waiting: int

    def to_dict(self) -> dict[str, bool | int | str | None]:
        """返回可直接交给 msgspec/JSON 的独立字典。"""
        return {
            "metrics_schema_version": self.metrics_schema_version,
            "coalescing_enabled": self.coalescing_enabled,
            "update_coalescing_config_json": (
                self.update_coalescing_config_json
            ),
            "update_events_accepted": self.update_events_accepted,
            "update_policy_works_admitted": self.update_policy_works_admitted,
            "update_policy_works_completed": self.update_policy_works_completed,
            "update_policy_works_cancelled_by_recovery": (
                self.update_policy_works_cancelled_by_recovery
            ),
            "update_policy_works_outstanding": (
                self.update_policy_works_outstanding
            ),
            "update_works_admitted": self.update_works_admitted,
            "update_works_completed": self.update_works_completed,
            "eager_update_prefill_tokens": self.eager_update_prefill_tokens,
            "policy_admitted_update_prefill_tokens": (
                self.policy_admitted_update_prefill_tokens
            ),
            "policy_saved_update_prefill_tokens": (
                self.policy_saved_update_prefill_tokens
            ),
            "policy_extra_update_prefill_tokens": (
                self.policy_extra_update_prefill_tokens
            ),
            "admitted_update_prefill_tokens": (
                self.admitted_update_prefill_tokens
            ),
            "scheduled_update_prefill_tokens": (
                self.scheduled_update_prefill_tokens
            ),
            "saved_update_prefill_tokens": self.saved_update_prefill_tokens,
            "extra_update_prefill_tokens": self.extra_update_prefill_tokens,
            "update_tokens_invalidated": self.update_tokens_invalidated,
            "update_tokens_invalidated_by_snapshot": (
                self.update_tokens_invalidated_by_snapshot
            ),
            "update_execution_attempts_started": (
                self.update_execution_attempts_started
            ),
            "update_execution_attempts_completed": (
                self.update_execution_attempts_completed
            ),
            "update_execution_attempts_preempted": (
                self.update_execution_attempts_preempted
            ),
            "update_recovery_works_created": self.update_recovery_works_created,
            "update_tokens_scheduled_total": self.update_tokens_scheduled_total,
            "update_tokens_scheduled_for_recovery": (
                self.update_tokens_scheduled_for_recovery
            ),
            "prefix_append_tokens_scheduled_for_recovery": (
                self.prefix_append_tokens_scheduled_for_recovery
            ),
            "update_tokens_discarded_by_preemption": (
                self.update_tokens_discarded_by_preemption
            ),
            "prefix_append_tokens_discarded_by_preemption": (
                self.prefix_append_tokens_discarded_by_preemption
            ),
            "question_tokens_discarded_by_preemption": (
                self.question_tokens_discarded_by_preemption
            ),
            "prompt_stream_running_preemptions": (
                self.prompt_stream_running_preemptions
            ),
            "prompt_stream_dormant_evictions": self.prompt_stream_dormant_evictions,
            "prompt_stream_retained_waiting_evictions": (
                self.prompt_stream_retained_waiting_evictions
            ),
            "prompt_stream_prefix_cache_resets": (
                self.prompt_stream_prefix_cache_resets
            ),
            "prompt_stream_allocation_failures_running": (
                self.prompt_stream_allocation_failures_running
            ),
            "prompt_stream_allocation_failures_waiting": (
                self.prompt_stream_allocation_failures_waiting
            ),
        }


@dataclass
class PromptStreamState:
    """保存一个 Request 内部的多源流式 prompt 状态。"""

    # 流式 source 之前的固定 prompt 前缀，例如 system prompt。
    prefix_token_ids: list[int]
    # 从 source_id 到 PromptSourceState 的映射。
    sources: dict[str, PromptSourceState]
    # source 的稳定到达顺序；最终布局会按 source_type 再分段。
    source_order: list[str]
    # 所有 Append source 按事件实际提交顺序形成的请求级单调 token 日志。
    append_token_ids: list[int]
    # PROMPT_FINISHED 事件携带的最终用户问题 / prompt suffix。
    suffix_token_ids: list[int]
    # 请求级流式 prompt 是否结束。
    finished: bool
    # prompt 每次真实变化后递增，用于防止使用旧状态 decode。
    revision: int
    # 已经写入 UPDATE KV、但因新快照 LCP 缩短而失效的 token 累计。
    total_tokens_invalidated: int
    # 请求生命周期内通过协议校验且真实改变内容的 Update 事件数。
    update_events_accepted: int
    # 算法真正创建的 Update work 数；preemption 恢复不重复增加。
    update_works_admitted: int
    # 已完成的算法 admission work 数；恢复用的合成 work 不计入。
    update_works_completed: int
    # 逐 Update 立即执行反事实下需要 Prefill 的 Update token 累计。
    total_eager_update_prefill_tokens: int
    # 合并策略真正 admission 的 Update token 累计。
    total_admitted_update_prefill_tokens: int
    # Scheduler 实际推进的 Update token 累计，preemption 重放会重复增加。
    total_scheduled_update_prefill_tokens: int
    # 稳定 admission ledger；终态只由完成或显式 recovery 取消推进。
    update_admissions: dict[int, UpdateAdmissionRecord] = field(
        default_factory=dict
    )
    next_update_admission_id: int = 1
    # 物理 attempt 与 recovery token 统计独立于 policy admission。
    next_update_execution_attempt_id: int = 1
    update_execution_attempts_started: int = 0
    update_execution_attempts_completed: int = 0
    update_execution_attempts_preempted: int = 0
    update_recovery_works_created: int = 0
    total_recovery_update_prefill_tokens: int = 0
    total_recovery_prefix_append_prefill_tokens: int = 0
    total_update_tokens_discarded_by_preemption: int = 0
    total_prefix_append_tokens_discarded_by_preemption: int = 0
    total_question_tokens_discarded_by_preemption: int = 0
    # 请求级 pressure 计数由 Scheduler 通过 Request 接口更新。
    prompt_stream_running_preemptions: int = 0
    prompt_stream_dormant_evictions: int = 0
    prompt_stream_retained_waiting_evictions: int = 0
    prompt_stream_prefix_cache_resets: int = 0
    prompt_stream_allocation_failures_running: int = 0
    prompt_stream_allocation_failures_waiting: int = 0
    # reset 后 bootstrap/Append 物理重建使用 recovery origin。
    prefix_append_recovery_pending: bool = False
    # prefix+append 物理域已经完成 prefill 的 token 数，包含尾部 padding。
    prefix_append_num_computed_tokens: int = 0
    # 当前 update 快照已经完成 prefill 的 token 数。
    update_num_computed_tokens: int = 0
    # 当前实际物化到 UPDATE KV 域的完整快照。它可能落后于 source.token_ids，
    # 因为后到 Update work 会先在 pending 队列中等待前一代快照完成。
    update_committed_token_ids: list[int] = field(default_factory=list)
    # 最终用户问题已经完成 prefill 的 token 数。
    question_num_computed_tokens: int = 0
    # scheduler 首次准备执行视图后记录的统一 block size。
    block_size: int | None = None
    # PREFIX_APPEND 尾部 padding 使用的确定性 token id。
    padding_token_id: int = 0
    # 已完整提交到 PREFIX_APPEND KV 域的逻辑 token 数，包含固定模板但不含
    # block 对齐 padding。Update 只能读取这个稳定快照，不能读取尚未被调度的
    # Append 事件。
    prefix_append_committed_logical_len: int = 0
    # 固定模板是否已经独立完成 bootstrap prefill。它是唯一强制优先的阶段。
    prefix_append_bootstrap_completed: bool = False
    # 模板完成后等待 scheduler 选择的 Append/Update work item FIFO 队列。
    pending_work_items: deque[PromptStreamWorkItem] = field(default_factory=deque)
    # 当前正在 chunked prefill 的 work item；一个 work item 中途不可切换，
    # 防止覆写 PREFIX_APPEND 尾部 padding 后再用旧 padding 作为 Update 前缀。
    active_work_item: PromptStreamWorkItem | None = None
    # 下一次选中 Update work 时需要执行的 UPDATE 域截断边界。事件到达时不
    # 立刻释放块，避免破坏仍在运行或排队等待的旧 Update work。
    pending_update_kv_truncation_keep_tokens: int | None = None
    # 启用实验开关后保存 Append gate；默认关闭路径保持 None。
    append_update_gate: AppendUpdateGateState | None = None
    # 唯一尚未 admission 的 Update 窗口；默认关闭路径保持 None。
    update_coalescing: UpdateCoalescingState | None = None
    # 显式 Question work 的去重与完成状态。
    question_work_enqueued: bool = False
    question_work_completed: bool = False


@dataclass(frozen=True)
class PromptStreamLayout:
    """描述流式 prompt 在全局 KV 序列中的分段布局。

    该布局对应 P 节点最终要传给 D 节点的 KV 顺序：
    prefix/template + append segment + update segment + suffix/user question。
    """

    prefix_len: int
    append_len: int
    update_len: int
    suffix_len: int
    prefix_append_padding_len: int = 0

    @property
    def total_len(self) -> int:
        """返回不包含物理 padding 的逻辑 prompt token 数。"""
        return self.prefix_len + self.append_len + self.update_len + self.suffix_len

    @property
    def physical_total_len(self) -> int:
        """返回 P/D transfer 使用的物理 prompt token 数。"""
        return self.total_len + self.prefix_append_padding_len


class Request:
    def __init__(
        self,
        request_id: str,
        prompt_token_ids: list[int] | None,
        sampling_params: SamplingParams | None,
        pooling_params: PoolingParams | None,
        client_index: int = 0,
        arrival_time: float | None = None,
        prompt_embeds: torch.Tensor | None = None,
        prompt_is_token_ids: list[bool] | None = None,
        mm_features: list[MultiModalFeatureSpec] | None = None,
        lora_request: "LoRARequest | None" = None,
        cache_salt: str | None = None,
        priority: int = 0,
        trace_headers: Mapping[str, str] | None = None,
        block_hasher: Callable[["Request"], list["BlockHash"]] | None = None,
        resumable: bool = False,
        reasoning_ended: bool | None = None,
        reasoning_parser_kwargs: dict[str, Any] | None = None,
        abort_immediately: bool = False,
        prompt_stream_event: PromptStreamEvent | None = None,
    ) -> None:
        self.request_id = request_id
        self.client_index = client_index
        self.priority = priority
        self.sampling_params = sampling_params
        self.pooling_params = pooling_params
        self.lora_request = lora_request
        self.structured_output_request = StructuredOutputRequest.from_sampling_params(
            sampling_params
        )
        if self.structured_output_request is not None:
            self.structured_output_request.reasoning_ended = reasoning_ended
            self.structured_output_request.reasoning_parser_kwargs = (
                reasoning_parser_kwargs
            )
        self.arrival_time = arrival_time if arrival_time is not None else time.time()

        self.status = RequestStatus.WAITING
        self.events: list[EngineCoreEvent] = []
        self.stop_reason: int | str | None = None

        # P/D: Connector-specific KV transfer parameters.
        self.kv_transfer_params: dict[str, Any] | None = None

        if pooling_params is not None:
            # Pooling models.
            self.max_tokens = 1
        elif sampling_params is not None:
            # Generative models.
            assert sampling_params.max_tokens is not None
            self.max_tokens = sampling_params.max_tokens
            if self.structured_output_request is not None:
                self.status = RequestStatus.WAITING_FOR_STRUCTURED_OUTPUT_GRAMMAR

            if sampling_params.extra_args is not None:
                self.kv_transfer_params = sampling_params.extra_args.get(
                    "kv_transfer_params"
                )
        else:
            raise ValueError("sampling_params and pooling_params can't both be unset")

        self.prompt_token_ids = prompt_token_ids
        self.prompt_embeds = prompt_embeds
        # Per-position mask used in mixed-mode (chat completion with
        # prompt_embeds). `None` except when both `prompt_token_ids` and
        # `prompt_embeds` are set and their positions are interleaved.
        self.prompt_is_token_ids = prompt_is_token_ids
        # Cache per-block prompt-embed hashes to avoid rehashing the same
        # tensor slices when generating extra keys.
        self._prompt_embeds_per_block_hashes: dict[tuple[int, int], bytes] = {}
        self.num_prompt_tokens = length_from_prompt_token_ids_or_embeds(
            prompt_token_ids, prompt_embeds
        )
        self._output_token_ids: list[int] = []
        self._all_token_ids: list[int] = (
            self.prompt_token_ids.copy()
            if self.prompt_token_ids is not None
            else [0] * self.num_prompt_tokens
        )

        # Used in async scheduling.
        self.num_output_placeholders = 0
        self.async_tokens_to_discard = 0

        # V2+PP+async: Enforces `pp_size` cadence between same-request decode steps
        # so the worker's broadcast slot ring stays consistent.
        self.next_decode_eligible_step = 0

        # Seq of the most recent step this request was scheduled in; fences
        # deferred block freeing (see Scheduler._free_request_blocks).
        self.last_sched_seq = 0

        self.spec_token_ids: list[int] = []
        self.num_computed_tokens = 0
        self.cache_salt: str | None = cache_salt

        # Multi-modal related
        self.mm_features = mm_features or []

        # Read-only views
        # Prevent directly appending to these lists since
        # they should also be updated simultaneously.
        self.output_token_ids = ConstantList(self._output_token_ids)
        self.all_token_ids = ConstantList(self._all_token_ids)
        # trace_headers
        self.trace_headers = trace_headers

        # True if this request is scheduled as a non-final prefill chunk.
        self.is_prefill_chunk = False

        # The number of NaNs in logits. A value greater than 0
        # indicates that the output is corrupted
        self.num_nans_in_logits = 0

        # The number of times this request has been preempted by the scheduler.
        self.num_preemptions = 0

        self.prefill_stats: PrefillStats | None = PrefillStats()

        self.block_hashes: list[BlockHash] = []
        # Store the block hasher without binding self to avoid creating a
        # reference cycle (Request -> partial -> Request) that prevents
        # immediate garbage collection via reference counting.
        self._block_hasher: Callable[[Request], list[BlockHash]] | None = block_hasher
        self.update_block_hashes()

        self.skip_reading_prefix_cache = self.get_skip_reading_prefix_cache()
        # D 节点收到的 prompt-stream KV 是分段近似 attention 的结果，不能
        # 命中或写入普通全量 prompt 的 APC hash namespace。
        if self.uses_prompt_stream_kv_transfer():
            self.skip_reading_prefix_cache = True

        # Used for streaming
        self.resumable = resumable
        # None entry in the queue means finished.
        self.streaming_queue: deque[StreamingUpdate | None] | None = None

        # If True, request should be aborted immediately after being added to
        # the scheduler so the connector's request_finished hook runs.
        self.abort_immediately = abort_immediately

        # 流式 prefill 的入站事件。对于普通请求为 None。
        # 对于 duplicate request id 的流式事件，这个字段用于传给 scheduler。
        self.prompt_stream_event = prompt_stream_event

        # 流式 prefill 的长期状态。只有 scheduler 中的长期 Request 才应初始化。
        self.prompt_stream: PromptStreamState | None = None

        # scheduler/worker 当前共同使用的物理 token 与 KV segment 视图。
        self.prompt_stream_execution_view: PromptStreamExecutionView | None = None

        # prompt 发生真实变化后，worker 侧 cached request 需要刷新 token/block 状态。
        self.prompt_stream_needs_worker_refresh = False
        # D 请求无 PromptStreamState，必须独立保留 P 已声明的坐标语义。
        self._transferred_padding_range = parse_prompt_stream_padding_range(
            self.kv_transfer_params, self.num_prompt_tokens
        )
        self.prompt_stream_masked_padding = self._transferred_padding_range is not None

    @classmethod
    def from_engine_core_request(
        cls,
        request: EngineCoreRequest,
        block_hasher: Callable[["Request"], list["BlockHash"]] | None,
    ) -> "Request":
        return cls(
            request_id=request.request_id,
            client_index=request.client_index,
            prompt_token_ids=request.prompt_token_ids,
            prompt_embeds=request.prompt_embeds,
            prompt_is_token_ids=request.prompt_is_token_ids,
            mm_features=request.mm_features,
            sampling_params=request.sampling_params,
            pooling_params=request.pooling_params,
            arrival_time=request.arrival_time,
            lora_request=request.lora_request,
            cache_salt=request.cache_salt,
            priority=request.priority,
            trace_headers=request.trace_headers,
            block_hasher=block_hasher,
            resumable=request.resumable,
            reasoning_ended=request.reasoning_ended,
            reasoning_parser_kwargs=request.reasoning_parser_kwargs,
            abort_immediately=request.abort_immediately,
            prompt_stream_event=request.prompt_stream_event,
        )

    def append_output_token_ids(
        self,
        token_ids: int | list[int],
    ) -> None:
        if isinstance(token_ids, int):
            self._output_token_ids.append(token_ids)
            self._all_token_ids.append(token_ids)
        else:
            self._output_token_ids.extend(token_ids)
            self._all_token_ids.extend(token_ids)

        self.update_block_hashes()

    def update_block_hashes(self) -> None:
        """Compute block hashes for any new full blocks and append them."""
        if self._block_hasher is not None:
            self.block_hashes.extend(self._block_hasher(self))

    @property
    def use_structured_output(self) -> bool:
        return self.structured_output_request is not None

    @property
    def num_tokens(self) -> int:
        return len(self._all_token_ids)

    @property
    def num_tokens_with_spec(self) -> int:
        return len(self._all_token_ids) + len(self.spec_token_ids)

    @property
    def num_output_tokens(self) -> int:
        return len(self._output_token_ids)

    @property
    def num_encoder_inputs(self) -> int:
        return len(self.mm_features)

    @property
    def has_encoder_inputs(self) -> bool:
        return self.num_encoder_inputs > 0

    def get_skip_reading_prefix_cache(self) -> bool:
        if (
            self.sampling_params is not None
            and self.sampling_params.skip_reading_prefix_cache is not None
        ):
            return self.sampling_params.skip_reading_prefix_cache
        elif (
            self.pooling_params is not None
            and self.pooling_params.skip_reading_prefix_cache is not None
        ):
            return self.pooling_params.skip_reading_prefix_cache
        return False

    def is_finished(self) -> bool:
        return RequestStatus.is_finished(self.status)

    def get_finished_reason(self) -> FinishReason | None:
        return RequestStatus.get_finished_reason(self.status)

    def get_num_encoder_embeds(self, input_id: int) -> int:
        assert input_id < len(self.mm_features)
        return self.mm_features[input_id].mm_position.get_num_embeds()

    def record_event(
        self,
        event_type: EngineCoreEventType,
        timestamp: float | None = None,
    ) -> None:
        self.events.append(EngineCoreEvent.new_event(event_type, timestamp))

    def take_events(self) -> list[EngineCoreEvent] | None:
        if not self.events:
            return None
        events, self.events = self.events, []
        return events

    def take_prefill_stats(self) -> PrefillStats | None:
        if self.prefill_stats is None:
            return None
        prefill_stats = self.prefill_stats
        self.prefill_stats = None
        return prefill_stats

    def __lt__(self, other: "Request") -> bool:
        """
        Compare two requests based on priority, arrival time, and request ID.
        Used in priority scheduling.
        """
        if self.priority != other.priority:
            return self.priority < other.priority
        if self.arrival_time != other.arrival_time:
            return self.arrival_time < other.arrival_time
        if self.request_id != other.request_id:
            return self.request_id < other.request_id
        return id(self) < id(other)

    def is_prompt_streaming(self) -> bool:
        """
        判断是不是流式 prompt
        """
        return self.prompt_stream is not None

    def _create_prompt_stream_update_admission(
        self,
        target_token_ids: tuple[int, ...],
        required_append_epoch: int,
        policy_prefill_tokens: int,
    ) -> int:
        """创建稳定 Update admission，并更新 policy 层累计值。"""
        prompt_stream = self._ensure_prompt_stream_state()
        admission_id = prompt_stream.next_update_admission_id
        prompt_stream.next_update_admission_id += 1
        prompt_stream.update_admissions[admission_id] = UpdateAdmissionRecord(
            admission_id=admission_id,
            target_token_ids=target_token_ids,
            required_append_epoch=required_append_epoch,
            policy_prefill_tokens=policy_prefill_tokens,
        )
        prompt_stream.update_works_admitted += 1
        prompt_stream.total_admitted_update_prefill_tokens += policy_prefill_tokens
        return admission_id

    def _get_prompt_stream_update_admission(
        self,
        admission_id: int,
    ) -> UpdateAdmissionRecord:
        """返回 admission 记录；缺失 ID 表示内部生命周期已经损坏。"""
        assert self.prompt_stream is not None
        record = self.prompt_stream.update_admissions.get(admission_id)
        if record is None:
            raise RuntimeError(
                "Prompt stream UPDATE work references an unknown admission: "
                f"request_id={self.request_id}, admission_id={admission_id}."
            )
        return record

    def _mark_prompt_stream_update_admission_active(
        self,
        admission_id: int | None,
    ) -> None:
        """在对应 work 真正进入物理执行时推进 admission 状态。"""
        if admission_id is None:
            return
        record = self._get_prompt_stream_update_admission(admission_id)
        if record.status == UpdateAdmissionStatus.PENDING:
            record.status = UpdateAdmissionStatus.ACTIVE
            return
        if record.status != UpdateAdmissionStatus.ACTIVE:
            raise RuntimeError(
                "Prompt stream UPDATE admission cannot become active from "
                f"status={record.status.value}: request_id={self.request_id}, "
                f"admission_id={admission_id}."
            )

    def _complete_prompt_stream_update_admission(
        self,
        admission_id: int | None,
    ) -> None:
        """按稳定 ID 结算 policy admission；recovery-only work 不结算。"""
        if admission_id is None:
            return
        record = self._get_prompt_stream_update_admission(admission_id)
        if record.status == UpdateAdmissionStatus.COMPLETED:
            return
        if record.status == UpdateAdmissionStatus.CANCELLED_BY_RECOVERY:
            raise RuntimeError(
                "Cancelled prompt stream UPDATE admission completed unexpectedly: "
                f"request_id={self.request_id}, admission_id={admission_id}."
            )
        record.status = UpdateAdmissionStatus.COMPLETED
        assert self.prompt_stream is not None
        self.prompt_stream.update_works_completed += 1

    def _start_prompt_stream_update_execution_attempt(
        self,
        work_item: PromptStreamWorkItem,
    ) -> None:
        """为首次 active 的 Update work 分配物理 attempt ID。"""
        if work_item.execution_attempt_id is not None:
            return
        assert self.prompt_stream is not None
        work_item.execution_attempt_id = (
            self.prompt_stream.next_update_execution_attempt_id
        )
        self.prompt_stream.next_update_execution_attempt_id += 1
        self.prompt_stream.update_execution_attempts_started += 1
        self._mark_prompt_stream_update_admission_active(
            work_item.update_admission_id
        )

    def _complete_prompt_stream_update_work(
        self,
        work_item: PromptStreamWorkItem,
    ) -> None:
        """同时结算一次物理 attempt 与其可选 policy admission。"""
        self._start_prompt_stream_update_execution_attempt(work_item)
        assert self.prompt_stream is not None
        self.prompt_stream.update_execution_attempts_completed += 1
        self._complete_prompt_stream_update_admission(
            work_item.update_admission_id
        )

    def record_prompt_stream_allocation_failure(
        self,
        *,
        is_running: bool,
    ) -> None:
        """记录一次 Prompt Stream KV slot 分配失败。

        普通请求调用该接口是 no-op，避免 Scheduler 主路径读取内部状态。
        """
        if self.prompt_stream is None:
            return
        if is_running:
            self.prompt_stream.prompt_stream_allocation_failures_running += 1
        else:
            self.prompt_stream.prompt_stream_allocation_failures_waiting += 1

    def build_prompt_stream_stats(
        self,
    ) -> dict[str, bool | int | str | None] | None:
        """构造当前 Prompt Stream 请求的独立可序列化统计快照。

        该接口只读取请求生命周期累计值，不会清空或转移所有权，因此
        Scheduler 日志和最终网络响应可以安全地重复读取。返回字典与
        Request 内部状态没有共享的可变引用，修改它不会回写累计器。

        Returns:
            Prompt Stream 已初始化时返回只读统计；普通请求返回 None。
            未启用 coalescer 或没有有效 Update 事件时，eager/saved/extra
            均为 None，避免把“不适用”误写为零收益。
        """
        prompt_stream = self.prompt_stream
        if prompt_stream is None:
            return None

        coalescing_state = prompt_stream.update_coalescing
        coalescing_enabled = coalescing_state is not None
        update_coalescing_config_json: str | None = None
        if coalescing_state is not None:
            config = coalescing_state.config
            update_coalescing_config_json = json.dumps(
                {
                    config_field.name: getattr(config, config_field.name)
                    for config_field in fields(config)
                },
                sort_keys=True,
                separators=(",", ":"),
            )
        has_update_events = prompt_stream.update_events_accepted > 0
        if coalescing_enabled and has_update_events:
            eager_tokens: int | None = (
                prompt_stream.total_eager_update_prefill_tokens
            )
            net_tokens_avoided = (
                prompt_stream.total_eager_update_prefill_tokens
                - prompt_stream.total_admitted_update_prefill_tokens
            )
            saved_tokens: int | None = max(0, net_tokens_avoided)
            extra_tokens: int | None = max(0, -net_tokens_avoided)
        else:
            eager_tokens = None
            saved_tokens = None
            extra_tokens = None

        cancelled_works = sum(
            record.status == UpdateAdmissionStatus.CANCELLED_BY_RECOVERY
            for record in prompt_stream.update_admissions.values()
        )
        outstanding_works = sum(
            record.status
            in (UpdateAdmissionStatus.PENDING, UpdateAdmissionStatus.ACTIVE)
            for record in prompt_stream.update_admissions.values()
        )

        stats = PromptStreamTokenStats(
            metrics_schema_version=2,
            coalescing_enabled=coalescing_enabled,
            update_coalescing_config_json=update_coalescing_config_json,
            update_events_accepted=prompt_stream.update_events_accepted,
            update_policy_works_admitted=prompt_stream.update_works_admitted,
            update_policy_works_completed=prompt_stream.update_works_completed,
            update_policy_works_cancelled_by_recovery=cancelled_works,
            update_policy_works_outstanding=outstanding_works,
            update_works_admitted=prompt_stream.update_works_admitted,
            update_works_completed=prompt_stream.update_works_completed,
            eager_update_prefill_tokens=eager_tokens,
            policy_admitted_update_prefill_tokens=(
                prompt_stream.total_admitted_update_prefill_tokens
            ),
            policy_saved_update_prefill_tokens=saved_tokens,
            policy_extra_update_prefill_tokens=extra_tokens,
            admitted_update_prefill_tokens=(
                prompt_stream.total_admitted_update_prefill_tokens
            ),
            scheduled_update_prefill_tokens=(
                prompt_stream.total_scheduled_update_prefill_tokens
            ),
            saved_update_prefill_tokens=saved_tokens,
            extra_update_prefill_tokens=extra_tokens,
            update_tokens_invalidated=prompt_stream.total_tokens_invalidated,
            update_tokens_invalidated_by_snapshot=(
                prompt_stream.total_tokens_invalidated
            ),
            update_execution_attempts_started=(
                prompt_stream.update_execution_attempts_started
            ),
            update_execution_attempts_completed=(
                prompt_stream.update_execution_attempts_completed
            ),
            update_execution_attempts_preempted=(
                prompt_stream.update_execution_attempts_preempted
            ),
            update_recovery_works_created=(
                prompt_stream.update_recovery_works_created
            ),
            update_tokens_scheduled_total=(
                prompt_stream.total_scheduled_update_prefill_tokens
            ),
            update_tokens_scheduled_for_recovery=(
                prompt_stream.total_recovery_update_prefill_tokens
            ),
            prefix_append_tokens_scheduled_for_recovery=(
                prompt_stream.total_recovery_prefix_append_prefill_tokens
            ),
            update_tokens_discarded_by_preemption=(
                prompt_stream.total_update_tokens_discarded_by_preemption
            ),
            prefix_append_tokens_discarded_by_preemption=(
                prompt_stream.total_prefix_append_tokens_discarded_by_preemption
            ),
            question_tokens_discarded_by_preemption=(
                prompt_stream.total_question_tokens_discarded_by_preemption
            ),
            prompt_stream_running_preemptions=(
                prompt_stream.prompt_stream_running_preemptions
            ),
            prompt_stream_dormant_evictions=(
                prompt_stream.prompt_stream_dormant_evictions
            ),
            prompt_stream_retained_waiting_evictions=(
                prompt_stream.prompt_stream_retained_waiting_evictions
            ),
            prompt_stream_prefix_cache_resets=(
                prompt_stream.prompt_stream_prefix_cache_resets
            ),
            prompt_stream_allocation_failures_running=(
                prompt_stream.prompt_stream_allocation_failures_running
            ),
            prompt_stream_allocation_failures_waiting=(
                prompt_stream.prompt_stream_allocation_failures_waiting
            ),
        )
        return stats.to_dict()

    def enable_prompt_stream_update_coalescing(
        self,
        config: "PromptStreamUpdateCoalescingConfig",
        now: float | None = None,
    ) -> None:
        """为当前长期 Request 启用单 Update 合并窗口。

        Scheduler 只在 P 节点显式传入启动级配置时调用该接口。默认路径不
        调用，因此仍保持“一条 Update 立即生成一个 work”的既有语义。

        Args:
            config: 已完成配置校验的 Update 合并配置。
            now: EngineCore 单调时钟；省略时使用 ``time.monotonic()``。

        Raises:
            ValueError: 当前请求已经使用不同配置启用，或已有多个真实 source。
        """
        prompt_stream = self._ensure_prompt_stream_state()
        current = prompt_stream.update_coalescing
        if current is not None:
            if current.config != config:
                raise ValueError(
                    "Prompt stream coalescing config changed for "
                    f"request_id={self.request_id}."
                )
            return

        has_real_source = any(
            source_id != "__prompt_stream_start__"
            for source_id in prompt_stream.source_order
        )
        if (
            prompt_stream.active_work_item is not None
            or prompt_stream.pending_work_items
            or has_real_source
            or prompt_stream.append_token_ids
            or prompt_stream.update_committed_token_ids
            or prompt_stream.finished
        ):
            raise ValueError(
                "Prompt stream Update coalescing must be enabled before any "
                "real source work is accepted or scheduled: "
                f"request_id={self.request_id}."
            )

        gate = AppendUpdateGateState()
        # start 使用的空 APPEND 事件只是 Request 类型标记，不占用真实 source。
        for source_id in prompt_stream.source_order:
            if source_id == "__prompt_stream_start__":
                continue
            source = prompt_stream.sources[source_id]
            if source.source_type == PromptSourceType.APPEND:
                if gate.append_source_id is not None:
                    raise ValueError(
                        "Prompt stream Update coalescing supports exactly one "
                        f"Append source: request_id={self.request_id}."
                    )
                gate.append_source_id = source_id
                gate.append_source_finished = source.finished
            else:
                if gate.update_source_id is not None:
                    raise ValueError(
                        "Prompt stream Update coalescing supports exactly one "
                        f"Update source: request_id={self.request_id}."
                    )
                gate.update_source_id = source_id
                gate.update_source_finished = source.finished

        start_time = time.monotonic() if now is None else now
        prompt_stream.append_update_gate = gate
        prompt_stream.update_coalescing = UpdateCoalescingState(
            config=config,
            request_start_time=start_time,
            last_admitted_token_ids=tuple(
                prompt_stream.update_committed_token_ids
            ),
            last_admitted_append_epoch=gate.update_committed_append_epoch,
            stats_last_admitted_token_ids=tuple(
                prompt_stream.update_committed_token_ids
            ),
            stats_last_admitted_append_epoch=(
                gate.update_committed_append_epoch
            ),
        )

    @property
    def prompt_stream_update_coalescing_state(
        self,
    ) -> UpdateCoalescingState | None:
        """返回可供 Scheduler controller 读取的单窗口状态。"""
        if self.prompt_stream is None:
            return None
        return self.prompt_stream.update_coalescing

    def is_prompt_stream_update_coalescing_enabled(self) -> bool:
        """判断当前 Request 是否启用了实验性 Update 合并。"""
        return self.prompt_stream_update_coalescing_state is not None

    def is_prompt_stream_update_admission_allowed(self) -> bool:
        """判断 Append gate 是否允许把当前窗口物化为 Update work。"""
        state = self.prompt_stream_update_coalescing_state
        if state is None:
            return True
        assert self.prompt_stream is not None
        gate = self.prompt_stream.append_update_gate
        assert gate is not None
        if state.config.append_gate == "none":
            return True
        if gate.append_source_finished:
            return True
        return bool(
            state.config.append_gate
            == "hold_until_append_or_update_finished"
            and gate.update_source_finished
        )

    def on_prompt_stream_update_snapshot(
        self,
        previous_token_ids: list[int],
        new_token_ids: list[int],
        now: float | None = None,
    ) -> None:
        """把合法 Update 快照合并进唯一未 admission 窗口。

        该接口只维护逻辑快照、deadline 和 EWMA，不修改 UPDATE KV，也不
        创建 work。Scheduler 随后调用 ``flush_prompt_stream_update`` 完成
        policy 与 gate 的二阶段判定。

        Args:
            previous_token_ids: 同一 source 的上一条完整原始快照。
            new_token_ids: 本次已经通过 version/sequence 校验的完整快照。
            now: EngineCore 单调时钟。
        """
        state = self.prompt_stream_update_coalescing_state
        if state is None:
            return
        assert self.prompt_stream is not None
        gate = self.prompt_stream.append_update_gate
        assert gate is not None
        timestamp = time.monotonic() if now is None else now
        config = state.config

        current_append_epoch = gate.append_epoch
        same_observed_append_context = (
            state.last_observed_update_append_epoch is None
            or state.last_observed_update_append_epoch == current_append_epoch
        )
        raw_lcp = (
            self._get_common_prefix_len(previous_token_ids, new_token_ids)
            if same_observed_append_context
            else 0
        )
        eager_tokens = len(new_token_ids) - raw_lcp
        # 合并收益属于 policy 层，必须相对跨 recovery 不回退的
        # admission 链尾计算。物理 KV 丢失不是新 Update 带来的收益。
        base_token_ids = list(state.stats_last_admitted_token_ids)
        same_admitted_append_context = (
            not base_token_ids
            or state.stats_last_admitted_append_epoch == current_append_epoch
        )
        base_lcp = (
            self._get_common_prefix_len(base_token_ids, new_token_ids)
            if same_admitted_append_context
            else 0
        )
        merged_tokens = len(new_token_ids) - base_lcp
        merge_gain = max(0, eager_tokens - merged_tokens)

        # 事件通过协议校验且内容真实变化后，才进入 eager 反事实累计。
        self.prompt_stream.update_events_accepted += 1
        self.prompt_stream.total_eager_update_prefill_tokens += eager_tokens

        if state.last_observed_update_time is not None:
            interval = max(
                0.0,
                timestamp - state.last_observed_update_time,
            )
            alpha = config.interval_ewma_alpha
            state.interval_ewma = (
                interval
                if state.interval_ewma is None
                else (1.0 - alpha) * state.interval_ewma + alpha * interval
            )
            state.interval_observation_count += 1
        state.last_observed_update_time = timestamp
        state.last_observed_update_append_epoch = current_append_epoch

        gain_alpha = config.gain_ewma_alpha
        state.merge_gain_ewma = (
            float(merge_gain)
            if state.merge_gain_ewma is None
            else (1.0 - gain_alpha) * state.merge_gain_ewma
            + gain_alpha * merge_gain
        )
        state.gain_observation_count += 1

        if state.first_pending_time is None:
            state.first_pending_time = timestamp
            if config.update_policy in (
                "fixed_window",
                "quiet_debounce",
                "adaptive",
            ):
                state.hard_deadline = (
                    timestamp + config.hard_deadline_ms / 1000.0
                )
            if (
                config.update_policy
                in ("fixed_window", "quiet_debounce", "adaptive")
                and config.ttft_deadline_ms is not None
            ):
                state.ttft_deadline = (
                    state.request_start_time
                    + config.ttft_deadline_ms / 1000.0
                )

        state.latest_pending_token_ids = tuple(new_token_ids)
        state.last_update_time = timestamp
        state.pending_event_count += 1
        state.hypothetical_eager_tokens += eager_tokens
        state.generation += 1

        # sticky flush_ready 一旦置位，新快照只能覆盖目标，不能重新延后执行。
        if state.flush_ready:
            return
        if config.update_policy == "fixed_window":
            assert state.first_pending_time is not None
            state.soft_deadline = (
                state.first_pending_time + config.soft_window_ms / 1000.0
            )
        elif config.update_policy == "quiet_debounce":
            state.soft_deadline = timestamp + config.soft_window_ms / 1000.0
        elif config.update_policy == "adaptive":
            wait_s = self._get_prompt_stream_adaptive_wait_s(prefill_queue_load=0.0)
            state.soft_deadline = timestamp + wait_s
            state.adaptive_deadline_needs_refresh = True
        else:
            state.soft_deadline = None

    def on_prompt_stream_source_finished(
        self,
        source_type: PromptSourceType,
        now: float | None = None,
    ) -> None:
        """更新 source barrier；Append finished 只开 gate，不强制 flush。"""
        del now  # 接口与 controller 时钟保持一致，当前状态更新本身不计时。
        state = self.prompt_stream_update_coalescing_state
        if state is None:
            return
        assert self.prompt_stream is not None
        gate = self.prompt_stream.append_update_gate
        assert gate is not None
        if source_type == PromptSourceType.APPEND:
            gate.append_source_finished = True
            gate.final_append_epoch = gate.append_epoch
            gate.phase = AppendGatePhase.APPEND_FINISHED
        else:
            gate.update_source_finished = True
            if (
                state.config.append_gate
                == "hold_until_append_or_update_finished"
                and not gate.append_source_finished
            ):
                gate.phase = AppendGatePhase.EARLY_RELEASE

    def get_prompt_stream_update_deadline(self) -> float | None:
        """返回当前窗口下一次需要 EngineCore 唤醒的绝对 monotonic 时间。"""
        state = self.prompt_stream_update_coalescing_state
        if (
            state is None
            or state.latest_pending_token_ids is None
            or state.flush_ready
        ):
            return None
        deadlines = (
            deadline
            for deadline in (
                state.soft_deadline,
                state.hard_deadline,
                state.ttft_deadline,
            )
            if deadline is not None
        )
        return min(deadlines, default=None)

    def _get_prompt_stream_adaptive_wait_s(
        self,
        prefill_queue_load: float,
    ) -> float:
        """按 EWMA 收益模型计算自适应 soft window，单位为秒。"""
        state = self.prompt_stream_update_coalescing_state
        if state is None:
            return 0.0
        config = state.config
        max_wait_s = config.soft_window_ms / 1000.0
        prior_weight = config.cold_start_prior_weight

        observed_rate = (
            None
            if state.interval_ewma is None
            else 1.0 / max(state.interval_ewma, 1e-9)
        )
        observed_gain = state.merge_gain_ewma

        def blend(
            observed: float | None,
            observation_count: int,
            prior: float | None,
        ) -> float | None:
            weighted_sum = 0.0
            total_weight = 0.0
            if observed is not None and observation_count > 0:
                weighted_sum += observation_count * observed
                total_weight += observation_count
            # prior=None 表示没有先验，不能稀释真实观测。
            if prior is not None and prior_weight > 0:
                weighted_sum += prior_weight * prior
                total_weight += prior_weight
            if total_weight <= 0:
                return None
            return weighted_sum / total_weight

        arrival_rate = blend(
            observed_rate,
            state.interval_observation_count,
            config.arrival_rate_prior_per_s,
        )
        merge_gain = blend(
            observed_gain,
            state.gain_observation_count,
            config.merge_gain_prior_tokens,
        )
        if arrival_rate is None or merge_gain is None or arrival_rate <= 0:
            return 0.0

        load_price = 1.0 + config.load_price_scale * max(0.0, prefill_queue_load)
        delay_price = config.delay_price
        marginal_value = load_price * merge_gain * arrival_rate
        if marginal_value <= delay_price or marginal_value <= 0:
            return 0.0
        if delay_price == 0:
            return max_wait_s
        wait_s = math.log(marginal_value / delay_price) / arrival_rate
        return min(max(wait_s, 0.0), max_wait_s)

    def is_prompt_stream_update_flush_due(
        self,
        now: float,
        prefill_queue_load: float = 0.0,
    ) -> bool:
        """判断当前 Update policy 是否已经要求 flush。"""
        state = self.prompt_stream_update_coalescing_state
        if state is None or state.latest_pending_token_ids is None:
            return False
        if state.flush_ready:
            return True
        assert self.prompt_stream is not None
        gate = self.prompt_stream.append_update_gate
        assert gate is not None
        policy = state.config.update_policy
        if policy == "immediate":
            return True
        if policy == "full_merge":
            return gate.update_source_finished or self.prompt_stream.finished
        if (
            policy == "adaptive"
            and state.last_update_time is not None
            and state.adaptive_deadline_needs_refresh
        ):
            adaptive_deadline = state.last_update_time + (
                self._get_prompt_stream_adaptive_wait_s(prefill_queue_load)
            )
            state.soft_deadline = adaptive_deadline
            state.adaptive_deadline_needs_refresh = False
        return any(
            deadline is not None and now >= deadline
            for deadline in (
                state.soft_deadline,
                state.hard_deadline,
                state.ttft_deadline,
            )
        ) or gate.update_source_finished or self.prompt_stream.finished

    def flush_prompt_stream_update(
        self,
        now: float,
        reason: str,
        *,
        force: bool = False,
        prefill_queue_load: float = 0.0,
    ) -> bool:
        """在 policy 与 gate 都允许时把唯一窗口冻结为 Update work。

        Returns:
            成功创建 work 返回 True；继续合并或被 gate 阻挡返回 False。
        """
        state = self.prompt_stream_update_coalescing_state
        if state is None or state.latest_pending_token_ids is None:
            self.enqueue_question_if_ready()
            return False
        should_flush = force or self.is_prompt_stream_update_flush_due(
            now,
            prefill_queue_load,
        )
        if not should_flush:
            return False
        if not self.is_prompt_stream_update_admission_allowed():
            # ready 是黏性的；后续 Update 仅覆盖 target，不撤销到期事实。
            if not state.flush_ready:
                state.flush_ready = True
                state.flush_ready_reason = reason
                state.generation += 1
            return False

        assert self.prompt_stream is not None
        gate = self.prompt_stream.append_update_gate
        assert gate is not None
        target = state.latest_pending_token_ids
        base = state.last_admitted_token_ids
        required_append_epoch = gate.append_epoch
        same_append_context = (
            not base
            or state.last_admitted_append_epoch == required_append_epoch
        )
        common_prefix_len = (
            self._get_common_prefix_len(list(base), list(target))
            if same_append_context
            else 0
        )

        # admission 统计使用独立的逻辑链尾，不能跟随 preemption 将物理
        # execution base 回退到空快照，否则同一目标会被重复计数。
        stats_base = state.stats_last_admitted_token_ids
        same_stats_append_context = (
            not stats_base
            or state.stats_last_admitted_append_epoch == required_append_epoch
        )
        stats_common_prefix_len = (
            self._get_common_prefix_len(list(stats_base), list(target))
            if same_stats_append_context
            else 0
        )
        is_new_logical_admission = (
            target != stats_base
            or (
                bool(target)
                and state.stats_last_admitted_append_epoch
                != required_append_epoch
            )
        )
        admission_id: int | None = None
        if is_new_logical_admission:
            admitted_tokens = len(target) - stats_common_prefix_len
            admission_id = self._create_prompt_stream_update_admission(
                target,
                required_append_epoch,
                admitted_tokens,
            )
            state.stats_last_admitted_token_ids = target
            state.stats_last_admitted_append_epoch = required_append_epoch

        self.prompt_stream.pending_work_items.append(
            PromptStreamWorkItem(
                phase=PromptStreamExecutionPhase.UPDATE,
                prefix_append_logical_len=(
                    self.get_prompt_stream_prefix_append_len()
                ),
                update_token_ids=target,
                execution_base_token_ids=base,
                # 空基准不复用任何 UPDATE KV，因此不绑定 Append epoch。
                # 这也覆盖 empty snapshot 经 recovery 后再变为非空的路径。
                execution_base_append_epoch=(
                    state.last_admitted_append_epoch if base else None
                ),
                required_append_epoch=required_append_epoch,
                update_common_prefix_len=common_prefix_len,
                update_admission_id=admission_id,
            )
        )
        state.last_admitted_token_ids = target
        state.last_admitted_append_epoch = required_append_epoch
        state.latest_pending_token_ids = None
        state.first_pending_time = None
        state.last_update_time = None
        state.soft_deadline = None
        state.hard_deadline = None
        state.ttft_deadline = None
        state.adaptive_deadline_needs_refresh = False
        state.flush_ready = False
        state.flush_ready_reason = None
        state.pending_event_count = 0
        state.hypothetical_eager_tokens = 0
        state.generation += 1
        self.enqueue_question_if_ready()
        return True

    def enqueue_question_if_ready(self) -> bool:
        """把最终用户问题作为唯一显式 QUESTION work 追加到 FIFO 尾部。"""
        if self.prompt_stream is None:
            return False
        prompt_stream = self.prompt_stream
        if (
            not prompt_stream.finished
            or prompt_stream.question_work_enqueued
            or not prompt_stream.suffix_token_ids
        ):
            return False
        state = prompt_stream.update_coalescing
        gate = prompt_stream.append_update_gate
        if state is not None:
            assert gate is not None
            if (
                not gate.append_source_finished
                or not gate.update_source_finished
                or state.latest_pending_token_ids is not None
                or state.flush_ready
            ):
                return False
            update_source = (
                prompt_stream.sources.get(gate.update_source_id)
                if gate.update_source_id is not None
                else None
            )
            latest_update = tuple(update_source.token_ids) if update_source else ()
            if state.last_admitted_token_ids != latest_update:
                return False
            required_append_epoch = gate.append_epoch
        else:
            required_append_epoch = 0

        prompt_stream.pending_work_items.append(
            PromptStreamWorkItem(
                phase=PromptStreamExecutionPhase.QUESTION,
                prefix_append_logical_len=(
                    self.get_prompt_stream_prefix_append_len()
                ),
                required_append_epoch=required_append_epoch,
                update_token_ids=tuple(
                    self._get_prompt_stream_segment_token_ids(
                        PromptSourceType.UPDATE
                    )
                ),
                question_token_ids=tuple(prompt_stream.suffix_token_ids),
            )
        )
        prompt_stream.question_work_enqueued = True
        return True

    def should_make_prompt_stream_dormant(self) -> bool:
        """判断请求是否只在等待事件、deadline 或 Append gate。"""
        if self.prompt_stream is None or self.prompt_stream.update_coalescing is None:
            return False
        prompt_stream = self.prompt_stream
        if not prompt_stream.prefix_append_bootstrap_completed:
            return False
        if prompt_stream.active_work_item is not None:
            return False
        if prompt_stream.pending_work_items:
            return False
        return not self.can_decode_from_prompt_stream()

    def uses_prompt_stream_kv_transfer(self) -> bool:
        """判断当前 D 请求是否加载 prompt-stream 的精确 block manifest。

        P 节点上的长期 Request 通过 prompt_stream 保存状态；D 节点收到的
        仍是普通 completion Request，因此必须依靠 transfer 参数识别。
        """
        return bool(
            self.kv_transfer_params
            and (
                self.kv_transfer_params.get("prompt_stream_exact_transfer")
                # 即使重新本地 prefill，带空洞的语义也不能复用普通 APC。
                or "prompt_stream_masked_padding" in self.kv_transfer_params
            )
        )

    def get_prompt_stream_padding_range(self) -> tuple[int, int] | None:
        """返回本轮 worker 使用的物理空洞；普通请求及旧近似模式返回 None。"""
        if not self.prompt_stream_masked_padding:
            return None
        if self.prompt_stream_execution_view is not None:
            return self.prompt_stream_execution_view.padding_range
        return self._transferred_padding_range

    def is_prompt_stream_finished(self) -> bool:
        """判断 prompt 输入是否结束。

        普通非流式请求视为 finished，因为它的 prompt 在入队时就是完整的。
        """
        return self.prompt_stream is None or self.prompt_stream.finished

    def can_decode_from_prompt_stream(self) -> bool:
        """判断当前请求是否已经越过流式 prompt 的 decode barrier。

        普通请求没有流式 prompt barrier，可以直接返回 True。流式 prompt
        请求必须同时满足两个条件：

        - 请求级 prompt 输入已经结束。
        - 当前 prompt token 已经全部 prefill 完成。

        Returns:
            如果 scheduler 可以进入 decode 阶段，返回 True。
        """
        if self.prompt_stream is None:
            return True
        if not self.prompt_stream.finished:
            return False
        if self.prompt_stream.block_size is None:
            return False

        layout = self.get_prompt_stream_layout()
        outstanding_update_admission = any(
            record.status
            in (UpdateAdmissionStatus.PENDING, UpdateAdmissionStatus.ACTIVE)
            for record in self.prompt_stream.update_admissions.values()
        )
        latest_update_token_ids = (
            self._get_latest_prompt_stream_update_snapshot()
        )
        latest_update_materialized = (
            latest_update_token_ids is None
            or tuple(self.prompt_stream.update_committed_token_ids)
            == latest_update_token_ids
        )
        gate = self.prompt_stream.append_update_gate
        final_append_materialized = (
            gate is None
            or gate.final_append_epoch is None
            or gate.committed_append_epoch == gate.final_append_epoch
        )
        return (
            self.prompt_stream.prefix_append_bootstrap_completed
            and self.prompt_stream.active_work_item is None
            and not self.prompt_stream.pending_work_items
            and self.prompt_stream.question_work_completed
            and not outstanding_update_admission
            and latest_update_materialized
            and final_append_materialized
            and (
                self.prompt_stream.update_coalescing is None
                or (
                    self.prompt_stream.update_coalescing.latest_pending_token_ids
                    is None
                    and not self.prompt_stream.update_coalescing.flush_ready
                )
            )
            and self.prompt_stream.prefix_append_committed_logical_len
            >= layout.prefix_len + layout.append_len
            and self.prompt_stream.prefix_append_num_computed_tokens
            >= layout.prefix_len + layout.append_len + layout.prefix_append_padding_len
            and self.prompt_stream.update_num_computed_tokens >= layout.update_len
            and self.prompt_stream.question_num_computed_tokens >= layout.suffix_len
            and self.num_computed_tokens >= self.num_prompt_tokens
        )

    def should_discard_prompt_stream_sampled_tokens(self) -> bool:
        """判断本轮采样结果是否只是中间 prefill 阶段的临时 logits。

        REST 事件可以在 GPU 完成前连续到达。即使当前物理执行视图已经完成，
        后面仍可能有 Update 或 Question 阶段等待计算；此时 worker 为简化
        chunked prefill 而采样出的 token 必须在 scheduler 侧丢弃。
        """
        return (
            self.prompt_stream is not None and not self.can_decode_from_prompt_stream()
        )

    def get_prompt_stream_revision(self) -> int:
        """返回当前流式 prompt 的版本号。

        普通请求没有流式 prompt 状态，统一视为 revision 0。
        scheduler 后续可以把这个值写入调度快照，避免旧 prompt 版本的
        执行结果被误用于新 prompt。
        """
        if self.prompt_stream is None:
            return 0
        return self.prompt_stream.revision

    def take_prompt_stream_update_kv_truncation(self) -> int | None:
        """取出本轮选中 Update work 需要的 UPDATE KV 截断边界。

        Update 事件到达时只把不可变快照追加到请求内队列，不能立即截断
        UPDATE blocks，否则仍在执行的上一代 Update work 会失去自己的
        KV 前缀。scheduler 在准备好下一轮执行视图之后调用本方法，并在
        分配新 slots 之前按返回的 token 边界完成实际截断。

        Returns:
            UPDATE segment 内应保留的 token 数；没有待处理截断时返回 None。

        Side Effects:
            成功读取后清空暂存边界，保证同一 work 的截断只执行一次。
        """
        if self.prompt_stream is None:
            return None
        keep_tokens = self.prompt_stream.pending_update_kv_truncation_keep_tokens
        self.prompt_stream.pending_update_kv_truncation_keep_tokens = None
        return keep_tokens

    def apply_prompt_stream_event(
        self,
        event: PromptStreamEvent,
    ) -> PromptStreamMutation:
        """应用一次流式 prompt 事件，并返回对调度层有用的变化摘要。

        这个方法是 scheduler 面向流式 prompt 的主入口。scheduler 不需要
        理解 append、update、source finished 的内部语义；它只需根据返回的
        `PromptStreamMutation` 更新请求状态，并在 work 被真正选中时处理
        对应的 segment KV 变化。

        Args:
            event: 本次入站的流式 prompt 事件。

        Returns:
            描述本次事件是否改变 prompt、公共前缀长度、失效 token 数和
            prompt revision 的 mutation。
        """
        if event.request_id != self.request_id:
            raise ValueError(
                f"Prompt stream event request_id={event.request_id} does not "
                f"match request_id={self.request_id}."
            )

        self._discard_unfinished_prompt_stream_outputs()

        if event.event_type == PromptStreamEventType.APPEND:
            return self._apply_prompt_stream_append(event)

        if event.event_type == PromptStreamEventType.UPDATE:
            return self._apply_prompt_stream_update(event)

        if event.event_type == PromptStreamEventType.SOURCE_FINISHED:
            return self._mark_prompt_source_finished(event)

        if event.event_type == PromptStreamEventType.PROMPT_FINISHED:
            return self._mark_prompt_stream_finished(event)

        raise ValueError(f"Unsupported prompt stream event type: {event.event_type}")

    def _discard_unfinished_prompt_stream_outputs(self) -> None:
        """丢弃流式 prompt 结束前被 prefill 顺手采样出的 output token。

        vLLM 的 generate 路径会在 prefill 当前 prompt 最后一个 token 时采样
        第一个 output token。对于流式 prompt，请求级 `PROMPT_FINISHED` 到达
        前这些 token 只是过早产生的临时结果，不能让它们阻止后续 append /
        update / finish 继续修改 prompt。
        """
        if self.prompt_stream is None or self.can_decode_from_prompt_stream():
            return
        if not self._output_token_ids:
            return

        self._output_token_ids.clear()
        self._all_token_ids = self._current_prompt_token_ids()
        self.output_token_ids = ConstantList(self._output_token_ids)
        self.all_token_ids = ConstantList(self._all_token_ids)
        self.spec_token_ids.clear()
        self.num_output_placeholders = 0
        self.async_tokens_to_discard = 0
        self.num_computed_tokens = min(
            self.num_computed_tokens,
            self.num_prompt_tokens,
        )
        self.block_hashes = []
        self.update_block_hashes()

    def _ensure_prompt_stream_state(self) -> PromptStreamState:
        """确保当前请求已经初始化流式 prompt 状态。"""
        if self.prompt_stream is None:
            prefix_token_ids = self._current_prompt_token_ids()
            self.prompt_stream = PromptStreamState(
                prefix_token_ids=prefix_token_ids,
                sources={},
                source_order=[],
                append_token_ids=[],
                suffix_token_ids=[],
                finished=False,
                revision=0,
                total_tokens_invalidated=0,
                update_events_accepted=0,
                update_works_admitted=0,
                update_works_completed=0,
                total_eager_update_prefill_tokens=0,
                total_admitted_update_prefill_tokens=0,
                total_scheduled_update_prefill_tokens=0,
                prefix_append_num_computed_tokens=0,
                update_num_computed_tokens=0,
                question_num_computed_tokens=0,
                padding_token_id=prefix_token_ids[-1] if prefix_token_ids else 0,
            )
            # Segment KV 依赖事件到达时的上下文，不能复用普通全局 prefix cache。
            self.skip_reading_prefix_cache = True
        return self.prompt_stream

    def _validate_prompt_stream_can_mutate(self, event: PromptStreamEvent) -> None:
        """校验当前请求是否仍然允许修改 prompt。"""
        if self._output_token_ids:
            raise ValueError(
                "Cannot mutate streaming prompt after output tokens have been "
                f"generated for request_id={self.request_id}."
            )
        if self.prompt_embeds is not None or self.prompt_is_token_ids is not None:
            raise ValueError(
                "Streaming prompt mutation currently supports token-id prompts "
                f"only, but request_id={self.request_id} uses prompt embeds or "
                "mixed token/embed inputs."
            )
        if (
            event.event_type
            in (
                PromptStreamEventType.APPEND,
                PromptStreamEventType.UPDATE,
                PromptStreamEventType.PROMPT_FINISHED,
            )
            and event.token_ids is None
        ):
            raise ValueError(
                f"Prompt stream {event.event_type} event for "
                f"request_id={self.request_id} must carry token_ids."
            )

    def _get_or_create_prompt_source(
        self,
        event: PromptStreamEvent,
        expected_source_type: PromptSourceType,
    ) -> PromptSourceState:
        """获取或创建某个 source 的长期状态。"""
        prompt_stream = self._ensure_prompt_stream_state()
        source = prompt_stream.sources.get(event.source_id)
        if source is not None:
            if event.source_id == "__prompt_stream_start__":
                raise ValueError(
                    "The synthetic prompt-stream start source cannot carry "
                    f"another event: request_id={self.request_id}."
                )
            if source.source_type != expected_source_type:
                raise ValueError(
                    f"Prompt stream source_id={event.source_id} changed type "
                    f"from {source.source_type} to {expected_source_type}."
                )
            return source

        if event.source_type is not None and event.source_type != expected_source_type:
            raise ValueError(
                f"Prompt stream event source_type={event.source_type} does not "
                f"match event_type={event.event_type}."
            )
        if event.source_id == "__prompt_stream_start__" and (
            expected_source_type != PromptSourceType.APPEND
            or bool(event.token_ids)
            or bool(prompt_stream.source_order)
        ):
            raise ValueError(
                "The reserved prompt-stream start source only accepts the "
                f"initial empty APPEND marker: request_id={self.request_id}."
            )
        if event.sequence_no != 0:
            raise ValueError(
                "The first prompt-stream source event must use sequence_no=0: "
                f"request_id={self.request_id}, source_id={event.source_id}, "
                f"received={event.sequence_no}."
            )

        gate = prompt_stream.append_update_gate
        if expected_source_type == PromptSourceType.UPDATE:
            for existing_source_id in prompt_stream.source_order:
                existing_source = prompt_stream.sources[existing_source_id]
                if existing_source.source_type == PromptSourceType.UPDATE:
                    raise ValueError(
                        "Streaming prompt currently supports at most one "
                        f"update source, but request_id={self.request_id} "
                        f"already has source_id={existing_source_id}."
                    )
        elif gate is not None and event.source_id != "__prompt_stream_start__":
            if gate.append_source_id is not None:
                raise ValueError(
                    "Prompt stream Update coalescing supports exactly one "
                    f"Append source, but request_id={self.request_id} already "
                    f"has source_id={gate.append_source_id}."
                )

        source = PromptSourceState(
            source_id=event.source_id,
            source_type=expected_source_type,
            token_ids=[],
            version=-1,
            next_sequence_no=0,
            seen_chunk_ids=set(),
            pending_events={},
            finished=False,
            last_arrival_time=event.arrival_time,
        )
        prompt_stream.sources[event.source_id] = source
        prompt_stream.source_order.append(event.source_id)
        if gate is not None and event.source_id != "__prompt_stream_start__":
            if expected_source_type == PromptSourceType.APPEND:
                gate.append_source_id = event.source_id
            else:
                gate.update_source_id = event.source_id
        return source

    def _record_prompt_stream_update_kv_truncation(self, keep_tokens: int) -> None:
        """记录下一代 Update work 开始前的延后 UPDATE KV 截断。

        Args:
            keep_tokens: 当前物化快照和下一代快照可安全复用的 token 前缀长度。

        Raises:
            ValueError: 当截断边界为负数时抛出，保护 segment-local 进度不变量。
        """
        if keep_tokens < 0:
            raise ValueError(
                "Prompt stream UPDATE KV truncation boundary must be non-negative: "
                f"request_id={self.request_id}, keep_tokens={keep_tokens}."
            )
        prompt_stream = self._ensure_prompt_stream_state()
        # 每次只会有一个 active Update work。若连续处理纯截断的零 token work，
        # 后一个 work 的 LCP 不会超过前一个快照长度，覆盖为最新边界即可。
        prompt_stream.pending_update_kv_truncation_keep_tokens = keep_tokens

    def _apply_prompt_stream_append(
        self,
        event: PromptStreamEvent,
    ) -> PromptStreamMutation:
        """应用 append 语义的流式 prompt 事件。"""
        self._validate_prompt_stream_can_mutate(event)
        prompt_stream = self._ensure_prompt_stream_state()
        if prompt_stream.finished:
            raise ValueError(
                f"Cannot append to finished prompt stream request_id={self.request_id}."
            )

        source = self._get_or_create_prompt_source(event, PromptSourceType.APPEND)
        if source.finished:
            raise ValueError(
                f"Cannot append to finished prompt source_id={event.source_id}."
            )
        if event.chunk_id is not None and event.chunk_id in source.seen_chunk_ids:
            if event.sequence_no < source.next_sequence_no:
                return PromptStreamMutation.noop(
                    self.num_prompt_tokens,
                    prompt_stream.revision,
                )
            raise ValueError(
                "Prompt stream APPEND reused a chunk_id for a new sequence: "
                f"request_id={self.request_id}, source_id={event.source_id}, "
                f"chunk_id={event.chunk_id}."
            )
        if event.sequence_no != source.next_sequence_no:
            raise ValueError(
                "Prompt stream APPEND events must arrive in source order: "
                f"request_id={self.request_id}, source_id={event.source_id}, "
                f"expected={source.next_sequence_no}, "
                f"received={event.sequence_no}."
            )
        if event.version < source.version:
            raise ValueError(
                "Prompt stream APPEND version cannot move backwards: "
                f"request_id={self.request_id}, source_id={event.source_id}, "
                f"current={source.version}, received={event.version}."
            )

        old_prompt_token_ids = self._rebuild_prompt_from_stream_state()

        # Append 的 KV 语义是请求级尾部追加。source.token_ids 负责来源级
        # 去重和完成状态，物理顺序则必须使用全局事件提交顺序，不能按
        # source 分组重排交错到达的 chunks。
        source.token_ids.extend(event.token_ids or [])
        prompt_stream.append_token_ids.extend(event.token_ids or [])
        gate = prompt_stream.append_update_gate
        if gate is not None and event.token_ids:
            # epoch 只随真实 Append 内容增长；空 start marker 不推进上下文版本。
            gate.append_epoch += 1
        source.version = max(source.version, event.version)
        source.next_sequence_no += 1
        source.last_arrival_time = event.arrival_time
        if event.chunk_id is not None:
            source.seen_chunk_ids.add(event.chunk_id)

        # 新 Append 不立刻覆写旧 PREFIX_APPEND 末尾 padding。它先成为一个
        # 可竞争 work item；只有 scheduler 选中它时，才从已提交逻辑长度开始
        # 写入并覆盖旧 padding。这样先被选中的 Update 仍能读取旧前缀快照。
        if event.token_ids:
            prompt_stream.pending_work_items.append(
                PromptStreamWorkItem(
                    phase=PromptStreamExecutionPhase.PREFIX_APPEND,
                    prefix_append_logical_len=(
                        self.get_prompt_stream_prefix_append_len()
                    ),
                    target_append_epoch=(gate.append_epoch if gate is not None else 0),
                )
            )
        return self._sync_prompt_from_stream_state(old_prompt_token_ids)

    def _apply_prompt_stream_update(
        self,
        event: PromptStreamEvent,
    ) -> PromptStreamMutation:
        """应用 update 语义的流式 prompt 快照事件。"""
        self._validate_prompt_stream_can_mutate(event)
        prompt_stream = self._ensure_prompt_stream_state()
        if prompt_stream.finished:
            raise ValueError(
                f"Cannot update finished prompt stream request_id={self.request_id}."
            )

        source = self._get_or_create_prompt_source(event, PromptSourceType.UPDATE)
        if source.finished:
            raise ValueError(
                f"Cannot update finished prompt source_id={event.source_id}."
            )
        if event.chunk_id is not None and event.chunk_id in source.seen_chunk_ids:
            if event.sequence_no < source.next_sequence_no:
                return PromptStreamMutation.noop(
                    self.num_prompt_tokens,
                    prompt_stream.revision,
                )
            raise ValueError(
                "Prompt stream UPDATE reused a chunk_id for a new sequence: "
                f"request_id={self.request_id}, source_id={event.source_id}, "
                f"chunk_id={event.chunk_id}."
            )
        if event.sequence_no != source.next_sequence_no:
            raise ValueError(
                "Prompt stream UPDATE events must arrive in source order: "
                f"request_id={self.request_id}, source_id={event.source_id}, "
                f"expected={source.next_sequence_no}, "
                f"received={event.sequence_no}."
            )
        if event.version < source.version:
            raise ValueError(
                "Prompt stream UPDATE version cannot move backwards: "
                f"request_id={self.request_id}, source_id={event.source_id}, "
                f"current={source.version}, received={event.version}."
            )
        if event.version == source.version and source.token_ids != event.token_ids:
            raise ValueError(
                f"Prompt stream source_id={event.source_id} received conflicting "
                f"content for version={event.version}."
            )

        old_update_token_ids = list(source.token_ids)
        new_update_token_ids = list(event.token_ids or [])
        update_common_prefix_len = self._get_common_prefix_len(
            old_update_token_ids,
            new_update_token_ids,
        )
        old_prompt_token_ids = self._rebuild_prompt_from_stream_state()
        source.token_ids = new_update_token_ids
        source.version = event.version
        source.next_sequence_no += 1
        source.last_arrival_time = event.arrival_time
        if event.chunk_id is not None:
            source.seen_chunk_ids.add(event.chunk_id)

        # Update 只在来源内部做 LCP。这里不把 prefix/template 或 append
        # 纳入公共前缀比较，否则 append 到达会错误扩大 update 的可复用范围。
        #
        # 关键不变量：新快照绝不能取消旧 Update work。active work 仍要使用
        # 已冻结的旧快照继续写完；pending work 也必须按到达顺序逐个计算。
        # 因此不能在这里回退全局 update 进度或截断 UPDATE blocks。两项操作
        # 都会延后到对应 work 被 scheduler 选中、前一代已完整物化之后执行。
        if new_update_token_ids != old_update_token_ids:
            if prompt_stream.update_coalescing is not None:
                # 合并开启时只覆盖唯一窗口；work 只能由显式 flush 接口创建。
                self.on_prompt_stream_update_snapshot(
                    old_update_token_ids,
                    new_update_token_ids,
                    time.monotonic(),
                )
            else:
                # 默认关闭路径保持每个 Update 事件立即生成不可变 FIFO work。
                eager_tokens = len(new_update_token_ids) - update_common_prefix_len
                prompt_stream.update_events_accepted += 1
                prompt_stream.total_eager_update_prefill_tokens += eager_tokens
                admission_id = self._create_prompt_stream_update_admission(
                    tuple(new_update_token_ids),
                    required_append_epoch=0,
                    policy_prefill_tokens=eager_tokens,
                )
                prompt_stream.pending_work_items.append(
                    PromptStreamWorkItem(
                        phase=PromptStreamExecutionPhase.UPDATE,
                        update_token_ids=tuple(new_update_token_ids),
                        execution_base_token_ids=tuple(old_update_token_ids),
                        update_common_prefix_len=update_common_prefix_len,
                        update_admission_id=admission_id,
                    )
                )

        return self._sync_prompt_from_stream_state(old_prompt_token_ids)

    def _mark_prompt_source_finished(
        self,
        event: PromptStreamEvent,
    ) -> PromptStreamMutation:
        """校验并标记单个真实 prompt source 已结束。"""
        prompt_stream = self._ensure_prompt_stream_state()
        if event.source_id == "__prompt_stream_start__":
            raise ValueError(
                "The synthetic prompt-stream start marker cannot be finished: "
                f"request_id={self.request_id}."
            )
        source_type = event.source_type
        source = prompt_stream.sources.get(event.source_id)
        if source is None:
            if source_type is None:
                raise ValueError(
                    "SOURCE_FINISHED must identify source_type for an unknown "
                    f"source: request_id={self.request_id}, "
                    f"source_id={event.source_id}."
                )
            if event.sequence_no != 0:
                raise ValueError(
                    "SOURCE_FINISHED cannot create a source with a sequence "
                    f"gap: request_id={self.request_id}, "
                    f"source_id={event.source_id}, "
                    f"sequence_no={event.sequence_no}."
                )
            source = self._get_or_create_prompt_source(
                event,
                source_type,
            )
        elif source_type is not None and source.source_type != source_type:
            raise ValueError(
                f"Prompt stream source_id={event.source_id} changed type "
                f"from {source.source_type} to {source_type}."
            )

        if source.finished:
            if (
                source.finished_version == event.version
                and source.finished_sequence_no == event.sequence_no
                and source.finished_chunk_id == event.chunk_id
            ):
                return PromptStreamMutation.noop(
                    self.num_prompt_tokens,
                    prompt_stream.revision,
                )
            raise ValueError(
                "Prompt stream source received conflicting SOURCE_FINISHED: "
                f"request_id={self.request_id}, source_id={event.source_id}."
            )

        if source.pending_events:
            raise ValueError(
                "Cannot finish prompt source while out-of-order events remain: "
                f"request_id={self.request_id}, source_id={event.source_id}, "
                f"pending_sequences={sorted(source.pending_events)}."
            )
        if event.sequence_no != source.next_sequence_no:
            raise ValueError(
                "SOURCE_FINISHED sequence does not follow the last accepted "
                f"event: request_id={self.request_id}, "
                f"source_id={event.source_id}, "
                f"expected={source.next_sequence_no}, "
                f"received={event.sequence_no}."
            )
        expected_version = max(0, source.version)
        if event.version != expected_version:
            raise ValueError(
                "SOURCE_FINISHED version does not match the latest source "
                f"snapshot: request_id={self.request_id}, "
                f"source_id={event.source_id}, expected={expected_version}, "
                f"received={event.version}."
            )
        if event.chunk_id is not None and event.chunk_id in source.seen_chunk_ids:
            raise ValueError(
                "SOURCE_FINISHED reused an existing chunk_id: "
                f"request_id={self.request_id}, source_id={event.source_id}, "
                f"chunk_id={event.chunk_id}."
            )

        source.finished = True
        source.version = event.version
        source.finished_version = event.version
        source.finished_sequence_no = event.sequence_no
        source.finished_chunk_id = event.chunk_id
        source.next_sequence_no += 1
        source.last_arrival_time = event.arrival_time
        if event.chunk_id is not None:
            source.seen_chunk_ids.add(event.chunk_id)
        if prompt_stream.update_coalescing is not None:
            now = time.monotonic()
            self.on_prompt_stream_source_finished(source.source_type, now)
            # Update finished 是终止性 flush；Append finished 只开 gate 并复查
            # 已经由 policy 判定到期的原窗口。
            self.flush_prompt_stream_update(
                now,
                reason=f"{source.source_type.value}_source_finished",
                force=source.source_type == PromptSourceType.UPDATE,
            )
            self.enqueue_question_if_ready()
        return PromptStreamMutation.noop(
            self.num_prompt_tokens,
            prompt_stream.revision,
        )

    def _mark_prompt_stream_finished(
        self,
        event: PromptStreamEvent,
    ) -> PromptStreamMutation:
        """封口流式 prompt，并写入最终用户问题。

        `PROMPT_FINISHED` 是请求级 barrier。它携带的 `token_ids` 被视为
        suffix，也就是 RAG 场景中最后到达的用户问题。只有这段 suffix 也
        prefill 完成后，scheduler 才允许该请求进入 decode。
        """
        self._validate_prompt_stream_can_mutate(event)
        prompt_stream = self._ensure_prompt_stream_state()
        new_suffix_token_ids = list(event.token_ids or [])
        if not new_suffix_token_ids:
            raise ValueError(
                "PROMPT_FINISHED must carry a non-empty final question for "
                f"request_id={self.request_id}."
            )

        if prompt_stream.finished:
            if prompt_stream.suffix_token_ids != new_suffix_token_ids:
                raise ValueError(
                    "Finished streaming prompt received conflicting suffix "
                    f"for request_id={self.request_id}."
                )
            return PromptStreamMutation.noop(
                self.num_prompt_tokens,
                prompt_stream.revision,
            )

        old_prompt_token_ids = self._rebuild_prompt_from_stream_state()
        if prompt_stream.update_coalescing is not None:
            gate = prompt_stream.append_update_gate
            assert gate is not None
            append_source = (
                prompt_stream.sources.get(gate.append_source_id)
                if gate.append_source_id is not None
                else None
            )
            update_source = (
                prompt_stream.sources.get(gate.update_source_id)
                if gate.update_source_id is not None
                else None
            )
            for source, source_type in (
                (append_source, PromptSourceType.APPEND),
                (update_source, PromptSourceType.UPDATE),
            ):
                if source is None:
                    # 请求级 finish 同时声明该可选来源不会再出现。
                    self.on_prompt_stream_source_finished(source_type)
                    continue
                if (
                    source.source_type != source_type
                    or source.pending_events
                ):
                    raise ValueError(
                        "PROMPT_FINISHED requires every present source to be "
                        "type-consistent with no pending events: "
                        f"request_id={self.request_id}, "
                        f"source_id={source.source_id}, "
                        f"finished={source.finished}."
                    )
                if not source.finished:
                    # SOURCE_FINISHED 可提前打开 gate；若前端省略它，请求级
                    # barrier 仍可在确认没有乱序缺口后封闭该来源。
                    source.finished = True
                    source.last_arrival_time = event.arrival_time
                    self.on_prompt_stream_source_finished(source_type)
            if not gate.append_source_finished or not gate.update_source_finished:
                raise RuntimeError(
                    "Prompt stream gate state disagrees with sealed sources: "
                    f"request_id={self.request_id}."
                )

        prompt_stream.suffix_token_ids = new_suffix_token_ids
        # finished 只表示用户问题已到达；显式 QUESTION work 才表示可调度。
        prompt_stream.finished = True
        prompt_stream.question_num_computed_tokens = 0
        prompt_stream.question_work_completed = False
        if prompt_stream.update_coalescing is not None:
            self.flush_prompt_stream_update(
                time.monotonic(),
                reason="prompt_finished",
                force=True,
            )
        self.enqueue_question_if_ready()
        return self._sync_prompt_from_stream_state(old_prompt_token_ids)

    def _current_prompt_token_ids(self) -> list[int]:
        """返回当前 prompt token 的可变副本。"""
        if self.prompt_token_ids is None:
            return []
        return list(self.prompt_token_ids)

    def _get_prompt_stream_segment_token_ids(
        self,
        source_type: PromptSourceType,
    ) -> list[int]:
        """按 source 到达顺序收集某个分段的 token。"""
        assert self.prompt_stream is not None
        if source_type == PromptSourceType.APPEND:
            return list(self.prompt_stream.append_token_ids)

        token_ids: list[int] = []
        for source_id in self.prompt_stream.source_order:
            source = self.prompt_stream.sources[source_id]
            if source.source_type == source_type:
                token_ids.extend(source.token_ids)
        return token_ids

    def get_prompt_stream_prefix_append_len(self) -> int:
        """返回 prefix+append 段的逻辑 token 数。"""
        if self.prompt_stream is None:
            return self.num_prompt_tokens
        append_len = len(
            self._get_prompt_stream_segment_token_ids(PromptSourceType.APPEND)
        )
        return len(self.prompt_stream.prefix_token_ids) + append_len

    def get_prompt_stream_update_len(self) -> int:
        """返回 update 段的逻辑 token 数。"""
        if self.prompt_stream is None:
            return 0
        return len(self._get_prompt_stream_segment_token_ids(PromptSourceType.UPDATE))

    def get_prompt_stream_question_len(self) -> int:
        """返回最终用户问题 / suffix 段的逻辑 token 数。"""
        if self.prompt_stream is None:
            return 0
        return len(self.prompt_stream.suffix_token_ids)

    def get_prompt_stream_segment_key(
        self,
        segment: PromptStreamKVSegment,
    ) -> str:
        """返回某个 KV segment 在 KV manager 中使用的 request key。

        block 仍然归属于同一个逻辑 request_id，但 update 与后续 question
        共用一个派生 key，避免和 prefix+append 段共用生命周期。
        """
        if segment == PromptStreamKVSegment.PREFIX_APPEND:
            return self.request_id
        return f"{self.request_id}::prompt_stream::{segment.value}"

    def get_prompt_stream_required_segments(
        self,
    ) -> tuple[PromptStreamKVSegment, ...]:
        """返回 worker 当前执行视图需要安装的 KV segment 顺序。"""
        if self.prompt_stream_execution_view is None:
            return ()
        return self.prompt_stream_execution_view.required_segments

    def get_prompt_stream_write_segment(self) -> PromptStreamKVSegment:
        """返回当前 forward 应写入的物理 KV segment。"""
        if self.prompt_stream_execution_view is None:
            raise RuntimeError(
                "Prompt stream execution view is not prepared for "
                f"request_id={self.request_id}."
            )
        return self.prompt_stream_execution_view.write_segment

    def get_prompt_stream_segment_num_computed_tokens(self) -> int:
        """返回当前写入 segment 内部已经计算的 token 数。"""
        if self.prompt_stream_execution_view is None:
            raise RuntimeError(
                "Prompt stream execution view is not prepared for "
                f"request_id={self.request_id}."
            )
        return max(
            0,
            self.num_computed_tokens - self.prompt_stream_execution_view.segment_offset,
        )

    def get_prompt_stream_layout(self) -> PromptStreamLayout:
        """返回当前 prompt 在 KV cache 中应呈现的逻辑布局。

        非流式请求被视为只有 prefix/template 段。流式请求固定采用：
        prefix/template + append segment + update segment + suffix/user question。
        """
        if self.prompt_stream is None:
            return PromptStreamLayout(
                prefix_len=self.num_prompt_tokens,
                append_len=0,
                update_len=0,
                suffix_len=0,
            )

        append_token_ids = self._get_prompt_stream_segment_token_ids(
            PromptSourceType.APPEND
        )
        update_token_ids = self._get_prompt_stream_segment_token_ids(
            PromptSourceType.UPDATE
        )
        prefix_append_len = len(self.prompt_stream.prefix_token_ids) + len(
            append_token_ids
        )
        padding_len = 0
        if self.prompt_stream.block_size is not None and prefix_append_len > 0:
            padding_len = (-prefix_append_len) % self.prompt_stream.block_size
        return PromptStreamLayout(
            prefix_len=len(self.prompt_stream.prefix_token_ids),
            append_len=len(append_token_ids),
            update_len=len(update_token_ids),
            suffix_len=len(self.prompt_stream.suffix_token_ids),
            prefix_append_padding_len=padding_len,
        )

    def _rebuild_prompt_from_stream_state(self) -> list[int]:
        """重建不包含物理 padding 的最终逻辑 prompt。"""
        assert self.prompt_stream is not None
        prompt_token_ids = list(self.prompt_stream.prefix_token_ids)
        prompt_token_ids.extend(
            self._get_prompt_stream_segment_token_ids(PromptSourceType.APPEND)
        )
        prompt_token_ids.extend(
            self._get_prompt_stream_segment_token_ids(PromptSourceType.UPDATE)
        )
        prompt_token_ids.extend(self.prompt_stream.suffix_token_ids)
        return prompt_token_ids

    def _get_prefix_append_physical_token_ids(
        self,
        logical_len: int | None = None,
    ) -> list[int]:
        """返回指定稳定逻辑前缀对应的 PREFIX_APPEND 物理 token 序列。

        Args:
            logical_len: 要暴露给本轮 attention 的逻辑前缀长度（包含固定
                模板、不包含 padding）。省略时使用当前所有 Append token，
                用于最终 P/D transfer 和最终用户问题阶段。

        Returns:
            所选逻辑前缀加上 block 对齐 padding 后的物理 token 序列。
        """
        assert self.prompt_stream is not None
        if self.prompt_stream.block_size is None:
            raise RuntimeError(
                "Prompt stream block size has not been initialized for "
                f"request_id={self.request_id}."
            )

        prefix_token_ids = self.prompt_stream.prefix_token_ids
        append_token_ids = self._get_prompt_stream_segment_token_ids(
            PromptSourceType.APPEND
        )
        full_logical_len = len(prefix_token_ids) + len(append_token_ids)
        if logical_len is None:
            logical_len = full_logical_len
        if not len(prefix_token_ids) <= logical_len <= full_logical_len:
            raise ValueError(
                "Prompt stream PREFIX_APPEND logical length is outside the "
                f"available range for request_id={self.request_id}: "
                f"logical_len={logical_len}, prefix_len={len(prefix_token_ids)}, "
                f"full_logical_len={full_logical_len}."
            )

        # Update work 只能读取已经提交的 Append 前缀；尚未被选中的 chunk
        # 不在此切片中，因此不会污染该次 Update 的 attention 上下文。
        num_append_tokens = logical_len - len(prefix_token_ids)
        token_ids = list(prefix_token_ids)
        token_ids.extend(append_token_ids[:num_append_tokens])
        if token_ids:
            # PREFIX_APPEND 和 UPDATE 使用不同 block table。为了让二者能
            # 在 worker/D 节点侧按 block list 直接拼接，PREFIX_APPEND 必须
            # 在物理序列上补齐到 block 边界。
            num_padding_tokens = (-len(token_ids)) % self.prompt_stream.block_size
            token_ids.extend([self.prompt_stream.padding_token_id] * num_padding_tokens)
        return token_ids

    def get_prompt_stream_physical_prompt_token_ids(self) -> list[int]:
        """返回 P/D transfer 必须共同使用的最终物理 prompt token 序列。"""
        if self.prompt_stream is None:
            return self._current_prompt_token_ids()
        token_ids = self._get_prefix_append_physical_token_ids()
        token_ids.extend(
            self._get_prompt_stream_segment_token_ids(PromptSourceType.UPDATE)
        )
        token_ids.extend(self.prompt_stream.suffix_token_ids)
        return token_ids

    def _start_next_prompt_stream_work_item(self) -> None:
        """在固定模板完成后，从统一队列选出下一项 Append/Update 工作。

        一个长期 Request 仍是 vLLM scheduler 的单个实体，因而不能在同一批次
        里将它的两个事件作为两个独立 request 并行执行。这里把事件建成
        work item：模板 bootstrap 后，队首事件就是该 Request 下一次进入
        scheduler 时被选中的工作，Append 不再具有额外硬编码优先级。
        """
        prompt_stream = self._ensure_prompt_stream_state()
        if prompt_stream.active_work_item is not None:
            return

        prefix_len = len(prompt_stream.prefix_token_ids)
        if not prompt_stream.prefix_append_bootstrap_completed:
            if prefix_len == 0:
                # 空模板无需写 KV，直接开放后续 Append/Update 竞争。
                prompt_stream.prefix_append_bootstrap_completed = True
                prompt_stream.prefix_append_committed_logical_len = 0
                prompt_stream.prefix_append_recovery_pending = False
            else:
                prompt_stream.active_work_item = PromptStreamWorkItem(
                    phase=PromptStreamExecutionPhase.PREFIX_APPEND,
                    prefix_append_logical_len=prefix_len,
                    num_computed_tokens=0,
                    execution_origin=(
                        PromptStreamExecutionOrigin.RECOVERY
                        if prompt_stream.prefix_append_recovery_pending
                        else PromptStreamExecutionOrigin.POLICY
                    ),
                )
                prompt_stream.prefix_append_recovery_pending = False
                return

        # 一个 Update 快照可能只是上一代的严格前缀，此时只需要截断而不需要
        # forward。循环会同步提交这种零 token work，随后继续寻找真正需要计算
        # 的 work，避免把它留成永远无法 advance 的 active work。
        while prompt_stream.pending_work_items:
            work_item = prompt_stream.pending_work_items[0]
            gate = prompt_stream.append_update_gate
            if (
                gate is not None
                and work_item.phase != PromptStreamExecutionPhase.PREFIX_APPEND
                and gate.committed_append_epoch
                != work_item.required_append_epoch
            ):
                if gate.committed_append_epoch < work_item.required_append_epoch:
                    # 依赖尚未物化，保留 FIFO 队首等待 Append work。
                    return
                raise RuntimeError(
                    "Prompt stream work observed a newer Append epoch than its "
                    f"frozen dependency: request_id={self.request_id}, "
                    f"committed_epoch={gate.committed_append_epoch}, "
                    f"required_epoch={work_item.required_append_epoch}."
                )
            prompt_stream.pending_work_items.popleft()
            if work_item.phase == PromptStreamExecutionPhase.PREFIX_APPEND:
                # 新 Append 从已提交的逻辑末尾开始写，复用该位置之后的旧 padding
                # 槽位。一个 Append work 开始后必须运行到新的 block 边界，不能
                # 在半个 block 内切给 Update。
                work_item.num_computed_tokens = (
                    prompt_stream.prefix_append_committed_logical_len
                )
                prompt_stream.active_work_item = work_item
                return

            if work_item.phase == PromptStreamExecutionPhase.QUESTION:
                question_token_ids = work_item.question_token_ids
                if question_token_ids is None:
                    raise RuntimeError(
                        "Prompt stream QUESTION work is missing its frozen suffix: "
                        f"request_id={self.request_id}."
                    )
                frozen_update = work_item.update_token_ids or ()
                if tuple(prompt_stream.update_committed_token_ids) != frozen_update:
                    raise RuntimeError(
                        "Prompt stream QUESTION work does not follow the "
                        "committed Update snapshot: "
                        f"request_id={self.request_id}."
                    )
                state = prompt_stream.update_coalescing
                if (
                    gate is not None
                    and state is not None
                    and state.config.append_gate
                    == "hold_until_append_finished"
                    and frozen_update
                    and gate.update_committed_append_epoch
                    != work_item.required_append_epoch
                ):
                    raise RuntimeError(
                        "Strict prompt stream QUESTION observed UPDATE KV from "
                        "the wrong Append epoch: "
                        f"request_id={self.request_id}, "
                        f"committed_epoch={gate.update_committed_append_epoch}, "
                        f"required_epoch={work_item.required_append_epoch}."
                    )
                work_item.prefix_append_token_ids = tuple(
                    self._get_prefix_append_physical_token_ids(
                        work_item.prefix_append_logical_len
                    )
                )
                work_item.num_computed_tokens = (
                    prompt_stream.question_num_computed_tokens
                )
                prompt_stream.active_work_item = work_item
                return

            # Update 快照必须在事件到达时已经冻结；若这里缺失，说明某条控制流
            # 绕过了 _apply_prompt_stream_update()，继续执行会把旧 work 错算成
            # 最新 source 快照，因此直接 fail closed。
            update_token_ids = work_item.update_token_ids
            if update_token_ids is None:
                raise RuntimeError(
                    "Prompt stream UPDATE work is missing its frozen snapshot: "
                    f"request_id={self.request_id}."
                )
            if not 0 <= work_item.update_common_prefix_len <= len(
                update_token_ids
            ):
                raise RuntimeError(
                    "Prompt stream UPDATE work has an invalid source-local LCP: "
                    f"request_id={self.request_id}, "
                    f"lcp={work_item.update_common_prefix_len}, "
                    f"snapshot_len={len(update_token_ids)}."
                )

            self._start_prompt_stream_update_execution_attempt(work_item)
            # 默认即时路径在 work 获调度时冻结可见 Append；合并路径已在
            # flush 时冻结逻辑长度与 epoch，这里只安装对应的物理 token 视图。
            committed_logical_len = prompt_stream.prefix_append_committed_logical_len
            if prompt_stream.update_coalescing is None:
                work_item.prefix_append_logical_len = committed_logical_len
            elif committed_logical_len != work_item.prefix_append_logical_len:
                raise RuntimeError(
                    "Prompt stream UPDATE work does not match its frozen "
                    f"Append prefix: request_id={self.request_id}, "
                    f"committed_len={committed_logical_len}, "
                    f"required_len={work_item.prefix_append_logical_len}."
                )
            work_item.prefix_append_token_ids = tuple(
                self._get_prefix_append_physical_token_ids(
                    work_item.prefix_append_logical_len
                )
            )

            execution_base = work_item.execution_base_token_ids
            if (
                gate is not None
                # 空基准没有 UPDATE KV 可以复用，Append epoch 与执行无关。
                and execution_base
                and work_item.execution_base_append_epoch
                != gate.update_committed_append_epoch
            ):
                raise RuntimeError(
                    "Prompt stream UPDATE execution base uses the wrong "
                    f"Append epoch: request_id={self.request_id}, "
                    f"base_epoch={work_item.execution_base_append_epoch}, "
                    f"committed_epoch={gate.update_committed_append_epoch}."
                )
            if (
                execution_base is not None
                and tuple(prompt_stream.update_committed_token_ids) != execution_base
            ):
                raise RuntimeError(
                    "Prompt stream UPDATE execution base does not match committed "
                    f"KV: request_id={self.request_id}."
                )

            # UPDATE KV 域仍保存上一代已经物化的快照。只能复用“上一代实际
            # 已计算进度”和“相邻 Update 快照 LCP”两者都覆盖的部分。
            old_update_num_computed_tokens = prompt_stream.update_num_computed_tokens
            work_item.num_computed_tokens = min(
                old_update_num_computed_tokens,
                work_item.update_common_prefix_len,
            )
            prompt_stream.update_num_computed_tokens = work_item.num_computed_tokens
            self._record_prompt_stream_update_kv_truncation(
                work_item.num_computed_tokens
            )
            prompt_stream.total_tokens_invalidated += max(
                0,
                old_update_num_computed_tokens - work_item.num_computed_tokens,
            )

            if work_item.num_computed_tokens >= len(update_token_ids):
                # 严格前缀或内容完全相同的下一代快照没有 delta prefill；仍需
                # 把逻辑上的 UPDATE KV 状态切换到该快照，并让 scheduler 完成
                # 延后截断，然后继续消费后续 work / Question。
                prompt_stream.update_num_computed_tokens = len(update_token_ids)
                prompt_stream.update_committed_token_ids = list(update_token_ids)
                if gate is not None:
                    gate.update_committed_append_epoch = (
                        work_item.required_append_epoch
                    )
                # 零 forward 截断同样结算物理 attempt 与 policy admission。
                self._complete_prompt_stream_update_work(work_item)
                continue

            prompt_stream.active_work_item = work_item
            return

    def _select_prompt_stream_execution_view(
        self,
    ) -> PromptStreamExecutionView:
        """根据 bootstrap、事件队列和已提交进度选择下一阶段执行视图。

        固定模板是唯一必须先完成的前缀。之后 Append/Update 从同一 FIFO work
        队列竞争；Update 被选中时冻结当前已提交的 PREFIX_APPEND 快照。这样
        Update 先获调度时不会读取后来 Append chunk 的 KV，反之则会读取已经
        完成提交的 Append KV。
        """
        assert self.prompt_stream is not None
        self._start_next_prompt_stream_work_item()
        prompt_stream = self.prompt_stream
        active_work_item = prompt_stream.active_work_item

        if active_work_item is not None:
            if active_work_item.phase == PromptStreamExecutionPhase.PREFIX_APPEND:
                prefix_append_token_ids = self._get_prefix_append_physical_token_ids(
                    active_work_item.prefix_append_logical_len
                )
                return PromptStreamExecutionView(
                    phase=PromptStreamExecutionPhase.PREFIX_APPEND,
                    write_segment=PromptStreamKVSegment.PREFIX_APPEND,
                    required_segments=(PromptStreamKVSegment.PREFIX_APPEND,),
                    token_ids=tuple(prefix_append_token_ids),
                    num_computed_tokens=active_work_item.num_computed_tokens,
                    prefix_append_physical_len=len(prefix_append_token_ids),
                    segment_offset=0,
                    padding_range=(
                        active_work_item.prefix_append_logical_len,
                        len(prefix_append_token_ids),
                    ),
                )

            if active_work_item.phase == PromptStreamExecutionPhase.QUESTION:
                update_token_ids = active_work_item.update_token_ids or ()
                question_token_ids = active_work_item.question_token_ids
                assert question_token_ids is not None
                prefix_append_token_ids = active_work_item.prefix_append_token_ids
                execution_token_ids = (
                    list(prefix_append_token_ids)
                    + list(update_token_ids)
                    + list(question_token_ids)
                )
                prefix_append_physical_len = len(prefix_append_token_ids)
                return PromptStreamExecutionView(
                    phase=PromptStreamExecutionPhase.QUESTION,
                    write_segment=PromptStreamKVSegment.UPDATE,
                    required_segments=(
                        PromptStreamKVSegment.PREFIX_APPEND,
                        PromptStreamKVSegment.UPDATE,
                    ),
                    token_ids=tuple(execution_token_ids),
                    num_computed_tokens=(
                        prefix_append_physical_len
                        + len(update_token_ids)
                        + active_work_item.num_computed_tokens
                    ),
                    prefix_append_physical_len=prefix_append_physical_len,
                    segment_offset=prefix_append_physical_len,
                    padding_range=(
                        active_work_item.prefix_append_logical_len,
                        prefix_append_physical_len,
                    ),
                )

            # Update delta 基于 flush 时冻结的 PREFIX_APPEND KV 和 work 自己
            # 保存的 source-local LCP KV。新 KV 仍然只写入 UPDATE segment。
            update_token_ids = active_work_item.update_token_ids
            assert update_token_ids is not None
            execution_token_ids = list(active_work_item.prefix_append_token_ids) + list(
                update_token_ids
            )
            prefix_append_physical_len = len(active_work_item.prefix_append_token_ids)
            return PromptStreamExecutionView(
                phase=PromptStreamExecutionPhase.UPDATE,
                write_segment=PromptStreamKVSegment.UPDATE,
                required_segments=(
                    PromptStreamKVSegment.PREFIX_APPEND,
                    PromptStreamKVSegment.UPDATE,
                ),
                token_ids=tuple(execution_token_ids),
                num_computed_tokens=(
                    prefix_append_physical_len + active_work_item.num_computed_tokens
                ),
                prefix_append_physical_len=prefix_append_physical_len,
                segment_offset=prefix_append_physical_len,
                padding_range=(
                    active_work_item.prefix_append_logical_len,
                    prefix_append_physical_len,
                ),
            )

        # 所有 source work 已提交后，最终 question 才能使用最终的物理布局。
        prefix_append_token_ids = self._get_prefix_append_physical_token_ids()
        if prompt_stream.update_coalescing is None:
            update_token_ids = self._get_prompt_stream_segment_token_ids(
                PromptSourceType.UPDATE
            )
        else:
            # 未 admission 的窗口只属于逻辑状态；idle view 只能暴露已写 KV。
            update_token_ids = list(prompt_stream.update_committed_token_ids)
        question_token_ids = prompt_stream.suffix_token_ids
        prefix_append_physical_len = len(prefix_append_token_ids)

        # idle/最终 decode 视图使用已完成 Append 的边界，而非某个旧 work。
        layout = self.get_prompt_stream_layout()
        padding_range = (
            layout.prefix_len + layout.append_len,
            prefix_append_physical_len,
        )

        # 没有 active work 时只暴露已经物化的内容。Question 必须先由
        # enqueue_question_if_ready() 显式进入 FIFO，不能通过 idle view
        # 绕过最终 Update barrier。
        committed_question_token_ids = (
            question_token_ids if prompt_stream.question_work_completed else []
        )
        if update_token_ids or committed_question_token_ids:
            execution_token_ids = (
                prefix_append_token_ids
                + update_token_ids
                + committed_question_token_ids
            )
            phase = (
                PromptStreamExecutionPhase.QUESTION
                if prompt_stream.question_work_completed
                else PromptStreamExecutionPhase.UPDATE
            )
            return PromptStreamExecutionView(
                phase=phase,
                write_segment=PromptStreamKVSegment.UPDATE,
                required_segments=(
                    PromptStreamKVSegment.PREFIX_APPEND,
                    PromptStreamKVSegment.UPDATE,
                ),
                token_ids=tuple(execution_token_ids),
                num_computed_tokens=len(execution_token_ids),
                prefix_append_physical_len=prefix_append_physical_len,
                segment_offset=prefix_append_physical_len,
                padding_range=padding_range,
            )

        return PromptStreamExecutionView(
            phase=PromptStreamExecutionPhase.PREFIX_APPEND,
            write_segment=PromptStreamKVSegment.PREFIX_APPEND,
            required_segments=(PromptStreamKVSegment.PREFIX_APPEND,),
            token_ids=tuple(prefix_append_token_ids),
            num_computed_tokens=len(prefix_append_token_ids),
            prefix_append_physical_len=prefix_append_physical_len,
            segment_offset=0,
            padding_range=padding_range,
        )

    def prepare_prompt_stream_execution(self, block_size: int) -> bool:
        """准备 scheduler/worker 下一轮共同使用的 segment 执行视图。

        Args:
            block_size: scheduler 使用的统一 KV block size。

        Returns:
            如果 token、计算位置或所需 block table 发生变化，返回 True。

        Side Effects:
            会刷新 Request 的普通调度字段，并要求 scheduler 通过
            `NewRequestData` 完整覆盖 worker 的缓存请求状态。
        """
        prompt_stream = self._ensure_prompt_stream_state()
        if block_size <= 0:
            raise ValueError(
                f"Prompt stream block_size must be positive: {block_size}."
            )
        if prompt_stream.block_size is None:
            prompt_stream.block_size = block_size
        elif prompt_stream.block_size != block_size:
            raise ValueError(
                "Prompt stream block size changed for "
                f"request_id={self.request_id}: old={prompt_stream.block_size}, "
                f"new={block_size}."
            )

        new_view = self._select_prompt_stream_execution_view()
        if new_view == self.prompt_stream_execution_view:
            return False
        old_view = self.prompt_stream_execution_view
        if (
            old_view is not None
            and old_view.phase == new_view.phase
            and old_view.write_segment == new_view.write_segment
            and old_view.required_segments == new_view.required_segments
            and old_view.token_ids == new_view.token_ids
            and old_view.prefix_append_physical_len
            == new_view.prefix_append_physical_len
            and old_view.segment_offset == new_view.segment_offset
            and old_view.padding_range == new_view.padding_range
        ):
            # 同一执行视图的 chunked prefill 只推进进度；cached request 路径
            # 会收到新的 num_computed_tokens，无需重装 token 和 block table。
            self.prompt_stream_execution_view = new_view
            return False

        # 中间阶段不允许携带上一个物理视图采样出的临时 output token。
        if self._output_token_ids and not self.can_decode_from_prompt_stream():
            self._output_token_ids.clear()
            self.output_token_ids = ConstantList(self._output_token_ids)
            self.spec_token_ids.clear()
            self.num_output_placeholders = 0
            self.async_tokens_to_discard = 0

        self.prompt_stream_execution_view = new_view
        self.prompt_token_ids = list(new_view.token_ids)
        self._all_token_ids = self.prompt_token_ids + list(self._output_token_ids)
        self.num_prompt_tokens = len(self.prompt_token_ids)
        self.num_computed_tokens = new_view.num_computed_tokens
        self.all_token_ids = ConstantList(self._all_token_ids)
        self.output_token_ids = ConstantList(self._output_token_ids)
        self.prompt_stream_needs_worker_refresh = True

        # Prompt stream 不进入普通 prefix cache，但清空 hashes 可避免调试时
        # 把旧的逻辑 prompt hash 误认为当前物理执行视图。
        self.block_hashes.clear()
        self._prompt_embeds_per_block_hashes.clear()
        return True

    def advance_prompt_stream_computed_tokens(self, num_scheduled_tokens: int) -> None:
        """提交一轮 forward 完成的 segment 进度并推进当前 work item。"""
        if self.prompt_stream is None or self.prompt_stream_execution_view is None:
            self.num_computed_tokens += num_scheduled_tokens
            return
        if num_scheduled_tokens <= 0:
            raise ValueError(
                "Prompt stream scheduled token count must be positive for "
                f"request_id={self.request_id}: {num_scheduled_tokens}."
            )

        prompt_stream = self.prompt_stream
        view = self.prompt_stream_execution_view
        new_num_computed_tokens = self.num_computed_tokens + num_scheduled_tokens
        active_work_item = prompt_stream.active_work_item
        if view.phase == PromptStreamExecutionPhase.UPDATE:
            # 该接口在 Scheduler 完成本轮 schedule 后推进进度，因此这里
            # 诚实记录 scheduled token；preemption 后的重放会再次累计。
            prompt_stream.total_scheduled_update_prefill_tokens += (
                num_scheduled_tokens
            )
            if (
                active_work_item is not None
                and active_work_item.execution_origin
                == PromptStreamExecutionOrigin.RECOVERY
            ):
                prompt_stream.total_recovery_update_prefill_tokens += (
                    num_scheduled_tokens
                )
        elif (
            view.phase == PromptStreamExecutionPhase.PREFIX_APPEND
            and active_work_item is not None
            and active_work_item.execution_origin
            == PromptStreamExecutionOrigin.RECOVERY
        ):
            prompt_stream.total_recovery_prefix_append_prefill_tokens += (
                num_scheduled_tokens
            )
        if view.phase == PromptStreamExecutionPhase.PREFIX_APPEND:
            # PREFIX_APPEND work 的全局位置就是 segment-local 位置。
            local_num_computed_tokens = min(
                new_num_computed_tokens,
                len(view.token_ids),
            )
            prompt_stream.prefix_append_num_computed_tokens = local_num_computed_tokens
            if active_work_item is not None:
                active_work_item.num_computed_tokens = local_num_computed_tokens
                if local_num_computed_tokens >= len(view.token_ids):
                    # 只有补齐到新的 block 边界，Append 才能成为后续 Update
                    # 可见的稳定 KV 前缀。
                    prompt_stream.prefix_append_committed_logical_len = (
                        active_work_item.prefix_append_logical_len
                    )
                    prompt_stream.prefix_append_bootstrap_completed = True
                    gate = prompt_stream.append_update_gate
                    if gate is not None:
                        gate.committed_append_epoch = max(
                            gate.committed_append_epoch,
                            active_work_item.target_append_epoch,
                        )
                    prompt_stream.active_work_item = None
        elif view.phase == PromptStreamExecutionPhase.UPDATE:
            local_num_computed_tokens = max(
                0,
                new_num_computed_tokens - view.segment_offset,
            )
            if active_work_item is not None:
                update_token_ids = active_work_item.update_token_ids
                assert update_token_ids is not None
                update_len = len(update_token_ids)
            else:
                update_token_ids = None
                update_len = self.get_prompt_stream_update_len()
            local_num_computed_tokens = min(
                local_num_computed_tokens,
                update_len,
            )
            prompt_stream.update_num_computed_tokens = local_num_computed_tokens
            if active_work_item is not None:
                active_work_item.num_computed_tokens = local_num_computed_tokens
                if local_num_computed_tokens >= update_len:
                    # 到这里该 Update work 的完整快照已经真实写入 UPDATE 域，
                    # 才能推进 committed 状态并允许下一代 work 复用其 LCP。
                    assert update_token_ids is not None
                    prompt_stream.update_committed_token_ids = list(update_token_ids)
                    gate = prompt_stream.append_update_gate
                    if gate is not None:
                        gate.update_committed_append_epoch = (
                            active_work_item.required_append_epoch
                        )
                    self._complete_prompt_stream_update_work(
                        active_work_item
                    )
                    prompt_stream.active_work_item = None
        else:
            local_num_computed_tokens = max(
                0,
                new_num_computed_tokens
                - view.segment_offset
                - self.get_prompt_stream_update_len(),
            )
            question_len = self.get_prompt_stream_question_len()
            prompt_stream.question_num_computed_tokens = min(
                local_num_computed_tokens,
                question_len,
            )
            if active_work_item is not None:
                active_work_item.num_computed_tokens = (
                    prompt_stream.question_num_computed_tokens
                )
                if prompt_stream.question_num_computed_tokens >= question_len:
                    prompt_stream.question_work_completed = True
                    prompt_stream.active_work_item = None

        self.num_computed_tokens = new_num_computed_tokens

    def _get_latest_prompt_stream_update_snapshot(
        self,
    ) -> tuple[int, ...] | None:
        """返回最新 Update 逻辑快照，并区分空快照与没有 Update source。"""
        assert self.prompt_stream is not None
        has_update_source = False
        token_ids: list[int] = []
        for source_id in self.prompt_stream.source_order:
            source = self.prompt_stream.sources[source_id]
            if source.source_type != PromptSourceType.UPDATE:
                continue
            has_update_source = True
            token_ids.extend(source.token_ids)
        return tuple(token_ids) if has_update_source else None

    def _record_prompt_stream_recovery_reason(
        self,
        reason: PromptStreamRecoveryReason,
    ) -> None:
        """把一次物理 KV recovery 归因到互斥的请求级原因。"""
        assert self.prompt_stream is not None
        if reason == PromptStreamRecoveryReason.RUNNING_PREEMPTION:
            self.prompt_stream.prompt_stream_running_preemptions += 1
        elif reason == PromptStreamRecoveryReason.DORMANT_EVICTION:
            self.prompt_stream.prompt_stream_dormant_evictions += 1
        elif reason == PromptStreamRecoveryReason.RETAINED_WAITING_EVICTION:
            self.prompt_stream.prompt_stream_retained_waiting_evictions += 1
        elif reason == PromptStreamRecoveryReason.PREFIX_CACHE_RESET:
            self.prompt_stream.prompt_stream_prefix_cache_resets += 1

    def reset_prompt_stream_computed_state(
        self,
        reason: PromptStreamRecoveryReason = PromptStreamRecoveryReason.UNKNOWN,
    ) -> PromptStreamRecoveryDelta | None:
        """按 latest-only 语义重建物理 KV，并保留 policy/window 状态。

        资源 recovery 可以折叠尚未完成的旧 admission，但必须给每个旧 ID
        写入唯一终态。尚未 admission 的 coalescing window、绝对 deadline、
        sticky ready 和 EWMA 均保持不变，避免 repeated preemption 延长等待。

        Args:
            reason: Scheduler 提供的互斥 recovery 原因。

        Returns:
            Prompt Stream 请求返回本次 recovery delta；普通请求返回 None。
        """
        if self.prompt_stream is None:
            return None
        prompt_stream = self.prompt_stream
        latest_update_token_ids = self._get_latest_prompt_stream_update_snapshot()
        state = prompt_stream.update_coalescing
        has_unadmitted_window = (
            state is not None and state.latest_pending_token_ids is not None
        )

        # 物理 KV 被释放前，先按 segment 记录实际丢失的有效计算进度。
        prefix_append_tokens_discarded = (
            prompt_stream.prefix_append_num_computed_tokens
        )
        update_tokens_discarded = prompt_stream.update_num_computed_tokens
        question_tokens_discarded = prompt_stream.question_num_computed_tokens
        prompt_stream.total_prefix_append_tokens_discarded_by_preemption += (
            prefix_append_tokens_discarded
        )
        prompt_stream.total_update_tokens_discarded_by_preemption += (
            update_tokens_discarded
        )
        prompt_stream.total_question_tokens_discarded_by_preemption += (
            question_tokens_discarded
        )
        self._record_prompt_stream_recovery_reason(reason)

        # 只有已经成为 active 的 Update work 才拥有物理 attempt。
        active_work_item = prompt_stream.active_work_item
        if (
            active_work_item is not None
            and active_work_item.phase == PromptStreamExecutionPhase.UPDATE
            and active_work_item.execution_attempt_id is not None
        ):
            prompt_stream.update_execution_attempts_preempted += 1

        # 若没有更新的未放行窗口，保留与 source 最新快照对应的最后一个
        # outstanding admission；其余旧 ID 由 recovery 显式取消。
        outstanding_records = [
            record
            for record in prompt_stream.update_admissions.values()
            if record.status
            in (UpdateAdmissionStatus.PENDING, UpdateAdmissionStatus.ACTIVE)
        ]
        carried_admission_id: int | None = None
        if not has_unadmitted_window and latest_update_token_ids is not None:
            matching_records = [
                record
                for record in outstanding_records
                if record.target_token_ids == latest_update_token_ids
            ]
            if matching_records:
                carried_admission_id = max(
                    matching_records,
                    key=lambda record: record.admission_id,
                ).admission_id

        cancelled_admission_ids: list[int] = []
        for record in outstanding_records:
            if record.admission_id == carried_admission_id:
                # 新 recovery attempt 延续同一 admission，不增加 policy 计数。
                record.status = UpdateAdmissionStatus.PENDING
                continue
            record.status = UpdateAdmissionStatus.CANCELLED_BY_RECOVERY
            cancelled_admission_ids.append(record.admission_id)

        # 清空只属于物理执行层的状态；source、ledger 和 window 不回退。
        prompt_stream.prefix_append_num_computed_tokens = 0
        prompt_stream.update_num_computed_tokens = 0
        prompt_stream.update_committed_token_ids.clear()
        prompt_stream.question_num_computed_tokens = 0
        prompt_stream.prefix_append_committed_logical_len = 0
        prompt_stream.prefix_append_bootstrap_completed = False
        prompt_stream.prefix_append_recovery_pending = True
        prompt_stream.active_work_item = None
        prompt_stream.pending_work_items.clear()
        prompt_stream.pending_update_kv_truncation_keep_tokens = None
        prompt_stream.question_work_enqueued = False
        prompt_stream.question_work_completed = False
        gate = prompt_stream.append_update_gate
        if gate is not None:
            gate.committed_append_epoch = 0
            gate.update_committed_append_epoch = None

        # recovery 必须先重建最终 PA。模板 bootstrap 由 work selector 创建；
        # 若存在 Append，再追加一个从模板末尾追到完整 PA 的 recovery work。
        if prompt_stream.append_token_ids:
            prompt_stream.pending_work_items.append(
                PromptStreamWorkItem(
                    phase=PromptStreamExecutionPhase.PREFIX_APPEND,
                    prefix_append_logical_len=(
                        self.get_prompt_stream_prefix_append_len()
                    ),
                    target_append_epoch=(
                        gate.append_epoch if gate is not None else 0
                    ),
                    execution_origin=PromptStreamExecutionOrigin.RECOVERY,
                )
            )

        # window 存在时保持其原始 policy 状态，等 Scheduler 按原 deadline
        # admission；否则直接为 source 最新快照建立 recovery-only 执行链。
        if state is not None:
            state.last_admitted_token_ids = ()
            state.last_admitted_append_epoch = None
        should_recover_update = (
            not has_unadmitted_window
            and latest_update_token_ids is not None
            and (carried_admission_id is not None or bool(latest_update_token_ids))
        )
        if should_recover_update:
            required_append_epoch = gate.append_epoch if gate is not None else 0
            prompt_stream.pending_work_items.append(
                PromptStreamWorkItem(
                    phase=PromptStreamExecutionPhase.UPDATE,
                    prefix_append_logical_len=(
                        self.get_prompt_stream_prefix_append_len()
                    ),
                    update_token_ids=latest_update_token_ids,
                    execution_base_token_ids=(),
                    execution_base_append_epoch=None,
                    required_append_epoch=required_append_epoch,
                    update_common_prefix_len=0,
                    update_admission_id=carried_admission_id,
                    execution_origin=PromptStreamExecutionOrigin.RECOVERY,
                )
            )
            prompt_stream.update_recovery_works_created += 1
            if state is not None:
                state.last_admitted_token_ids = latest_update_token_ids
                state.last_admitted_append_epoch = required_append_epoch
        elif state is not None and latest_update_token_ids == ():
            # 已完成的空快照无需 forward，但逻辑链尾仍应明确为空。
            state.last_admitted_token_ids = ()
            state.last_admitted_append_epoch = (
                gate.append_epoch if gate is not None else 0
            )

        # Question 可以提前排在 recovery work 后面，但执行时仍受 FIFO、
        # final Append epoch 和最新 Update committed 状态三重 barrier 保护。
        self.enqueue_question_if_ready()
        self.prompt_stream_execution_view = None
        self.prompt_stream_needs_worker_refresh = True
        self._output_token_ids.clear()
        self.output_token_ids = ConstantList(self._output_token_ids)
        self.spec_token_ids.clear()
        self.num_computed_tokens = 0

        return PromptStreamRecoveryDelta(
            reason=reason,
            cancelled_update_admission_ids=tuple(cancelled_admission_ids),
            carried_update_admission_id=carried_admission_id,
            prefix_append_tokens_discarded=prefix_append_tokens_discarded,
            update_tokens_discarded=update_tokens_discarded,
            question_tokens_discarded=question_tokens_discarded,
        )

    @staticmethod
    def _get_common_prefix_len(
        old_token_ids: list[int],
        new_token_ids: list[int],
    ) -> int:
        """计算两个 token 序列的最长公共前缀长度。"""
        common_prefix_len = 0
        for old_token_id, new_token_id in zip(old_token_ids, new_token_ids):
            if old_token_id != new_token_id:
                break
            common_prefix_len += 1
        return common_prefix_len

    def _sync_prompt_from_stream_state(
        self,
        old_prompt_token_ids: list[int],
    ) -> PromptStreamMutation:
        """提交逻辑 prompt revision，物理执行视图由 scheduler 后续准备。

        Args:
            old_prompt_token_ids: 修改流式状态前的逻辑 prompt token。

        Returns:
            本次逻辑变化的统计摘要。Update KV 的截断不会在事件处理函数中
            立即发生，而是由 scheduler 在对应 work 被选中时按 segment 执行。
        """
        assert self.prompt_stream is not None
        new_prompt_token_ids = self._rebuild_prompt_from_stream_state()
        common_prefix_len = self._get_common_prefix_len(
            old_prompt_token_ids,
            new_prompt_token_ids,
        )
        if old_prompt_token_ids == new_prompt_token_ids:
            return PromptStreamMutation.noop(
                self.num_prompt_tokens,
                self.prompt_stream.revision,
            )
        old_num_prompt_tokens = len(old_prompt_token_ids)
        self.prompt_stream.revision += 1
        # 事件可能在 GPU forward 之前连续到达；执行视图由 scheduler 按
        # PREFIX_APPEND -> UPDATE -> QUESTION 的提交进度动态选择。
        self.prompt_stream_needs_worker_refresh = True

        return PromptStreamMutation(
            applied=True,
            old_num_prompt_tokens=old_num_prompt_tokens,
            new_num_prompt_tokens=len(new_prompt_token_ids),
            common_prefix_len=common_prefix_len,
            prompt_revision=self.prompt_stream.revision,
        )


class RequestStatus(enum.IntEnum):
    """Status of a request."""

    WAITING = enum.auto()
    WAITING_FOR_STRUCTURED_OUTPUT_GRAMMAR = enum.auto()
    WAITING_FOR_REMOTE_KVS = enum.auto()
    WAITING_FOR_STREAMING_REQ = enum.auto()
    RUNNING = enum.auto()
    PREEMPTED = enum.auto()
    # Note: anything after PREEMPTED will be considered
    # as a finished status.
    FINISHED_STOPPED = enum.auto()
    FINISHED_LENGTH_CAPPED = enum.auto()
    FINISHED_ABORTED = enum.auto()
    FINISHED_IGNORED = enum.auto()
    FINISHED_ERROR = enum.auto()
    FINISHED_REPETITION = enum.auto()

    def __str__(self) -> str:
        return self.name

    @staticmethod
    def is_finished(status: "RequestStatus") -> bool:
        return status > RequestStatus.PREEMPTED

    @staticmethod
    def get_finished_reason(status: "RequestStatus") -> FinishReason | None:
        return _FINISHED_REASON_MAP.get(status)


# Mapping of finished statuses to their finish reasons.
# NOTE: The ignored requests are the requests whose prompt lengths
# are longer than the model's length cap. Therefore, the stop
# reason should also be "length" as in OpenAI API.
_FINISHED_REASON_MAP = {
    RequestStatus.FINISHED_STOPPED: FinishReason.STOP,
    RequestStatus.FINISHED_LENGTH_CAPPED: FinishReason.LENGTH,
    RequestStatus.FINISHED_ABORTED: FinishReason.ABORT,
    RequestStatus.FINISHED_IGNORED: FinishReason.LENGTH,
    RequestStatus.FINISHED_ERROR: FinishReason.ERROR,
    RequestStatus.WAITING_FOR_STREAMING_REQ: FinishReason.STOP,
    RequestStatus.FINISHED_REPETITION: FinishReason.REPETITION,
}
