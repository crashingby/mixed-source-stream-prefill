# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import enum
import time
from collections import deque
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
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
)
from vllm.v1.metrics.stats import PrefillStats
from vllm.v1.structured_output.request import StructuredOutputRequest
from vllm.v1.utils import ConstantList

if TYPE_CHECKING:
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


@dataclass
class PromptStreamWorkItem:
    """表示同一长期请求内一项可竞争的流式 prefill 工作。

    固定模板完成 bootstrap 后，Append 和 Update 事件都会产生一个 work item，
    并按进入 Request 的顺序进入同一个队列。这样二者不再由阶段选择函数的
    固定优先级决定。

    对 Update，`update_token_ids` 在事件到达时冻结，`prefix_append_token_ids`
    则在 work item 被 scheduler 选中时冻结。前者保证多个 Update 快照会按
    到达顺序逐个计算，后者保证 Update 后到达、但尚未被调度的 Append chunk
    不会悄悄进入这次 Update 的 attention 上下文。
    """

    # 本 work item 要写入的物理 KV 域。
    phase: PromptStreamExecutionPhase
    # Append work 完成时 PREFIX_APPEND 应达到的逻辑长度（含固定模板）。
    prefix_append_logical_len: int = 0
    # Update 事件到达时冻结的完整快照；None 只表示 PREFIX_APPEND work。
    update_token_ids: tuple[int, ...] | None = None
    # 当前快照相对上一条 Update 快照的 source-local LCP。该值既是本 work
    # 可以复用的 UPDATE KV 上界，也是切换到本 work 时的延后截断位置。
    update_common_prefix_len: int = 0
    # Update 被选中时冻结的 PREFIX_APPEND 物理 token 视图。
    prefix_append_token_ids: tuple[int, ...] = ()
    # 当前 work item 已完成的 segment-local token 数。
    num_computed_tokens: int = 0


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

    # 重复 chunk、旧版本 update、乱序 pending 等，没有实际改 prompt。
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
    # 暂存乱序到达但还不能应用的事件；MVP 可以先不消费这个队列。
    pending_events: dict[int, PromptStreamEvent]
    # 该来源是否结束。
    finished: bool
    # 该来源最近一次事件到达时间。
    last_arrival_time: float


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
    # 累计失效 token 数。
    total_tokens_invalidated: int
    # 累计重算 token 数。
    total_tokens_recomputed: int
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

    def uses_prompt_stream_kv_transfer(self) -> bool:
        """判断当前 D 请求是否加载 prompt-stream 的精确 block manifest。

        P 节点上的长期 Request 通过 prompt_stream 保存状态；D 节点收到的
        仍是普通 completion Request，因此必须依靠 transfer 参数识别。
        """
        return bool(
            self.kv_transfer_params
            and self.kv_transfer_params.get("prompt_stream_exact_transfer")
        )

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
        return (
            self.prompt_stream.prefix_append_bootstrap_completed
            and self.prompt_stream.active_work_item is None
            and not self.prompt_stream.pending_work_items
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
                total_tokens_recomputed=0,
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

        if expected_source_type == PromptSourceType.UPDATE:
            for existing_source_id in prompt_stream.source_order:
                existing_source = prompt_stream.sources[existing_source_id]
                if existing_source.source_type == PromptSourceType.UPDATE:
                    raise ValueError(
                        "Streaming prompt currently supports at most one "
                        f"update source, but request_id={self.request_id} "
                        f"already has source_id={existing_source_id}."
                    )

        source = PromptSourceState(
            source_id=event.source_id,
            source_type=expected_source_type,
            token_ids=[],
            version=-1,
            next_sequence_no=event.sequence_no,
            seen_chunk_ids=set(),
            pending_events={},
            finished=False,
            last_arrival_time=event.arrival_time,
        )
        prompt_stream.sources[event.source_id] = source
        prompt_stream.source_order.append(event.source_id)
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
            return PromptStreamMutation.noop(
                self.num_prompt_tokens,
                prompt_stream.revision,
            )
        if event.sequence_no != source.next_sequence_no:
            # TODO: 后续可以把乱序事件放进 pending_events，再在缺口补齐后消费。
            source.pending_events[event.sequence_no] = event
            return PromptStreamMutation.noop(
                self.num_prompt_tokens,
                prompt_stream.revision,
            )

        old_prompt_token_ids = self._rebuild_prompt_from_stream_state()

        # Append 的 KV 语义是请求级尾部追加。source.token_ids 负责来源级
        # 去重和完成状态，物理顺序则必须使用全局事件提交顺序，不能按
        # source 分组重排交错到达的 chunks。
        source.token_ids.extend(event.token_ids or [])
        prompt_stream.append_token_ids.extend(event.token_ids or [])
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
            return PromptStreamMutation.noop(
                self.num_prompt_tokens,
                prompt_stream.revision,
            )
        if event.sequence_no != source.next_sequence_no:
            # TODO: 后续可以为 update source 增加 debounce 和乱序合并策略。
            source.pending_events[event.sequence_no] = event
            return PromptStreamMutation.noop(
                self.num_prompt_tokens,
                prompt_stream.revision,
            )
        if event.version < source.version:
            return PromptStreamMutation.noop(
                self.num_prompt_tokens,
                prompt_stream.revision,
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
            prompt_stream.pending_work_items.append(
                PromptStreamWorkItem(
                    phase=PromptStreamExecutionPhase.UPDATE,
                    update_token_ids=tuple(new_update_token_ids),
                    update_common_prefix_len=update_common_prefix_len,
                )
            )

        return self._sync_prompt_from_stream_state(old_prompt_token_ids)

    def _mark_prompt_source_finished(
        self,
        event: PromptStreamEvent,
    ) -> PromptStreamMutation:
        """标记单个 prompt source 已结束。"""
        prompt_stream = self._ensure_prompt_stream_state()
        source = prompt_stream.sources.get(event.source_id)
        if source is None:
            if event.source_type is None:
                return PromptStreamMutation.noop(
                    self.num_prompt_tokens,
                    prompt_stream.revision,
                )
            source = self._get_or_create_prompt_source(event, event.source_type)

        source.finished = True
        source.last_arrival_time = event.arrival_time
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
        prompt_stream.suffix_token_ids = new_suffix_token_ids
        prompt_stream.finished = True
        prompt_stream.question_num_computed_tokens = 0
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
            else:
                prompt_stream.active_work_item = PromptStreamWorkItem(
                    phase=PromptStreamExecutionPhase.PREFIX_APPEND,
                    prefix_append_logical_len=prefix_len,
                    num_computed_tokens=0,
                )
                return

        # 一个 Update 快照可能只是上一代的严格前缀，此时只需要截断而不需要
        # forward。循环会同步提交这种零 token work，随后继续寻找真正需要计算
        # 的 work，避免把它留成永远无法 advance 的 active work。
        while prompt_stream.pending_work_items:
            work_item = prompt_stream.pending_work_items.popleft()
            if work_item.phase == PromptStreamExecutionPhase.PREFIX_APPEND:
                # 新 Append 从已提交的逻辑末尾开始写，复用该位置之后的旧 padding
                # 槽位。一个 Append work 开始后必须运行到新的 block 边界，不能
                # 在半个 block 内切给 Update。
                work_item.num_computed_tokens = (
                    prompt_stream.prefix_append_committed_logical_len
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

            # Update 在真正获调度时才冻结可见的 PREFIX_APPEND。此刻仍在队列
            # 里的 Append 尚未写入 KV，因此不会被拼到该次 Update 前缀中。
            committed_logical_len = prompt_stream.prefix_append_committed_logical_len
            work_item.prefix_append_logical_len = committed_logical_len
            work_item.prefix_append_token_ids = tuple(
                self._get_prefix_append_physical_token_ids(committed_logical_len)
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
                )

            # Update delta 基于获调度时冻结的 PREFIX_APPEND KV 和 work 自己
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
            )

        # 所有 source work 已提交后，最终 question 才能使用最终的物理布局。
        prefix_append_token_ids = self._get_prefix_append_physical_token_ids()
        update_token_ids = self._get_prompt_stream_segment_token_ids(
            PromptSourceType.UPDATE
        )
        question_token_ids = prompt_stream.suffix_token_ids
        prefix_append_physical_len = len(prefix_append_token_ids)

        # Question 与 Update 共用 UPDATE 物理域，避免制造第三个 partial-block
        # 边界。它只会在所有已确认 source work 完成后开始计算。
        if prompt_stream.finished and prompt_stream.question_num_computed_tokens < len(
            question_token_ids
        ):
            execution_token_ids = (
                prefix_append_token_ids + update_token_ids + question_token_ids
            )
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
                    + prompt_stream.question_num_computed_tokens
                ),
                prefix_append_physical_len=prefix_append_physical_len,
                segment_offset=prefix_append_physical_len,
            )

        # 没有待计算阶段时保留最终完整视图。未封口请求仍会被 scheduler 的
        # decode barrier 截住；封口请求则从该视图进入正常 decode 路径。
        if update_token_ids or question_token_ids:
            execution_token_ids = (
                prefix_append_token_ids + update_token_ids + question_token_ids
            )
            phase = (
                PromptStreamExecutionPhase.QUESTION
                if prompt_stream.finished
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
            )

        return PromptStreamExecutionView(
            phase=PromptStreamExecutionPhase.PREFIX_APPEND,
            write_segment=PromptStreamKVSegment.PREFIX_APPEND,
            required_segments=(PromptStreamKVSegment.PREFIX_APPEND,),
            token_ids=tuple(prefix_append_token_ids),
            num_computed_tokens=len(prefix_append_token_ids),
            prefix_append_physical_len=prefix_append_physical_len,
            segment_offset=0,
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
                    prompt_stream.active_work_item = None
        else:
            local_num_computed_tokens = max(
                0,
                new_num_computed_tokens
                - view.segment_offset
                - self.get_prompt_stream_update_len(),
            )
            prompt_stream.question_num_computed_tokens = min(
                local_num_computed_tokens,
                self.get_prompt_stream_question_len(),
            )

        self.num_computed_tokens = new_num_computed_tokens

    def reset_prompt_stream_computed_state(self) -> None:
        """在 preemption 后重建流式工作队列，但保留长期 source 状态。"""
        if self.prompt_stream is None:
            return
        prompt_stream = self.prompt_stream
        prompt_stream.prefix_append_num_computed_tokens = 0
        prompt_stream.update_num_computed_tokens = 0
        prompt_stream.update_committed_token_ids.clear()
        prompt_stream.question_num_computed_tokens = 0
        prompt_stream.prefix_append_committed_logical_len = 0
        prompt_stream.prefix_append_bootstrap_completed = False
        prompt_stream.active_work_item = None
        prompt_stream.pending_work_items.clear()
        prompt_stream.pending_update_kv_truncation_keep_tokens = None

        # Preemption 会释放两个物理域的 blocks。重新执行时不能重放已经丢失的
        # 到达交错，只能以当前逻辑快照构造安全顺序：Append 完整追平后再算 Update。
        if prompt_stream.append_token_ids:
            prompt_stream.pending_work_items.append(
                PromptStreamWorkItem(
                    phase=PromptStreamExecutionPhase.PREFIX_APPEND,
                    prefix_append_logical_len=(
                        self.get_prompt_stream_prefix_append_len()
                    ),
                )
            )
        update_token_ids = self._get_prompt_stream_segment_token_ids(
            PromptSourceType.UPDATE
        )
        if update_token_ids:
            prompt_stream.pending_work_items.append(
                PromptStreamWorkItem(
                    phase=PromptStreamExecutionPhase.UPDATE,
                    update_token_ids=tuple(update_token_ids),
                    update_common_prefix_len=0,
                )
            )

        self.prompt_stream_execution_view = None
        self.prompt_stream_needs_worker_refresh = True
        self._output_token_ids.clear()
        self.output_token_ids = ConstantList(self._output_token_ids)
        self.spec_token_ids.clear()
        self.num_computed_tokens = 0

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
