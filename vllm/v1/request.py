# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import enum
import time
from collections import deque
from collections.abc import Callable, Mapping
from dataclasses import dataclass
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
    #重复 chunk、旧版本 update、乱序 pending 等，没有实际改 prompt。
    applied: bool
    old_num_prompt_tokens: int
    new_num_prompt_tokens: int
    #新旧 prompt 的最长公共前缀。
    common_prefix_len: int
    #已经 prefill 过但现在失效的 token 数。
    num_tokens_to_invalidate: int
    #每次 prompt 真实变化都递增，后面防止 scheduler 用旧状态 decode。
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
            num_tokens_to_invalidate=0,
            prompt_revision=prompt_revision,
        )

@dataclass
class PromptSourceState:
    """保存单个 prompt source 的流式状态。"""
    #来源 ID。
    source_id: str
    #来源语义，创建后不应该被更改
    source_type: PromptSourceType
    #该来源当前有效 token。
    token_ids: list[int]
    #该来源当前有效版本。
    version: int
    #下一条期望收到的序号。
    next_sequence_no: int
    #已经应用过的 chunk，用于去重。
    seen_chunk_ids: set[str]
    #暂存乱序到达但还不能应用的事件；MVP 可以先不消费这个队列。
    pending_events: dict[int, PromptStreamEvent]
    #该来源是否结束。
    finished: bool
    #该来源最近一次事件到达时间。
    last_arrival_time: float

@dataclass
class PromptStreamState:
    """保存一个 Request 内部的多源流式 prompt 状态。"""
    #流式 source 之前的固定 prompt 前缀，例如 system prompt 或用户原始问题。
    prefix_token_ids: list[int]
    #从 source_id 到 PromptSourceState 的映射。
    sources: dict[str, PromptSourceState]
    #最终交给decode节点的排布顺序；TODO:这个有待思考
    source_order: list[str]
    #流式 source 之后的固定 prompt 后缀，后续可用于放 answer prefix。
    suffix_token_ids: list[int]
    #请求级流式 prompt 是否结束。
    finished: bool
    #prompt 每次真实变化后递增，用于防止 scheduler 或 model runner 使用旧状态 decode。
    revision: int
    #累计失效 token 数。
    total_tokens_invalidated: int
    #累计重算 token 数。
    total_tokens_recomputed: int

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
        return (
            self.prompt_stream is None
            or (
                self.prompt_stream.finished
                and self.num_computed_tokens >= self.num_prompt_tokens
            )
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

    def apply_prompt_stream_event(
        self,
        event: PromptStreamEvent,
    ) -> PromptStreamMutation:
        """应用一次流式 prompt 事件，并返回对调度层有用的变化摘要。

        这个方法是 scheduler 面向流式 prompt 的主入口。scheduler 不需要
        理解 append、update、source finished 的内部语义，只需要根据返回的
        `PromptStreamMutation` 决定是否释放失效 KV、是否继续 prefill，以及
        什么时候允许 decode。

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

        if event.event_type == PromptStreamEventType.APPEND:
            return self._apply_prompt_stream_append(event)

        if event.event_type == PromptStreamEventType.UPDATE:
            return self._apply_prompt_stream_update(event)

        if event.event_type == PromptStreamEventType.SOURCE_FINISHED:
            return self._mark_prompt_source_finished(event)

        if event.event_type == PromptStreamEventType.REQUEST_FINISHED:
            return self._mark_prompt_stream_finished(event)

        raise ValueError(f"Unsupported prompt stream event type: {event.event_type}")

    def _ensure_prompt_stream_state(self) -> PromptStreamState:
        """确保当前请求已经初始化流式 prompt 状态。"""
        if self.prompt_stream is None:
            self.prompt_stream = PromptStreamState(
                prefix_token_ids=self._current_prompt_token_ids(),
                sources={},
                source_order=[],
                suffix_token_ids=[],
                finished=False,
                revision=0,
                total_tokens_invalidated=0,
                total_tokens_recomputed=0,
            )
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
        if event.event_type in (
            PromptStreamEventType.APPEND,
            PromptStreamEventType.UPDATE,
        ) and event.token_ids is None:
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

        old_prompt_token_ids = self._current_prompt_token_ids()
        source.token_ids.extend(event.token_ids or [])
        source.version = max(source.version, event.version)
        source.next_sequence_no += 1
        source.last_arrival_time = event.arrival_time
        if event.chunk_id is not None:
            source.seen_chunk_ids.add(event.chunk_id)

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

        old_prompt_token_ids = self._current_prompt_token_ids()
        source.token_ids = list(event.token_ids or [])
        source.version = event.version
        source.next_sequence_no += 1
        source.last_arrival_time = event.arrival_time
        if event.chunk_id is not None:
            source.seen_chunk_ids.add(event.chunk_id)

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
        """标记请求级流式 prompt 输入已经结束。"""
        prompt_stream = self._ensure_prompt_stream_state()
        prompt_stream.finished = True
        return PromptStreamMutation.noop(
            self.num_prompt_tokens,
            prompt_stream.revision,
        )

    def _current_prompt_token_ids(self) -> list[int]:
        """返回当前 prompt token 的可变副本。"""
        if self.prompt_token_ids is None:
            return []
        return list(self.prompt_token_ids)

    def _rebuild_prompt_from_stream_state(self) -> list[int]:
        """按近似注意力布局重建全局 prompt token 序列。

        当前研究语义把 Append 和 Update 的 KV 分域理解为：

        - Append 源稳定累积，先放在全局 prompt 中。
        - Update 源按版本快照更新，放在 Append 源之后。

        因此 update 事件只会影响 update segment 的 KV；append 事件会扩展
        append segment，并使后面的 update segment 需要基于新的 append KV
        重新计算。
        """
        assert self.prompt_stream is not None
        prompt_token_ids = list(self.prompt_stream.prefix_token_ids)
        for source_id in self.prompt_stream.source_order:
            source = self.prompt_stream.sources[source_id]
            if source.source_type == PromptSourceType.APPEND:
                prompt_token_ids.extend(source.token_ids)
        for source_id in self.prompt_stream.source_order:
            source = self.prompt_stream.sources[source_id]
            if source.source_type == PromptSourceType.UPDATE:
                prompt_token_ids.extend(source.token_ids)
        prompt_token_ids.extend(self.prompt_stream.suffix_token_ids)
        return prompt_token_ids

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
        """把 PromptStreamState 同步回 Request 的调度字段。"""
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
        old_num_computed_tokens = self.num_computed_tokens
        num_tokens_to_invalidate = max(
            0,
            old_num_computed_tokens - common_prefix_len,
        )

        self.prompt_token_ids = new_prompt_token_ids
        self._all_token_ids = new_prompt_token_ids.copy()
        self.num_prompt_tokens = len(new_prompt_token_ids)
        self.num_computed_tokens = min(
            self.num_computed_tokens,
            common_prefix_len,
        )
        self.prompt_stream.revision += 1
        self.prompt_stream.total_tokens_invalidated += num_tokens_to_invalidate
        self.prompt_stream_needs_worker_refresh = True

        if common_prefix_len < old_num_prompt_tokens:
            self.block_hashes.clear()
            self._prompt_embeds_per_block_hashes.clear()
        self.update_block_hashes()

        return PromptStreamMutation(
            applied=True,
            old_num_prompt_tokens=old_num_prompt_tokens,
            new_num_prompt_tokens=self.num_prompt_tokens,
            common_prefix_len=common_prefix_len,
            num_tokens_to_invalidate=num_tokens_to_invalidate,
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
