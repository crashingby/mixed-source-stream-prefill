# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import asyncio
import copy
import time
from dataclasses import dataclass, field
from inspect import signature
from typing import Any

from vllm.engine.protocol import EngineClient
from vllm.entrypoints.serve.prompt_stream.protocol import (
    PromptStreamEventEnvelope,
    PromptStreamEventRequest,
    PromptStreamEventResponse,
    PromptStreamFinishResponse,
    PromptStreamStartRequest,
    PromptStreamStartResponse,
)
from vllm.outputs import RequestOutput
from vllm.sampling_params import SamplingParams
from vllm.v1.engine import EngineCoreRequest
from vllm.v1.engine.output_processor import RequestOutputCollector
from vllm.v1.engine.prompt_stream import (
    PromptSourceType,
    PromptStreamEvent,
    PromptStreamEventType,
)
from vllm.v1.request import Request


@dataclass
class PromptStreamSession:
    """P 节点上的单个 prompt stream 会话状态。"""

    request_id: str
    collector: RequestOutputCollector
    sampling_params: SamplingParams
    length_request: Request
    created_time: float
    last_event_time: float
    finishing: bool = False
    aborting: bool = False
    # Append 与 Update 来源可以并发调用 REST endpoint。长度镜像和 EngineCore
    # 入队必须使用同一串行顺序，避免两个 handler 从同一个旧快照同时提交。
    event_lock: asyncio.Lock = field(default_factory=asyncio.Lock, repr=False)


@dataclass(frozen=True)
class PromptStreamPrefillOutput:
    """P 节点 finish 时交给 proxy 的首 token 和可选 transfer 元数据。"""

    kv_transfer_params: dict[str, Any]
    token_ids: list[int]
    text: str
    finish_reason: str | None
    stop_reason: int | str | None
    prompt_stream_stats: dict[str, Any] | None


class PromptStreamServing:
    """P 节点 prompt stream RESTFUL 服务逻辑。

    该类只管理 P 节点本地会话状态。proxy 负责把同一个 request id 的
    所有事件固定路由到同一个 P 节点。
    """

    def __init__(self, engine_client: EngineClient) -> None:
        self.engine_client = engine_client
        self.sessions: dict[str, PromptStreamSession] = {}

    async def start(
        self,
        request: PromptStreamStartRequest,
    ) -> PromptStreamStartResponse:
        """创建流式 prompt 会话，并注册 output collector。"""
        request_id = request.request_id or f"prompt-stream-{time.time_ns()}"
        sampling_params = self._build_prefill_sampling_params(
            request.get_sampling_params_dict()
        )

        # start 阶段使用普通输入处理链路，确保 prefix/template 按模型 tokenizer
        # 和现有 vLLM prompt 规则处理。
        prefix_token_ids = self._resolve_start_prompt_token_ids(
            request.prompt, add_special_tokens=request.add_special_tokens
        )
        probe_token_ids = None
        if request.probe_query is not None:
            probe_token_ids = self._resolve_token_ids(
                PromptStreamEventRequest(
                    source_id="__probe__",
                    event_type="prompt_finished",
                    text=request.probe_query,
                )
            )
            if not probe_token_ids:
                raise ValueError("probe_query must contain at least one token.")
            config = self.engine_client.vllm_config
            backend = config.attention_config.backend
            if backend is None or backend.name != "CHUNK_SPARSE_ATTN":
                raise ValueError(
                    "Query probe requires --attention-backend CHUNK_SPARSE_ATTN."
                )
            # 首版使用 eager 的单 GPU Triton 路径，避免图捕获缓存动态策略。
            if (
                not config.model_config.enforce_eager
                or config.parallel_config.tensor_parallel_size != 1
                or not config.scheduler_config.prompt_stream_masked_padding
                or self._block_size() != 16
                or config.cache_config.cache_dtype != "auto"
            ):
                raise ValueError(
                    "Query probe requires --enforce-eager, TP=1 and "
                    "--prompt-stream-masked-padding, block_size=16 and auto KV dtype."
                )
        elif request.probe_keep_ratio is not None:
            raise ValueError("probe_keep_ratio requires probe_query.")
        start_required_len = self._align_to_block_size(len(prefix_token_ids)) + 1
        if probe_token_ids is not None:
            start_required_len += len(probe_token_ids)
        if start_required_len > self._max_model_len():
            raise ValueError(
                "Prompt stream start exceeds the P node model limit after "
                "segment alignment and reserving the P output token: "
                f"prefix_tokens={len(prefix_token_ids)}, "
                f"required_sequence_tokens={start_required_len}, "
                f"max_model_len={self._max_model_len()}."
            )
        collector = await self.engine_client.add_prompt_stream_request(
            request_id=request_id,
            prompt=request.prompt,
            sampling_params=sampling_params,
            arrival_time=time.time(),
            # 实际 engine 分词必须与长度镜像使用同一标记。
            tokenization_kwargs={"add_special_tokens": request.add_special_tokens},
            probe_token_ids=probe_token_ids,
            probe_keep_ratio=(
                request.probe_keep_ratio
                if request.probe_keep_ratio is not None
                else 0.5
            ),
        )

        # `add_request()` 会把外部 request id 随机化为内部 request id。后续
        # prompt stream 事件必须使用内部 id 才能命中 scheduler 中的长期 Request。
        internal_request_id = collector.request_id
        now = time.time()
        length_request = self._build_length_request(
            request_id=internal_request_id,
            prefix_token_ids=prefix_token_ids,
            sampling_params=sampling_params,
        )
        if probe_token_ids is not None:
            from vllm.v1.query_probe import QueryProbeState

            length_request.query_probe = QueryProbeState(
                probe_token_ids,
                request.probe_keep_ratio
                if request.probe_keep_ratio is not None
                else 0.5,
            )
        self.sessions[internal_request_id] = PromptStreamSession(
            request_id=internal_request_id,
            collector=collector,
            sampling_params=sampling_params,
            length_request=length_request,
            created_time=now,
            last_event_time=now,
        )
        return PromptStreamStartResponse(
            request_id=internal_request_id,
            token_ids=prefix_token_ids,
        )

    async def add_event(
        self,
        envelope: PromptStreamEventEnvelope,
    ) -> PromptStreamEventResponse:
        """应用 Append / Update / Source Finished 事件并返回 ack。"""
        session = self._get_session(envelope.request_id)
        async with session.event_lock:
            if session.finishing:
                raise ValueError(
                    "Prompt stream finish is already in progress for "
                    f"request_id={session.request_id}."
                )
            event_request = envelope.event
            if event_request.event_type == PromptStreamEventType.PROMPT_FINISHED.value:
                raise ValueError("prompt_finished must use /v1/prompt_stream/finish.")
            prompt_stream_event = self._build_prompt_stream_event(
                request_id=session.request_id,
                event_request=event_request,
            )
            length_request = await self._validate_length_event(
                session,
                prompt_stream_event,
            )
            engine_core_request = self._build_event_engine_core_request(
                session=session,
                prompt_stream_event=prompt_stream_event,
            )

            # event 只进入 EngineCore，不注册新的 output collector。只有入队
            # 成功后才提交长度镜像，失败时仍可安全重试同一个事件。
            await self.engine_client.add_prompt_stream_event(engine_core_request)
            session.length_request = length_request
            session.last_event_time = time.time()
        return PromptStreamEventResponse(
            request_id=session.request_id,
            event_type=event_request.event_type,
            token_ids=prompt_stream_event.token_ids,
        )

    async def finish(
        self,
        envelope: PromptStreamEventEnvelope,
    ) -> PromptStreamFinishResponse:
        """提交 PROMPT_FINISHED，并等待 P 节点返回首 token 和可选 KV 元数据。"""
        session = self._get_session(envelope.request_id)
        async with session.event_lock:
            if session.finishing:
                raise ValueError(
                    "Prompt stream finish is already in progress for "
                    f"request_id={session.request_id}."
                )
            session.finishing = True
            try:
                if (
                    envelope.event.event_type
                    != PromptStreamEventType.PROMPT_FINISHED.value
                ):
                    raise ValueError(
                        "prompt_stream finish endpoint requires "
                        "event_type=prompt_finished, "
                        f"but got {envelope.event.event_type}."
                    )
                event_request = envelope.event
                prompt_stream_event = self._build_prompt_stream_event(
                    request_id=session.request_id,
                    event_request=event_request,
                )
                if not prompt_stream_event.token_ids:
                    raise ValueError(
                        "prompt_stream finish requires a non-empty final question."
                    )
                length_request = await self._validate_length_event(
                    session,
                    prompt_stream_event,
                )
                engine_core_request = self._build_event_engine_core_request(
                    session=session,
                    prompt_stream_event=prompt_stream_event,
                )

                # finish 仍然是 core-only 入队；输出从 start 阶段保存的
                # collector 读取。入队完成后不能重试 PROMPT_FINISHED。
                await self.engine_client.add_prompt_stream_event(engine_core_request)
                session.length_request = length_request
                session.last_event_time = time.time()
            except Exception:
                if session.request_id in self.sessions and not session.aborting:
                    session.finishing = False
                raise

        try:
            prefill_output = await self._wait_for_finish_output(session)
            # 无 NIXL 的 TTFT-only 输出也携带语义布局。复制后再取出 token
            # view，避免修改 collector 保存的共享结果或意外丢掉 mask 区间。
            kv_transfer_params = dict(prefill_output.kv_transfer_params)
            scheduler_config = self.engine_client.vllm_config.scheduler_config
            if (
                scheduler_config.prompt_stream_masked_padding
                and "prompt_stream_masked_padding" not in kv_transfer_params
            ):
                raise ValueError(
                    "Masked-padding finish output is missing its semantic manifest: "
                    f"request_id={session.request_id}."
                )
            physical_prompt_token_ids = kv_transfer_params.pop(
                "prompt_stream_prompt_token_ids",
                None,
            )
            response = PromptStreamFinishResponse(
                request_id=session.request_id,
                kv_transfer_params=kv_transfer_params,
                token_ids=prompt_stream_event.token_ids,
                prompt_token_ids=physical_prompt_token_ids,
                output_token_ids=prefill_output.token_ids,
                output_text=prefill_output.text,
                output_finish_reason=prefill_output.finish_reason,
                output_stop_reason=prefill_output.stop_reason,
                prompt_stream_stats=prefill_output.prompt_stream_stats,
            )
            self.sessions.pop(session.request_id, None)
            return response
        except Exception:
            # PROMPT_FINISHED 已经入队，不能通过第二次 finish 重试。保留
            # finishing 状态和 session，调用方仍可走幂等 abort 做最终清理。
            raise

    async def abort(self, request_id: str) -> None:
        """取消 prompt stream 会话并释放本地 session 状态。"""
        session = self.sessions.get(request_id)
        if session is not None:
            async with session.event_lock:
                session.finishing = True
                session.aborting = True
                session.collector.close()
                # Session key comes from collector.request_id and is already
                # the EngineCore internal ID. External-ID abort would be a no-op.
                await self.engine_client.abort(request_id, internal=True)
                self.sessions.pop(request_id, None)
            return
        await self.engine_client.abort(request_id, internal=True)

    async def _validate_length_event(
        self,
        session: PromptStreamSession,
        event: PromptStreamEvent,
    ) -> Request:
        """构造并验证事件应用后的候选长度镜像。

        REST ack 只表示事件已进入 EngineCore 队列，并不表示 scheduler 已经
        应用它。因此长度检查必须放在 serving 层，避免超长 Request 被 core
        终止后，同 ID 的后续事件再次创建一个新的 Request。候选镜像仅在
        EngineCore 入队成功后由调用方提交，入队异常不会污染长期状态。
        """
        length_request = copy.deepcopy(session.length_request)
        length_request.apply_prompt_stream_event(event)
        layout = length_request.get_prompt_stream_layout()
        required_sequence_len = layout.physical_total_len + 1
        if length_request.query_probe is not None:
            probe_len = (
                self._align_to_block_size(layout.prefix_len + layout.append_len)
                + len(length_request.query_probe.token_ids)
                + 1
            )
            required_sequence_len = max(required_sequence_len, probe_len)
        max_model_len = self._max_model_len()
        if required_sequence_len <= max_model_len:
            return length_request

        session.finishing = True
        session.aborting = True
        session.collector.close()
        await self.engine_client.abort(session.request_id, internal=True)
        self.sessions.pop(session.request_id, None)
        raise ValueError(
            "Prompt stream event exceeds the P node model limit after segment "
            "alignment and reserving the P output token; the session was "
            f"aborted: request_id={session.request_id}, "
            f"physical_prompt_tokens={layout.physical_total_len}, "
            f"required_sequence_tokens={required_sequence_len}, "
            f"max_model_len={max_model_len}."
        )

    def _get_session(self, request_id: str) -> PromptStreamSession:
        """读取 P 节点本地 prompt stream 会话。"""
        try:
            return self.sessions[request_id]
        except KeyError as err:
            raise ValueError(f"Unknown prompt stream request_id={request_id}.") from err

    def _build_prefill_sampling_params(
        self,
        raw_params: dict[str, Any],
    ) -> SamplingParams:
        """构造 P 节点 prefill 使用的 SamplingParams。

        用户请求中的 decode 参数会保存在 proxy 的 base request 里给 D 节点使用。
        P 节点只需要完成 prefill 并产出首个 output / `kv_transfer_params`，
        因此这里强制 `max_tokens=1`，并把非 SamplingParams 原生字段放入
        `extra_args`，例如 `kv_transfer_params`。
        """
        params = dict(raw_params)
        if "max_completion_tokens" in params and "max_tokens" not in params:
            params["max_tokens"] = params.pop("max_completion_tokens")

        # P 节点只负责 remote decode 前的 prefill 和首 token 协调。
        params["max_tokens"] = 1
        params["min_tokens"] = 0

        sampling_param_names = set(signature(SamplingParams).parameters)
        sampling_kwargs: dict[str, Any] = {}
        extra_args = dict(params.pop("extra_args", {}) or {})
        for key, value in params.items():
            if key in sampling_param_names:
                sampling_kwargs[key] = value
            else:
                extra_args[key] = value
        if extra_args:
            sampling_kwargs["extra_args"] = extra_args
        return SamplingParams(**sampling_kwargs, skip_clone=True)

    def _build_length_request(
        self,
        request_id: str,
        prefix_token_ids: list[int],
        sampling_params: SamplingParams,
    ) -> Request:
        """构造只在 REST 进程中维护的无 KV Request 长度镜像。"""
        request = Request(
            request_id=request_id,
            prompt_token_ids=list(prefix_token_ids),
            sampling_params=sampling_params.clone(),
            pooling_params=None,
        )
        request.apply_prompt_stream_event(
            PromptStreamEvent(
                request_id=request_id,
                source_id="__prompt_stream_start__",
                event_type=PromptStreamEventType.APPEND,
                source_type=PromptSourceType.APPEND,
                chunk_id="__prompt_stream_start__",
                version=0,
                sequence_no=0,
                token_ids=[],
                arrival_time=time.time(),
            )
        )
        assert request.prompt_stream is not None
        request.prompt_stream.block_size = self._block_size()
        return request

    def _block_size(self) -> int:
        """返回 P 节点 scheduler 使用的逻辑 KV block size。"""
        return int(self.engine_client.vllm_config.cache_config.block_size)

    def _max_model_len(self) -> int:
        """返回 P 节点实际配置的最大序列长度。"""
        return int(self.engine_client.vllm_config.model_config.max_model_len)

    def _align_to_block_size(self, num_tokens: int) -> int:
        """把 token 数向上对齐到 P 节点 KV block 边界。"""
        block_size = self._block_size()
        return (num_tokens + block_size - 1) // block_size * block_size

    def _resolve_start_prompt_token_ids(
        self, prompt: str | list[int], *, add_special_tokens: bool = True
    ) -> list[int]:
        """把 start 阶段 prefix/template 归一化为 P 节点实际 token ids。

        Args:
            prompt: RESTFUL start 请求里的原始 prompt。
            add_special_tokens: 已渲染模型模板时关闭，避免重复 BOS。

        Returns:
            P 节点用于初始化长期 Request 的 prefix token ids。
        """
        if isinstance(prompt, list):
            return list(prompt)

        tokenizer = self.engine_client.input_processor.get_tokenizer()
        try:
            return list(tokenizer.encode(prompt, add_special_tokens=add_special_tokens))
        except TypeError:
            return list(tokenizer.encode(prompt))

    def _resolve_token_ids(
        self,
        event_request: PromptStreamEventRequest,
    ) -> list[int] | None:
        """把 RESTFUL 事件内容归一化为 token ids。"""
        if event_request.token_ids is not None:
            return list(event_request.token_ids)
        if event_request.text is None:
            if event_request.event_type == PromptStreamEventType.SOURCE_FINISHED.value:
                return None
            raise ValueError("Prompt stream event must carry either token_ids or text.")

        tokenizer = self.engine_client.input_processor.get_tokenizer()
        try:
            return list(
                tokenizer.encode(
                    event_request.text,
                    add_special_tokens=False,
                )
            )
        except TypeError:
            return list(tokenizer.encode(event_request.text))

    def _build_prompt_stream_event(
        self,
        request_id: str,
        event_request: PromptStreamEventRequest,
    ) -> PromptStreamEvent:
        """把 RESTFUL 事件转换成 EngineCoreRequest 内部事件。"""
        source_type = (
            PromptSourceType(event_request.source_type)
            if event_request.source_type is not None
            else None
        )
        return PromptStreamEvent(
            request_id=request_id,
            source_id=event_request.source_id,
            event_type=PromptStreamEventType(event_request.event_type),
            source_type=source_type,
            chunk_id=event_request.chunk_id,
            version=event_request.version,
            sequence_no=event_request.sequence_no,
            token_ids=self._resolve_token_ids(event_request),
            arrival_time=event_request.arrival_time or time.time(),
        )

    def _build_event_engine_core_request(
        self,
        session: PromptStreamSession,
        prompt_stream_event: PromptStreamEvent,
    ) -> EngineCoreRequest:
        """构造只携带 prompt stream event 的 EngineCoreRequest。"""
        return EngineCoreRequest(
            request_id=session.request_id,
            prompt_token_ids=[],
            mm_features=None,
            sampling_params=session.sampling_params.clone(),
            pooling_params=None,
            arrival_time=prompt_stream_event.arrival_time,
            lora_request=None,
            cache_salt=None,
            data_parallel_rank=None,
            prompt_stream_event=prompt_stream_event,
        )

    async def _wait_for_finish_output(
        self,
        session: PromptStreamSession,
    ) -> PromptStreamPrefillOutput:
        """等待 P 节点返回首个输出 token 和可选 KV transfer 元数据。

        Returns:
            P 节点为了完成 PD prefill 采样出的首个 decode token、文本、停止
            原因、可选 connector 参数以及 Prompt Stream 最终统计。

        Notes:
            `finish_ttft` 不会把请求交给 D 节点，因此 P 可在未启用 KV
            connector 时正常完成。完整 P/D 路径是否要求 transfer 参数由
            proxy 的 `/v1/prompt_stream/finish` 路由单独校验。
        """
        while True:
            output = await session.collector.get()
            if not isinstance(output, RequestOutput):
                raise ValueError(
                    "Prompt stream finish expected RequestOutput, "
                    f"but got {type(output).__name__}."
                )
            # 有 transfer 参数时，P/D 与 TTFT-only 都可直接使用该完成输出。
            # 没有参数时，只有 Request 已 finished 才表示无 connector 的 P-only
            # prefill 已正常结束；中间 streaming 输出仍需继续等待最终输出。
            if output.kv_transfer_params or output.finished:
                if len(output.outputs) != 1:
                    raise ValueError(
                        "Prompt stream finish requires exactly one completion "
                        f"branch, but got {len(output.outputs)}."
                    )
                completion_output = output.outputs[0]
                return PromptStreamPrefillOutput(
                    # 无 connector 时可能只有 padding 的语义清单；有字典
                    # 不等于存在可加载的远端 KV，proxy 另行检查 transfer 标记。
                    kv_transfer_params=output.kv_transfer_params or {},
                    token_ids=list(completion_output.token_ids),
                    text=completion_output.text,
                    finish_reason=completion_output.finish_reason,
                    stop_reason=completion_output.stop_reason,
                    prompt_stream_stats=output.prompt_stream_stats,
                )
