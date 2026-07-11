# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import time
from dataclasses import dataclass
from inspect import signature
from typing import Any

from vllm.engine.protocol import EngineClient
from vllm.outputs import RequestOutput
from vllm.sampling_params import SamplingParams
from vllm.v1.engine import EngineCoreRequest
from vllm.v1.engine.output_processor import RequestOutputCollector
from vllm.v1.engine.prompt_stream import (
    PromptSourceType,
    PromptStreamEvent,
    PromptStreamEventType,
)
from vllm.entrypoints.serve.prompt_stream.protocol import (
    PromptStreamEventEnvelope,
    PromptStreamEventRequest,
    PromptStreamFinishResponse,
    PromptStreamStartRequest,
    PromptStreamStartResponse,
)


@dataclass
class PromptStreamSession:
    """P 节点上的单个 prompt stream 会话状态。"""

    request_id: str
    collector: RequestOutputCollector
    sampling_params: SamplingParams
    created_time: float
    last_event_time: float


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
        collector = await self.engine_client.add_prompt_stream_request(
            request_id=request_id,
            prompt=request.prompt,
            sampling_params=sampling_params,
            arrival_time=time.time(),
        )

        # `add_request()` 会把外部 request id 随机化为内部 request id。后续
        # prompt stream 事件必须使用内部 id 才能命中 scheduler 中的长期 Request。
        internal_request_id = collector.request_id
        now = time.time()
        self.sessions[internal_request_id] = PromptStreamSession(
            request_id=internal_request_id,
            collector=collector,
            sampling_params=sampling_params,
            created_time=now,
            last_event_time=now,
        )
        return PromptStreamStartResponse(request_id=internal_request_id)

    async def add_event(
        self,
        envelope: PromptStreamEventEnvelope,
    ) -> None:
        """应用 Append / Update / Source Finished 事件并返回 ack。"""
        session = self._get_session(envelope.request_id)
        event_request = envelope.event
        prompt_stream_event = self._build_prompt_stream_event(
            request_id=session.request_id,
            event_request=event_request,
        )
        engine_core_request = self._build_event_engine_core_request(
            session=session,
            prompt_stream_event=prompt_stream_event,
        )

        # event 只进入 EngineCore，不注册新的 output collector。
        await self.engine_client.add_prompt_stream_event(engine_core_request)
        session.last_event_time = time.time()

    async def finish(
        self,
        envelope: PromptStreamEventEnvelope,
    ) -> PromptStreamFinishResponse:
        """提交 PROMPT_FINISHED，并等待 P 节点返回 kv_transfer_params。"""
        session = self._get_session(envelope.request_id)
        if envelope.event.event_type != PromptStreamEventType.PROMPT_FINISHED.value:
            raise ValueError(
                "prompt_stream finish endpoint requires event_type=prompt_finished, "
                f"but got {envelope.event.event_type}."
            )

        event_request = envelope.event
        prompt_stream_event = self._build_prompt_stream_event(
            request_id=session.request_id,
            event_request=event_request,
        )
        engine_core_request = self._build_event_engine_core_request(
            session=session,
            prompt_stream_event=prompt_stream_event,
        )

        # finish 仍然是 core-only 入队；输出从 start 阶段保存的 collector 读取。
        await self.engine_client.add_prompt_stream_event(engine_core_request)
        session.last_event_time = time.time()

        kv_transfer_params = await self._wait_for_kv_transfer_params(session)
        return PromptStreamFinishResponse(
            request_id=session.request_id,
            kv_transfer_params=kv_transfer_params,
        )

    async def abort(self, request_id: str) -> None:
        """取消 prompt stream 会话并释放本地 session 状态。"""
        session = self.sessions.pop(request_id, None)
        if session is not None:
            session.collector.close()
        await self.engine_client.abort(request_id)

    def _get_session(self, request_id: str) -> PromptStreamSession:
        """读取 P 节点本地 prompt stream 会话。"""
        try:
            return self.sessions[request_id]
        except KeyError as err:
            raise ValueError(
                f"Unknown prompt stream request_id={request_id}."
            ) from err

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
            raise ValueError(
                "Prompt stream event must carry either token_ids or text."
            )

        tokenizer = self.engine_client.input_processor.get_tokenizer()
        try:
            return list(tokenizer.encode(
                event_request.text,
                add_special_tokens=False,
            ))
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

    async def _wait_for_kv_transfer_params(
        self,
        session: PromptStreamSession,
    ) -> dict[str, Any]:
        """等待 P 节点首个携带 kv_transfer_params 的输出。"""
        while True:
            output = await session.collector.get()
            if not isinstance(output, RequestOutput):
                raise ValueError(
                    "Prompt stream finish expected RequestOutput, "
                    f"but got {type(output).__name__}."
                )
            if output.kv_transfer_params:
                return output.kv_transfer_params
            if output.finished:
                raise ValueError(
                    "Prompt stream finished without kv_transfer_params for "
                    f"request_id={session.request_id}."
                )
