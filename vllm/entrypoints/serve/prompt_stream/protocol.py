# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import Any, Literal

from pydantic import Field

from vllm.entrypoints.openai.engine.protocol import OpenAIBaseModel


class PromptStreamEventRequest(OpenAIBaseModel):
    """RESTFUL 层的单次 prompt stream 事件。

    RESTFUL 层允许 `text` / `token_ids` 双模式。进入 EngineCore 之前，P
    节点 serving 层会把 `text` 归一化为 `token_ids`。
    """

    source_id: str
    event_type: Literal[
        "append",
        "update",
        "source_finished",
        "prompt_finished",
    ]
    source_type: Literal["append", "update"] | None = None
    chunk_id: str | None = None
    version: int = 0
    sequence_no: int = 0
    text: str | None = None
    token_ids: list[int] | None = None
    arrival_time: float | None = None


class PromptStreamStartRequest(OpenAIBaseModel):
    """创建 prompt stream 会话的请求。

    `prompt` 是固定 prefix/template。其它采样字段可以直接放在请求顶层，
    也可以放入 `sampling_params`。
    """

    model: str | None = None
    prompt: str | list[int]
    request_id: str | None = None
    stream: bool = True
    sampling_params: dict[str, Any] = Field(default_factory=dict)

    def get_sampling_params_dict(self) -> dict[str, Any]:
        """合并顶层扩展字段和 `sampling_params` 字段。"""
        ignored_fields = {
            "api",
            "model",
            "prompt",
            "request_id",
            "stream",
            "sampling_params",
        }
        params = dict(self.sampling_params)
        for key, value in (self.model_extra or {}).items():
            if key not in ignored_fields:
                params[key] = value
        return params


class PromptStreamEventEnvelope(OpenAIBaseModel):
    """带 request_id 的 prompt stream 事件请求。"""

    request_id: str
    event: PromptStreamEventRequest


class PromptStreamStartResponse(OpenAIBaseModel):
    """prompt stream start 的 ack 响应。"""

    request_id: str
    status: Literal["started"] = "started"
    token_ids: list[int] | None = None


class PromptStreamEventResponse(OpenAIBaseModel):
    """prompt stream event 的 ack 响应。"""

    request_id: str
    status: Literal["accepted"] = "accepted"
    event_type: str
    token_ids: list[int] | None = None


class PromptStreamFinishResponse(OpenAIBaseModel):
    """prompt stream finish 的 P 节点响应。"""

    request_id: str
    status: Literal["prefill_finished"] = "prefill_finished"
    kv_transfer_params: dict[str, Any]
    token_ids: list[int] | None = None
    prompt_token_ids: list[int] | None = None
    output_token_ids: list[int] | None = None
    output_text: str = ""
    output_finish_reason: str | None = None
    output_stop_reason: int | str | None = None
    # 统计与 KV 传输参数相互独立，由 proxy 原样返回给实验客户端。
    prompt_stream_stats: dict[str, Any] | None = None
