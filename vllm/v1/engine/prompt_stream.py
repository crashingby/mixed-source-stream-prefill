# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import enum
from collections.abc import Mapping
from typing import Any

import msgspec


def parse_prompt_stream_padding_range(
    params: Mapping[str, Any] | None,
    num_prompt_tokens: int,
) -> tuple[int, int] | None:
    """校验 P/D manifest 中的 padding 坐标，返回物理半开区间。

    在 API 输入处理和 Request 构建时共用，避免坏 manifest 进入 worker。
    普通请求返回 None；即使区间为空，存在版本标记也表示启用了新语义。

    Args:
        params: 原样转发的 KV transfer metadata，也可包含 P-only 输出描述。
        num_prompt_tokens: D 输入的物理 token 数，包含可能追加的 P 首 token。

    Returns:
        经过校验的 (start, end)，或未启用时的 None。
    """
    if not params or "prompt_stream_masked_padding" not in params:
        return None
    marker = params["prompt_stream_masked_padding"]
    if not isinstance(marker, dict) or set(marker) != {"version", "start", "end"}:
        raise ValueError("Invalid prompt_stream_masked_padding manifest fields.")
    if type(marker["version"]) is not int or marker["version"] != 1:
        raise ValueError("Unsupported prompt_stream_masked_padding version.")
    start, end = marker["start"], marker["end"]
    physical_len = params.get("prompt_stream_physical_prompt_len")
    logical_len = params.get("prompt_stream_logical_prompt_len")
    output_tokens = params.get("prompt_stream_prefill_output_tokens", 0)
    # bool 也是 int 子类，传输长度必须排除这种歧义。
    if any(
        type(value) is not int
        for value in (start, end, physical_len, logical_len, output_tokens)
    ):
        raise ValueError("Masked padding bounds and lengths must be integers.")
    if not 0 <= start <= end <= physical_len:
        raise ValueError("Masked padding range is outside the physical prompt.")
    if logical_len != physical_len - (end - start):
        raise ValueError("Masked padding logical/physical prompt lengths disagree.")
    if output_tokens not in (0, 1) or num_prompt_tokens != physical_len + output_tokens:
        raise ValueError(
            "Masked padding prompt length does not match the P token tail: "
            f"decode_prompt_tokens={num_prompt_tokens}, "
            f"physical_prompt_tokens={physical_len}, "
            f"prefill_output_tokens={output_tokens}."
        )
    return start, end


class PromptStreamEventType(str, enum.Enum):
    """PromptStreamEventType 表示这次事件到底是什么操作。"""
    APPEND = "append"
    UPDATE = "update"
    SOURCE_FINISHED = "source_finished"
    PROMPT_FINISHED = "prompt_finished"


class PromptSourceType(str, enum.Enum):
    """PromptSourceType 表示某个来源采用哪种语义。"""
    APPEND = "append"
    UPDATE = "update"


class PromptStreamEvent(msgspec.Struct, omit_defaults=True):
    """PromptStreamEvent 是传输层事件，回答的是：这一次来了什么？"""

    # Engine 内部稳定请求 ID；同一混合流请求的所有事件必须一致
    request_id: str
    # 同一请求内的来源实例，例如 web_search_0、anns_0
    source_id: str
    # 区分数据、单来源结束和请求输入流结束，避免多个布尔组合出非法状态
    event_type: PromptStreamEventType
    # 来源采用 Append 还是 Update 语义；创建后不可改变
    source_type: PromptSourceType | None
    # 事件幂等键；同一来源内必须唯一，重试发送不得重复应用
    chunk_id: str | None
    # 来源内容版本；Append 可等于 sequence_no，Update 必须严格递增
    version: int
    # 来源内传输顺序，从 0 或 1 单调递增，用于检测缺口和乱序
    sequence_no: int
    # Append 为新增 token；Update 为该来源当前版本的完整 token 快照
    token_ids: list[int] | None
    # 前端观察到内容可用的时间，使用单调时钟计算排队与 debounce 延迟
    arrival_time: float
