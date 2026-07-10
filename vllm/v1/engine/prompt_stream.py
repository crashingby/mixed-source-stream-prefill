# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import enum

import msgspec


class PromptStreamEventType(str, enum.Enum):
    """PromptStreamEventType 表示这次事件到底是什么操作。"""
    APPEND = "append"
    UPDATE = "update"
    SOURCE_FINISHED = "source_finished"
    REQUEST_FINISHED = "request_finished"


class PromptSourceType(str, enum.Enum):
    """PromptSourceType 表示某个来源采用哪种语义。"""
    APPEND = "append"
    UPDATE = "update"

class PromptStreamEvent(msgspec.Struct, omit_defaults=True):
    """PromptStreamEvent 是传输层事件，回答的是：这一次来了什么？"""

    #Engine 内部稳定请求 ID；同一混合流请求的所有事件必须一致
    request_id: str
    #同一请求内的来源实例，例如 web_search_0、anns_0
    source_id: str
    #区分数据、单来源结束和请求输入流结束，避免多个布尔组合出非法状态
    event_type: PromptStreamEventType
    #来源采用 Append 还是 Update 语义；创建后不可改变
    source_type: PromptSourceType | None
    #事件幂等键；同一来源内必须唯一，重试发送不得重复应用
    chunk_id: str | None
    #来源内容版本；Append 可等于 sequence_no，Update 必须严格递增
    version: int
    #来源内传输顺序，从 0 或 1 单调递增，用于检测缺口和乱序
    sequence_no: int
    #Append 为新增 token；Update 为该来源当前版本的完整 token 快照
    token_ids: list[int] | None
    #前端观察到内容可用的时间，使用单调时钟计算排队与 debounce 延迟
    arrival_time: float