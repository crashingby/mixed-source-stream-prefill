# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""失效统计实验的布局、计划事件和请求结果。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

from research.scripts.replay.common.models import RequestMetric

Layout = Literal["update_first", "append_first", "segmented"]
WHOLE_SOURCE_ID = "whole_context"


@dataclass(frozen=True)
class PlannedEvent:
    """一个原始数据事件；布局只改变正文建模，不改变计划到达时间。"""

    offset_s: float  # 相对 start ACK 的秒数。
    source_id: str  # 原始来源 ID，用于跨模式配对。
    source_type: Literal["append", "update"]
    event: dict[str, Any]


@dataclass(frozen=True)
class OutboundEvent:
    """原始输入与实际 REST 事件的对应关系，不包含运行时 arrival_time。"""

    planned: PlannedEvent
    event: dict[str, Any]


@dataclass
class InvalidationMetric(RequestMetric):
    """保留公共服务端计量，附加失效统计的主指标及输入指纹。"""

    input_sha256: str = ""  # 冻结整个输入样本，包含所有中间快照。
    plan_sha256: str = ""  # 所有原始事件的内容、顺序及计划时间。
    evidence_sha256: str = ""  # 最终 Append、Update、问题，忽略布局次序。
    logical_prompt_sha256: str = ""  # 实际布局下最终文本的指纹。
    original_append_chunks: int = 0
    original_update_snapshots: int = 0
    outbound_append_events: int = 0
    outbound_update_events: int = 0
    changed_update_events: int = 0  # 同快照 ACK 不产生新的服务端 admission。
    source_finished_events: int = 0
    potential_replaced_suffix_tokens: int = 0  # ACK LCP 的逻辑后缀，仅作诊断。
    snapshot_invalidated_tokens: int | None = None  # 只保存正式有效的服务端主指标。
    preemption_discarded_tokens: int | None = None  # 抢占丢弃单列，不混入主指标。
