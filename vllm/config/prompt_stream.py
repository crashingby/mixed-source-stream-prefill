# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import Literal

from pydantic import Field

from vllm.config.utils import config

PromptStreamAppendGate = Literal[
    "none",
    "hold_until_append_finished",
    "hold_until_append_or_update_finished",
]
PromptStreamUpdatePolicy = Literal[
    "immediate",
    "fixed_window",
    "quiet_debounce",
    "adaptive",
    "full_merge",
]


@config
class PromptStreamUpdateCoalescingConfig:
    """流式 Prompt Update 合并配置。

    传入该配置即启用 Update 合并；配置为 ``None`` 时，PromptStream 保持
    每个 Update 事件立即创建 work 的原有行为。Append gate 仅控制已满足
    Update policy 的窗口能否 admission，不影响窗口统计与最新快照覆盖。
    """

    append_gate: PromptStreamAppendGate = "hold_until_append_finished"
    """Append 未封闭时是否允许把 Update 窗口 admission 为 work。"""

    update_policy: PromptStreamUpdatePolicy = "fixed_window"
    """决定 Update 窗口何时满足 flush 条件的策略。"""

    soft_window_ms: float = Field(default=20.0, ge=0.0)
    """固定窗口、静默窗口或自适应候选等待时间，单位为毫秒。"""

    hard_deadline_ms: float = Field(default=80.0, gt=0.0)
    """Update 窗口允许等待的最长时间，单位为毫秒。"""

    ttft_deadline_ms: float | None = Field(default=None, gt=0.0)
    """从 Request start 开始计算的可选 TTFT 总预算，单位为毫秒。"""

    interval_ewma_alpha: float = Field(default=0.25, gt=0.0, le=1.0)
    """Update 到达间隔 EWMA 的更新系数。"""

    gain_ewma_alpha: float = Field(default=0.25, gt=0.0, le=1.0)
    """Update 合并收益 EWMA 的更新系数。"""

    cold_start_prior_weight: float = Field(default=8.0, ge=0.0)
    """自适应策略冷启动先验的等效观测权重。"""

    arrival_rate_prior_per_s: float | None = Field(default=None, gt=0.0)
    """可选的 Update 到达率冷启动先验，单位为每秒事件数。"""

    merge_gain_prior_tokens: float | None = Field(default=None, ge=0.0)
    """可选的单次合并 token 收益冷启动先验。"""

    delay_price: float = Field(default=1.0, ge=0.0)
    """自适应决策中等待时延的基础价格。"""

    load_price_scale: float = Field(default=0.0, ge=0.0)
    """P 节点负载对等待价格的缩放系数。"""
