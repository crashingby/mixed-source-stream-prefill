# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import Literal

from pydantic import Field

from vllm.config.utils import config

PromptStreamUpdatePolicy = Literal[
    "immediate",
    "fixed_window",
    "quiet_debounce",
    "load_aware",
    "full_merge",
]


@config
class PromptStreamUpdateCoalescingConfig:
    """流式 Prompt Update 合并配置。

    传入该配置即启用 Update 合并；配置为 ``None`` 时，PromptStream 保持
    每个 Update 事件立即创建 work 的原有行为。启用后只由 Update policy
    决定窗口何时 admission，不要求 Append source 已经结束。
    """

    update_policy: PromptStreamUpdatePolicy = "fixed_window"
    """决定 Update 窗口何时 flush；load_aware 使用执行背压与全局计算余量。"""

    soft_window_ms: float = Field(default=20.0, ge=0.0)
    """fixed_window 或 quiet_debounce 的候选等待时间，单位为毫秒。"""

    hard_deadline_ms: float = Field(default=80.0, gt=0.0)
    """仅定时策略使用的最大窗口等待时间；load_aware 不使用时间上限。"""

    ttft_deadline_ms: float | None = Field(default=None, gt=0.0)
    """仅定时策略使用的 Request 起点放行预算；不保证实际 TTFT。"""

    load_aware_token_pressure_threshold: float = Field(
        default=1.0, gt=0.0, allow_inf_nan=False
    )
    """load_aware 放行要求 token 需求 / 单轮预算小于此值；1 表示一轮预算。"""

    load_aware_slot_pressure_threshold: float = Field(
        default=1.0, gt=0.0, allow_inf_nan=False
    )
    """load_aware 放行要求可执行请求数 / max_num_seqs 小于此值。"""
