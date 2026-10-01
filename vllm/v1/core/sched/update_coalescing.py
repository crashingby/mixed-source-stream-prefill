# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""按配置创建 Update 放行策略；窗口和 work 生命周期仍由 Request 管理。"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING

from vllm.config.prompt_stream import PromptStreamUpdateCoalescingConfig

if TYPE_CHECKING:
    from vllm.v1.request import UpdateCoalescingState


@dataclass(frozen=True)
class UpdateCoalescingContext:
    """保存一次准入判定的只读快照，压力值由 Scheduler 统一计算。

    token_pressure 与 slot_pressure 分别以单轮 token budget 和
    max_num_seqs 归一化。这里不读取队列，不修改 GPU 或 KV 状态。
    """

    now: float  # EngineCore 的 monotonic 秒数。
    has_outstanding_update: bool = False  # 包含 pending、active 与 GPU 在途。
    token_pressure: float = 0.0  # 可执行 token 需求与单轮预算之比。
    slot_pressure: float = 0.0  # 可执行请求数量与运行名额上限之比。


class UpdateCoalescingPolicy(ABC):
    """定义窗口更新和准入判定，不负责创建 work 或终止事件强制放行。

    每个策略只保存配置。请求的快照、deadline 和统计保存在 state 中，
    因而策略可以独立测试，不需要访问 Scheduler 或 Request 的执行队列。
    """

    requires_scheduler_feedback: bool = False  # 只有背压策略依赖执行进度。

    def __init__(self, config: PromptStreamUpdateCoalescingConfig) -> None:
        """保存启动配置；后续判定不得原地修改配置。"""
        self.config = config

    def on_snapshot(self, state: UpdateCoalescingState, now: float) -> None:
        """接收完整快照后清空定时器，供没有时间条件的策略使用。

        Request 已设置 first_pending_time、last_update_time 与最新快照。
        此方法只更新策略状态，不改变已放行的 work。
        """
        # immediate/full_merge/load_aware 都不能遗留旧窗口的定时器。
        state.soft_deadline = None
        state.hard_deadline = None
        state.ttft_deadline = None

    @abstractmethod
    def should_flush(
        self, state: UpdateCoalescingState, context: UpdateCoalescingContext
    ) -> bool:
        """判断非终止窗口是否放行；结束事件由 Request 在调用前统一处理。"""
        raise NotImplementedError


class ImmediateUpdateCoalescingPolicy(UpdateCoalescingPolicy):
    """每次内容变化立即放行，用作启用 controller 的即时计算对照。"""

    def should_flush(
        self, state: UpdateCoalescingState, context: UpdateCoalescingContext
    ) -> bool:
        """已有合法 pending 快照时立即允许创建 work。"""
        return True


class FullMergeUpdateCoalescingPolicy(UpdateCoalescingPolicy):
    """持续保留最新快照，只接受 Request 层的结束事件或显式强制放行。"""

    def should_flush(
        self, state: UpdateCoalescingState, context: UpdateCoalescingContext
    ) -> bool:
        """普通调度机会不放行，等待唯一最终快照。"""
        return False


class LoadAwareUpdateCoalescingPolicy(UpdateCoalescingPolicy):
    """按请求执行背压与全局计算余量实施负载感知放行。

    此策略不设置时间 deadline。
    结束事件由 Request 提前处理，因此最终快照不会被本请求的旧 work
    或全局压力拦住。普通窗口只有在两个压力都低于阈值时才能放行。
    """

    requires_scheduler_feedback = True  # 计算完成和队列变化都要重新检查。

    def should_flush(
        self, state: UpdateCoalescingState, context: UpdateCoalescingContext
    ) -> bool:
        """上一个 Update 真正完成且全局有余量时允许普通准入。"""
        # 已准入的工作不能被合并取消，先等待它的最后一个 GPU 回包。
        if context.has_outstanding_update:
            return False
        # token 和名额是两种独立限制；任意一项饱和都继续保留可覆盖快照。
        return (
            context.token_pressure < self.config.load_aware_token_pressure_threshold
            and context.slot_pressure < self.config.load_aware_slot_pressure_threshold
        )


class _TimedUpdateCoalescingPolicy(UpdateCoalescingPolicy):
    """共享固定 hard deadline 与可选 Request 起点截止时间。"""

    def on_snapshot(self, state: UpdateCoalescingState, now: float) -> None:
        """保持窗口第一次到达时刻固定，后续更新不能推迟硬截止时间。"""
        # Request 必须先建立窗口，避免毫秒配置与不完整状态混用。
        assert state.first_pending_time is not None
        state.hard_deadline = (
            state.first_pending_time + self.config.hard_deadline_ms / 1000.0
        )
        state.ttft_deadline = (
            None
            if self.config.ttft_deadline_ms is None
            else state.request_start_time + self.config.ttft_deadline_ms / 1000.0
        )

    def should_flush(
        self, state: UpdateCoalescingState, context: UpdateCoalescingContext
    ) -> bool:
        """任一绝对 deadline 到期即可放行，不重复延长等待。"""
        return any(
            deadline is not None and context.now >= deadline
            for deadline in (
                state.soft_deadline,
                state.hard_deadline,
                state.ttft_deadline,
            )
        )


class FixedWindowUpdateCoalescingPolicy(_TimedUpdateCoalescingPolicy):
    """窗口从第一条 Update 计时，后续覆盖不会刷新 soft deadline。"""

    def on_snapshot(self, state: UpdateCoalescingState, now: float) -> None:
        """以当前窗口起点设置软截止时间，同时维护共享硬截止时间。"""
        super().on_snapshot(state, now)
        assert state.first_pending_time is not None
        state.soft_deadline = (
            state.first_pending_time + self.config.soft_window_ms / 1000.0
        )


class QuietDebounceUpdateCoalescingPolicy(_TimedUpdateCoalescingPolicy):
    """每次 Update 刷新静默窗口，固定 hard deadline 限制最长等待。"""

    def on_snapshot(self, state: UpdateCoalescingState, now: float) -> None:
        """以最近事件时间刷新 soft deadline，不改变首次事件的硬截止。"""
        super().on_snapshot(state, now)
        state.soft_deadline = now + self.config.soft_window_ms / 1000.0


def create_update_coalescing_policy(
    config: PromptStreamUpdateCoalescingConfig,
) -> UpdateCoalescingPolicy:
    """根据启动配置创建独立策略；未知策略显式报错，避免静默降级。"""
    policy_types: dict[str, type[UpdateCoalescingPolicy]] = {
        "immediate": ImmediateUpdateCoalescingPolicy,
        "fixed_window": FixedWindowUpdateCoalescingPolicy,
        "quiet_debounce": QuietDebounceUpdateCoalescingPolicy,
        "load_aware": LoadAwareUpdateCoalescingPolicy,
        "full_merge": FullMergeUpdateCoalescingPolicy,
    }
    policy_type = policy_types.get(config.update_policy)
    if policy_type is None:
        raise ValueError(f"Unknown Update coalescing policy: {config.update_policy!r}")
    return policy_type(config)
