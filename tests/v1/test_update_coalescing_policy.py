# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""用确定性的调度状态验证各个 Update 准入策略，不启动 GPU 或服务。"""

import pytest
from pydantic import ValidationError

from vllm.config.prompt_stream import (
    PromptStreamUpdateCoalescingConfig,
    PromptStreamUpdatePolicy,
)
from vllm.v1.core.sched.update_coalescing import (
    AdaptiveUpdateCoalescingPolicy,
    FixedWindowUpdateCoalescingPolicy,
    FullMergeUpdateCoalescingPolicy,
    ImmediateUpdateCoalescingPolicy,
    QuietDebounceUpdateCoalescingPolicy,
    UpdateCoalescingContext,
    UpdateCoalescingPolicy,
    create_update_coalescing_policy,
)
from vllm.v1.request import UpdateCoalescingState


def _window(config: PromptStreamUpdateCoalescingConfig) -> UpdateCoalescingState:
    """构造已经接收最新快照、但尚未放行的独立窗口。"""
    return UpdateCoalescingState(
        config=config,
        request_start_time=5.0,
        first_pending_time=10.0,
        last_update_time=10.0,
        latest_pending_token_ids=(1, 2, 3),
    )


@pytest.mark.parametrize(
    ("name", "expected_type"),
    [
        ("immediate", ImmediateUpdateCoalescingPolicy),
        ("fixed_window", FixedWindowUpdateCoalescingPolicy),
        ("quiet_debounce", QuietDebounceUpdateCoalescingPolicy),
        ("adaptive", AdaptiveUpdateCoalescingPolicy),
        ("full_merge", FullMergeUpdateCoalescingPolicy),
    ],
)
def test_factory_selects_distinct_policy(
    name: PromptStreamUpdatePolicy, expected_type: type[UpdateCoalescingPolicy]
) -> None:
    """配置直接选择各自的实现类；只有 adaptive 依赖执行反馈。"""
    policy = create_update_coalescing_policy(
        PromptStreamUpdateCoalescingConfig(update_policy=name)
    )
    assert type(policy) is expected_type
    assert policy.requires_scheduler_feedback == (name == "adaptive")


@pytest.mark.parametrize(
    ("outstanding", "token_pressure", "slot_pressure", "expected"),
    [
        (False, 0.5, 0.5, True),
        (True, 0.0, 0.0, False),
        (False, 1.0, 0.0, False),
        (False, 0.0, 1.0, False),
        (False, 0.999, 0.999, True),
    ],
)
def test_adaptive_waits_for_execution_and_both_resources(
    outstanding: bool,
    token_pressure: float,
    slot_pressure: float,
    expected: bool,
) -> None:
    """分别覆盖执行背压、token 满额和 running 名额满额三种阻挡。"""
    config = PromptStreamUpdateCoalescingConfig(update_policy="adaptive")
    state = _window(config)
    policy = create_update_coalescing_policy(config)
    policy.on_snapshot(state, now=10.0)
    assert policy.should_flush(
        state,
        UpdateCoalescingContext(
            now=10000.0,
            has_outstanding_update=outstanding,
            token_pressure=token_pressure,
            slot_pressure=slot_pressure,
        ),
    ) is expected


def test_adaptive_never_registers_a_time_deadline() -> None:
    """即使配置带时间参数，也不能绕过持续的执行背压。"""
    config = PromptStreamUpdateCoalescingConfig(
        update_policy="adaptive",
        soft_window_ms=0.0,
        hard_deadline_ms=1.0,
        ttft_deadline_ms=1.0,
    )
    state = _window(config)
    state.soft_deadline = state.hard_deadline = state.ttft_deadline = 1.0
    policy = create_update_coalescing_policy(config)
    policy.on_snapshot(state, now=10.0)
    assert state.soft_deadline is None
    assert state.hard_deadline is None
    assert state.ttft_deadline is None
    assert not policy.should_flush(
        state, UpdateCoalescingContext(now=10000.0, has_outstanding_update=True)
    )
    # 背压解除即可放行，完全不需要新事件或定时器唤醒。
    assert policy.should_flush(state, UpdateCoalescingContext(now=10000.0))


def test_adaptive_uses_configured_pressure_thresholds() -> None:
    """两个无量纲阈值独立生效，不把 token budget 当作唯一限制。"""
    config = PromptStreamUpdateCoalescingConfig(
        update_policy="adaptive",
        adaptive_token_pressure_threshold=0.75,
        adaptive_slot_pressure_threshold=0.5,
    )
    state = _window(config)
    policy = create_update_coalescing_policy(config)
    assert policy.should_flush(
        state,
        UpdateCoalescingContext(now=10.0, token_pressure=0.7, slot_pressure=0.4),
    )
    assert not policy.should_flush(
        state,
        UpdateCoalescingContext(now=10.0, token_pressure=0.7, slot_pressure=0.5),
    )


@pytest.mark.parametrize(
    "field",
    ["adaptive_token_pressure_threshold", "adaptive_slot_pressure_threshold"],
)
@pytest.mark.parametrize("invalid", [0.0, -1.0, float("inf"), float("nan")])
def test_adaptive_rejects_invalid_pressure_thresholds(
    field: str, invalid: float
) -> None:
    """非法阈值不能让所有窗口永久挂起或无条件放行。"""
    with pytest.raises(ValidationError):
        PromptStreamUpdateCoalescingConfig(**{field: invalid})


@pytest.mark.parametrize(
    ("name", "expected_soft"),
    [("fixed_window", 10.020), ("quiet_debounce", 10.035)],
)
def test_timed_policy_preserves_hard_deadline_when_replaced(
    name: PromptStreamUpdatePolicy, expected_soft: float
) -> None:
    """相同覆盖事件只刷新 debounce 的软窗口，硬窗口仍固定于首事件。"""
    config = PromptStreamUpdateCoalescingConfig(
        update_policy=name, soft_window_ms=20.0, hard_deadline_ms=80.0
    )
    state = _window(config)
    policy = create_update_coalescing_policy(config)
    policy.on_snapshot(state, now=10.0)
    state.last_update_time = 10.015
    policy.on_snapshot(state, now=10.015)
    assert state.soft_deadline == pytest.approx(expected_soft)
    assert state.hard_deadline == pytest.approx(10.080)
    assert not policy.should_flush(state, UpdateCoalescingContext(now=10.019))
    assert policy.should_flush(state, UpdateCoalescingContext(now=expected_soft))


def test_full_merge_ignores_elapsed_time() -> None:
    """full_merge 仍只由 Request 的结束事件放行，不引入时间上限。"""
    config = PromptStreamUpdateCoalescingConfig(update_policy="full_merge")
    state = _window(config)
    policy = create_update_coalescing_policy(config)
    policy.on_snapshot(state, now=10.0)
    assert state.hard_deadline is None
    assert not policy.should_flush(state, UpdateCoalescingContext(now=10000.0))
