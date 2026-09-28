# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""在 CPU 上验证 PromptStream 的停放、唤醒和 KV 恢复规则。"""

from collections import deque
from queue import Queue
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest

from tests.v1.test_prompt_stream_request import (
    _finish_active_view,
    _make_event,
    _make_kv_cache_manager,
    _make_request,
    _start_request,
)
from vllm.config import PromptStreamUpdateCoalescingConfig
from vllm.v1.core.sched.interface import PauseState
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.core.sched.request_queue import FCFSRequestQueue
from vllm.v1.core.sched.scheduler import (
    PromptStreamUpdateCoalescingController,
    Scheduler,
)
from vllm.v1.engine.prompt_stream import PromptSourceType, PromptStreamEventType
from vllm.v1.outputs import ModelRunnerOutput
from vllm.v1.request import (
    PromptStreamExecutionPhase,
    PromptStreamKVSegment,
    Request,
    RequestStatus,
)


def _make_scheduler(
    config: PromptStreamUpdateCoalescingConfig | None,
) -> Scheduler:
    """构造无模型、无 GPU 的调度器，保留真实请求队列和 KV block 管理。

    仅省略服务配置与 worker 初始化；被测 Scheduler 方法不做替换。
    Encoder cache 是边缘依赖，本测试使用纯文本请求。
    """
    scheduler = object.__new__(Scheduler)
    scheduler._prompt_stream_update_coalescer = (
        PromptStreamUpdateCoalescingController(config) if config else None
    )
    scheduler._prompt_stream_dormant_reasons = {}
    scheduler._prompt_stream_inflight_steps = {}
    scheduler._prompt_stream_retained_waiting_req_ids = set()
    scheduler._prompt_stream_worker_removal_req_ids = set()
    scheduler._inflight_prefills = set()
    scheduler.requests = {}
    scheduler.running = []
    scheduler.waiting = FCFSRequestQueue()
    scheduler.skipped_waiting = FCFSRequestQueue()
    scheduler.kv_cache_manager = _make_kv_cache_manager()
    scheduler.encoder_cache_manager = Mock()
    scheduler.connector = None
    scheduler.block_size = 4
    scheduler.max_model_len = 64
    scheduler.max_num_running_reqs = 1
    scheduler.max_num_scheduled_tokens = 64
    scheduler.scheduler_config = SimpleNamespace(
        prompt_stream_masked_padding=False,
        long_prefill_token_threshold=0,
    )
    scheduler.log_stats = False
    scheduler.perf_metrics = None
    scheduler.enable_return_routed_experts = False
    scheduler.defer_block_free = False
    scheduler.finished_req_ids = set()
    scheduler.finished_req_ids_dict = None
    scheduler.num_waiting_for_streaming_input = 0
    scheduler._pause_state = PauseState.UNPAUSED
    scheduler._terminal_prompt_stream_request_ids = set()
    scheduler._terminal_prompt_stream_request_order = deque()
    scheduler._max_prompt_stream_tombstones = 32
    return scheduler


def _add_running_request(
    scheduler: Scheduler,
    request_id: str = "prompt-stream-test",
) -> Request:
    """建立已分配模板 KV、但模板尚未完成的运行请求。"""
    request = _make_request([1, 2, 3, 4])
    scheduler._enable_prompt_stream_update_coalescing(request, now=10.0)
    _start_request(request)
    request.request_id = request_id
    request.status = RequestStatus.RUNNING
    scheduler.requests[request.request_id] = request
    scheduler.running.append(request)
    blocks = scheduler.kv_cache_manager.allocate_slots(request, num_new_tokens=4)
    assert blocks is not None
    return request


def _send_event(
    scheduler: Scheduler,
    request: Request,
    event_type: PromptStreamEventType,
    token_ids: list[int] | None,
    *,
    now: float = 10.0,
    version: int = 0,
    sequence_no: int | None = None,
    source_type: PromptSourceType | None = None,
) -> None:
    """经由真实 Scheduler 事件入口更新长期 Request。"""
    source_type = source_type or (
        PromptSourceType.UPDATE
        if event_type == PromptStreamEventType.UPDATE
        else PromptSourceType.APPEND
    )
    if event_type == PromptStreamEventType.PROMPT_FINISHED:
        source_type = None
    incoming = _make_request([1])
    incoming.prompt_stream_event = _make_event(
        event_type,
        token_ids,
        source_id="update" if source_type == PromptSourceType.UPDATE else "append",
        source_type=source_type,
        version=version,
        sequence_no=version if sequence_no is None else sequence_no,
    )
    incoming.prompt_stream_event.request_id = request.request_id
    with patch("vllm.v1.core.sched.scheduler.time.monotonic", return_value=now):
        scheduler._apply_prompt_stream_request(request, incoming)


@pytest.fixture(params=[None, "immediate"], ids=["disabled", "immediate"])
def scheduler(request: pytest.FixtureRequest) -> Scheduler:
    """以关闭/开启合并两种配置验证相同的停放规则。"""
    config = (
        PromptStreamUpdateCoalescingConfig(update_policy=request.param)
        if request.param is not None
        else None
    )
    return _make_scheduler(config)


def test_first_prompt_stream_request_waits_for_initial_admission(
    scheduler: Scheduler,
) -> None:
    """首次到达的模板没有计算，必须先进入 waiting，不能直接停放。"""
    request = _make_request([1, 2, 3, 4])
    request.prompt_stream_event = _make_event(
        PromptStreamEventType.APPEND,
        [],
        source_id="__prompt_stream_start__",
        source_type=PromptSourceType.APPEND,
    )
    scheduler.add_request(request)
    assert request.status == RequestStatus.WAITING
    assert list(scheduler.waiting) == [request]
    assert not scheduler.running
    assert not scheduler._prompt_stream_dormant_reasons


def test_prompt_stream_parks_only_after_all_work_completes(
    scheduler: Scheduler,
) -> None:
    """模板、pending 和 active work 均阻止停放，完成后才释放运行名额。"""
    request = _add_running_request(scheduler)
    assert not request.should_make_prompt_stream_dormant()
    scheduler._reconcile_prompt_stream_request(request)
    assert scheduler.running == [request]

    _finish_active_view(request)
    _send_event(scheduler, request, PromptStreamEventType.APPEND, [5, 6])
    assert request.prompt_stream is not None
    assert request.prompt_stream.pending_work_items
    assert not request.should_make_prompt_stream_dormant()
    scheduler._prepare_prompt_stream_execution(request)
    assert request.prompt_stream.active_work_item is not None
    assert not scheduler._make_prompt_stream_request_dormant(request)

    assert scheduler.kv_cache_manager.allocate_slots(request, num_new_tokens=2)
    _finish_active_view(request)
    block_ids = scheduler.kv_cache_manager.get_prompt_stream_block_ids(request)
    scheduler._reconcile_prompt_stream_request(request)
    assert request.should_make_prompt_stream_dormant()
    assert not scheduler.running
    assert not scheduler.waiting
    assert not scheduler.skipped_waiting
    assert scheduler._prompt_stream_dormant_reasons == {
        request.request_id: "stream_input"
    }
    assert request.status == RequestStatus.WAITING
    # 停放只卸下 worker 请求状态，真实分段 blocks 仍然属于该请求。
    assert request.request_id in scheduler._prompt_stream_worker_removal_req_ids
    assert scheduler.kv_cache_manager.get_prompt_stream_block_ids(request) == block_ids
    assert scheduler.kv_cache_manager.has_prompt_stream_segment_blocks(
        request, PromptStreamKVSegment.PREFIX_APPEND
    )
    assert scheduler.get_num_unfinished_requests() == 1
    assert scheduler.has_requests()
    scheduler._prompt_stream_worker_removal_req_ids.clear()
    assert not scheduler.has_runnable_engine_work()


@pytest.mark.parametrize(
    "event_type",
    [PromptStreamEventType.APPEND, PromptStreamEventType.PROMPT_FINISHED],
)
def test_new_work_wakes_dormant_request_to_waiting(
    scheduler: Scheduler,
    event_type: PromptStreamEventType,
) -> None:
    """新 Append 或问题唤醒请求，但必须重新等待准入，不能直接回 running。"""
    request = _add_running_request(scheduler)
    _finish_active_view(request)
    scheduler._reconcile_prompt_stream_request(request)
    assert not scheduler.running

    _send_event(scheduler, request, event_type, [5, 6])
    assert request.request_id not in scheduler._prompt_stream_dormant_reasons
    assert list(scheduler.waiting) == [request]
    assert not scheduler.running
    assert request.prompt_stream_needs_worker_refresh
    assert request.request_id in scheduler._prompt_stream_retained_waiting_req_ids
    # 多次协调同一个已唤醒请求，不可重复入队或重复保留 running 名额。
    scheduler._reconcile_prompt_stream_request(request)
    assert list(scheduler.waiting) == [request]


def test_pending_coalescing_window_stays_dormant_until_deadline() -> None:
    """仅覆盖合并窗口不产生 work；deadline flush 后才唤醒请求。"""
    scheduler = _make_scheduler(
        PromptStreamUpdateCoalescingConfig(
            update_policy="fixed_window",
            soft_window_ms=20.0,
            hard_deadline_ms=80.0,
        )
    )
    request = _add_running_request(scheduler)
    _finish_active_view(request)
    scheduler._reconcile_prompt_stream_request(request)
    _send_event(scheduler, request, PromptStreamEventType.UPDATE, [10, 11])
    assert request.request_id in scheduler._prompt_stream_dormant_reasons
    assert not scheduler.waiting
    assert not scheduler.running
    assert scheduler.get_next_prompt_stream_deadline() == pytest.approx(10.02)
    assert scheduler.flush_due_update_coalescing(now=10.01) == 0

    with patch("vllm.v1.core.sched.scheduler.time.monotonic", return_value=10.03):
        assert scheduler.flush_due_update_coalescing(now=10.03) == 1
    assert request.request_id not in scheduler._prompt_stream_dormant_reasons
    assert list(scheduler.waiting) == [request]
    assert not scheduler.running
    assert scheduler.get_next_prompt_stream_deadline() is None


def test_inflight_prompt_stream_cannot_park(scheduler: Scheduler) -> None:
    """调度时的乐观进度不能替代 worker 回包；最后一步确认后才能停放。"""
    request = _add_running_request(scheduler)
    output = SchedulerOutput.make_empty()
    output.num_scheduled_tokens = {request.request_id: 2}
    output.total_num_scheduled_tokens = 2
    # 模拟两次调度先后提交，但 GPU 尚未返回任何一步。
    scheduler._update_after_schedule(output)
    scheduler._update_after_schedule(output)
    assert request.should_make_prompt_stream_dormant()
    assert scheduler._prompt_stream_inflight_steps[request.request_id] == 2
    scheduler._reconcile_prompt_stream_request(request)
    assert scheduler.running == [request]
    assert not scheduler._prompt_stream_dormant_reasons

    runner_output = ModelRunnerOutput(
        req_ids=[request.request_id],
        req_id_to_index={request.request_id: 0},
    )
    scheduler.update_from_output(output, runner_output)
    assert scheduler._prompt_stream_inflight_steps[request.request_id] == 1
    assert scheduler.running == [request]
    assert not scheduler._prompt_stream_dormant_reasons

    scheduler.update_from_output(output, runner_output)
    assert not scheduler._prompt_stream_inflight_steps
    assert not scheduler.running
    assert request.request_id in scheduler._prompt_stream_dormant_reasons


@pytest.mark.parametrize(
    "finished_status",
    [RequestStatus.FINISHED_ABORTED, RequestStatus.FINISHED_STOPPED],
)
def test_finishing_dormant_request_cleans_state_and_kv(
    scheduler: Scheduler,
    finished_status: RequestStatus,
) -> None:
    """结束或取消 dormant 请求，应清理停放记录、worker 清理记录与 KV。"""
    request = _add_running_request(scheduler)
    _finish_active_view(request)
    scheduler._reconcile_prompt_stream_request(request)
    scheduler.finish_requests(request.request_id, finished_status)
    assert request.request_id not in scheduler.requests
    assert not scheduler._prompt_stream_dormant_reasons
    assert not scheduler._prompt_stream_worker_removal_req_ids
    assert not scheduler._prompt_stream_retained_waiting_req_ids
    assert not scheduler._prompt_stream_inflight_steps
    assert request.request_id in scheduler.finished_req_ids
    assert not scheduler.kv_cache_manager.has_prompt_stream_segment_blocks(
        request, PromptStreamKVSegment.PREFIX_APPEND
    )


def test_preempted_prompt_stream_waits_for_recovery(scheduler: Scheduler) -> None:
    """没有新事件的请求被抢占后仍有恢复工作，不能误停放。"""
    request = _add_running_request(scheduler)
    _finish_active_view(request)
    scheduler.running.remove(request)
    scheduler._preempt_request(request, timestamp=10.0)
    assert request.status == RequestStatus.PREEMPTED
    assert list(scheduler.waiting) == [request]
    assert not scheduler._prompt_stream_dormant_reasons
    assert not request.should_make_prompt_stream_dormant()
    assert not scheduler.kv_cache_manager.has_prompt_stream_segment_blocks(
        request, PromptStreamKVSegment.PREFIX_APPEND
    )


def test_dormant_eviction_waits_for_recovery(scheduler: Scheduler) -> None:
    """显存压力驱逐 dormant KV 后，必须重新进入 waiting 重算。"""
    request = _add_running_request(scheduler)
    _finish_active_view(request)
    scheduler._reconcile_prompt_stream_request(request)
    evicted: list[Request] = []
    assert scheduler._evict_one_dormant_prompt_stream_request(10.0, evicted)
    scheduler._finalize_dormant_prompt_stream_evictions(evicted)
    assert evicted == [request]
    assert request.status == RequestStatus.PREEMPTED
    assert list(scheduler.waiting) == [request]
    assert not scheduler._prompt_stream_dormant_reasons
    assert not request.should_make_prompt_stream_dormant()
    assert not scheduler.kv_cache_manager.has_prompt_stream_segment_blocks(
        request, PromptStreamKVSegment.PREFIX_APPEND
    )


def _make_adaptive_scheduler() -> Scheduler:
    """创建无定时兜底的执行背压策略，使用较小预算显式触发压力。"""
    return _make_scheduler(
        PromptStreamUpdateCoalescingConfig(update_policy="adaptive")
    )


def _add_dormant_request(
    scheduler: Scheduler,
    request_id: str = "prompt-stream-test",
) -> Request:
    """先完成真实模板 work，再停放请求并保留其 KV。"""
    request = _add_running_request(scheduler, request_id)
    _finish_active_view(request)
    scheduler._reconcile_prompt_stream_request(request)
    assert request.request_id in scheduler._prompt_stream_dormant_reasons
    return request


def _make_step(request: Request, num_tokens: int) -> SchedulerOutput:
    """构造一轮 CPU 模拟输出，阶段标识由真实提交方法登记。"""
    output = SchedulerOutput.make_empty()
    output.num_scheduled_tokens = {request.request_id: num_tokens}
    output.total_num_scheduled_tokens = num_tokens
    return output


def _return_step(scheduler: Scheduler, output: SchedulerOutput) -> None:
    """通过真实回包路径归还一个已提交步骤，不生成 decode token。"""
    request_ids = list(output.num_scheduled_tokens)
    scheduler.update_from_output(
        output,
        ModelRunnerOutput(
            req_ids=request_ids,
            req_id_to_index={rid: index for index, rid in enumerate(request_ids)},
        ),
    )


def test_adaptive_backpressure_tracks_pending_active_and_gpu_completion() -> None:
    """Update 提前清空 active 后仍须等待所有 GPU 回包才能放行下一代。"""
    scheduler = _make_adaptive_scheduler()
    request = _add_dormant_request(scheduler)
    _send_event(scheduler, request, PromptStreamEventType.UPDATE, [10, 11, 12, 13])
    stream = request.prompt_stream
    assert stream is not None
    state = request.prompt_stream_update_coalescing_state
    assert state is not None
    assert stream.pending_work_items
    assert request.has_outstanding_prompt_stream_update()

    # 同一请求内部 active 转换保持背压，与等待队列位置无关。
    scheduler.waiting.remove_request(request)
    scheduler.running.append(request)
    request.status = RequestStatus.RUNNING
    scheduler._prepare_prompt_stream_execution(request)
    assert stream.active_work_item is not None
    assert request.has_outstanding_prompt_stream_update()
    first_step = _make_step(request, 2)
    second_step = _make_step(request, 2)
    scheduler._update_after_schedule(first_step)
    scheduler._update_after_schedule(second_step)
    assert first_step.prompt_stream_update_req_ids == {request.request_id}
    assert second_step.prompt_stream_update_req_ids == {request.request_id}
    assert stream.active_work_item is None
    assert not stream.pending_work_items
    assert stream.num_update_inflight_steps == 2
    assert request.has_outstanding_prompt_stream_update()

    _send_event(
        scheduler, request, PromptStreamEventType.UPDATE, [10, 20], version=1
    )
    assert state.latest_pending_token_ids == (10, 20)
    assert not stream.pending_work_items
    _return_step(scheduler, first_step)
    assert stream.num_update_inflight_steps == 1
    assert state.latest_pending_token_ids == (10, 20)
    assert not stream.pending_work_items

    # 最后一个回包主动重评估，不需要新 Update 或 deadline 来触发。
    _return_step(scheduler, second_step)
    assert stream.num_update_inflight_steps == 0
    assert state.latest_pending_token_ids is None
    assert len(stream.pending_work_items) == 1
    assert stream.pending_work_items[0].update_token_ids == (10, 20)
    assert stream.pending_work_items[0].update_common_prefix_len == 1


def test_adaptive_window_has_no_time_based_flush() -> None:
    """即使配置了旧定时参数且经过很久，执行背压也不能被时间绕过。"""
    scheduler = _make_scheduler(
        PromptStreamUpdateCoalescingConfig(
            update_policy="adaptive",
            soft_window_ms=1.0,
            hard_deadline_ms=2.0,
            ttft_deadline_ms=3.0,
        )
    )
    request = _add_dormant_request(scheduler)
    _send_event(scheduler, request, PromptStreamEventType.UPDATE, [10, 11])
    _send_event(
        scheduler, request, PromptStreamEventType.UPDATE, [10, 12], version=1
    )
    state = request.prompt_stream_update_coalescing_state
    assert state is not None
    assert state.latest_pending_token_ids == (10, 12)
    assert state.soft_deadline is None
    assert state.hard_deadline is None
    assert state.ttft_deadline is None
    assert request.get_prompt_stream_update_deadline() is None
    assert scheduler.get_next_prompt_stream_deadline() is None
    assert scheduler.flush_due_update_coalescing(now=1_000_000.0) == 0
    assert state.latest_pending_token_ids == (10, 12)


@pytest.mark.parametrize("pressure_kind", ["tokens", "slots"])
def test_adaptive_token_and_slot_pressure_independently_hold_updates(
    pressure_kind: str,
) -> None:
    """预算和运行名额任一用满都可暂缓，解除后无需新事件即可唤醒。"""
    scheduler = _make_adaptive_scheduler()
    scheduler.max_num_scheduled_tokens = 4 if pressure_kind == "tokens" else 64
    scheduler.max_num_running_reqs = 16 if pressure_kind == "tokens" else 1
    blocker = _add_running_request(scheduler, "busy-template")
    request = _add_dormant_request(scheduler)
    context = scheduler._get_prompt_stream_update_context(request, now=10.0)
    if pressure_kind == "tokens":
        assert context.token_pressure == 1.0
        assert context.slot_pressure == pytest.approx(1 / 16)
    else:
        assert context.token_pressure == pytest.approx(4 / 64)
        assert context.slot_pressure == 1.0

    _send_event(scheduler, request, PromptStreamEventType.UPDATE, [10, 11])
    state = request.prompt_stream_update_coalescing_state
    assert state is not None
    assert state.latest_pending_token_ids == (10, 11)
    assert request.request_id in scheduler._prompt_stream_dormant_reasons

    _finish_active_view(blocker)
    scheduler._reconcile_prompt_stream_request(blocker)
    assert scheduler.flush_due_update_coalescing(now=10.1) == 1
    assert state.latest_pending_token_ids is None
    assert list(scheduler.waiting) == [request]
    assert not scheduler.running


@pytest.mark.parametrize(
    "blocked_status",
    [
        RequestStatus.WAITING_FOR_REMOTE_KVS,
        RequestStatus.WAITING_FOR_STRUCTURED_OUTPUT_GRAMMAR,
        RequestStatus.WAITING_FOR_STREAMING_REQ,
    ],
)
def test_adaptive_pressure_excludes_dormant_and_external_waiters(
    blocked_status: RequestStatus,
) -> None:
    """保留 KV 的空闲请求与等待外部条件的请求不构成可执行积压。"""
    scheduler = _make_adaptive_scheduler()
    request = _add_dormant_request(scheduler)
    _add_dormant_request(scheduler, "idle-stream")
    blocked = _make_request([10] * 32)
    blocked.request_id = "external-waiter"
    blocked.status = blocked_status
    scheduler.requests[blocked.request_id] = blocked
    scheduler.skipped_waiting.add_request(blocked)
    context = scheduler._get_prompt_stream_update_context(request, now=10.0)
    assert context.token_pressure == 0.0
    assert context.slot_pressure == 0.0
    _send_event(scheduler, request, PromptStreamEventType.UPDATE, [10, 11])
    state = request.prompt_stream_update_coalescing_state
    assert state is not None
    assert state.latest_pending_token_ids is None
    assert list(scheduler.waiting) == [request]


def test_adaptive_admissions_recompute_pressure_between_candidates() -> None:
    """一轮只剩一个 slot 时不能把所有 dormant 候选同时放行。"""
    scheduler = _make_adaptive_scheduler()
    blocker = _add_running_request(scheduler, "busy-template")
    requests = [
        _add_dormant_request(scheduler, f"candidate-{index}") for index in range(3)
    ]
    for request in requests:
        _send_event(scheduler, request, PromptStreamEventType.UPDATE, [10, 11])
    controller = scheduler._prompt_stream_update_coalescer
    assert controller is not None
    assert list(controller.adaptive_request_ids) == [r.request_id for r in requests]

    _finish_active_view(blocker)
    scheduler._reconcile_prompt_stream_request(blocker)
    assert scheduler.flush_due_update_coalescing(now=10.1) == 1
    assert list(scheduler.waiting) == [requests[0]]
    assert list(controller.adaptive_request_ids) == [
        request.request_id for request in requests[1:]
    ]
    assert scheduler.flush_due_update_coalescing(now=10.2) == 0


def test_adaptive_idle_engine_check_admits_pending_window() -> None:
    """全局只剩窗口时，EngineCore 的空闲检查必须主动形成可执行 work。"""
    scheduler = _make_adaptive_scheduler()
    blocker = _add_running_request(scheduler, "busy-template")
    request = _add_dormant_request(scheduler)
    _send_event(scheduler, request, PromptStreamEventType.UPDATE, [10, 11])
    _finish_active_view(blocker)
    scheduler._reconcile_prompt_stream_request(blocker)
    scheduler._prompt_stream_worker_removal_req_ids.clear()
    assert not scheduler.running
    assert not scheduler.waiting
    # 调度器查询保持只读；真实 EngineCore 在进入输入等待前重评估窗口。
    from vllm.v1.engine.core import EngineCoreProc, EngineShutdownState

    engine = object.__new__(EngineCoreProc)
    engine.scheduler = scheduler
    engine.engines_running = False
    engine.batch_queue = None
    engine.shutdown_state = EngineShutdownState.RUNNING
    engine.prompt_stream_update_coalescer_enabled = True
    engine.input_queue = Queue()
    engine.aborts_queue = Queue()
    engine.process_input_queue_block = False
    engine._notify_idle_state_callbacks = Mock()
    assert not engine.has_work()
    engine._process_input_queue()
    assert engine.has_work()
    assert list(scheduler.waiting) == [request]
    assert request.request_id not in scheduler._prompt_stream_dormant_reasons


@pytest.mark.parametrize("terminal_kind", ["source_finished", "question"])
def test_adaptive_terminal_event_bypasses_backpressure_preserves_fifo(
    terminal_kind: str,
) -> None:
    """最终快照不再值得覆盖，即使旧 Update 未完成也要排入其后。"""
    scheduler = _make_adaptive_scheduler()
    request = _add_dormant_request(scheduler)
    _send_event(scheduler, request, PromptStreamEventType.UPDATE, [10, 11])
    _send_event(
        scheduler, request, PromptStreamEventType.UPDATE, [10, 12], version=1
    )
    stream = request.prompt_stream
    assert stream is not None
    state = request.prompt_stream_update_coalescing_state
    assert state is not None and state.latest_pending_token_ids == (10, 12)
    if terminal_kind == "question":
        _send_event(scheduler, request, PromptStreamEventType.PROMPT_FINISHED, [30])
    else:
        _send_event(
            scheduler,
            request,
            PromptStreamEventType.SOURCE_FINISHED,
            None,
            version=1,
            sequence_no=2,
            source_type=PromptSourceType.UPDATE,
        )
    assert state.latest_pending_token_ids is None
    work_items = list(stream.pending_work_items)
    expected_phases = [PromptStreamExecutionPhase.UPDATE] * 2
    if terminal_kind == "question":
        expected_phases.append(PromptStreamExecutionPhase.QUESTION)
    assert [work.phase for work in work_items] == expected_phases
    assert work_items[0].update_token_ids == (10, 11)
    assert work_items[1].update_token_ids == (10, 12)
    assert work_items[1].execution_base_token_ids == (10, 11)
    assert not request.can_decode_from_prompt_stream()


def test_adaptive_zero_forward_work_releases_backpressure() -> None:
    """严格前缀截断无需 GPU 回包，消费后应正常放行最新 Update。"""
    scheduler = _make_adaptive_scheduler()
    request = _add_dormant_request(scheduler)
    _send_event(scheduler, request, PromptStreamEventType.UPDATE, [10, 11])
    scheduler._prepare_prompt_stream_execution(request)
    _finish_active_view(request)
    scheduler._reconcile_prompt_stream_request(request)
    _send_event(scheduler, request, PromptStreamEventType.UPDATE, [10], version=1)
    _send_event(
        scheduler, request, PromptStreamEventType.UPDATE, [10, 12], version=2
    )
    state = request.prompt_stream_update_coalescing_state
    assert state is not None
    assert state.latest_pending_token_ids == (10, 12)
    assert request.has_outstanding_prompt_stream_update()
    scheduler._prepare_prompt_stream_execution(request)
    assert not request.has_outstanding_prompt_stream_update()
    scheduler._reconcile_prompt_stream_request(request)
    assert state.latest_pending_token_ids is None
    assert request.has_outstanding_prompt_stream_update()


def test_adaptive_abort_removes_candidate_and_wakes_other_request() -> None:
    """取消阻挡请求释放压力，且取消候选不能遗留待检查的幽灵窗口。"""
    scheduler = _make_adaptive_scheduler()
    blocker = _add_running_request(scheduler, "busy-template")
    cancelled = _add_dormant_request(scheduler, "cancelled-candidate")
    request = _add_dormant_request(scheduler)
    for candidate in (cancelled, request):
        _send_event(scheduler, candidate, PromptStreamEventType.UPDATE, [10, 11])
    controller = scheduler._prompt_stream_update_coalescer
    assert controller is not None
    scheduler.finish_requests(cancelled.request_id, RequestStatus.FINISHED_ABORTED)
    assert cancelled.request_id not in controller.adaptive_request_ids
    scheduler.finish_requests(blocker.request_id, RequestStatus.FINISHED_ABORTED)
    # abort 后由 EngineCore 下一轮/空闲检查统一评估可执行机会。
    assert scheduler.flush_due_update_coalescing(now=10.1) == 1
    assert request.request_id not in controller.adaptive_request_ids
    assert list(scheduler.waiting) == [request]


def test_adaptive_pressure_counts_inflight_normal_prefill_until_reply() -> None:
    """普通请求已乐观推进到末尾时仍占用真实在途预算，回包后才解除。"""
    scheduler = _make_adaptive_scheduler()
    scheduler.max_num_scheduled_tokens = 4
    scheduler.max_num_running_reqs = 8
    ordinary = _make_request([20, 21, 22, 23])
    ordinary.request_id = "ordinary-prefill"
    ordinary.status = RequestStatus.RUNNING
    scheduler.requests[ordinary.request_id] = ordinary
    scheduler.running.append(ordinary)
    output = _make_step(ordinary, 4)
    scheduler._update_after_schedule(output)
    assert ordinary.num_computed_tokens == ordinary.num_tokens
    request = _add_dormant_request(scheduler)
    _send_event(scheduler, request, PromptStreamEventType.UPDATE, [10, 11])
    state = request.prompt_stream_update_coalescing_state
    assert state is not None and state.latest_pending_token_ids == (10, 11)
    context = scheduler._get_prompt_stream_update_context(request, now=10.0)
    assert context.token_pressure == 1.0
    _return_step(scheduler, output)
    assert state.latest_pending_token_ids is None
    assert list(scheduler.waiting) == [request]


def test_adaptive_append_source_finished_does_not_bypass_pressure() -> None:
    """Append 封闭不是最终 Update 出口，不得使用默认空负载上下文放行。"""
    scheduler = _make_adaptive_scheduler()
    _add_running_request(scheduler, "busy-template")
    request = _add_dormant_request(scheduler)
    _send_event(scheduler, request, PromptStreamEventType.APPEND, [])
    _send_event(scheduler, request, PromptStreamEventType.UPDATE, [10, 11])
    _send_event(
        scheduler,
        request,
        PromptStreamEventType.SOURCE_FINISHED,
        None,
        version=0,
        sequence_no=1,
        source_type=PromptSourceType.APPEND,
    )
    state = request.prompt_stream_update_coalescing_state
    assert state is not None and state.latest_pending_token_ids == (10, 11)
    assert not request.has_outstanding_prompt_stream_update()
    assert request.request_id in scheduler._prompt_stream_dormant_reasons


def test_adaptive_recovery_preserves_window_and_candidate_registration() -> None:
    """抢占保留未放行窗口，并允许旧 admission 按原恢复规则取消。"""
    scheduler = _make_adaptive_scheduler()
    request = _add_dormant_request(scheduler)
    _send_event(scheduler, request, PromptStreamEventType.UPDATE, [10, 11])
    _send_event(
        scheduler, request, PromptStreamEventType.UPDATE, [10, 12], version=1
    )
    blocker = _add_running_request(scheduler, "busy-template")
    scheduler.waiting.remove_request(request)
    request.status = RequestStatus.RUNNING
    scheduler._preempt_request(request, timestamp=10.1)
    controller = scheduler._prompt_stream_update_coalescer
    state = request.prompt_stream_update_coalescing_state
    assert controller is not None and state is not None
    assert state.latest_pending_token_ids == (10, 12)
    assert request.request_id in controller.adaptive_request_ids
    assert request.get_prompt_stream_update_deadline() is None
    # 原恢复语义取消尚未完成的旧 admission，只保留最新窗口与 PA 恢复。
    assert not request.has_outstanding_prompt_stream_update()
    assert list(scheduler.waiting) == [request]
    assert scheduler.flush_due_update_coalescing(now=1000.0) == 0
    _finish_active_view(blocker)
    scheduler._reconcile_prompt_stream_request(blocker)
    scheduler._prepare_prompt_stream_execution(request)
    _finish_active_view(request)
    scheduler._reconcile_prompt_stream_request(request)
    assert state.latest_pending_token_ids is None
    assert request.request_id not in controller.adaptive_request_ids
    assert request.has_outstanding_prompt_stream_update()
