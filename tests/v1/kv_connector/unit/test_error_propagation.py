# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import Callable
from unittest.mock import Mock

import pytest

from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.request import FinishReason, Request, RequestStatus

from .utils import (
    create_model_runner_output,
    create_request,
    create_scheduler,
    create_vllm_config,
)

pytestmark = pytest.mark.cpu_test


def _make_get_num_new_matched_tokens(
    req_num_new_matched_tokens: dict[str, int],
    async_load: bool,
) -> Callable[[Request, int], tuple[int, bool]]:
    def get_num_new_matched_tokens(request: Request, _: int) -> tuple[int, bool]:
        value = req_num_new_matched_tokens.get(request.request_id, 0)
        return value, async_load

    return get_num_new_matched_tokens


@pytest.fixture
def fail_scheduler():
    """scheduler with kv_load_failure_policy='fail'"""
    vllm_config = create_vllm_config()
    vllm_config.kv_transfer_config.kv_load_failure_policy = "fail"
    return create_scheduler(vllm_config)


def test_error_propagation_sync_load(fail_scheduler: Scheduler):
    """test invalid_block_ids with fail policy -> FINISHED_ERROR (sync load)"""
    num_prompt_blocks = 100
    num_external_computed_blocks = 99
    invalid_block_idx = 50

    num_prompt_tokens = num_prompt_blocks * fail_scheduler.block_size
    num_external_computed_tokens = (
        num_external_computed_blocks * fail_scheduler.block_size
    )

    request = create_request(num_tokens=num_prompt_tokens)
    fail_scheduler.add_request(request=request)

    req_num_new_matched_tokens = {
        request.request_id: num_external_computed_tokens,
    }

    fail_scheduler.connector = Mock()
    fail_scheduler.connector.get_num_new_matched_tokens.side_effect = (
        _make_get_num_new_matched_tokens(req_num_new_matched_tokens, False)
    )
    fail_scheduler.connector.request_finished.return_value = (False, None)
    fail_scheduler.connector.take_events.return_value = ()

    scheduler_output = fail_scheduler.schedule()

    assert len(fail_scheduler.running) == 1
    assert len(scheduler_output.scheduled_new_reqs) == 1
    assert fail_scheduler.connector.get_num_new_matched_tokens.call_count == 1

    req_block_ids = scheduler_output.scheduled_new_reqs[0].block_ids[0]
    invalid_block_ids = {req_block_ids[invalid_block_idx]}
    model_runner_output = create_model_runner_output(
        [request],
        invalid_block_ids=invalid_block_ids,
        use_eos=True,
    )

    outputs = fail_scheduler.update_from_output(scheduler_output, model_runner_output)

    assert request.status == RequestStatus.FINISHED_ERROR
    assert request.get_finished_reason() == FinishReason.ERROR

    assert len(outputs) == 1
    engine_outputs = next(iter(outputs.values()))
    assert len(engine_outputs.outputs) == 1
    output = engine_outputs.outputs[0]
    assert output.request_id == request.request_id
    assert output.finish_reason == FinishReason.ERROR

    assert len(fail_scheduler.running) == 0


def test_error_propagation_async_load(fail_scheduler: Scheduler):
    """test invalid_block_ids with fail policy -> FINISHED_ERROR (async load)"""
    num_prompt_blocks = 100
    num_external_computed_blocks = 99
    invalid_block_idx = 50

    num_prompt_tokens = num_prompt_blocks * fail_scheduler.block_size
    num_external_computed_tokens = (
        num_external_computed_blocks * fail_scheduler.block_size
    )

    request = create_request(num_tokens=num_prompt_tokens)
    fail_scheduler.add_request(request=request)

    req_num_new_matched_tokens = {
        request.request_id: num_external_computed_tokens,
    }

    fail_scheduler.connector = Mock()
    fail_scheduler.connector.get_num_new_matched_tokens.side_effect = (
        _make_get_num_new_matched_tokens(req_num_new_matched_tokens, True)
    )
    fail_scheduler.connector.request_finished.return_value = (False, None)
    fail_scheduler.connector.take_events.return_value = ()

    scheduler_output = fail_scheduler.schedule()

    assert len(fail_scheduler.skipped_waiting) == 1
    assert request.status == RequestStatus.WAITING_FOR_REMOTE_KVS
    assert request.num_computed_tokens == num_external_computed_tokens

    (req_block_ids,) = fail_scheduler.kv_cache_manager.get_block_ids(request.request_id)
    invalid_block_ids = {req_block_ids[invalid_block_idx]}
    model_runner_output = create_model_runner_output(
        reqs=[],
        finished_recving=set(),
        invalid_block_ids=invalid_block_ids,
        use_eos=True,
    )

    outputs = fail_scheduler.update_from_output(scheduler_output, model_runner_output)

    assert request.status == RequestStatus.FINISHED_ERROR
    assert request.get_finished_reason() == FinishReason.ERROR

    assert len(outputs) == 1
    engine_outputs = next(iter(outputs.values()))
    assert len(engine_outputs.outputs) == 1
    output = engine_outputs.outputs[0]
    assert output.request_id == request.request_id
    assert output.finish_reason == FinishReason.ERROR

    assert len(fail_scheduler.waiting) == 0
    assert len(fail_scheduler.skipped_waiting) == 0


def _schedule_async_load(
    scheduler: Scheduler,
) -> tuple[Request, SchedulerOutput]:
    """创建一个进入 WAITING_FOR_REMOTE_KVS 的轻量请求。"""
    num_prompt_tokens = scheduler.block_size * 4
    num_external_tokens = scheduler.block_size * 3
    request = create_request(num_tokens=num_prompt_tokens)
    scheduler.add_request(request=request)

    scheduler.connector = Mock()
    scheduler.connector.get_num_new_matched_tokens.side_effect = (
        _make_get_num_new_matched_tokens(
            {request.request_id: num_external_tokens},
            True,
        )
    )
    scheduler.connector.request_finished.return_value = (False, None)
    scheduler.connector.take_events.return_value = ()

    scheduler_output = scheduler.schedule()
    assert request.status == RequestStatus.WAITING_FOR_REMOTE_KVS
    return request, scheduler_output


def test_request_level_load_error_fail_policy(
    fail_scheduler: Scheduler,
) -> None:
    """请求级加载失败在 fail policy 下直接终止异步请求。"""
    request, scheduler_output = _schedule_async_load(fail_scheduler)
    model_runner_output = create_model_runner_output(
        reqs=[],
        finished_recving={request.request_id},
        failed_recving={request.request_id},
    )

    outputs = fail_scheduler.update_from_output(
        scheduler_output,
        model_runner_output,
    )

    assert request.status == RequestStatus.FINISHED_ERROR
    engine_outputs = next(iter(outputs.values()))
    assert engine_outputs.outputs[0].request_id == request.request_id
    assert engine_outputs.outputs[0].finish_reason == FinishReason.ERROR
    assert request.request_id not in fail_scheduler.finished_recving_kv_req_ids


def test_request_level_load_error_recompute_policy() -> None:
    """请求级加载失败清零进度，并复用远端等待状态完成恢复。"""
    vllm_config = create_vllm_config()
    vllm_config.kv_transfer_config.kv_load_failure_policy = "recompute"
    scheduler = create_scheduler(vllm_config)
    request, scheduler_output = _schedule_async_load(scheduler)
    model_runner_output = create_model_runner_output(
        reqs=[],
        finished_recving={request.request_id},
        failed_recving={request.request_id},
    )

    outputs = scheduler.update_from_output(
        scheduler_output,
        model_runner_output,
    )

    assert not outputs[request.client_index].outputs
    assert request.status == RequestStatus.WAITING_FOR_REMOTE_KVS
    assert request.num_computed_tokens == 0
    assert request.request_id in scheduler.finished_recving_kv_req_ids
    assert request.request_id in scheduler.failed_recving_kv_req_ids

    assert scheduler._try_promote_blocked_waiting_request(request)
    assert request.status == RequestStatus.WAITING
    assert request.request_id not in scheduler.finished_recving_kv_req_ids
    assert request.request_id not in scheduler.failed_recving_kv_req_ids
