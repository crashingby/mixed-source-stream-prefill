# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""验证 Query probe 的事件边界、回包提交、缓存所有权和恢复语义。"""

import pytest

from tests.v1.test_prompt_stream_request import _make_kv_cache_manager
from vllm.sampling_params import SamplingParams
from vllm.v1.engine.prompt_stream import PromptStreamEvent
from vllm.v1.query_probe import QueryProbeResult
from vllm.v1.request import PromptStreamKVSegment, Request


def make_request() -> Request:
    """创建有固定 Query、非块对齐模板的请求。"""
    request = Request(
        request_id="probe-test",
        prompt_token_ids=[1, 2, 3],
        sampling_params=SamplingParams(max_tokens=1),
        pooling_params=None,
        probe_token_ids=[21, 22, 23],
        probe_keep_ratio=0.5,
    )
    request.apply_prompt_stream_event(
        PromptStreamEvent(
            version=0,
            arrival_time=0.0,
            request_id=request.request_id,
            source_id="__prompt_stream_start__",
            event_type="append",
            source_type="append",
            chunk_id="start",
            token_ids=[],
            sequence_no=0,
        )
    )
    request.prepare_prompt_stream_execution(16)
    request.advance_prompt_stream_computed_tokens(16)
    return request


def append(request: Request, count: int, sequence: int) -> None:
    """向同一来源追加真实事件，保留顺序与稳定块边界。"""
    request.apply_prompt_stream_event(
        PromptStreamEvent(
            version=0,
            arrival_time=0.0,
            request_id=request.request_id,
            source_id="append",
            event_type="append",
            source_type="append",
            chunk_id=str(sequence),
            token_ids=[5] * count,
            sequence_no=sequence,
        )
    )
    request.prepare_prompt_stream_execution(16)


def test_joint_commit_and_private_blocks() -> None:
    """联合执行的乐观进度不能发布策略，释放 Query 不能丢掉正式块。"""
    request = make_request()
    manager = _make_kv_cache_manager(16)
    append(request, 62, 0)
    view = request.prompt_stream_execution_view
    assert view is not None and view.probe_start == 80
    assert request.num_computed_tokens == 3
    work = request.build_query_probe_work(80)
    assert work is not None and work.candidate_blocks == [1, 2, 3]
    assert manager.allocate_slots(request, 80) is not None
    assert (
        len(manager.get_prompt_stream_execution_blocks(request).get_block_ids()[0]) == 6
    )
    assert len(manager.get_prompt_stream_blocks(request).get_block_ids()[0]) == 5
    request.advance_prompt_stream_computed_tokens(80)
    assert request.prompt_stream.active_work_item is not None
    assert not request.query_probe.chunks[0].published
    assert request.complete_query_probe_step(
        work,
        QueryProbeResult(
            work.attempt_id,
            3,
            [0.6, 0.9, 0.3],
            [1.2, 1.8, 0.6],
        ),
    )
    manager.coordinator.free(
        request.get_prompt_stream_segment_key(PromptStreamKVSegment.PROBE)
    )
    assert len(manager.get_prompt_stream_blocks(request).get_block_ids()[0]) == 5
    assert request.query_probe.chunks[0].kept_blocks == [1, 2]
    append(request, 47, 1)
    next_work = request.build_query_probe_work(50)
    assert next_work.dropped_blocks == [3]
    # 第二个 Append 从真实末尾继续覆盖 padding，边界块 4 始终保护。
    assert request.num_computed_tokens == 65
    assert next_work.candidate_blocks == [5, 6]


def test_chunked_query_and_recovery() -> None:
    """跨 step 分数统一归一化；恢复保留旧策略并拒绝旧 attempt 回包。"""
    request = make_request()
    append(request, 62, 0)
    first = request.build_query_probe_work(78)
    request.advance_prompt_stream_computed_tokens(78)
    assert not request.complete_query_probe_step(
        first,
        QueryProbeResult(
            first.attempt_id,
            1,
            [0.2, 0.3, 0.1],
            [0.4, 0.6, 0.2],
        ),
    )
    request.prepare_prompt_stream_execution(16)
    last = request.build_query_probe_work(2)
    request.advance_prompt_stream_computed_tokens(2)
    assert request.complete_query_probe_step(
        last,
        QueryProbeResult(
            last.attempt_id,
            2,
            [0.4, 0.6, 0.2],
            [0.8, 1.2, 0.4],
        ),
    )
    assert request.query_probe.chunks[0].raw_scores == pytest.approx([0.2, 0.3, 0.1])
    append(request, 47, 1)
    stale = request.build_query_probe_work(50)
    request.reset_prompt_stream_computed_state()
    assert [
        work.prefix_append_logical_len
        for work in request.prompt_stream.pending_work_items
    ] == [65, 112]
    assert request.query_probe.chunks[0].published
    assert not request.complete_query_probe_step(
        stale,
        QueryProbeResult(
            stale.attempt_id,
            3,
            [0.1, 0.2],
            [0.2, 0.4],
        ),
    )


def test_joint_allocation_is_atomic() -> None:
    """正式段与 Query 的总容量不足时，不留下半份分配。"""
    request = make_request()
    manager = _make_kv_cache_manager(16)
    append(request, 16 * 40, 0)
    free_before = manager.block_pool.get_num_free_blocks()
    assert (
        manager.allocate_slots(
            request, request.num_tokens - request.num_computed_tokens
        )
        is None
    )
    assert manager.block_pool.get_num_free_blocks() == free_before
    assert not manager.has_prompt_stream_segment_blocks(
        request, PromptStreamKVSegment.PROBE
    )


def test_finish_query_mismatch() -> None:
    """问题不一致在入队前拒绝，不污染正式 suffix。"""
    request = make_request()
    with pytest.raises(ValueError, match="registered probe_query"):
        request.apply_prompt_stream_event(
            PromptStreamEvent(
                version=0,
                arrival_time=0.0,
                request_id=request.request_id,
                source_id="finish",
                event_type="prompt_finished",
                source_type=None,
                chunk_id=None,
                sequence_no=0,
                token_ids=[99],
            )
        )
    assert not request.prompt_stream.finished
