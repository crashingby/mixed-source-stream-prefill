# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from unittest.mock import patch

import pytest
import torch

from vllm.distributed.kv_transfer.kv_connector.v1.nixl.pull_scheduler import (
    NixlPullConnectorScheduler,
    _trim_prompt_stream_local_blocks,
    _trim_prompt_stream_local_blocks_to_manifest,
)
from vllm.sampling_params import SamplingParams
from vllm.v1.core.kv_cache_manager import KVCacheManager
from vllm.v1.engine.prompt_stream import (
    PromptSourceType,
    PromptStreamEvent,
    PromptStreamEventType,
)
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
)
from vllm.v1.request import (
    PromptStreamExecutionPhase,
    PromptStreamKVSegment,
    Request,
)


def _make_request(prompt_token_ids: list[int]) -> Request:
    """构造只包含 token-id prompt 的生成请求。"""
    return Request(
        request_id="prompt-stream-test",
        prompt_token_ids=prompt_token_ids,
        sampling_params=SamplingParams(max_tokens=2),
        pooling_params=None,
    )


def _make_prompt_stream_decode_request(
    prompt_token_ids: list[int],
) -> Request:
    """构造携带 P 侧精确 block manifest 标记的 D 节点请求。"""
    return Request(
        request_id="prompt-stream-decode-test",
        prompt_token_ids=prompt_token_ids,
        sampling_params=SamplingParams(
            max_tokens=2,
            extra_args={
                "kv_transfer_params": {
                    "prompt_stream_exact_transfer": True,
                }
            },
        ),
        pooling_params=None,
    )


def _make_kv_cache_manager(block_size: int = 4) -> KVCacheManager:
    """构造单一 full-attention group 的 CPU KV manager。"""
    kv_cache_config = KVCacheConfig(
        num_blocks=32,
        kv_cache_tensors=[],
        kv_cache_groups=[
            KVCacheGroupSpec(
                ["layer"],
                FullAttentionSpec(
                    block_size=block_size,
                    num_kv_heads=1,
                    head_size=1,
                    dtype=torch.float32,
                ),
            )
        ],
    )
    return KVCacheManager(
        kv_cache_config=kv_cache_config,
        max_model_len=64,
        scheduler_block_size=block_size,
        hash_block_size=block_size,
        max_num_batched_tokens=64,
        enable_caching=False,
    )


def _make_event(
    event_type: PromptStreamEventType,
    token_ids: list[int] | None,
    *,
    source_id: str,
    source_type: PromptSourceType | None,
    version: int = 0,
    sequence_no: int = 0,
) -> PromptStreamEvent:
    """构造确定性的流式 prompt 测试事件。"""
    return PromptStreamEvent(
        request_id="prompt-stream-test",
        source_id=source_id,
        event_type=event_type,
        source_type=source_type,
        chunk_id=f"{source_id}-{version}-{sequence_no}",
        version=version,
        sequence_no=sequence_no,
        token_ids=token_ids,
        arrival_time=float(sequence_no),
    )


def _start_request(request: Request, block_size: int = 4) -> None:
    """应用 start 空事件并准备固定模板的 bootstrap 视图。"""
    request.apply_prompt_stream_event(
        _make_event(
            PromptStreamEventType.APPEND,
            [],
            source_id="__start__",
            source_type=PromptSourceType.APPEND,
        )
    )
    request.prepare_prompt_stream_execution(block_size)


def _finish_active_view(request: Request) -> None:
    """一次性提交当前执行视图剩余的全部 token。"""
    num_scheduled_tokens = request.num_tokens - request.num_computed_tokens
    assert num_scheduled_tokens > 0
    request.advance_prompt_stream_computed_tokens(num_scheduled_tokens)


def _assert_token_views_consistent(request: Request) -> None:
    """保护 Request 可变 token 与只读 ConstantList 始终一致。"""
    assert list(request.all_token_ids) == request._all_token_ids
    assert (
        request.prompt_token_ids == request._all_token_ids[: request.num_prompt_tokens]
    )


def test_rapid_events_preserve_prefill_phase_order() -> None:
    """Finish 快速到达时仍必须按 PA、Update、Question 顺序提交。"""
    request = _make_request([1, 2, 3])
    _start_request(request)
    _finish_active_view(request)

    request.apply_prompt_stream_event(
        _make_event(
            PromptStreamEventType.APPEND,
            [4, 5],
            source_id="append",
            source_type=PromptSourceType.APPEND,
        )
    )
    request.apply_prompt_stream_event(
        _make_event(
            PromptStreamEventType.UPDATE,
            [10, 11],
            source_id="update",
            source_type=PromptSourceType.UPDATE,
        )
    )
    request.apply_prompt_stream_event(
        _make_event(
            PromptStreamEventType.PROMPT_FINISHED,
            [20],
            source_id="question",
            source_type=None,
        )
    )

    request.prepare_prompt_stream_execution(4)
    view = request.prompt_stream_execution_view
    assert view is not None
    assert view.phase == PromptStreamExecutionPhase.PREFIX_APPEND
    assert list(view.token_ids) == [1, 2, 3, 4, 5, 3, 3, 3]
    assert view.required_segments == (PromptStreamKVSegment.PREFIX_APPEND,)
    _finish_active_view(request)

    request.prepare_prompt_stream_execution(4)
    view = request.prompt_stream_execution_view
    assert view is not None
    assert view.phase == PromptStreamExecutionPhase.UPDATE
    assert view.num_computed_tokens == 8
    assert list(view.token_ids) == [1, 2, 3, 4, 5, 3, 3, 3, 10, 11]
    _finish_active_view(request)

    request.prepare_prompt_stream_execution(4)
    view = request.prompt_stream_execution_view
    assert view is not None
    assert view.phase == PromptStreamExecutionPhase.QUESTION
    assert view.num_computed_tokens == 10
    assert not request.can_decode_from_prompt_stream()
    _finish_active_view(request)

    assert request.can_decode_from_prompt_stream()
    assert request.get_prompt_stream_physical_prompt_token_ids() == [
        1,
        2,
        3,
        4,
        5,
        3,
        3,
        3,
        10,
        11,
        20,
    ]
    _assert_token_views_consistent(request)


def test_update_can_run_before_later_append_and_freezes_prefix_snapshot() -> None:
    """Update 先获调度时，不得读取尚未提交的 Append KV。"""
    request = _make_request([1, 2, 3])
    _start_request(request)
    _finish_active_view(request)

    # 两个 source 都就绪，但 Update 事件先进入统一 work 队列。
    request.apply_prompt_stream_event(
        _make_event(
            PromptStreamEventType.UPDATE,
            [10, 11],
            source_id="update",
            source_type=PromptSourceType.UPDATE,
        )
    )
    request.apply_prompt_stream_event(
        _make_event(
            PromptStreamEventType.APPEND,
            [4],
            source_id="append",
            source_type=PromptSourceType.APPEND,
        )
    )

    request.prepare_prompt_stream_execution(4)
    view = request.prompt_stream_execution_view
    assert view is not None
    assert view.phase == PromptStreamExecutionPhase.UPDATE
    # 模板 [1, 2, 3] 的物理尾部 padding 仍在；后到的 Append token 4 不可见。
    assert list(view.token_ids) == [1, 2, 3, 3, 10, 11]
    _finish_active_view(request)

    request.prepare_prompt_stream_execution(4)
    view = request.prompt_stream_execution_view
    assert view is not None
    assert view.phase == PromptStreamExecutionPhase.PREFIX_APPEND
    assert list(view.token_ids) == [1, 2, 3, 4]
    _finish_active_view(request)

    assert request.prompt_stream is not None
    assert request.prompt_stream.prefix_append_committed_logical_len == 4


def test_empty_final_question_is_rejected() -> None:
    """空 Question 没有 final forward，必须 fail closed 避免时序卡死。"""
    request = _make_request([1, 2, 3, 4])
    _start_request(request)
    _finish_active_view(request)

    try:
        request.apply_prompt_stream_event(
            _make_event(
                PromptStreamEventType.PROMPT_FINISHED,
                [],
                source_id="question",
                source_type=None,
            )
        )
    except ValueError as error:
        assert "non-empty final question" in str(error)
    else:
        raise AssertionError("Empty final question should be rejected.")


def test_append_after_update_does_not_recompute_update() -> None:
    """Append 新 chunk 只推进 PA 域，并保留 Update 已提交进度。"""
    request = _make_request([1, 2, 3, 4])
    manager = _make_kv_cache_manager()
    _start_request(request)
    assert manager.allocate_slots(request, num_new_tokens=4) is not None
    _finish_active_view(request)

    request.apply_prompt_stream_event(
        _make_event(
            PromptStreamEventType.UPDATE,
            [10, 11],
            source_id="update",
            source_type=PromptSourceType.UPDATE,
        )
    )
    request.prepare_prompt_stream_execution(4)
    assert manager.allocate_slots(request, num_new_tokens=2) is not None
    _finish_active_view(request)
    assert request.prompt_stream is not None
    assert request.prompt_stream.update_num_computed_tokens == 2
    update_key = request.get_prompt_stream_segment_key(PromptStreamKVSegment.UPDATE)
    update_block_ids_before = manager.get_blocks(update_key).get_block_ids()

    request.apply_prompt_stream_event(
        _make_event(
            PromptStreamEventType.APPEND,
            [5],
            source_id="append",
            source_type=PromptSourceType.APPEND,
        )
    )
    request.prepare_prompt_stream_execution(4)
    view = request.prompt_stream_execution_view
    assert view is not None
    assert view.phase == PromptStreamExecutionPhase.PREFIX_APPEND
    assert view.num_computed_tokens == 4
    assert request.num_tokens - request.num_computed_tokens == 4
    assert request.prompt_stream.update_num_computed_tokens == 2

    # 新 Append 跨过旧 PA block 边界时只给 PA 再申请一块，Update block id
    # 必须保持不变，后续 worker 仍按 PA + Update 顺序安装整表。
    assert manager.allocate_slots(request, num_new_tokens=4) is not None
    _finish_active_view(request)
    request.prepare_prompt_stream_execution(4)
    view = request.prompt_stream_execution_view
    assert view is not None
    assert view.phase == PromptStreamExecutionPhase.UPDATE
    assert view.num_computed_tokens == len(view.token_ids)
    assert request.prompt_stream.update_num_computed_tokens == 2
    assert manager.get_blocks(update_key).get_block_ids() == update_block_ids_before
    pa_block_ids = manager.get_blocks(request.request_id).get_block_ids()
    assert manager.get_prompt_stream_execution_blocks(request).get_block_ids() == (
        tuple(
            pa_group + update_group
            for pa_group, update_group in zip(
                pa_block_ids,
                update_block_ids_before,
            )
        )
    )


def test_interleaved_append_sources_keep_global_tail_order() -> None:
    """多个 Append source 交错提交时仍保持请求级尾部追加。"""
    request = _make_request([1, 2, 3, 4])
    _start_request(request)
    _finish_active_view(request)

    request.apply_prompt_stream_event(
        _make_event(
            PromptStreamEventType.APPEND,
            [10],
            source_id="append-a",
            source_type=PromptSourceType.APPEND,
        )
    )
    request.apply_prompt_stream_event(
        _make_event(
            PromptStreamEventType.APPEND,
            [20],
            source_id="append-b",
            source_type=PromptSourceType.APPEND,
        )
    )
    request.apply_prompt_stream_event(
        _make_event(
            PromptStreamEventType.APPEND,
            [11],
            source_id="append-a",
            source_type=PromptSourceType.APPEND,
            version=1,
            sequence_no=1,
        )
    )

    # 三个 Append 事件分别成为 work item，并保持它们的全局到达顺序。
    request.prepare_prompt_stream_execution(4)
    view = request.prompt_stream_execution_view
    assert view is not None
    assert view.phase == PromptStreamExecutionPhase.PREFIX_APPEND
    assert list(view.token_ids) == [1, 2, 3, 4, 10, 4, 4, 4]
    _finish_active_view(request)

    request.prepare_prompt_stream_execution(4)
    view = request.prompt_stream_execution_view
    assert view is not None
    assert list(view.token_ids) == [1, 2, 3, 4, 10, 20, 4, 4]
    _finish_active_view(request)

    request.prepare_prompt_stream_execution(4)
    view = request.prompt_stream_execution_view
    assert view is not None
    assert list(view.token_ids) == [1, 2, 3, 4, 10, 20, 11, 4]


def test_update_uses_source_local_lcp() -> None:
    """Update 版本变化只回退 UPDATE 域内 LCP 后缀。"""
    request = _make_request([1, 2, 3, 4])
    _start_request(request)
    _finish_active_view(request)

    request.apply_prompt_stream_event(
        _make_event(
            PromptStreamEventType.UPDATE,
            [10, 11, 12],
            source_id="update",
            source_type=PromptSourceType.UPDATE,
        )
    )
    request.prepare_prompt_stream_execution(4)
    _finish_active_view(request)

    mutation = request.apply_prompt_stream_event(
        _make_event(
            PromptStreamEventType.UPDATE,
            [10, 13, 14],
            source_id="update",
            source_type=PromptSourceType.UPDATE,
            version=1,
            sequence_no=1,
        )
    )
    assert mutation.kv_segment_to_truncate == PromptStreamKVSegment.UPDATE
    assert mutation.kv_segment_keep_tokens == 1
    assert mutation.num_tokens_to_invalidate == 2
    assert request.prompt_stream is not None
    assert request.prompt_stream.update_num_computed_tokens == 1

    request.prepare_prompt_stream_execution(4)
    view = request.prompt_stream_execution_view
    assert view is not None
    assert view.phase == PromptStreamExecutionPhase.UPDATE
    assert view.num_computed_tokens == 5
    assert request.num_tokens - request.num_computed_tokens == 2


def test_preemption_replays_all_segments() -> None:
    """物理 blocks 被释放后，长期 source 状态仍能按阶段完整 replay。"""
    request = _make_request([1, 2, 3, 4])
    manager = _make_kv_cache_manager()
    _start_request(request)
    assert manager.allocate_slots(request, num_new_tokens=4) is not None
    _finish_active_view(request)
    request.apply_prompt_stream_event(
        _make_event(
            PromptStreamEventType.UPDATE,
            [10],
            source_id="update",
            source_type=PromptSourceType.UPDATE,
        )
    )
    request.prepare_prompt_stream_execution(4)
    assert manager.allocate_slots(request, num_new_tokens=1) is not None
    _finish_active_view(request)

    manager.free(request)
    assert not manager.has_prompt_stream_segment_blocks(
        request,
        PromptStreamKVSegment.PREFIX_APPEND,
    )
    assert not manager.has_prompt_stream_segment_blocks(
        request,
        PromptStreamKVSegment.UPDATE,
    )
    request.reset_prompt_stream_computed_state()
    request.prepare_prompt_stream_execution(4)
    view = request.prompt_stream_execution_view
    assert view is not None
    assert view.phase == PromptStreamExecutionPhase.PREFIX_APPEND
    assert view.num_computed_tokens == 0
    assert request.prompt_stream is not None
    assert request.prompt_stream.update_num_computed_tokens == 0
    assert manager.allocate_slots(request, num_new_tokens=4) is not None


def test_kv_manager_allocates_two_independent_segments() -> None:
    """allocator 只写活跃域，worker 视图按 PA + UPDATE 顺序合并。"""
    request = _make_request([1, 2, 3, 4])
    manager = _make_kv_cache_manager()
    _start_request(request)

    new_pa_blocks = manager.allocate_slots(request, num_new_tokens=4)
    assert new_pa_blocks is not None
    assert [len(group) for group in new_pa_blocks.blocks] == [1]
    _finish_active_view(request)

    request.apply_prompt_stream_event(
        _make_event(
            PromptStreamEventType.UPDATE,
            [10, 11, 12, 13, 14],
            source_id="update",
            source_type=PromptSourceType.UPDATE,
        )
    )
    request.prepare_prompt_stream_execution(4)
    new_update_blocks = manager.allocate_slots(request, num_new_tokens=5)
    assert new_update_blocks is not None
    assert [len(group) for group in new_update_blocks.blocks] == [2]

    execution_blocks = manager.get_prompt_stream_execution_blocks(request)
    assert [len(group) for group in execution_blocks.blocks] == [3]
    pa_block_ids = manager.get_blocks(request.request_id).get_block_ids()
    update_key = request.get_prompt_stream_segment_key(PromptStreamKVSegment.UPDATE)
    update_block_ids = manager.get_blocks(update_key).get_block_ids()
    assert execution_blocks.get_block_ids() == tuple(
        pa_group + update_group
        for pa_group, update_group in zip(pa_block_ids, update_block_ids)
    )
    assert manager.has_prompt_stream_segment_blocks(
        request,
        PromptStreamKVSegment.PREFIX_APPEND,
    )
    assert manager.has_prompt_stream_segment_blocks(
        request,
        PromptStreamKVSegment.UPDATE,
    )


def test_question_reuses_update_segment_and_final_block_order() -> None:
    """Question 复用 UPDATE partial block，最终只导出 PA + UPDATE 两段。"""
    request = _make_request([1, 2, 3])
    manager = _make_kv_cache_manager()
    _start_request(request)
    assert manager.allocate_slots(request, num_new_tokens=4) is not None
    _finish_active_view(request)

    request.apply_prompt_stream_event(
        _make_event(
            PromptStreamEventType.UPDATE,
            [10, 11, 12, 13, 14],
            source_id="update",
            source_type=PromptSourceType.UPDATE,
        )
    )
    request.prepare_prompt_stream_execution(4)
    assert manager.allocate_slots(request, num_new_tokens=5) is not None
    _finish_active_view(request)

    request.apply_prompt_stream_event(
        _make_event(
            PromptStreamEventType.PROMPT_FINISHED,
            [20, 21],
            source_id="question",
            source_type=None,
        )
    )
    request.prepare_prompt_stream_execution(4)
    question_blocks = manager.allocate_slots(request, num_new_tokens=2)
    assert question_blocks is not None
    assert [len(group) for group in question_blocks.blocks] == [0]
    _finish_active_view(request)

    pa_ids = manager.get_blocks(request.request_id).get_block_ids()
    update_key = request.get_prompt_stream_segment_key(PromptStreamKVSegment.UPDATE)
    update_ids = manager.get_blocks(update_key).get_block_ids()
    final_ids = manager.get_prompt_stream_block_ids(request)
    assert final_ids == tuple(
        pa_group + update_group for pa_group, update_group in zip(pa_ids, update_ids)
    )
    assert [len(group) for group in final_ids] == [3]
    assert len(request.get_prompt_stream_physical_prompt_token_ids()) == 11
    assert len(manager.get_prompt_stream_segment_keys(request)) == 2


def test_nixl_load_ignores_d_only_tail_block() -> None:
    """D 首 token 跨 block 时，NIXL 只加载 remote prefix 对应 blocks。"""
    manager = _make_kv_cache_manager(block_size=4)
    local_block_ids = ([10, 11, 12],)
    trimmed_block_ids = _trim_prompt_stream_local_blocks(
        local_block_ids,
        manager.kv_cache_config,
        num_external_tokens=8,
    )
    assert trimmed_block_ids == ([10, 11],)


def test_update_truncation_never_frees_prefix_append_blocks() -> None:
    """Update LCP 截断只改变 UPDATE 域的 block 数。"""
    request = _make_request([1, 2, 3, 4])
    manager = _make_kv_cache_manager()
    _start_request(request)
    assert manager.allocate_slots(request, num_new_tokens=4) is not None
    _finish_active_view(request)

    request.apply_prompt_stream_event(
        _make_event(
            PromptStreamEventType.UPDATE,
            [10, 11, 12, 13, 14],
            source_id="update",
            source_type=PromptSourceType.UPDATE,
        )
    )
    request.prepare_prompt_stream_execution(4)
    assert manager.allocate_slots(request, num_new_tokens=5) is not None
    _finish_active_view(request)
    pa_block_ids_before = manager.get_blocks(request.request_id).get_block_ids()

    mutation = request.apply_prompt_stream_event(
        _make_event(
            PromptStreamEventType.UPDATE,
            [10, 20],
            source_id="update",
            source_type=PromptSourceType.UPDATE,
            version=1,
            sequence_no=1,
        )
    )
    assert mutation.kv_segment_to_truncate == PromptStreamKVSegment.UPDATE
    manager.truncate_prompt_stream_segment_blocks(
        request,
        mutation.kv_segment_to_truncate,
        mutation.kv_segment_keep_tokens,
    )

    assert manager.get_blocks(request.request_id).get_block_ids() == pa_block_ids_before
    update_key = request.get_prompt_stream_segment_key(PromptStreamKVSegment.UPDATE)
    assert [len(group) for group in manager.get_blocks(update_key).blocks] == [1]


def test_nixl_manifest_ignores_d_only_tail_block() -> None:
    """D 的 16N+1 tail block 不属于 P manifest，不得参与 NIXL 读取。"""
    local_block_ids = ([10, 11, 12],)
    trimmed_block_ids = _trim_prompt_stream_local_blocks_to_manifest(
        local_block_ids,
        expected_block_counts=(2,),
    )
    assert trimmed_block_ids == ([10, 11],)


def test_nixl_manifest_rejects_insufficient_local_blocks() -> None:
    """D 本地 blocks 少于 P manifest 时必须 fail closed。"""
    with pytest.raises(
        ValueError,
        match="local block count is smaller than manifest",
    ):
        _trim_prompt_stream_local_blocks_to_manifest(
            ([10],),
            expected_block_counts=(2,),
        )


def test_prompt_stream_decode_request_skips_normal_prefix_cache() -> None:
    """精确 prompt-stream transfer 请求不得读普通 APC namespace。"""
    request = _make_prompt_stream_decode_request([1, 2, 3])

    assert request.uses_prompt_stream_kv_transfer()
    assert request.skip_reading_prefix_cache


def test_prompt_stream_decode_request_is_not_written_to_normal_cache() -> None:
    """精确 prompt-stream transfer 的近似 KV 不得污染普通 APC。"""
    request = _make_prompt_stream_decode_request([1, 2, 3])
    manager = _make_kv_cache_manager()
    manager.enable_caching = True

    with patch.object(manager.coordinator, "cache_blocks") as cache_blocks:
        manager.cache_blocks(request, num_computed_tokens=3)

    cache_blocks.assert_not_called()


def test_prompt_stream_decode_forward_is_not_cached_during_allocation() -> None:
    """D 后续本地 forward 分配 slots 时仍不得写入普通 APC。"""
    request = _make_prompt_stream_decode_request([1, 2, 3, 4, 5])
    manager = _make_kv_cache_manager()
    manager.enable_caching = True

    initial_blocks = manager.allocate_slots(
        request,
        num_new_tokens=1,
        num_external_computed_tokens=4,
        delay_cache_blocks=True,
    )
    assert initial_blocks is not None

    request.num_computed_tokens = 5
    request.append_output_token_ids(6)
    with patch.object(manager.coordinator, "cache_blocks") as cache_blocks:
        next_blocks = manager.allocate_slots(
            request,
            num_new_tokens=1,
            delay_cache_blocks=False,
        )

    assert next_blocks is not None
    cache_blocks.assert_not_called()


def test_prompt_stream_exact_transfer_uses_manifest_token_boundary() -> None:
    """16N+1 D prompt 只加载 16N remote tokens，并忽略重算阈值。"""
    num_remote_tokens = 32
    request = _make_prompt_stream_decode_request(list(range(num_remote_tokens + 1)))
    assert request.kv_transfer_params is not None
    request.kv_transfer_params.update(
        {
            "do_remote_prefill": True,
            "remote_num_tokens": num_remote_tokens,
        }
    )
    scheduler = object.__new__(NixlPullConnectorScheduler)
    scheduler.kv_recompute_threshold = num_remote_tokens + 1

    num_external_tokens, is_async = scheduler.get_num_new_matched_tokens(
        request,
        num_computed_tokens=0,
    )

    assert num_external_tokens == num_remote_tokens
    assert is_async
