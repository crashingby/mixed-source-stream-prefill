# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from copy import deepcopy
from typing import Any
from unittest.mock import Mock

import pytest

from vllm.config import SchedulerConfig
from vllm.sampling_params import SamplingParams
from vllm.v1.core.sched.output import NewRequestData
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.engine.prompt_stream import (
    PromptSourceType,
    PromptStreamEvent,
    PromptStreamEventType,
    parse_prompt_stream_padding_range,
)
from vllm.v1.request import PromptStreamExecutionPhase, Request, RequestStatus


def _event(kind: str, tokens: list[int], version: int = 0) -> PromptStreamEvent:
    """构造按 source 独立递增的事件，不需要启动 EngineCore。"""
    return PromptStreamEvent(
        request_id="masked-test",
        source_id=kind,
        event_type=PromptStreamEventType(kind),
        source_type=(PromptSourceType(kind) if kind != "prompt_finished" else None),
        chunk_id=f"{kind}-{version}",
        version=version,
        sequence_no=version,
        token_ids=tokens,
        arrival_time=float(version),
    )


def _request(masked: bool = True) -> Request:
    """固定模板长 3，block size 4，可覆盖非对齐分段。"""
    request = Request("masked-test", [1, 2, 3], SamplingParams(max_tokens=1), None)
    request.prompt_stream_masked_padding = masked
    request.apply_prompt_stream_event(_event("append", []))
    request.prepare_prompt_stream_execution(4)
    return request


def _complete(request: Request) -> None:
    """提交当前视图剩余 token，模拟真实 forward 完成回调。"""
    request.advance_prompt_stream_computed_tokens(
        request.num_tokens - request.num_computed_tokens
    )


def test_range_follows_frozen_work_and_append_overwrite() -> None:
    """新 Append 到达不能修改 active Update 的 mask，切换后才能覆盖旧洞。"""
    request = _request()
    assert request.get_prompt_stream_padding_range() == (3, 4)
    _complete(request)
    request.apply_prompt_stream_event(_event("update", [10, 11, 12]))
    request.prepare_prompt_stream_execution(4)
    request.advance_prompt_stream_computed_tokens(1)
    request.apply_prompt_stream_event(_event("append", [4, 5], 1))
    request.prepare_prompt_stream_execution(4)
    assert (
        request.prompt_stream_execution_view.phase == PromptStreamExecutionPhase.UPDATE
    )
    assert request.get_prompt_stream_padding_range() == (3, 4)
    assert request.prompt_token_ids == [1, 2, 3, 3, 10, 11, 12]
    _complete(request)

    request.prepare_prompt_stream_execution(4)
    assert request.get_prompt_stream_padding_range() == (5, 8)
    # 旧 padding 槽位 3 已变成真实 Append token，不能继续屏蔽。
    assert request.prompt_token_ids[:5] == [1, 2, 3, 4, 5]
    data = NewRequestData.from_request(request, ([1, 2],))
    assert data.prompt_stream_padding_range == (5, 8)
    _complete(request)
    request.apply_prompt_stream_event(_event("prompt_finished", [20, 21]))
    request.prepare_prompt_stream_execution(4)
    phase = request.prompt_stream_execution_view.phase
    assert phase == PromptStreamExecutionPhase.QUESTION
    assert request.get_prompt_stream_padding_range() == (5, 8)
    _complete(request)
    assert request.can_decode_from_prompt_stream()


def test_recovery_reinstalls_mask_and_default_stays_unmasked() -> None:
    """抢占恢复重新冻结执行范围；未开启开关的历史模式不传 mask。"""
    request = _request()
    _complete(request)
    request.apply_prompt_stream_event(_event("update", [10, 11, 12]))
    request.prepare_prompt_stream_execution(4)
    request.advance_prompt_stream_computed_tokens(1)
    request.reset_prompt_stream_computed_state()
    request.prepare_prompt_stream_execution(4)
    assert request.get_prompt_stream_padding_range() == (3, 4)
    _complete(request)
    request.prepare_prompt_stream_execution(4)
    assert request.get_prompt_stream_padding_range() == (3, 4)
    assert _request(masked=False).get_prompt_stream_padding_range() is None


def _manifest() -> dict[str, Any]:
    """P 已物化 10 个位置，D prompt 在其后附加一个尚无 KV 的首 token。"""
    return {
        "prompt_stream_masked_padding": {"version": 1, "start": 3, "end": 4},
        "prompt_stream_physical_prompt_len": 10,
        "prompt_stream_logical_prompt_len": 9,
        "prompt_stream_prefill_output_tokens": 1,
        "prompt_stream_exact_transfer": True,
    }


def test_decode_request_preserves_mask_without_stream_state() -> None:
    """D 是普通 completion，也必须保留 P 的区间并隔离普通 APC。"""
    request = Request(
        "decode",
        list(range(11)),
        SamplingParams(max_tokens=2, extra_args={"kv_transfer_params": _manifest()}),
        None,
    )
    assert request.prompt_stream is None
    assert request.get_prompt_stream_padding_range() == (3, 4)
    assert request.skip_reading_prefix_cache
    request.kv_transfer_params["do_remote_prefill"] = False
    request.append_output_token_ids(42)
    assert request.get_prompt_stream_padding_range() == (3, 4)
    data = NewRequestData.from_request(request, ([1],))
    assert data.prompt_stream_padding_range == (3, 4)


@pytest.mark.parametrize(
    "field,value",
    [
        ("version", 2),
        ("version", True),
        ("start", -1),
        ("end", 11),
        ("end", 2),
        ("start", "3"),
    ],
)
def test_invalid_transfer_range_is_rejected(field: str, value: object) -> None:
    """拒绝未知版本、越界及非整数，不能把不可信坐标交给 CUDA。"""
    manifest = deepcopy(_manifest())
    manifest["prompt_stream_masked_padding"][field] = value
    with pytest.raises(ValueError):
        parse_prompt_stream_padding_range(manifest, 11)


def test_manifest_checks_lengths_and_empty_hole() -> None:
    """物理长度含 padding，逻辑长度不含；空区间仍保留开启标记。"""
    manifest = _manifest()
    with pytest.raises(ValueError, match="tail"):
        parse_prompt_stream_padding_range(manifest, 10)
    manifest["prompt_stream_logical_prompt_len"] = 10
    with pytest.raises(ValueError, match="lengths"):
        parse_prompt_stream_padding_range(manifest, 11)
    manifest["prompt_stream_masked_padding"]["start"] = 4
    assert parse_prompt_stream_padding_range(manifest, 11) == (4, 4)
    assert parse_prompt_stream_padding_range(None, 11) is None


def test_masked_padding_changes_scheduler_graph_hash() -> None:
    """新旧语义不能共用编译缓存，但关闭配置仍是默认行为。"""
    ordinary = SchedulerConfig.default_factory()
    masked = SchedulerConfig.default_factory(prompt_stream_masked_padding=True)
    assert not ordinary.prompt_stream_masked_padding
    assert ordinary.compute_hash() != masked.compute_hash()


@pytest.mark.parametrize("aborted", [False, True])
def test_p_only_finish_manifest_without_connector(aborted: bool) -> None:
    """不创建 scheduler/worker，只验证释放请求时的 P-only 语义清单。"""
    request = _request()
    _complete(request)
    request.apply_prompt_stream_event(_event("prompt_finished", [20, 21]))
    request.prepare_prompt_stream_execution(4)
    _complete(request)
    request.append_output_token_ids(42)
    request.status = (
        RequestStatus.FINISHED_ABORTED
        if aborted
        else RequestStatus.FINISHED_LENGTH_CAPPED
    )
    # 模拟无 connector 的资源释放边界，不能隐式创建 GPU 或模型。
    scheduler = Scheduler.__new__(Scheduler)
    scheduler._prompt_stream_update_coalescer = None
    scheduler._prompt_stream_retained_waiting_req_ids = set()
    scheduler._prompt_stream_worker_removal_req_ids = set()
    scheduler._inflight_prefills = set()
    scheduler.connector = None
    scheduler.encoder_cache_manager = Mock()
    scheduler.finished_req_ids = set()
    scheduler.finished_req_ids_dict = None
    scheduler._free_blocks = Mock()

    manifest = scheduler._free_request(request)
    scheduler._free_blocks.assert_called_once_with(request)
    assert request.request_id in scheduler.finished_req_ids
    if aborted:
        assert manifest is None
        return
    assert manifest is not None
    assert manifest["prompt_stream_prompt_token_ids"] == [1, 2, 3, 3, 20, 21]
    assert manifest["prompt_stream_physical_prompt_len"] == 6
    assert manifest["prompt_stream_logical_prompt_len"] == 5
    assert manifest["prompt_stream_prefill_output_tokens"] == 1
    assert parse_prompt_stream_padding_range(manifest, 7) == (3, 4)
    assert "do_remote_prefill" not in manifest
    assert "remote_block_ids" not in manifest
