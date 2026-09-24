# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from vllm.sampling_params import SamplingParams
from vllm.v1.core.sched.output import NewRequestData
from vllm.v1.request import Request
from vllm.v1.utils import CpuGpuBuffer
from vllm.v1.worker.gpu_input_batch import CachedRequestState, InputBatch
from vllm.v1.worker.gpu_model_runner import GPUModelRunner


def _request(
    req_id: str, padding_range: tuple[int, int] | None
) -> CachedRequestState:
    """构造无需模型、GPU 或分布式初始化的 worker 请求。"""
    return CachedRequestState(
        req_id=req_id,
        prompt_token_ids=list(range(32)),
        mm_features=[],
        sampling_params=SamplingParams(temperature=0),
        generator=None,
        block_ids=([1, 2],),
        num_computed_tokens=0,
        output_token_ids=[],
        prompt_stream_padding_range=padding_range,
    )


def test_padding_ranges_follow_batch_rows() -> None:
    """覆盖重排、删除压缩、普通请求复用行与 refresh 的范围生命周期。"""
    batch = InputBatch(
        max_num_reqs=4,
        max_model_len=64,
        max_num_batched_tokens=64,
        device=torch.device("cpu"),
        pin_memory=False,
        vocab_size=128,
        block_sizes=[16],
        kernel_block_sizes=[16],
    )
    requests = [
        _request("a", (3, 16)),
        _request("b", None),
        _request("c", (20, 32)),
    ]
    for request in requests:
        batch.add_request(request)
    batch.refresh_metadata()
    batch.swap_states(0, 2)
    assert batch.req_ids == ["c", "b", "a"]
    assert batch.prompt_stream_padding_ranges[:3].tolist() == [
        [20, 32],
        [0, 0],
        [3, 16],
    ]
    batch.remove_request("b")
    batch.condense()
    assert batch.req_ids == ["c", "a"]
    assert batch.prompt_stream_padding_ranges[:3].tolist() == [
        [20, 32],
        [3, 16],
        [0, 0],
    ]
    batch.refresh_metadata()
    # 模拟 Request refresh：原 request 被移除，再携带新的执行视图加入。
    batch.remove_request("a")
    requests[0].prompt_stream_padding_range = (0, 0)
    batch.add_request(requests[0])
    assert batch.prompt_stream_padding_ranges[1].tolist() == [0, 0]
    batch.refresh_metadata()
    batch.remove_request("c")
    batch.add_request(_request("ordinary", None))
    assert batch.prompt_stream_padding_ranges[0].tolist() == [0, 0]


def test_model_positions_compact_without_moving_kv_positions() -> None:
    """跨 padding 边界及混合请求时，只移动 RoPE 位置，保留 KV 物理坐标。"""
    runner = GPUModelRunner.__new__(GPUModelRunner)
    runner.prompt_stream_padding_ranges = CpuGpuBuffer(
        4, 2, dtype=torch.int32, device=torch.device("cpu"), pin_memory=False
    )
    runner.input_batch = SimpleNamespace(
        prompt_stream_padding_ranges=np.array(
            [[3, 5], [0, 0], [6, 8], [99, 100]], dtype=np.int32
        )
    )
    runner.positions = torch.tensor([2, 3, 4, 5, 6, 10, 11, 8, 9])
    original_positions = runner.positions.clone()
    runner.model_positions = torch.empty_like(runner.positions)
    req_indices = torch.tensor([0, 0, 0, 0, 0, 1, 1, 2, 2])
    runner._prepare_prompt_stream_positions(3, 9, req_indices)

    torch.testing.assert_close(runner.positions, original_positions)
    torch.testing.assert_close(
        runner.model_positions, torch.tensor([2, 3, 4, 3, 4, 10, 11, 6, 7])
    )
    # batch 尾部属于图 padding，不能继承上一个 batch 的区间。
    assert runner.prompt_stream_padding_ranges.gpu[3].tolist() == [0, 0]


@pytest.mark.parametrize("padding_range", [(-1, 16), (17, 16), (3, 33)])
def test_worker_rejects_invalid_padding_range(
    padding_range: tuple[int, int],
) -> None:
    """范围错误必须在模型调用之前暴露，不能悄悄转成错误 mask。"""
    runner = GPUModelRunner.__new__(GPUModelRunner)
    runner.prompt_stream_masked_padding = True
    request = NewRequestData(
        req_id="invalid",
        prompt_token_ids=list(range(32)),
        mm_features=[],
        sampling_params=SamplingParams(temperature=0),
        pooling_params=None,
        block_ids=([1, 2],),
        num_computed_tokens=0,
        lora_request=None,
        prompt_stream_padding_range=padding_range,
    )
    with pytest.raises(ValueError, match="Invalid padding range"):
        runner._validate_prompt_stream_padding_request(request)


def test_ordinary_request_has_no_padding_metadata() -> None:
    """普通请求经 scheduler 序列化后保持旧语义，可进入未启用 mask 的 worker。"""
    request = Request(
        request_id="ordinary",
        prompt_token_ids=[1, 2, 3],
        sampling_params=SamplingParams(temperature=0),
        pooling_params=None,
    )
    request_data = NewRequestData.from_request(request, ([1],))
    assert request_data.prompt_stream_padding_range is None
    runner = GPUModelRunner.__new__(GPUModelRunner)
    runner.prompt_stream_masked_padding = False
    runner._validate_prompt_stream_padding_request(request_data)


@pytest.mark.parametrize("padding_range", [(0, 0), (3, 16)])
def test_disabled_worker_rejects_masked_metadata(
    padding_range: tuple[int, int],
) -> None:
    """即使本轮空洞长度为零，未启用能力的 worker 也不能接收该协议。"""
    request = Request(
        request_id="masked",
        prompt_token_ids=list(range(32)),
        sampling_params=SamplingParams(temperature=0),
        pooling_params=None,
    )
    request_data = NewRequestData.from_request(request, ([1, 2],))
    request_data.prompt_stream_padding_range = padding_range
    runner = GPUModelRunner.__new__(GPUModelRunner)
    runner.prompt_stream_masked_padding = False
    with pytest.raises(ValueError, match="disabled on this worker"):
        runner._validate_prompt_stream_padding_request(request_data)
