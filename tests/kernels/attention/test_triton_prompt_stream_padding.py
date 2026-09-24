# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""单独验证物理 KV 空洞，不启动模型、EngineCore 或 HTTP 服务。"""

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from vllm.v1.attention.ops.triton_unified_attention import unified_attention


def _reference_with_padding(
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    block_table: torch.Tensor,
    query_len: int,
    seq_len: int,
    padding_ranges: list[tuple[int, int]],
) -> torch.Tensor:
    """逐 query 抽取真实可见 KV，作为与物理空洞实现独立的参考。"""
    outputs: list[torch.Tensor] = []
    num_heads = query.shape[1]
    head_size = query.shape[2]
    for request_idx, (padding_start, padding_end) in enumerate(padding_ranges):
        # 使用同一 block table，但参考计算直接删除 padding 与未来位置。
        keys = key_cache[block_table[request_idx]].flatten(0, 1)[:seq_len]
        values = value_cache[block_table[request_idx]].flatten(0, 1)[:seq_len]
        keys = keys.repeat_interleave(num_heads // keys.shape[1], dim=1)
        values = values.repeat_interleave(num_heads // values.shape[1], dim=1)
        positions = torch.arange(seq_len, device=query.device)
        for local_query_idx in range(query_len):
            query_idx = request_idx * query_len + local_query_idx
            query_position = seq_len - query_len + local_query_idx
            visible = (positions <= query_position) & (
                (positions < padding_start) | (positions >= padding_end)
            )
            # 整行被掩码时定义输出为零，避免参考 softmax 也产生 NaN。
            if not visible.any():
                outputs.append(torch.zeros_like(query[query_idx]))
                continue
            scores = torch.einsum(
                "hd,khd->hk", query[query_idx].float(), keys[visible].float()
            ) * head_size**-0.5
            probabilities = scores.softmax(dim=-1)
            outputs.append(
                torch.einsum("hk,khd->hd", probabilities, values[visible].float())
                .to(query.dtype)
            )
    return torch.stack(outputs)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA")
@pytest.mark.parametrize(
    "use_3d,query_len,kv_layout",
    [
        (False, 1, "NHD"),
        (False, 7, "NHD"),
        (False, 256, "NHD"),
        (True, 1, "NHD"),
        (False, 7, "HND"),
        (True, 1, "HND"),
    ],
)
@pytest.mark.parametrize("padding_range", [(19, 32), (0, 128), (1, 255), (0, 256)])
@torch.inference_mode()
def test_prompt_stream_padding_attention(
    use_3d: bool,
    query_len: int,
    kv_layout: str,
    padding_range: tuple[int, int],
) -> None:
    """验证 prefill/decode、空分区、混合请求和 padding NaN 均正确处理。"""
    torch.manual_seed(2026)
    device = torch.device("cuda")
    num_reqs, seq_len, block_size = 3, 256, 16
    num_heads, num_kv_heads, head_size = 4, 2, 64
    num_blocks_per_req = seq_len // block_size
    num_blocks = num_reqs * num_blocks_per_req
    query = torch.randn(
        num_reqs * query_len, num_heads, head_size, dtype=torch.float16, device=device
    )
    if kv_layout == "HND":
        # NIXL 使用 HND 物理布局；外部仍呈现 [block, slot, head, dim] 视图。
        key_cache = torch.randn(
            num_blocks,
            num_kv_heads,
            block_size,
            head_size,
            dtype=query.dtype,
            device=device,
        ).transpose(1, 2)
    else:
        key_cache = torch.randn(
            num_blocks,
            block_size,
            num_kv_heads,
            head_size,
            dtype=query.dtype,
            device=device,
        )
    value_cache = torch.randn_like(key_cache)
    # 打散 block ID，检查区间是逻辑物理槽位置而非 GPU block ID。
    block_table = torch.randperm(num_blocks, device=device, dtype=torch.int32).reshape(
        num_reqs, num_blocks_per_req
    )
    padding_ranges = [padding_range, (0, 0), (63, 80)]
    ranges_tensor = torch.tensor(padding_ranges, dtype=torch.int32, device=device)
    query[:query_len].fill_(1.0)
    key_cache[block_table[0].long()] = -32.0
    # 第一个请求的合法 logits 全部很负，空分区若伪造 max=0 就会错误下溢。
    for request_idx, (start, end) in enumerate(padding_ranges):
        slots = torch.arange(start, end, device=device)
        blocks = block_table[request_idx, slots // block_size].long()
        key_cache[blocks, slots % block_size] = float("nan")
        value_cache[blocks, slots % block_size] = float("nan")

    output = torch.empty_like(query)
    num_segments = 16
    intermediate_shape = (num_reqs * query_len, num_heads, num_segments)
    segment_output = torch.empty(
        (*intermediate_shape, head_size), dtype=torch.float32, device=device
    )
    segment_max = torch.empty(intermediate_shape, dtype=torch.float32, device=device)
    segment_sum = torch.empty_like(segment_max)
    query_start = torch.arange(
        0, (num_reqs + 1) * query_len, query_len, dtype=torch.int32, device=device
    )
    seq_lens = torch.full((num_reqs,), seq_len, dtype=torch.int32, device=device)

    unified_attention(
        q=query,
        k=key_cache,
        v=value_cache,
        out=output,
        cu_seqlens_q=query_start,
        max_seqlen_q=query_len,
        seqused_k=seq_lens,
        max_seqlen_k=seq_len,
        softmax_scale=head_size**-0.5,
        causal=True,
        window_size=(-1, -1),
        block_table=block_table,
        softcap=0,
        q_descale=None,
        k_descale=None,
        v_descale=None,
        seq_threshold_3D=num_reqs if use_3d else 0,
        num_par_softmax_segments=num_segments,
        softmax_segm_output=segment_output,
        softmax_segm_max=segment_max,
        softmax_segm_expsum=segment_sum,
        prompt_stream_padding_ranges=ranges_tensor,
    )
    reference = _reference_with_padding(
        query, key_cache, value_cache, block_table, query_len, seq_len, padding_ranges
    )
    assert torch.isfinite(output).all()
    torch.testing.assert_close(output, reference, atol=2e-3, rtol=2e-3)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA")
@torch.inference_mode()
def test_prompt_stream_padding_cudagraph_range_changes() -> None:
    """先以空区间捕获，再修改同一 GPU 缓冲区，回放必须看到新空洞。"""
    device = torch.device("cuda")
    torch.manual_seed(2026)
    query = torch.randn(1, 4, 64, dtype=torch.float16, device=device)
    keys = torch.randn(4, 16, 2, 64, dtype=query.dtype, device=device)
    values = torch.randn_like(keys)
    output = torch.empty_like(query)
    block_table = torch.arange(4, dtype=torch.int32, device=device).reshape(1, 4)
    ranges = torch.zeros((1, 2), dtype=torch.int32, device=device)
    kwargs = dict(
        q=query,
        k=keys,
        v=values,
        out=output,
        cu_seqlens_q=torch.tensor([0, 1], dtype=torch.int32, device=device),
        max_seqlen_q=1,
        seqused_k=torch.tensor([64], dtype=torch.int32, device=device),
        max_seqlen_k=64,
        softmax_scale=64**-0.5,
        causal=True,
        window_size=(-1, -1),
        block_table=block_table,
        softcap=0,
        q_descale=None,
        k_descale=None,
        v_descale=None,
        prompt_stream_padding_ranges=ranges,
    )
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        unified_attention(**kwargs)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        unified_attention(**kwargs)

    # 图复用只更新数值、不替换指针；同时验证空洞恢复为空区间的情况。
    for start, end in [(19, 32), (16, 48), (0, 0)]:
        ranges.copy_(torch.tensor([[start, end]], dtype=torch.int32, device=device))
        graph.replay()
        expected = _reference_with_padding(
            query, keys, values, block_table, 1, 64, [(start, end)]
        )
        torch.testing.assert_close(output, expected, atol=2e-3, rtol=2e-3)


def _reference_rope(states: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
    """用独立的 split-half RoPE 参考实现检查物理空洞后的语义位置。"""
    head_size = states.shape[-1]
    inv_freq = 10000.0 ** (
        -torch.arange(0, head_size, 2, device=states.device).float() / head_size
    )
    angles = positions.float()[:, None] * inv_freq[None, :]
    angles = torch.cat((angles, angles), dim=-1)[:, None, :]
    left, right = states.float().chunk(2, dim=-1)
    rotated = torch.cat((-right, left), dim=-1)
    return (states.float() * angles.cos() + rotated * angles.sin()).to(states.dtype)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA")
@pytest.mark.parametrize("use_3d", [False, True])
@torch.inference_mode()
def test_prompt_stream_rope_and_mask_match_compact_attention(use_3d: bool) -> None:
    """真实 worker 位置转换 + HND Triton 与删除空洞的紧凑序列逐项对照。"""
    from vllm.v1.utils import CpuGpuBuffer
    from vllm.v1.worker.gpu_model_runner import GPUModelRunner

    torch.manual_seed(2026)
    device = torch.device("cuda")
    prefix_len, padding_end, logical_len = 19, 32, 96
    padding_len = padding_end - prefix_len
    physical_len = logical_len + padding_len
    query_len = 1 if use_3d else 7
    block_size, num_heads, num_kv_heads, head_size = 16, 4, 2, 64
    raw_keys = torch.randn(
        logical_len, num_kv_heads, head_size, dtype=torch.float16, device=device
    )
    raw_values = torch.randn_like(raw_keys)
    raw_queries = torch.randn(
        query_len, num_heads, head_size, dtype=torch.float16, device=device
    )
    logical_positions = torch.arange(logical_len, device=device)
    compact_keys = _reference_rope(raw_keys, logical_positions)
    compact_queries = _reference_rope(raw_queries, logical_positions[-query_len:])

    # 仅构造 runner 的坐标缓冲区，不初始化模型、分布式或推理服务。
    runner = GPUModelRunner.__new__(GPUModelRunner)
    runner.prompt_stream_padding_ranges = CpuGpuBuffer(
        1, 2, dtype=torch.int32, device=device, pin_memory=False
    )
    runner.input_batch = SimpleNamespace(
        prompt_stream_padding_ranges=np.array(
            [[prefix_len, padding_end]], dtype=np.int32
        )
    )
    runner.positions = torch.arange(physical_len, device=device)
    runner.model_positions = torch.empty_like(runner.positions)
    runner._prepare_prompt_stream_positions(
        1, physical_len, torch.zeros(physical_len, dtype=torch.int64, device=device)
    )
    real_slots = torch.cat(
        (
            torch.arange(prefix_len, device=device),
            torch.arange(padding_end, physical_len, device=device),
        )
    )
    torch.testing.assert_close(runner.model_positions[real_slots], logical_positions)
    # 补到 kernel tile 边界，尾部超出 seq_len 的槽位不参加 attention。
    physical_slots = 128
    expanded_keys = torch.zeros(
        physical_slots, num_kv_heads, head_size, dtype=torch.float16, device=device
    )
    expanded_values = torch.full_like(expanded_keys, float("nan"))
    expanded_keys[real_slots] = raw_keys
    expanded_values[real_slots] = raw_values
    expanded_keys[:physical_len] = _reference_rope(
        expanded_keys[:physical_len], runner.model_positions
    )
    query = _reference_rope(raw_queries, runner.model_positions[-query_len:])
    torch.testing.assert_close(query, compact_queries)
    # 转成真实 HND backing storage，再暴露 NHD 维度顺序的 strided view。
    key_cache = expanded_keys.reshape(-1, block_size, num_kv_heads, head_size)
    key_cache = key_cache.transpose(1, 2).contiguous().transpose(1, 2)
    value_cache = expanded_values.reshape(
        -1, block_size, num_kv_heads, head_size
    )
    value_cache = value_cache.transpose(1, 2).contiguous().transpose(1, 2)
    block_table = torch.arange(
        physical_slots // block_size, dtype=torch.int32, device=device
    ).reshape(1, -1)
    segment_shape = (query_len, num_heads, 16)
    output = torch.empty_like(query)
    unified_attention(
        q=query,
        k=key_cache,
        v=value_cache,
        out=output,
        cu_seqlens_q=torch.tensor([0, query_len], dtype=torch.int32, device=device),
        max_seqlen_q=query_len,
        seqused_k=torch.tensor([physical_len], dtype=torch.int32, device=device),
        max_seqlen_k=physical_len,
        softmax_scale=head_size**-0.5,
        causal=True,
        window_size=(-1, -1),
        block_table=block_table,
        softcap=0,
        q_descale=None,
        k_descale=None,
        v_descale=None,
        seq_threshold_3D=1 if use_3d else 0,
        num_par_softmax_segments=16,
        softmax_segm_output=torch.empty(
            (*segment_shape, head_size), dtype=torch.float32, device=device
        ),
        softmax_segm_max=torch.empty(segment_shape, dtype=torch.float32, device=device),
        softmax_segm_expsum=torch.empty(
            segment_shape, dtype=torch.float32, device=device
        ),
        prompt_stream_padding_ranges=runner.prompt_stream_padding_ranges.gpu,
    )
    # 参考 KV 不含任何 padding，RoPE 使用普通连续的 0..logical_len-1。
    reference = _reference_with_padding(
        compact_queries,
        compact_keys.reshape(-1, block_size, num_kv_heads, head_size),
        raw_values.reshape(-1, block_size, num_kv_heads, head_size),
        torch.arange(
            logical_len // block_size, dtype=torch.int32, device=device
        ).reshape(1, -1),
        query_len,
        logical_len,
        [(0, 0)],
    )
    torch.testing.assert_close(output, reference, atol=2e-3, rtol=2e-3)
