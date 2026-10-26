# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""对比显式行级 mask，覆盖 Query 边界、GQA 和非连续分页布局。"""

import pytest
import torch

from vllm.v1.attention.backends import chunk_sparse_attn, triton_attn
from vllm.v1.attention.ops.query_probe_attention import (
    QueryProbeAttention,
    query_probe_attention,
)
from vllm.v1.query_probe import QueryProbeWork


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA")
@pytest.mark.parametrize("computed", [15, 16, 17, 31, 32, 33, 79, 81])
@pytest.mark.parametrize("layout", ["NHD", "HND"])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@torch.inference_mode()
def test_mixed_rows(computed: int, layout: str, dtype: torch.dtype) -> None:
    """Query 完整读取被 Append 屏蔽的页，评分与独立 softmax 一致。"""
    torch.manual_seed(7)
    device = "cuda"
    end, probe_start, prefix_end = 87, 80, 75
    count = end - computed
    query = torch.randn(count, 4, 64, device=device, dtype=dtype)
    cache = torch.randn(12, 2, 16, 2, 64, device=device, dtype=query.dtype)
    if layout == "HND":
        cache = cache.transpose(2, 3).contiguous().transpose(2, 3)
    key, value = cache.unbind(1)
    table = torch.tensor([[5, 2, 9, 3, 7, 1]], device=device, dtype=torch.int32)
    keep = torch.tensor([[True, False, False, True, True, True]], device=device)
    work = QueryProbeWork(
        1, computed, count, prefix_end, probe_start, [3], [1, 2], (75, 80)
    )
    metadata = QueryProbeAttention.build(
        {"r": work}, ["r"], [0, count], "last", query.device
    )
    output = torch.empty_like(query)
    query_probe_attention(query, key, value, output, table, metadata, 0.125, "last")
    keys = key[table[0].long()].flatten(0, 1)[:end].repeat_interleave(2, dim=1)
    values = value[table[0].long()].flatten(0, 1)[:end].repeat_interleave(2, dim=1)
    qpos = torch.arange(computed, end, device=device)
    kpos = torch.arange(end, device=device)
    valid = (kpos[None, :] <= qpos[:, None]) & (
        (kpos[None, :] < 75) | (kpos[None, :] >= 80)
    )
    valid &= keep[0, kpos // 16][None, :] | (qpos[:, None] >= 80)
    logits = torch.einsum("qhd,khd->hqk", query.float(), keys.float()) * 0.125
    weights = logits.masked_fill(~valid[None, :, :], float("-inf")).softmax(-1)
    expected = torch.einsum("hqk,khd->qhd", weights, values.float())
    tolerance = 2e-2 if dtype == torch.bfloat16 else 3e-3
    torch.testing.assert_close(output.float(), expected, atol=tolerance, rtol=tolerance)
    result = metadata.collect_results()["r"]
    probe_rows = qpos >= 80
    mass = weights[:, probe_rows, 48:64].sum(-1).mean(0)
    assert result.num_query_tokens == int(probe_rows.sum())
    assert result.raw_sums == pytest.approx([mass.sum().item()], rel=1e-5)
    corrected = mass * valid[probe_rows].sum(-1) / 16
    assert result.corrected_sums == pytest.approx([corrected.sum().item()], rel=1e-5)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA")
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("layout", ["NHD", "HND"])
@pytest.mark.parametrize("probe_tokens", [37, 243])
@torch.inference_mode()
def test_ragged_batch_and_scores(
    dtype: torch.dtype, layout: str, probe_tokens: int
) -> None:
    """不同长度、padding、评分范围共批；覆盖 Qwen 的 8:1 GQA 和空候选。"""
    torch.manual_seed(19)
    device = torch.device("cuda")
    works = {
        "update": QueryProbeWork(1, 129, 83, 123, -1, [], [1, 3, 5], (123, 129)),
        "joint": QueryProbeWork(2, 81, 143, 175, 176, [6, 7, 8, 9], [1, 3], (175, 176)),
        "query_only": QueryProbeWork(
            3, 240, probe_tokens, 225, 240, [10, 11, 12, 13], [1, 4], (225, 240)
        ),
        "empty_candidates": QueryProbeWork(4, 15, 22, 15, 16, [], [], (15, 16)),
        "dense": QueryProbeWork(5, 64, 1, 64, -1, [], [], (0, 0)),
    }
    req_ids = list(works)
    offsets = [0]
    for work in works.values():
        offsets.append(offsets[-1] + work.scheduled_tokens)
    query = torch.randn(offsets[-1], 16, 128, device=device, dtype=dtype)
    cache = torch.randn(192, 2, 16, 2, 128, device=device, dtype=dtype)
    if layout == "HND":
        cache = cache.transpose(2, 3).contiguous().transpose(2, 3)
    key, value = cache.unbind(1)
    table = torch.randperm(192, device=device, dtype=torch.int32)[:160].view(5, 32)
    for index, work in enumerate(works.values()):
        end = work.computed_tokens + work.scheduled_tokens
        positions = torch.cat(
            [
                torch.arange(*work.padding_range, device=device),
                torch.arange(end, ((end + 15) // 16) * 16, device=device),
            ]
        )
        # 被 mask 的无效 KV 必须不参与读取，包含 NaN 时也不能污染输出。
        key[table[index, positions // 16].long(), positions % 16] = float("nan")
        value[table[index, positions // 16].long(), positions % 16] = float("nan")
    metadata = QueryProbeAttention.build(works, req_ids, offsets, "last", device)
    output = torch.empty_like(query)
    scale = 128**-0.5
    query_probe_attention(query, key, value, output, table, metadata, scale, "last")
    results = metadata.collect_results()
    for index, (req_id, work) in enumerate(works.items()):
        end = work.computed_tokens + work.scheduled_tokens
        keys = key[table[index].long()].flatten(0, 1)[:end].repeat_interleave(8, dim=1)
        values = (
            value[table[index].long()].flatten(0, 1)[:end].repeat_interleave(8, dim=1)
        )
        keys, values = keys.nan_to_num(), values.nan_to_num()
        qpos = torch.arange(work.computed_tokens, end, device=device)
        kpos = torch.arange(end, device=device)
        valid = (kpos[None, :] <= qpos[:, None]) & (
            (kpos[None, :] < work.padding_range[0])
            | (kpos[None, :] >= work.padding_range[1])
        )
        keep = torch.ones(32, device=device, dtype=torch.bool)
        keep[work.dropped_blocks] = False
        dense_rows = (work.probe_start >= 0) & (qpos >= work.probe_start)
        valid &= keep[kpos // 16][None, :] | dense_rows[:, None]
        q = query[offsets[index] : offsets[index + 1]]
        logits = torch.einsum("qhd,khd->hqk", q.float(), keys.float()) * scale
        weights = logits.masked_fill(~valid[None, :, :], float("-inf")).softmax(-1)
        expected = torch.einsum("hqk,khd->qhd", weights, values.float())
        tolerance = 2e-2 if dtype == torch.bfloat16 else 3e-3
        torch.testing.assert_close(
            output[offsets[index] : offsets[index + 1]].float(),
            expected,
            atol=tolerance,
            rtol=tolerance,
        )
        if req_id in results:
            result = results[req_id]
            assert result.num_query_tokens == int(dense_rows.sum())
            raw, corrected = [], []
            for block in work.candidate_blocks:
                mass = (
                    weights[:, dense_rows, block * 16 : (block + 1) * 16]
                    .sum(-1)
                    .mean(0)
                )
                raw.append(mass.sum().item())
                corrected.append((mass * valid[dense_rows].sum(-1) / 16).sum().item())
            assert result.raw_sums == pytest.approx(raw, rel=1e-5)
            assert result.corrected_sums == pytest.approx(corrected, rel=1e-5)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA")
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("layout", ["NHD", "HND"])
@pytest.mark.parametrize(
    "computed,count,padding",
    [
        (0, 7, (0, 0)),
        (0, 7, (0, 7)),
        (31, 65, (17, 31)),
        (1024, 1, (1000, 1024)),
        (1024, 513, (1000, 1024)),
    ],
)
@torch.inference_mode()
def test_dense_backend_routing(
    dtype: torch.dtype,
    layout: str,
    computed: int,
    count: int,
    padding: tuple[int, int],
) -> None:
    """无 probe 的 prefill/decode 使用新 kernel；原 Triton 保持 unified 路由。"""
    from unittest.mock import patch

    torch.manual_seed(31)
    device = torch.device("cuda")
    end = computed + count
    pages = (end + 15) // 16
    query = torch.randn(count, 16, 128, device=device, dtype=dtype)
    cache = torch.randn(pages, 2, 16, 2, 128, device=device, dtype=dtype)
    if layout == "HND":
        cache = cache.transpose(2, 3).contiguous().transpose(2, 3)
    table = torch.randperm(pages, device=device, dtype=torch.int32).view(1, -1)
    key, value = cache.unbind(1)
    invalid = torch.cat(
        (
            torch.arange(*padding, device=device),
            torch.arange(end, pages * 16, device=device),
        )
    )
    key[table[0, invalid // 16].long(), invalid % 16] = float("nan")
    value[table[0, invalid // 16].long(), invalid % 16] = float("nan")
    work = QueryProbeWork(0, computed, count, computed, -1, [], [], padding)
    metadata = QueryProbeAttention.build({"r": work}, ["r"], [0, count], "", device)
    assert metadata.tiles is not None
    assert metadata.score_specs == []
    attn = triton_attn.TritonAttentionMetadata(
        num_actual_tokens=count,
        max_query_len=count,
        query_start_loc=torch.tensor([0, count], device=device, dtype=torch.int32),
        max_seq_len=end,
        seq_lens=torch.tensor([end], device=device, dtype=torch.int32),
        block_table=table,
        slot_mapping=torch.empty(0, device=device, dtype=torch.int64),
        seq_threshold_3D=0,
        num_par_softmax_segments=16,
        softmax_segm_output=None,
        softmax_segm_max=None,
        softmax_segm_expsum=None,
        causal=True,
        use_cascade=False,
        common_prefix_len=0,
        cu_prefix_query_lens=None,
        prefix_kv_lens=None,
        suffix_kv_lens=None,
        prompt_stream_padding_ranges=torch.tensor(
            [padding], device=device, dtype=torch.int32
        ),
        query_probe=metadata,
    )
    layer = torch.nn.Module()
    layer.layer_name = "dense"
    layer._k_scale = layer._v_scale = torch.ones((), device=device)
    custom = chunk_sparse_attn.ChunkSparseAttentionImpl(
        16, 128, 128**-0.5, 2, None, None, "auto"
    )
    native = triton_attn.TritonAttentionImpl(16, 128, 128**-0.5, 2, None, None, "auto")
    output, expected = torch.empty_like(query), torch.empty_like(query)
    with (
        patch.object(
            chunk_sparse_attn, "query_probe_attention", wraps=query_probe_attention
        ) as custom_call,
        patch.object(
            triton_attn, "unified_attention", wraps=triton_attn.unified_attention
        ) as native_call,
    ):
        custom.forward(layer, query, key, value, cache, attn, output)
        assert custom_call.call_count == 1 and native_call.call_count == 0
        native.forward(layer, query, key, value, cache, attn, expected)
        assert custom_call.call_count == 1 and native_call.call_count == 1
    tolerance = 2e-2 if dtype == torch.bfloat16 else 3e-3
    torch.testing.assert_close(output, expected, atol=tolerance, rtol=tolerance)
    assert metadata.score_parts is None
    assert metadata.collect_results() == {}
