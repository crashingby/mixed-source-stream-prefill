# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""批量执行块稀疏 prefill，复用最后层 softmax 分母计算 Query 块分数。"""

from bisect import bisect_right
from dataclasses import dataclass, field

import torch

from vllm.triton_utils import tl, triton
from vllm.v1.query_probe import QueryProbeResult, QueryProbeWork

QUERY_TILE = 32
KEY_TILE = 128


@dataclass
class QueryProbeAttention:
    """保存一次 model forward 共用的紧凑页索引和 GPU 评分结果。

    请求长度与边界均放入运行时张量，不能作为 Triton 编译常量。
    分数留在 GPU，调用 collect_results 时才统一传回 CPU。
    """

    works: dict[str, QueryProbeWork]
    req_ids: list[str]
    score_layer: str
    tiles: torch.Tensor | None = None
    block_indices: torch.Tensor | None = None
    score_tiles: torch.Tensor | None = None
    score_specs: list[tuple[str, int, int, int, int]] = field(default_factory=list)
    score_parts: torch.Tensor | None = None

    @classmethod
    def build(
        cls,
        works: dict[str, QueryProbeWork],
        req_ids: list[str],
        query_offsets: list[int],
        score_layer: str,
        device: torch.device,
        block_size: int = 16,
    ) -> "QueryProbeAttention":
        """一次构造所有层复用的页列表，并在 Query 边界拆分行 tile。

        被丢弃的完整页不出现在稀疏索引中；Query tile 使用完整页列表。
        每个 tile 的页数按最后一行的因果边界截断，避免读取未来页。
        """
        # 全保留 Update、普通 dense 和最终 Query 同样准备完整页列表。
        metadata = cls(works, req_ids, score_layer)
        tiles: list[list[int]] = []
        pages: list[int] = []
        score_tiles: list[list[int]] = []
        for req_index, req_id in enumerate(req_ids):
            work = works[req_id]
            end = work.computed_tokens + work.scheduled_tokens
            first_query = max(work.computed_tokens, work.probe_start)
            has_query = work.probe_start >= 0 and first_query < end
            sparse_end = first_query if has_query else end
            # 两类行使用不同页列表，但仍在同一批、同一层的 kernel 中执行。
            dropped = set(work.dropped_blocks)
            all_pages = list(range(triton.cdiv(end, block_size)))
            ranges = [(work.computed_tokens, sparse_end, dropped)]
            if has_query:
                ranges.append((first_query, end, set()))
            for first, stop, excluded in ranges:
                if first == stop:
                    continue
                visible = [block for block in all_pages if block not in excluded]
                page_offset = len(pages)
                pages.extend(visible)
                for row in range(first, stop, QUERY_TILE):
                    count = min(QUERY_TILE, stop - row)
                    visible_count = bisect_right(
                        visible, (row + count - 1) // block_size
                    )
                    tiles.append(
                        [
                            req_index,
                            query_offsets[req_index] + row - work.computed_tokens,
                            count,
                            row,
                            *work.padding_range,
                            page_offset,
                            visible_count,
                        ]
                    )
            if has_query:
                candidate_start = len(score_tiles)
                for block in work.candidate_blocks:
                    score_tiles.append(
                        [
                            req_index,
                            block,
                            query_offsets[req_index]
                            + first_query
                            - work.computed_tokens,
                            end - first_query,
                            first_query,
                            *work.padding_range,
                        ]
                    )
                metadata.score_specs.append(
                    (
                        req_id,
                        work.attempt_id,
                        end - first_query,
                        candidate_start,
                        len(score_tiles),
                    )
                )
        # 一次异步 H2D 上传三段整数元数据，避免每个请求/每层单独拷贝。
        flat_tiles = [value for tile in tiles for value in tile]
        flat_scores = [value for tile in score_tiles for value in tile]
        packed = torch.tensor(
            flat_tiles + pages + flat_scores, dtype=torch.int32, pin_memory=True
        ).to(device=device, non_blocking=True)
        page_start = len(flat_tiles)
        score_start = page_start + len(pages)
        metadata.tiles = packed[:page_start].view(-1, 8)
        metadata.block_indices = packed[page_start:score_start]
        metadata.score_tiles = packed[score_start:].view(-1, 7)
        return metadata

    def collect_results(self) -> dict[str, QueryProbeResult]:
        """模型执行完毕后统一归约并回传评分，不在 attention 层内部同步。"""
        values = (
            self.score_parts.mean(dim=1).tolist()
            if self.score_parts is not None
            else []
        )
        return {
            req_id: QueryProbeResult(
                attempt,
                rows,
                [value[0] for value in values[start:end]],
                [value[1] for value in values[start:end]],
            )
            for req_id, attempt, rows, start, end in self.score_specs
        }


@triton.jit
def _query_probe_attention_kernel(
    Q,
    K,
    V,
    OUT,
    TABLE,
    TILES,
    PAGES,
    LSE,
    q_stride_t: tl.constexpr,
    q_stride_h: tl.constexpr,
    k_stride_b: tl.constexpr,
    k_stride_t: tl.constexpr,
    k_stride_h: tl.constexpr,
    v_stride_b: tl.constexpr,
    v_stride_t: tl.constexpr,
    v_stride_h: tl.constexpr,
    o_stride_t: tl.constexpr,
    o_stride_h: tl.constexpr,
    table_stride,
    NUM_HEADS: tl.constexpr,
    HEAD_SIZE: tl.constexpr,
    HEADS_PER_KV: tl.constexpr,
    SCALE: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    D: tl.constexpr,
    WRITE_LSE: tl.constexpr,
    M: tl.constexpr,
    N: tl.constexpr,
):
    """每层一次启动，直接遍历保留页，并复用 tile 内的 K/V 读取。"""
    tile = TILES + tl.program_id(0) * 8
    req = tl.load(tile)
    row_start = tl.load(tile + 1)
    count = tl.load(tile + 2)
    computed = tl.load(tile + 3)
    pad_start, pad_end = tl.load(tile + 4), tl.load(tile + 5)
    page_start, page_count = tl.load(tile + 6), tl.load(tile + 7)
    head = tl.program_id(1)
    rows, dims, cols = tl.arange(0, M), tl.arange(0, D), tl.arange(0, N)
    positions = computed + rows
    kv_head = head // HEADS_PER_KV
    q = tl.load(
        Q
        + (row_start + rows[:, None]) * q_stride_t
        + head * q_stride_h
        + dims[None, :],
        (rows[:, None] < count) & (dims[None, :] < HEAD_SIZE),
        0,
    )
    maximum = tl.full((M,), float("-inf"), tl.float32)
    denominator = tl.zeros((M,), tl.float32)
    accumulator = tl.zeros((M, D), tl.float32)
    for start in range(tl.cdiv(page_count * BLOCK_SIZE, N)):
        compact = start * N + cols
        valid_page = compact // BLOCK_SIZE < page_count
        logical_page = tl.load(
            PAGES + page_start + compact // BLOCK_SIZE, valid_page, 0
        )
        physical_page = tl.load(
            TABLE + req * table_stride + logical_page, valid_page, 0
        )
        key_positions = logical_page * BLOCK_SIZE + compact % BLOCK_SIZE
        # 尾页中尚未写入的 KV 可能含 NaN；必须在 load 前排除，不能只 mask logits。
        key_valid = (
            valid_page
            & (key_positions <= computed + count - 1)
            & ((key_positions < pad_start) | (key_positions >= pad_end))
        )
        k = tl.load(
            K
            + physical_page[None, :] * k_stride_b
            + (compact % BLOCK_SIZE)[None, :] * k_stride_t
            + kv_head * k_stride_h
            + dims[:, None],
            (dims[:, None] < HEAD_SIZE) & key_valid[None, :],
            0,
        )
        logits = tl.dot(q, k) * (SCALE * 1.4426950408889634)
        visible = key_valid[None, :] & (key_positions[None, :] <= positions[:, None])
        logits = tl.where(visible, logits, float("-inf"))
        next_max = tl.maximum(maximum, tl.max(logits, 1))
        # 即使某一行只有 padding，仍保持有限的归一化基准。
        safe_max = tl.where(next_max == float("-inf"), 0.0, next_max)
        alpha = tl.exp2(maximum - safe_max)
        probability = tl.exp2(logits - safe_max[:, None])
        v = tl.load(
            V
            + physical_page[:, None] * v_stride_b
            + (compact % BLOCK_SIZE)[:, None] * v_stride_t
            + kv_head * v_stride_h
            + dims[None, :],
            key_valid[:, None] & (dims[None, :] < HEAD_SIZE),
            0,
        )
        accumulator = accumulator * alpha[:, None] + tl.dot(probability.to(v.dtype), v)
        denominator = denominator * alpha + tl.sum(probability, 1)
        maximum = next_max
    result = accumulator / tl.where(denominator > 0, denominator, 1.0)[:, None]
    tl.store(
        OUT
        + (row_start + rows[:, None]) * o_stride_t
        + head * o_stride_h
        + dims[None, :],
        result,
        (rows[:, None] < count) & (dims[None, :] < HEAD_SIZE),
    )
    if WRITE_LSE:
        tl.store(
            LSE + (row_start + rows) * NUM_HEADS + head,
            maximum + tl.log2(denominator),
            rows < count,
        )


@triton.jit
def _query_probe_score_kernel(
    Q,
    K,
    TABLE,
    SCORES,
    LSE,
    OUT,
    q_stride_t: tl.constexpr,
    q_stride_h: tl.constexpr,
    k_stride_b: tl.constexpr,
    k_stride_t: tl.constexpr,
    k_stride_h: tl.constexpr,
    table_stride,
    NUM_HEADS: tl.constexpr,
    HEADS_PER_KV: tl.constexpr,
    HEAD_SIZE: tl.constexpr,
    D: tl.constexpr,
    SCALE: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """仅重算候选块 QK，使用 dense Query 的 LSE 保持全上下文分母。"""
    candidate, head = tl.program_id(0), tl.program_id(1)
    spec = SCORES + candidate * 7
    req, block = tl.load(spec), tl.load(spec + 1)
    row_start, count = tl.load(spec + 2), tl.load(spec + 3)
    first = tl.load(spec + 4)
    pad_start, pad_end = tl.load(spec + 5), tl.load(spec + 6)
    page = tl.load(TABLE + req * table_stride + block)
    rows, dims, cols = tl.arange(0, 16), tl.arange(0, D), tl.arange(0, BLOCK_SIZE)
    k = tl.load(
        K
        + page * k_stride_b
        + cols[None, :] * k_stride_t
        + (head // HEADS_PER_KV) * k_stride_h
        + dims[:, None],
        dims[:, None] < HEAD_SIZE,
        0,
    )
    raw, corrected = 0.0, 0.0
    for start in range(tl.cdiv(count, 16)):
        row = start * 16 + rows
        q = tl.load(
            Q
            + (row_start + row[:, None]) * q_stride_t
            + head * q_stride_h
            + dims[None, :],
            (row[:, None] < count) & (dims[None, :] < HEAD_SIZE),
            0,
        )
        normalizer = tl.load(LSE + (row_start + row) * NUM_HEADS + head, row < count, 0)
        logits = tl.dot(q, k) * (SCALE * 1.4426950408889634)
        mass = tl.sum(tl.exp2(logits - normalizer[:, None]), 1)
        mass = tl.where(row < count, mass, 0.0)
        # Query 可见 token 总数扣除物理 padding；分母从未限制到候选 Append。
        visible = (
            first
            + row
            + 1
            - tl.minimum(
                tl.maximum(first + row + 1 - pad_start, 0), pad_end - pad_start
            )
        )
        raw += tl.sum(mass, 0)
        corrected += tl.sum(mass * (visible / BLOCK_SIZE), 0)
    tl.store(OUT + (candidate * NUM_HEADS + head) * 2, raw)
    tl.store(OUT + (candidate * NUM_HEADS + head) * 2 + 1, corrected)


def query_probe_attention(
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    output: torch.Tensor,
    block_table: torch.Tensor,
    metadata: QueryProbeAttention,
    scale: float,
    layer_name: str,
) -> None:
    """提交批量 attention 和可选评分 kernel；本函数不读取任何 GPU 结果。"""
    assert metadata.tiles is not None
    assert metadata.score_tiles is not None
    score = layer_name == metadata.score_layer and bool(metadata.score_specs)
    num_heads, head_size = query.shape[1:]
    block_size = key_cache.shape[1]
    lse = (
        torch.empty(query.shape[:2], device=query.device, dtype=torch.float32)
        if score
        else None
    )
    _query_probe_attention_kernel[(metadata.tiles.shape[0], num_heads)](
        query,
        key_cache,
        value_cache,
        output,
        block_table,
        metadata.tiles,
        metadata.block_indices,
        lse,
        *query.stride()[:2],
        *key_cache.stride()[:3],
        *value_cache.stride()[:3],
        *output.stride()[:2],
        block_table.stride(0),
        num_heads,
        head_size,
        num_heads // key_cache.shape[2],
        scale,
        block_size,
        triton.next_power_of_2(head_size),
        score,
        QUERY_TILE,
        KEY_TILE,
        num_warps=4,
        num_stages=2,
    )
    if score and metadata.score_tiles.shape[0]:
        metadata.score_parts = torch.empty(
            (metadata.score_tiles.shape[0], num_heads, 2),
            device=query.device,
            dtype=torch.float32,
        )
        _query_probe_score_kernel[(metadata.score_tiles.shape[0], num_heads)](
            query,
            key_cache,
            block_table,
            metadata.score_tiles,
            lse,
            metadata.score_parts,
            *query.stride()[:2],
            *key_cache.stride()[:3],
            block_table.stride(0),
            num_heads,
            num_heads // key_cache.shape[2],
            head_size,
            triton.next_power_of_2(head_size),
            scale,
            block_size,
            num_warps=4,
        )
