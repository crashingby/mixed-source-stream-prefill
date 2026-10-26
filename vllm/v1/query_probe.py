# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""联合 Append / Query prefill 的逻辑块策略与跨进程数据。"""

import hashlib
import json
import math
from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class AppendBlockPolicy:
    """保存一个 Append 的真实范围；物理块号由当前 block table 解析。"""

    start: int  # 含模板的真实 token 起点。
    end: int  # 不含尾部 padding 的终点。
    source_id: str
    chunk_id: str | None
    sequence_no: int
    epoch: int
    token_digest: str = ""  # 事件分词结果的内容摘要。
    forward_count: int = 0  # 评分尝试的实际执行批次数，包含拆分和重试。
    joint_forward_count: int = 0  # 同一批同时处理真实 Append 与 Query。
    query_forward_count: int = 0  # 实际含 Query 行的批次，检测 Query 跨批。
    block_ids: list[int] = field(default_factory=list)  # 完整独占的逻辑块。
    raw_scores: list[float] = field(default_factory=list)
    corrected_scores: list[float] = field(default_factory=list)
    kept_blocks: list[int] = field(default_factory=list)
    published: bool = False  # 完成全部 Query 行归约后才能对后续 work 可见。


@dataclass
class QueryProbeWork:
    """冻结一批执行的读取规则和评分范围，不包含物理缓存所有权。"""

    attempt_id: int
    computed_tokens: int
    scheduled_tokens: int
    prefix_len: int  # 本次 Append 的真实终点，也用于 Query 语义位置。
    probe_start: int  # 私有 Query 的物理起点；-1 表示本批没有 probe。
    candidate_blocks: list[int]
    dropped_blocks: list[int]
    padding_range: tuple[int, int]
    is_final_step: bool = False
    suppress_sampling: bool = True


@dataclass
class QueryProbeResult:
    """worker 返回的 Query 行部分和，按全部有效行数统一归一化。"""

    attempt_id: int
    num_query_tokens: int
    raw_sums: list[float]
    corrected_sums: list[float]


@dataclass
class QueryProbeState:
    """管理固定 Query、不可变历史策略和当前评分尝试的累加器。"""

    token_ids: list[int]
    keep_ratio: float
    chunks: list[AppendBlockPolicy] = field(default_factory=list)
    attempt_id: int = 0  # 恢复或新 Append 都增加，拒绝迟到评分。
    active_end: int | None = None
    num_query_tokens: int = 0
    raw_sums: list[float] = field(default_factory=list)
    corrected_sums: list[float] = field(default_factory=list)
    forward_count: int = 0
    probe_forward_count: int = 0
    released_probe_blocks: int = 0

    def begin(self, end: int, block_size: int) -> None:
        """为未评分的新块建立尝试，并清空本次 Query 的部分和。"""
        chunk = self.chunk_at(end)
        chunk.block_ids = list(
            range((chunk.start + block_size - 1) // block_size, end // block_size)
        )
        self.attempt_id += 1
        self.active_end = end
        self.num_query_tokens = 0
        self.raw_sums = [0.0] * len(chunk.block_ids)
        self.corrected_sums = [0.0] * len(chunk.block_ids)

    def chunk_at(self, end: int) -> AppendBlockPolicy:
        """按真实终点找到事件；空事件不进入该清单。"""
        return next(chunk for chunk in self.chunks if chunk.end == end)

    def dropped_before(self, end: int) -> list[int]:
        """返回当前历史范围中被淘汰的独占块；保护块始终不在此列。"""
        dropped: list[int] = []
        for chunk in self.chunks:
            if chunk.end <= end and chunk.published:
                kept = set(chunk.kept_blocks)
                dropped.extend(block for block in chunk.block_ids if block not in kept)
        return dropped

    def accumulate(self, result: QueryProbeResult) -> None:
        """累加同一 attempt 的评分；恢复前的回包不能改变当前策略。"""
        if result.attempt_id != self.attempt_id or self.active_end is None:
            return
        self.num_query_tokens += result.num_query_tokens
        for index, score in enumerate(result.raw_sums):
            self.raw_sums[index] += score
            self.corrected_sums[index] += result.corrected_sums[index]

    def publish(self) -> None:
        """归一化并稳定选择 Top-K；只有完整 Query 的回包才能发布。"""
        assert self.active_end is not None
        assert self.num_query_tokens == len(self.token_ids)
        chunk = self.chunk_at(self.active_end)
        chunk.raw_scores = [score / self.num_query_tokens for score in self.raw_sums]
        chunk.corrected_scores = [
            score / self.num_query_tokens for score in self.corrected_sums
        ]
        num_keep = math.ceil(self.keep_ratio * len(chunk.block_ids))
        ranked = sorted(
            zip(chunk.block_ids, chunk.raw_scores), key=lambda item: (-item[1], item[0])
        )
        chunk.kept_blocks = sorted(block for block, _ in ranked[:num_keep])
        chunk.published = True
        self.active_end = None

    def reset_attempt(self) -> None:
        """抢占时丢弃临时进度，保留逻辑边界与已经发布的策略。"""
        self.attempt_id += 1
        self.active_end = None
        self.num_query_tokens = 0
        self.raw_sums.clear()
        self.corrected_sums.clear()

    def stats(self) -> dict[str, Any]:
        """返回可直接写入 replay 结果的策略和功能统计快照。"""
        return {
            "query_version": 1,
            "query_digest": hashlib.sha256(
                json.dumps(self.token_ids).encode()
            ).hexdigest(),
            "keep_ratio": self.keep_ratio,
            "query_tokens": len(self.token_ids),
            "forward_count": self.forward_count,
            "probe_forward_count": self.probe_forward_count,
            "released_probe_blocks": self.released_probe_blocks,
            "chunks": [asdict(chunk) for chunk in self.chunks],
        }
