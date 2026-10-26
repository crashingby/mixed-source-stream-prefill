# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""性能和质量回放共用的 Query probe 执行结果校验。"""

import math
from typing import Any


def validate_probe(stats: dict[str, Any], num_append: int, keep_ratio: float) -> None:
    """验证全部 Append 已发布策略、完整块预算及私有 Query 页已释放。"""
    probe = stats["query_probe"]
    chunks = probe["chunks"]
    if len(chunks) != num_append or any(not chunk["published"] for chunk in chunks):
        raise ValueError("Append 事件与已发布策略数量不一致")
    for chunk in chunks:
        blocks = chunk["block_ids"]
        expected = list(range((chunk["start"] + 15) // 16, chunk["end"] // 16))
        if blocks != expected:
            raise ValueError("候选包含跨 chunk 边界块，或漏掉完整独占块")
        if len(chunk["kept_blocks"]) != math.ceil(keep_ratio * len(blocks)):
            raise ValueError("保留块数量与固定比例不一致")
        ranked = sorted(
            zip(blocks, chunk["raw_scores"]), key=lambda item: (-item[1], item[0])
        )
        expected_kept = sorted(
            block for block, _ in ranked[: len(chunk["kept_blocks"])]
        )
        if chunk["kept_blocks"] != expected_kept:
            raise ValueError("保留集合不是 Query 分数的稳定 Top-K")
        for field in ("raw_scores", "corrected_scores"):
            scores = chunk[field]
            if len(scores) != len(blocks) or any(
                not math.isfinite(x) or x < 0 for x in scores
            ):
                raise ValueError(f"非法块评分：{field}")
    if probe["released_probe_blocks"] < num_append * math.ceil(
        probe["query_tokens"] / 16
    ):
        raise ValueError("临时 Query 页释放计数不足")
