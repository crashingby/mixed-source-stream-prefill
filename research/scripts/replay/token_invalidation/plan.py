# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""将相同原始输入时间线转换成三种上下文布局。"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from dataclasses import replace
from typing import Any

from research.scripts.replay.common.dataset import get_data_events
from research.scripts.replay.common.models import (
    RenderedReplayPrompt,
    SampleReplayInput,
)
from research.scripts.replay.common.prompt import (
    ChatTemplateRenderer,
    render_sample_prompt,
)
from research.scripts.replay.common.timeline import source_timing

from .models import WHOLE_SOURCE_ID, Layout, OutboundEvent, PlannedEvent


def fingerprint(value: Any) -> str:
    """计算 JSON 的稳定哈希，用于不同布局结果的请求级配对。"""
    encoded = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )
    return hashlib.sha256(encoded.encode()).hexdigest()


def make_plan(
    sample: SampleReplayInput, args: argparse.Namespace
) -> tuple[PlannedEvent, ...]:
    """按绝对时间排原始数据；同时刻先 Append，逐条发送，不折叠成一个 tick。

    Args:
        sample: 已按 sample_id 对齐的证据，当前实验支持每类最多一个来源。
        args: 两类来源的 initial delay 与 interval，单位秒。

    Returns:
        三种布局共同使用的原始数据计划。
    """
    counts = Counter(source.source_type for source in sample.source_streams)
    if any(count > 1 for count in counts.values()):
        raise ValueError(f"本实验每类最多一个来源：sample_id={sample.sample_id}")
    events: list[PlannedEvent] = []
    for source in sample.source_streams:
        initial, interval = source_timing(source.source_type, args)
        for index, record in enumerate(get_data_events(source)):
            event = record["event"]
            if (
                event["event_type"] != source.source_type
                or event.get("source_type", source.source_type) != source.source_type
            ):
                raise ValueError(f"事件与来源类型不符：{source.source_id}")
            events.append(
                PlannedEvent(
                    initial + index * interval,
                    source.source_id,
                    source.source_type,
                    dict(event),
                )
            )
    events.sort(
        key=lambda item: (
            round(item.offset_s, 9),
            0 if item.source_type == "append" else 1,
            item.source_id,
            int(item.event["sequence_no"]),
        )
    )
    return tuple(events)


def plan_fingerprint(plan: tuple[PlannedEvent, ...]) -> str:
    """冻结共同输入顺序；arrival_time 由实际发送时附加，不写入计划。"""
    return fingerprint(
        [
            {
                "offset_s": item.offset_s,
                "source_id": item.source_id,
                "source_type": item.source_type,
                "event": {
                    key: value
                    for key, value in item.event.items()
                    if key not in ("arrival_time", "token_ids")
                },
            }
            for item in plan
        ]
    )


def materialize(
    plan: tuple[PlannedEvent, ...], layout: Layout
) -> tuple[OutboundEvent, ...]:
    """整体布局每次发送全部当前正文；分段布局保留原始 Append/Update 事件。"""
    append_parts: list[str] = []  # 只追加稳定内容，保留其原始换行。
    update_text = ""  # 每代完全覆盖上一代，不拼接历史候选。
    outbound: list[OutboundEvent] = []
    for index, item in enumerate(plan):
        if item.source_type == "append":
            append_parts.append(str(item.event.get("text") or ""))
        else:
            update_text = str(item.event.get("text") or "")
        if layout == "segmented":
            event = dict(item.event)
        else:
            append_text = "".join(append_parts)
            text = (
                update_text + append_text
                if layout == "update_first"
                else append_text + update_text
            )
            event = {
                "source_id": WHOLE_SOURCE_ID,
                "source_type": "update",
                "event_type": "update",
                "sequence_no": index,
                "version": index,
                "chunk_id": f"whole-{index}",
                "text": text,
            }
        event.pop("arrival_time", None)
        event.pop("token_ids", None)
        outbound.append(OutboundEvent(item, event))
    return tuple(outbound)


def final_evidence(sample: SampleReplayInput) -> dict[str, str]:
    """分别冻结两类最终正文和问题，保证改变布局不会改变证据集合。"""
    append_parts: list[str] = []
    update = ""
    for source in sample.source_streams:
        data = get_data_events(source)
        if source.source_type == "append":
            append_parts.extend(str(row["event"].get("text") or "") for row in data)
        elif data:
            update = str(data[-1]["event"].get("text") or "")
    return {
        "append": "".join(append_parts),
        "update": update,
        "question": str(sample.finish_event.get("text") or ""),
    }


def render_prompt(
    sample: SampleReplayInput,
    renderer: ChatTemplateRenderer,
    layout: Layout,
    input_transport: str,
) -> RenderedReplayPrompt:
    """复用模型模板，只有 Update-first 改变 user 正文的空间排列。"""
    prompt = render_sample_prompt(sample, renderer, input_transport=input_transport)
    evidence = final_evidence(sample)
    body = (
        evidence["update"] + evidence["append"]
        if layout == "update_first"
        else evidence["append"] + evidence["update"]
    )
    text = prompt.start_text + body + prompt.finish_text
    if text != renderer.render_chat_prompt(
        str(sample.start_request["prompt"]).strip(), body + evidence["question"]
    ):
        raise ValueError(f"模板不能无损拆分：{sample.sample_id}")
    token_ids = prompt.logical_prompt_token_ids
    if input_transport == "token_ids" and layout != "segmented":
        token_ids = (
            prompt.start_token_ids
            + renderer.encode_content(body)
            + prompt.finish_token_ids
        )
    return replace(prompt, logical_prompt_text=text, logical_prompt_token_ids=token_ids)
