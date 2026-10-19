# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""最小文本前端，支持直连 P 或经 proxy 测试来源生命周期。"""

from __future__ import annotations

import argparse
import asyncio
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, cast

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from research.scripts.simple_test.frontend_common import (
    DemoEvent,
    add_common_arguments,
    run_demo,
)

PromptStreamEventKind = Literal["append", "update"]

DEFAULT_PREFIX = (
    "You are a helpful assistant. Read the streaming context and the latest "
    "mutable snapshot first, then answer the final user question.\n\n"
)

DEFAULT_MIXED_EVENTS: list[tuple[PromptStreamEventKind, str]] = [
    (
        "append",
        "Stable document A: vLLM uses EngineCore to keep long-lived request state.\n",
    ),
    (
        "update",
        (
            "Mutable search result v0: the proxy receives prompt stream events "
            "before decode.\n"
        ),
    ),
    (
        "append",
        "Stable document B: append chunks represent monotonic context growth.\n",
    ),
    (
        "update",
        (
            "Mutable search result v1: the proxy receives prompt stream events, "
            "lets the P node prefill them, then sends the final decode request to "
            "the D node.\n"
        ),
    ),
    (
        "append",
        (
            "Stable document C: P and D nodes communicate KV cache metadata through "
            "kv_transfer_params.\n"
        ),
    ),
]

DEFAULT_FINAL_QUESTION = (
    "\nUser question: Based on the stable documents and the latest mutable "
    "snapshot, what does the proxy do before decode?\nAnswer:"
)


@dataclass(frozen=True)
class MixedPromptStreamEvent:
    """描述混合模式脚本中的一个前端事件。

    Args:
        event_kind: 事件类型，只允许 `append` 或 `update`。
        text: 事件携带的原始文本。append 表示新增 chunk，update 表示
            同一个 update source 的完整当前快照。
    """

    event_kind: PromptStreamEventKind
    text: str


def _parse_mixed_event(raw_event: str) -> MixedPromptStreamEvent:
    """解析命令行传入的单个 mixed event。

    Args:
        raw_event: `--event` 的原始字符串，格式为 `append:text` 或
            `update:text`。

    Returns:
        结构化后的事件描述。

    Raises:
        ValueError: 当事件格式或事件类型非法时抛出。
    """
    event_kind, separator, event_text = raw_event.partition(":")
    if not separator:
        raise ValueError(
            "Mixed event must use '<append|update>:text' format, but got "
            f"{raw_event!r}."
        )

    normalized_kind = event_kind.strip().lower()
    if normalized_kind not in ("append", "update"):
        raise ValueError(
            f"Mixed event kind must be 'append' or 'update', but got {event_kind!r}."
        )
    if not event_text:
        raise ValueError(f"Mixed event text must be non-empty, but got {raw_event!r}.")

    return MixedPromptStreamEvent(
        event_kind=cast(PromptStreamEventKind, normalized_kind),
        text=event_text,
    )


def _get_mixed_events(args: argparse.Namespace) -> list[MixedPromptStreamEvent]:
    """返回本次测试要发送的混合事件序列。

    Args:
        args: 命令行参数。

    Returns:
        按发送顺序排列的事件列表。
    """
    if args.events is None:
        return [
            MixedPromptStreamEvent(event_kind=event_kind, text=text)
            for event_kind, text in DEFAULT_MIXED_EVENTS
        ]

    return [_parse_mixed_event(raw_event) for raw_event in args.events]


def parse_args() -> argparse.Namespace:
    """解析 mixed 文本前端参数；公共选项保持三个入口一致。"""
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_arguments(parser, DEFAULT_PREFIX, DEFAULT_FINAL_QUESTION)
    parser.add_argument("--event", action="append", dest="events")
    parser.add_argument("--append-source-id", default="demo_append_source")
    parser.add_argument("--update-source-id", default="demo_update_source")
    return parser.parse_args()


def run_mixed_demo(args: argparse.Namespace) -> None:
    """发送 mixed 数据、结束各来源并验证 finish；不在客户端分词。"""
    events = [
        DemoEvent(
            args.append_source_id
            if event.event_kind == "append"
            else args.update_source_id,
            event.event_kind,
            event.text,
        )
        for event in _get_mixed_events(args)
    ]
    asyncio.run(run_demo(args, events))


def main() -> None:
    """解析命令行并执行一次最小测试。"""
    run_mixed_demo(parse_args())


if __name__ == "__main__":
    main()
