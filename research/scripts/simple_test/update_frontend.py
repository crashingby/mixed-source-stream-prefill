# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""最小文本前端，支持直连 P 或经 proxy 测试来源生命周期。"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from research.scripts.simple_test.frontend_common import (
    DemoEvent,
    add_common_arguments,
    run_demo,
)

DEFAULT_PREFIX = (
    "You are a helpful assistant. Read the latest streaming context snapshot "
    "first, then answer the final user question.\n\n"
)

DEFAULT_UPDATE_SNAPSHOTS = [
    "Document snapshot: The proxy receives stream-prefill events before decode.\n",
    (
        "Document snapshot: The proxy receives stream-prefill events before it "
        "sends decode to the D node.\n"
    ),
    (
        "Document snapshot: The proxy receives stream-prefill events before it "
        "sends the final decode request to the D node.\n"
    ),
]

DEFAULT_FINAL_QUESTION = (
    "\nUser question: According to the latest document snapshot, what does the "
    "proxy do before decode?\nAnswer:"
)


def parse_args() -> argparse.Namespace:
    """解析 update 文本前端参数；公共选项保持三个入口一致。"""
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_arguments(parser, DEFAULT_PREFIX, DEFAULT_FINAL_QUESTION)
    parser.add_argument("--snapshot", action="append", dest="snapshots")
    parser.add_argument("--source-id", default="demo_update_source")
    return parser.parse_args()


def run_update_demo(args: argparse.Namespace) -> None:
    """发送 update 数据、结束各来源并验证 finish；不在客户端分词。"""
    events = [
        DemoEvent(args.source_id, "update", text)
        for text in (args.snapshots or DEFAULT_UPDATE_SNAPSHOTS)
    ]
    asyncio.run(run_demo(args, events))


def main() -> None:
    """解析命令行并执行一次最小测试。"""
    run_update_demo(parse_args())


if __name__ == "__main__":
    main()
