# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""以正式统计约束运行 mixed-source Update 合并回放。"""

from __future__ import annotations

import os
import sys
from pathlib import Path

DEFAULT_OUTPUT_DIR = Path("research/scripts/replay/update_coalescing/results")


def main() -> None:
    """委托标准流式回放入口，并注入 Update 合并实验约束。

    合并策略属于 P 节点的 Engine 启动配置。该入口不会在客户端模拟策略，
    而是复用 segmented-prefill 已验证的 source 并发与 REST 时序，仅增加实验
    套件标签、统计必需检查和独立结果目录。用户在本入口后传入的同名参数会
    覆盖这里的默认值，因此 disabled 基线可显式使用
    ``--expected-coalescing disabled``。
    """
    segmented_replay = (
        Path(__file__).resolve().parents[1]
        / "segmented_prefill"
        / "replay_prompt_stream.py"
    )
    delegated_args = [
        sys.executable,
        str(segmented_replay),
        "--experiment-suite",
        "update_coalescing",
        "--expected-coalescing",
        "enabled",
        "--require-prompt-stream-stats",
        "--output-dir",
        str(DEFAULT_OUTPUT_DIR),
        *sys.argv[1:],
    ]

    # exec 保留 Slurm 信号和退出码，使 fail-fast/取消行为与标准入口一致。
    os.execv(sys.executable, delegated_args)


if __name__ == "__main__":
    main()
