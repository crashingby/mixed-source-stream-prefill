# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""上传 TriviaQA Wikipedia，再为独立问题清单采集 Update 候选池。"""

from __future__ import annotations

import argparse
import logging
from collections.abc import Iterator
from pathlib import Path
from urllib.parse import quote

from append_chunks_common import normalize_body, source_path
from pipeline_pools_common import WikiDocument, add_arguments, run_pipeline


def iter_wikipedia(root: Path) -> Iterator[WikiDocument]:
    """按固定文件顺序读取整个证据集合，不使用逐问题的 gold 关联。"""
    found = False
    for path in sorted(root.rglob("*.txt")):
        found = True
        relative = path.relative_to(root).as_posix()
        title = path.stem.replace("_", " ")
        text = normalize_body(path.read_text(encoding="utf-8"))
        if not text.strip():
            logging.getLogger(__name__).warning("跳过空文章：%s", path)
            continue
        yield WikiDocument(
            source_id=f"triviaqa:wikipedia:{relative}",
            title=title,
            text=text,
            url="https://en.wikipedia.org/wiki/" + quote(path.stem, safe=""),
            source_file=source_path(path),
        )
    if not found:
        raise FileNotFoundError(f"没有 Wikipedia .txt 文件：{root}")


def main(argv: list[str] | None = None) -> int:
    """默认先入库，再采集现有 unfiltered Web dev 问题，支持自动续跑。"""
    parser = argparse.ArgumentParser(description=__doc__)
    add_arguments(parser, "triviaqa", 8091)
    args = parser.parse_args(argv)
    root = args.data_dir / "rc/evidence/wikipedia"
    return run_pipeline(
        args,
        "triviaqa",
        "web_unfiltered_dev_v1",
        root,
        lambda: iter_wikipedia(root),
    )


if __name__ == "__main__":
    raise SystemExit(main())
