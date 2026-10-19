# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""上传 HotpotQA Wikipedia 摘要语料，再逐问题采集 Update 候选池。"""

from __future__ import annotations

import argparse
import bz2
import json
import logging
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from append_chunks_common import normalize_body, source_path
from pipeline_pools_common import WikiDocument, add_arguments, run_pipeline


def article_text(value: Any) -> str:
    """兼容字符串、句子列表和段落列表，避免句子边界粘连。"""
    if isinstance(value, str):
        return normalize_body(value)
    if not isinstance(value, list):
        raise ValueError("Wikipedia text 必须是字符串或列表")
    parts = [article_text(item) for item in value]
    separator = "\n\n" if any(isinstance(item, list) for item in value) else " "
    return separator.join(parts)


def iter_wikipedia(root: Path) -> Iterator[WikiDocument]:
    """逐个流式解压语料分片；输入顺序与问题及标准标注无关。"""
    found = False
    for path in sorted(root.rglob("*.bz2")):
        found = True
        with bz2.open(path, "rt", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                item = json.loads(line)
                text = article_text(item["text"])
                if not text.strip():
                    logging.getLogger(__name__).warning(
                        "跳过空文章：%s:%d", path, line_number
                    )
                    continue
                yield WikiDocument(
                    source_id=f"hotpotqa:enwiki20171001:{item['id']}",
                    title=item["title"],
                    text=text,
                    url=item["url"],
                    source_file=f"{source_path(path)}:{line_number}",
                )
    if not found:
        raise FileNotFoundError(f"没有 Wikipedia .bz2 分片：{root}")


def main(argv: list[str] | None = None) -> int:
    """默认先入库，再采集现有 Fullwiki dev 问题，支持自动续跑。"""
    parser = argparse.ArgumentParser(description=__doc__)
    add_arguments(parser, "hotpotqa", 8092)
    args = parser.parse_args(argv)
    root = (
        args.data_dir / "corpus/enwiki-20171001-pages-meta-current-withlinks-abstracts"
    )
    return run_pipeline(
        args,
        "hotpotqa",
        "fullwiki_dev_context_v1",
        root,
        lambda: iter_wikipedia(root),
    )


if __name__ == "__main__":
    raise SystemExit(main())
