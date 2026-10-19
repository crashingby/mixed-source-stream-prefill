# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""将 HotpotQA Fullwiki QA 中的 context 生成独立 Append 样本。"""

from __future__ import annotations

import argparse
import logging
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from append_chunks_common import (
    DATA_ROOT,
    EvidenceDocument,
    InputRecord,
    add_common_arguments,
    generate_dataset,
    normalize_body,
    output_version,
    source_path,
    text_sha256,
    validate_arguments,
)


def normalize_pairs(value: Any, value_key: str) -> list[list[Any]]:
    """兼容 Hugging Face 列式结构和官方 JSON 的二元列表。"""
    if isinstance(value, dict):
        titles, values = value["title"], value[value_key]
        if len(titles) != len(values):
            raise ValueError(f"title 与 {value_key} 长度不一致")
        return [[title, item] for title, item in zip(titles, values)]
    if isinstance(value, list) and all(
        isinstance(pair, (list, tuple)) and len(pair) == 2 for pair in value
    ):
        return [list(pair) for pair in value]
    raise ValueError(f"不支持的 {value_key} 结构：{type(value).__name__}")


def iter_questions(files: list[tuple[str, Path]]) -> Iterator[InputRecord]:
    """分批读取 parquet，context 和标准标注分别交给生成与评价。"""
    try:
        import pyarrow.parquet as pq
    except ImportError as error:
        raise RuntimeError(
            "缺少 pyarrow，请用 uv 安装 data_processing/requirements.txt"
        ) from error
    for split, path in files:
        with pq.ParquetFile(path) as parquet:
            for batch in parquet.iter_batches(batch_size=128):
                for item in batch.to_pylist():
                    facts = item.get("supporting_facts")
                    yield InputRecord(
                        question_id=str(item.get("id", item.get("_id", ""))),
                        question=item["question"],
                        split=split,
                        source_file=path,
                        raw=item,
                        evaluation={
                            "protocol": "hotpotqa_answer_em_f1",
                            "answer": item.get("answer"),
                            "supporting_facts": normalize_pairs(facts, "sent_id")
                            if facts is not None
                            else None,
                            "type": item.get("type"),
                            "level": item.get("level"),
                        },
                    )


def extract_documents(
    record: InputRecord,
) -> tuple[list[EvidenceDocument], list[dict[str, Any]]]:
    """保留 context 文档与句子的原始顺序，选择过程不读取答案标注。"""
    documents: list[EvidenceDocument] = []
    issues: list[dict[str, Any]] = []
    for index, (title, sentences) in enumerate(
        normalize_pairs(record.raw["context"], "sentences")
    ):
        if not isinstance(title, str) or not isinstance(sentences, list):
            raise ValueError(f"context[{index}] 的标题或句子列表无效")
        parts: list[str] = []
        spans: list[dict[str, Any]] = []
        offset = 0
        last_character = ""
        for sentence_id, sentence in enumerate(sentences):
            if not isinstance(sentence, str):
                raise ValueError(f"context[{index}] 句子 {sentence_id} 不是字符串")
            normalized = normalize_body(sentence)
            if (
                normalized
                and last_character
                and not last_character.isspace()
                and not normalized[0].isspace()
            ):
                parts.append(" ")
                offset += 1
            spans.append(
                {
                    "sentence_id": sentence_id,
                    "body_span": [offset, offset + len(normalized)],
                }
            )
            parts.append(normalized)
            offset += len(normalized)
            if normalized:
                last_character = normalized[-1]
        body = "".join(parts)
        provenance = {
            "qa_file": source_path(record.source_file),
            "question_id": record.question_id,
            "context_index": index,
            "original_title": title,
        }
        if not body.strip():
            issues.append({**provenance, "reason": "empty_body"})
            continue
        documents.append(
            EvidenceDocument(
                document_id=f"hotpotqa:context:{title}@{text_sha256(body)}",
                title=title,
                body=body,
                source_type="dataset_context",
                provenance=provenance,
                sentence_spans=spans,
            )
        )
    return documents, issues


def resolve_inputs(data_dir: Path, split: str) -> list[tuple[str, Path]]:
    """检查 QA 分片的完整性，并将 validation 统一命名为 dev。"""
    files = []
    splits = ["train", "dev"] if split == "all" else [split]
    for current in splits:
        filenames = (
            ["validation-00000-of-00001.parquet"]
            if current == "dev"
            else ["train-00000-of-00002.parquet", "train-00001-of-00002.parquet"]
        )
        for filename in filenames:
            path = data_dir / "qa/fullwiki" / filename
            if not path.is_file():
                raise FileNotFoundError(f"缺少 QA 分片：{path}")
            files.append((current, path))
    return files


def main(argv: list[str] | None = None) -> int:
    """生成 HotpotQA context 证据与独立问题清单。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DATA_ROOT / "raw/hotpotqa")
    parser.add_argument("--split", choices=["dev", "train", "all"], default="dev")
    add_common_arguments(parser)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    validate_arguments(args)
    files = resolve_inputs(args.data_dir, args.split)
    version = output_version(f"fullwiki_{args.split}_context_v1", args)
    output_dir = args.output_dir or DATA_ROOT / "append_chunks/hotpotqa" / version
    questions_file = (
        args.questions_file or DATA_ROOT / "questions/hotpotqa" / f"{version}.jsonl"
    )
    metadata = generate_dataset(
        records=iter_questions(files),
        extractor=extract_documents,
        dataset="hotpotqa",
        generator="build_hotpotqa_append_chunks",
        source_files=[path for _, path in files],
        args=args,
        output_dir=output_dir,
        questions_file=questions_file,
        settings={
            "split": args.split,
            "selection": "dataset_context_original_order",
            "context_setting": "official_fullwiki_retrieval"
            if args.split == "dev"
            else "training_context_included",
            "deduplication": "title_and_body_sha256",
            "live_web_search": False,
            "uses_gold_for_selection": False,
        },
    )
    return 1 if metadata["completion_status"] == "complete_with_errors" else 0


if __name__ == "__main__":
    raise SystemExit(main())
