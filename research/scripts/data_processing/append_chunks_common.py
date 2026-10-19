# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""离线 Append 生成器共用的渲染、分段与文件写入。"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import shutil
import tempfile
from collections import Counter
from collections.abc import Callable, Iterable, Iterator
from contextlib import nullcontext
from dataclasses import dataclass, field
from itertools import islice
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)
PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATA_ROOT = PROJECT_ROOT / "research" / "data"
SCHEMA_VERSION = "append_chunks/v1"
RENDERER_VERSION = "markdown_sources/v1"


@dataclass
class EvidenceDocument:
    """一份尚未渲染的证据；正文及来源均由数据适配器提供。"""

    document_id: str
    title: str
    body: str
    source_type: str
    provenance: dict[str, Any]
    url: str | None = None
    sentence_spans: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class InputRecord:
    """一条原始问题及其来源；评价字段在写入时单独保存。"""

    question_id: str
    question: str
    split: str
    source_file: Path
    raw: dict[str, Any]
    evaluation: dict[str, Any]


Extractor = Callable[[InputRecord], tuple[list[EvidenceDocument], list[dict[str, Any]]]]
SampleBuilder = Callable[
    [str, InputRecord, list[EvidenceDocument], list[dict[str, Any]]], dict[str, Any]
]


def text_sha256(text: str) -> str:
    """计算 UTF-8 文本指纹。"""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def normalize_body(text: str) -> str:
    """只统一换行，保留正文空白及噪声。"""
    return text.replace("\r\n", "\n").replace("\r", "\n")


def source_path(path: Path) -> str:
    """默认返回相对于 data 的路径，外部输入保留绝对路径。"""
    resolved = path.resolve()
    try:
        return resolved.relative_to(DATA_ROOT.resolve()).as_posix()
    except ValueError:
        return str(resolved)


def fingerprint_file(path: Path) -> dict[str, Any]:
    """流式计算输入文件指纹。"""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return {
        "path": source_path(path),
        "bytes": path.stat().st_size,
        "sha256": digest.hexdigest(),
    }


def add_common_arguments(parser: argparse.ArgumentParser) -> None:
    """添加两种数据集共用的输出与内容选项。"""
    parser.add_argument("--output-dir", type=Path, help="覆盖默认 Append 输出目录")
    parser.add_argument("--questions-file", type=Path, help="覆盖默认问题清单路径")
    parser.add_argument("--max-requests", type=int, default=0, help="0 表示全部问题")
    parser.add_argument("--max-documents", type=int, default=0, help="0 表示全部证据")
    parser.add_argument(
        "--chunk-chars",
        type=int,
        default=0,
        help="连续分段的目标字符数；0 表示一份文档一块，保持标题完整",
    )
    parser.add_argument("--overwrite", action="store_true", help="重建本生成器的输出")
    parser.add_argument("--log-every", type=int, default=500, help="进度输出间隔")


def validate_arguments(args: argparse.Namespace) -> None:
    """校验非负选项，配置错误在创建文件前报告。"""
    for name in ["max_requests", "max_documents", "chunk_chars", "log_every"]:
        if getattr(args, name) < 0:
            raise ValueError(f"--{name.replace('_', '-')} 必须非负")


def output_version(base: str, args: argparse.Namespace) -> str:
    """为限制数量或改变分段的生成任务分配独立版本名。"""
    version = base
    for name, label in [
        ("max_requests", "first"),
        ("max_documents", "docs"),
        ("chunk_chars", "chars"),
    ]:
        value = getattr(args, name)
        if value:
            version += f"_{label}{value}"
    return version


def split_rendered_text(
    text: str, chunk_chars: int, header_chars: int
) -> Iterator[tuple[int, int, str]]:
    """连续分段，保持文档头完整以及重组后的字符不变。"""
    if chunk_chars < 0:
        raise ValueError("chunk_chars 必须非负")
    start = 0
    while start < len(text):
        end = len(text) if not chunk_chars else min(start + chunk_chars, len(text))
        if start == 0:
            end = max(end, header_chars)
        if end < len(text):
            lower = max(start + (end - start) // 2, header_chars if start == 0 else 0)
            boundary = text.rfind("\n", lower, end)
            if boundary < 0:
                boundary = text.rfind(" ", lower, end)
            if boundary >= lower:
                end = boundary + 1
        yield start, end, text[start:end]
        start = end


def build_sample(
    sample_id: str,
    documents: list[EvidenceDocument],
    issues: list[dict[str, Any]],
    chunk_chars: int = 0,
    max_documents: int = 0,
) -> dict[str, Any]:
    """构造模型可见的文本块，保存去重来源与完整文本指纹。"""
    if max_documents < 0:
        raise ValueError("max_documents 必须非负")
    unique: dict[str, EvidenceDocument] = {}
    duplicates: dict[str, list[dict[str, Any]]] = {}
    for document in documents:
        if document.document_id in unique:
            duplicates.setdefault(document.document_id, []).append(document.provenance)
        else:
            unique[document.document_id] = document
    selected = list(unique.values())
    excluded = max(0, len(selected) - max_documents) if max_documents else 0
    if max_documents:
        selected = selected[:max_documents]

    entries: list[dict[str, Any]] = []
    chunks: list[dict[str, Any]] = []
    for index, document in enumerate(selected, 1):
        citation_id = f"A{index}"
        title = " ".join(document.title.split()) or "Untitled document"
        header = f"\n\n### [{citation_id}] {title}\n"
        if document.url:
            header += f"Source: {' '.join(document.url.split())}\n"
        header += "\n"
        rendered = header + document.body + "\n\n---\n"
        entry = {
            "document_id": document.document_id,
            "citation_id": citation_id,
            "source_type": document.source_type,
            "title": document.title,
            "url": document.url,
            "provenance": document.provenance,
            "body_chars": len(document.body),
            "body_sha256": text_sha256(document.body),
            "body_start": len(header),
            "rendered_chars": len(rendered),
            "rendered_sha256": text_sha256(rendered),
        }
        if document.sentence_spans:
            entry["sentence_spans"] = document.sentence_spans
        if document.document_id in duplicates:
            entry["duplicate_provenance"] = duplicates[document.document_id]
        entries.append(entry)
        for start, end, text in split_rendered_text(rendered, chunk_chars, len(header)):
            sequence_no = len(chunks)
            chunks.append(
                {
                    "sequence_no": sequence_no,
                    "chunk_id": f"a{sequence_no:04d}",
                    "document_id": document.document_id,
                    "rendered_span": [start, end],
                    "text": text,
                }
            )
    append_text = "".join(chunk["text"] for chunk in chunks)
    return {
        "schema": SCHEMA_VERSION,
        "sample_id": sample_id,
        "status": "ready" if entries else "empty",
        "documents": entries,
        "chunks": chunks,
        "append_chars": len(append_text),
        "append_sha256": text_sha256(append_text),
        "diagnostics": {
            "issues": issues,
            "num_candidates": len(documents) + len(issues),
            "num_duplicates": len(documents) - len(unique),
            "num_excluded_by_limit": excluded,
        },
    }


def write_json(path: Path, value: dict[str, Any]) -> None:
    """写入 UTF-8 JSON 文件。"""
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def _check_destinations(
    output_dir: Path, questions_file: Path, generator: str, overwrite: bool
) -> bool:
    """校验输出归属，返回是否需要校验并复用独立存在的问题清单。"""
    if questions_file.is_relative_to(output_dir):
        raise ValueError("问题清单必须独立于 Append 输出目录")
    if output_dir.exists():
        if not overwrite:
            raise FileExistsError(f"输出已存在：{output_dir}；重建请使用 --overwrite")
        metadata_path = output_dir / "metadata.json"
        if not metadata_path.is_file():
            raise ValueError(f"拒绝覆盖非生成器目录：{output_dir}")
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if (
            metadata.get("generator") != generator
            or metadata.get("schema") != SCHEMA_VERSION
        ):
            raise ValueError(f"拒绝覆盖其他生成器数据：{output_dir}")
        unexpected = {path.name for path in output_dir.iterdir()} - {
            "samples",
            "manifest.jsonl",
            "metadata.json",
        }
        if unexpected:
            raise ValueError(f"输出目录含额外文件：{sorted(unexpected)}")
        if questions_file.exists() and metadata.get("questions_file") != str(
            questions_file
        ):
            raise ValueError(f"问题清单不属于该输出：{questions_file}")
        return False
    return questions_file.exists()


def _publish(
    stage: Path, output_dir: Path, questions_file: Path, reuse_questions: bool = False
) -> None:
    """发布 Append；复用相同的问题清单，否则替换并在失败时恢复。"""
    if reuse_questions and (
        questions_file.is_symlink()
        or fingerprint_file(questions_file)["sha256"]
        != fingerprint_file(stage / "questions.jsonl")["sha256"]
    ):
        raise ValueError(f"问题清单在生成期间发生变化，未发布 Append：{questions_file}")
    backup_output = stage / "previous"
    backup_questions = stage / "previous_questions.jsonl"
    output_published = False
    questions_published = False
    try:
        if output_dir.exists():
            output_dir.rename(backup_output)
        if questions_file.exists() and not reuse_questions:
            questions_file.rename(backup_questions)
        (stage / "dataset").rename(output_dir)
        output_published = True
        if not reuse_questions:
            (stage / "questions.jsonl").rename(questions_file)
            questions_published = True
    except BaseException:
        if output_published:
            shutil.rmtree(output_dir)
        if questions_published:
            questions_file.unlink()
        if backup_output.exists():
            backup_output.rename(output_dir)
        if backup_questions.exists():
            backup_questions.rename(questions_file)
        raise


def generate_dataset(
    records: Iterable[InputRecord],
    extractor: Extractor,
    dataset: str,
    generator: str,
    source_files: list[Path],
    args: argparse.Namespace,
    output_dir: Path,
    questions_file: Path,
    settings: dict[str, Any],
    sample_builder: SampleBuilder | None = None,
    generator_version: int = 1,
) -> dict[str, Any]:
    """流式生成样本、manifest 和问题清单，完成后发布文件。

    Args:
        records: 原始问题的迭代器。
        extractor: 按固定规则读取该问题的证据。
        dataset: 数据集命名空间。
        generator: 用于安全重建的生成器标识。
        source_files: 本次读取的 QA 文件。
        args: 共用命令行选项。
        output_dir: Append 数据版本目录。
        questions_file: 独立问题和评价清单。
        settings: 数据适配器的选择策略。
        sample_builder: 可选的按问题筛选与渲染函数；默认保留共用文档逻辑。
        generator_version: 内容生成规则的版本号。

    Returns:
        已写入 metadata.json 的统计和配置。
    """
    validate_arguments(args)
    output_dir, questions_file = output_dir.absolute(), questions_file.absolute()
    if output_dir.is_symlink() or questions_file.is_symlink():
        raise ValueError("输出位置不能是符号链接")
    output_dir, questions_file = output_dir.resolve(), questions_file.resolve()
    reuse_questions = _check_destinations(
        output_dir, questions_file, generator, args.overwrite
    )
    if reuse_questions:
        logger.info("校验并复用已有问题清单：%s", questions_file)
    inputs = [fingerprint_file(path) for path in source_files]
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    questions_file.parent.mkdir(parents=True, exist_ok=True)
    counts: Counter[str] = Counter()
    issue_counts: Counter[str] = Counter()
    seen: set[str] = set()
    limited_records = islice(records, args.max_requests or None)
    with tempfile.TemporaryDirectory(
        prefix=f".{output_dir.name}.build-", dir=output_dir.parent
    ) as temporary:
        stage = Path(temporary)
        staging_output = stage / "dataset"
        (staging_output / "samples").mkdir(parents=True)
        with (
            (staging_output / "manifest.jsonl").open("w", encoding="utf-8") as manifest,
            (stage / "questions.jsonl").open("w", encoding="utf-8") as questions,
            (
                questions_file.open("rb") if reuse_questions else nullcontext()
            ) as existing_questions,
        ):
            for record in limited_records:
                if not record.question_id or not record.question.strip():
                    raise ValueError(f"问题 ID 或正文为空：{record.source_file}")
                sample_id = f"{dataset}:{record.split}:{record.question_id}"
                if sample_id in seen:
                    raise ValueError(f"重复问题 ID：{sample_id}")
                seen.add(sample_id)
                question_line = (
                    json.dumps(
                        {
                            "sample_id": sample_id,
                            "dataset": dataset,
                            "split": record.split,
                            "question_id": record.question_id,
                            "question": record.question,
                            "evaluation": record.evaluation,
                            "source_file": source_path(record.source_file),
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                # 在读取正文和 BM25 计算前发现不兼容，避免覆盖其他实验的 QA。
                if existing_questions is not None and (
                    existing_questions.readline() != question_line.encode("utf-8")
                ):
                    raise ValueError(
                        f"已有问题清单与本次输入不一致：{questions_file}，{sample_id}；"
                        "请指定新的 --questions-file"
                    )
                try:
                    documents, issues = extractor(record)
                    if sample_builder is None:
                        sample = build_sample(
                            sample_id,
                            documents,
                            issues,
                            args.chunk_chars,
                            args.max_documents,
                        )
                    else:
                        sample = sample_builder(sample_id, record, documents, issues)
                except (OSError, ValueError, TypeError, KeyError) as error:
                    logger.error("%s: %s", sample_id, error)
                    sample = build_sample(
                        sample_id,
                        [],
                        [{"reason": "processing_error", "detail": str(error)}],
                    )
                    sample["status"] = "error"
                relative_path = f"samples/{text_sha256(sample_id)}.json"
                write_json(staging_output / relative_path, sample)
                issues = sample["diagnostics"]["issues"]
                entry = {
                    "sample_id": sample_id,
                    "path": relative_path,
                    "status": sample["status"],
                    "num_documents": len(sample["documents"]),
                    "num_chunks": len(sample["chunks"]),
                    "append_chars": sample["append_chars"],
                    "append_sha256": sample["append_sha256"],
                    "num_unavailable_candidates": len(issues),
                }
                if "append_tokens" in sample:
                    entry["append_tokens"] = sample["append_tokens"]
                    counts["append_tokens"] += sample["append_tokens"]
                manifest.write(json.dumps(entry, ensure_ascii=False) + "\n")
                questions.write(question_line)
                counts["requests"] += 1
                counts[sample["status"]] += 1
                counts["documents"] += len(sample["documents"])
                counts["chunks"] += len(sample["chunks"])
                counts["append_chars"] += sample["append_chars"]
                issue_counts.update(issue["reason"] for issue in issues)
                if args.log_every and counts["requests"] % args.log_every == 0:
                    logger.info("已生成 %d 个请求", counts["requests"])
            if existing_questions is not None and existing_questions.read(1):
                raise ValueError(
                    f"已有问题清单比本次输入包含更多请求：{questions_file}；"
                    "请指定新的 --questions-file"
                )
        if not counts["requests"]:
            raise ValueError("输入没有问题；未发布输出")
        metadata = {
            "schema": SCHEMA_VERSION,
            "generator": generator,
            "generator_version": generator_version,
            "dataset": dataset,
            "renderer": RENDERER_VERSION,
            "cleaning": "normalize_newlines/v1",
            "chunk_chars": args.chunk_chars,
            "max_documents": args.max_documents,
            "max_requests": args.max_requests,
            "settings": settings,
            "inputs": inputs,
            "questions_file": str(questions_file),
            "questions_sha256": fingerprint_file(stage / "questions.jsonl")["sha256"],
            "statistics": dict(counts),
            "unavailable_candidates": dict(issue_counts),
            "completion_status": "complete_with_errors"
            if counts["error"]
            else "complete",
        }
        write_json(staging_output / "metadata.json", metadata)
        _publish(stage, output_dir, questions_file, reuse_questions)
    logger.info("生成完成：%s", json.dumps(dict(counts), ensure_ascii=False))
    logger.info("Append: %s", output_dir)
    logger.info("Questions: %s", questions_file)
    return metadata
