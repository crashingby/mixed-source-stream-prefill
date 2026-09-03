# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""从 SQuAD v2 context 与 Stream2LLM ANNS trace 构造混合来源数据。

每个请求包含两个来源：

1. 根据 ANNS 问题自动定位 SQuAD v2 的官方 context，并把完整 context
   确定性地切成两个连续 Append chunk；
2. 将 ANNS trace 中每一行 PipelinePool 物化为完整 Update 快照；
3. 为两个来源分别追加 source_finished，最后保存 prompt_finished。

脚本完全离线，不执行网页搜索，也不保存事件到达时间。两个来源如何交错以及
事件间隔由回放脚本控制。输出继续使用 prompt_stream_mixed/v2，因此现有
normal、segmented-prefill 和 update-coalescing 回放器可以直接读取。
"""

from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import html
import json
import logging
import os
import re
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

logger = logging.getLogger(__name__)

SCHEMA_VERSION = "prompt_stream_mixed/v2"
DATA_REVISION = "squad_append_v1"
ANNS_SOURCE_TYPE: Literal["update"] = "update"
SQUAD_SOURCE_TYPE: Literal["append"] = "append"
NUM_SQUAD_APPEND_CHUNKS = 2
SQUAD_SPLIT_FILES = {
    "train": "train-v2.0.json",
    "dev": "dev-v2.0.json",
}
DEFAULT_PREFIX = (
    "You are a helpful assistant. Read the streamed retrieval context first. "
    "Some sources append stable evidence, while other sources publish mutable "
    "snapshots. Answer the final user question using the latest available "
    "evidence.\n\n"
)

SourceType = Literal["append", "update"]
SquadMatchMethod = Literal["exact", "normalized"]


@dataclass(frozen=True)
class AnnsTraceRow:
    """一条 ANNS PipelinePool 完整快照。

    Args:
        row_index: CSV 中从零开始的原始行号。
        start_iteration: 当前区间的起始 ANNS 迭代。
        end_iteration: 当前区间的结束 ANNS 迭代。
        document_ids: PipelinePool 中按原顺序排列的文档 id。
    """

    row_index: int
    start_iteration: int
    end_iteration: int
    document_ids: tuple[str, ...]


@dataclass(frozen=True)
class SquadAnswer:
    """SQuAD context 中的一条官方答案标注。"""

    text: str
    answer_start: int


@dataclass(frozen=True)
class SquadContextRecord:
    """一个问题对应的 SQuAD 官方 context 与来源信息。"""

    split: str
    title: str
    paragraph_index: int
    squad_id: str
    question: str
    context: str
    is_impossible: bool
    answers: tuple[SquadAnswer, ...]


@dataclass(frozen=True)
class SquadContextIndex:
    """同时支持精确问题和归一化问题查找的 SQuAD context 索引。"""

    exact_question_to_records: dict[str, tuple[SquadContextRecord, ...]]
    normalized_question_to_records: dict[str, tuple[SquadContextRecord, ...]]
    num_indexed_records: int


@dataclass(frozen=True)
class SquadContextMatch:
    """一个 ANNS 问题与 SQuAD context 的确定性匹配。"""

    method: SquadMatchMethod
    record: SquadContextRecord


@dataclass(frozen=True)
class SquadContextChunk:
    """SQuAD 官方 context 中的一个连续 Append chunk。"""

    chunk_index: int
    char_start: int
    char_end: int
    text: str
    contains_gold_answer: bool


def parse_args() -> argparse.Namespace:
    """解析混合来源数据生成参数。

    Returns:
        命令行参数命名空间。
    """
    parser = argparse.ArgumentParser(
        description=(
            "Build prompt-stream mixed-source samples from SQuAD v2 "
            "contexts and Stream2LLM ANNS traces."
        )
    )
    parser.add_argument(
        "--anns-dir",
        type=Path,
        default=Path("research/data/anns"),
        help="Directory containing query_trace_map_5k.json, res/, and corpus.",
    )
    parser.add_argument(
        "--squad-dir",
        type=Path,
        default=Path("research/data/squad_v2"),
        help="Directory containing train-v2.0.json and dev-v2.0.json.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("research/data/mix_source"),
        help="Dataset output directory.",
    )
    parser.add_argument(
        "--query-id",
        action="append",
        dest="query_ids",
        default=None,
        help="ANNS query id to build; repeat to select multiple ids.",
    )
    parser.add_argument(
        "--start-index",
        type=int,
        default=0,
        help="Start position in numerically sorted query ids.",
    )
    parser.add_argument(
        "--max-queries",
        type=int,
        default=1,
        help=(
            "Maximum queries to build when --query-id is omitted; use 0 for "
            "all remaining queries."
        ),
    )
    parser.add_argument(
        "--tokenizer-model",
        type=str,
        default=None,
        help=(
            "Local tokenizer path or locally cached HuggingFace repo used "
            "only for ANNS token budgets. Required when a --max-*-tokens "
            "budget is enabled."
        ),
    )
    parser.add_argument(
        "--max-anns-document-tokens",
        type=int,
        default=0,
        help=(
            "Truncate each formatted ANNS document to this many tokenizer "
            "tokens; 0 disables per-document token truncation."
        ),
    )
    parser.add_argument(
        "--max-update-snapshot-tokens",
        type=int,
        default=0,
        help=(
            "Truncate each complete ANNS Update snapshot to this many "
            "tokenizer tokens; 0 disables snapshot-level truncation."
        ),
    )
    parser.add_argument(
        "--max-anns-document-chars",
        type=int,
        default=0,
        help="Truncate each ANNS document to this many characters; 0 disables.",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=64,
        help="Default decode max_tokens stored in each start request.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Default decode temperature stored in each start request.",
    )
    parser.add_argument(
        "--prefix",
        type=str,
        default=DEFAULT_PREFIX,
        help="Fixed prompt prefix stored in each start request.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Rebuild current SQuAD-based request JSON files that already exist.",
    )
    parser.add_argument(
        "--fail-fast",
        action="store_true",
        help=(
            "Stop at the first trace, corpus, SQuAD match, or construction "
            "failure."
        ),
    )
    parser.add_argument(
        "--log-level",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
        default="INFO",
        help="Logging verbosity.",
    )
    return parser.parse_args()


def _atomic_write_text(path: Path, text: str) -> None:
    """原子写入文本，避免中断留下半个输出文件。

    函数会在目标目录创建临时文件，并原子替换目标文件。

    Args:
        path: 最终文件路径。
        text: 待写入的完整文本。
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary_path.write_text(text, encoding="utf-8")
    os.replace(temporary_path, path)


def _atomic_write_json(path: Path, value: Any) -> None:
    """以稳定、可读格式原子写入 JSON。"""
    serialized = json.dumps(
        value,
        ensure_ascii=False,
        indent=2,
        sort_keys=False,
    )
    _atomic_write_text(path, f"{serialized}\n")


def _load_json(path: Path) -> Any:
    """读取一个 UTF-8 JSON 文件。"""
    return json.loads(path.read_text(encoding="utf-8"))


def _numeric_query_sort_key(query_id: str) -> tuple[int, int | str]:
    """让纯数字 query id 按数值排序，其它 id 按字符串排序。"""
    if query_id.isdigit():
        return (0, int(query_id))
    return (1, query_id)


def _select_query_ids(
    query_map: dict[str, dict[str, str]],
    args: argparse.Namespace,
) -> list[str]:
    """根据显式 id 或位置切片选择本次处理的问题。

    Raises:
        ValueError: 参数越界或显式 query id 不存在。
    """
    if args.query_ids:
        missing_ids = [
            query_id for query_id in args.query_ids if query_id not in query_map
        ]
        if missing_ids:
            raise ValueError(f"Unknown ANNS query ids: {missing_ids}.")
        return list(dict.fromkeys(args.query_ids))

    if args.start_index < 0:
        raise ValueError("--start-index must be non-negative.")
    if args.max_queries < 0:
        raise ValueError("--max-queries must be non-negative.")

    query_ids = sorted(query_map, key=_numeric_query_sort_key)
    query_ids = query_ids[args.start_index :]
    if args.max_queries:
        query_ids = query_ids[: args.max_queries]
    return query_ids


def _normalize_question(question: str) -> str:
    """生成只用于 SQuAD 安全回退匹配的稳定问题键。

    归一化会消除大小写、重音、标点和重复空白，但不会修改最终请求中的原始
    问题。只有所有候选都指向同一个 context 时，归一化匹配才会被接受。
    """
    decomposed = unicodedata.normalize("NFKD", question).casefold()
    without_marks = "".join(
        character
        for character in decomposed
        if not unicodedata.combining(character)
    )
    alphanumeric_only = re.sub(
        r"[^\w]+",
        " ",
        without_marks,
        flags=re.UNICODE,
    )
    return " ".join(alphanumeric_only.split())


def _load_squad_context_index(
    squad_dir: Path,
    target_questions: set[str],
) -> SquadContextIndex:
    """只为本批 ANNS 问题构建 SQuAD context 索引。

    Args:
        squad_dir: 包含 SQuAD v2 train/dev JSON 的目录。
        target_questions: 本批实际需要匹配的 ANNS 问题原文。

    Returns:
        同时支持精确和归一化问题查找的只读索引。

    Raises:
        FileNotFoundError: 任一 SQuAD split 文件不存在。
        ValueError: SQuAD 文件结构或字段类型非法。
    """
    exact_targets = set(target_questions)
    normalized_targets = {
        _normalize_question(question) for question in target_questions
    }
    exact_records: dict[str, list[SquadContextRecord]] = defaultdict(list)
    normalized_records: dict[str, list[SquadContextRecord]] = defaultdict(list)
    num_indexed_records = 0

    # SQuAD 按 split -> article -> paragraph -> question 展开，只保留目标问题。
    for split, filename in SQUAD_SPLIT_FILES.items():
        split_path = squad_dir / filename
        if not split_path.is_file():
            raise FileNotFoundError(f"Missing SQuAD {split} split: {split_path}")
        split_payload = _load_json(split_path)
        articles = (
            split_payload.get("data")
            if isinstance(split_payload, dict)
            else None
        )
        if not isinstance(articles, list):
            raise ValueError(
                f"SQuAD split has no list-valued data: {split_path}"
            )

        for article in articles:
            if not isinstance(article, dict):
                raise ValueError(f"Invalid SQuAD article in {split_path}")
            title = article.get("title")
            paragraphs = article.get("paragraphs")
            if not isinstance(title, str) or not isinstance(paragraphs, list):
                raise ValueError(f"Invalid SQuAD article fields in {split_path}")

            for paragraph_index, paragraph in enumerate(paragraphs):
                if not isinstance(paragraph, dict):
                    raise ValueError(
                        f"Invalid SQuAD paragraph in {split_path}"
                    )
                context = paragraph.get("context")
                questions = paragraph.get("qas")
                if not isinstance(context, str) or not isinstance(
                    questions, list
                ):
                    raise ValueError(
                        f"Invalid SQuAD paragraph fields in {split_path}"
                    )

                for question_payload in questions:
                    if not isinstance(question_payload, dict):
                        raise ValueError(
                            f"Invalid SQuAD question in {split_path}"
                        )
                    question = question_payload.get("question")
                    squad_id = question_payload.get("id")
                    if not isinstance(question, str) or not isinstance(
                        squad_id, str
                    ):
                        raise ValueError(
                            f"Invalid SQuAD question fields in {split_path}"
                        )

                    normalized_question = _normalize_question(question)
                    if (
                        question not in exact_targets
                        and normalized_question not in normalized_targets
                    ):
                        continue

                    is_impossible = question_payload.get(
                        "is_impossible", False
                    )
                    raw_answers = question_payload.get("answers", [])
                    if not isinstance(is_impossible, bool) or not isinstance(
                        raw_answers, list
                    ):
                        raise ValueError(
                            f"Invalid SQuAD answer fields in {split_path}"
                        )

                    answers: list[SquadAnswer] = []
                    for raw_answer in raw_answers:
                        if not isinstance(raw_answer, dict):
                            raise ValueError(
                                f"Invalid SQuAD answer in {split_path}"
                            )
                        answer_text = raw_answer.get("text")
                        answer_start = raw_answer.get("answer_start")
                        if not isinstance(answer_text, str) or not isinstance(
                            answer_start, int
                        ):
                            raise ValueError(
                                f"Invalid SQuAD answer fields in {split_path}"
                            )
                        answer_end = answer_start + len(answer_text)
                        if context[answer_start:answer_end] != answer_text:
                            raise ValueError(
                                "SQuAD answer span does not match context: "
                                f"split={split}, squad_id={squad_id}."
                            )
                        answers.append(
                            SquadAnswer(
                                text=answer_text,
                                answer_start=answer_start,
                            )
                        )

                    record = SquadContextRecord(
                        split=split,
                        title=title,
                        paragraph_index=paragraph_index,
                        squad_id=squad_id,
                        question=question,
                        context=context,
                        is_impossible=is_impossible,
                        answers=tuple(answers),
                    )
                    exact_records[question].append(record)
                    normalized_records[normalized_question].append(record)
                    num_indexed_records += 1

    return SquadContextIndex(
        exact_question_to_records={
            question: tuple(records)
            for question, records in exact_records.items()
        },
        normalized_question_to_records={
            question: tuple(records)
            for question, records in normalized_records.items()
        },
        num_indexed_records=num_indexed_records,
    )


def _resolve_squad_context(
    question: str,
    squad_index: SquadContextIndex,
) -> SquadContextMatch:
    """为一个 ANNS 问题选择唯一的 SQuAD 官方 context。

    精确匹配优先。只有精确匹配为空时才使用归一化问题键；若多个候选来自
    不同 context，函数会拒绝静默选择。

    Raises:
        ValueError: 问题未匹配，或候选指向多个不同 context。
    """
    candidates = squad_index.exact_question_to_records.get(question, ())
    match_method: SquadMatchMethod = "exact"
    if not candidates:
        candidates = squad_index.normalized_question_to_records.get(
            _normalize_question(question),
            (),
        )
        match_method = "normalized"
    if not candidates:
        raise ValueError(f"No SQuAD context matched question {question!r}.")

    context_keys = {
        (
            record.split,
            record.title,
            record.paragraph_index,
            record.context,
        )
        for record in candidates
    }
    if len(context_keys) != 1:
        candidate_ids = sorted(record.squad_id for record in candidates)
        raise ValueError(
            "Ambiguous SQuAD contexts for question "
            f"{question!r}: squad_ids={candidate_ids}."
        )

    # 相同 context 下若有重复 QA，按 split 和 squad_id 稳定选择 provenance。
    selected_record = min(
        candidates,
        key=lambda record: (record.split, record.squad_id),
    )
    return SquadContextMatch(
        method=match_method,
        record=selected_record,
    )


def _answer_intervals(
    record: SquadContextRecord,
) -> tuple[tuple[int, int], ...]:
    """返回 context 内所有官方答案的半开字符区间。"""
    return tuple(
        (answer.answer_start, answer.answer_start + len(answer.text))
        for answer in record.answers
    )


def _is_safe_context_boundary(
    context: str,
    boundary: int,
    answer_intervals: tuple[tuple[int, int], ...],
) -> bool:
    """判断切分点能否形成两个非空 chunk 且不切断答案。"""
    if not context[:boundary].strip() or not context[boundary:].strip():
        return False
    return not any(
        answer_start < boundary < answer_end
        for answer_start, answer_end in answer_intervals
    )


def _choose_context_split_boundary(record: SquadContextRecord) -> int:
    """选择靠近 context 中点且不切断答案的稳定句子边界。

    优先使用段落或句末后的空白；如果没有可用句子边界，再退回普通空白。
    两个候选距中点相同时选择较早位置，使结果跨运行稳定。

    Raises:
        ValueError: context 无法切成两个非空且不破坏答案 span 的部分。
    """
    context = record.context
    target_boundary = len(context) / NUM_SQUAD_APPEND_CHUNKS
    answer_intervals = _answer_intervals(record)

    paragraph_boundaries = [
        match.end() for match in re.finditer(r"\n\s*\n", context)
    ]
    sentence_boundaries = [
        match.end()
        for match in re.finditer(r"""[.!?]["')\]]*\s+""", context)
    ]
    safe_boundaries = [
        boundary
        for boundary in paragraph_boundaries + sentence_boundaries
        if _is_safe_context_boundary(context, boundary, answer_intervals)
    ]

    # 极短或格式异常的 context 可能没有句末；此时只在普通空白处回退。
    if not safe_boundaries:
        safe_boundaries = [
            match.end()
            for match in re.finditer(r"\s+", context)
            if _is_safe_context_boundary(
                context,
                match.end(),
                answer_intervals,
            )
        ]
    if not safe_boundaries:
        raise ValueError(
            "SQuAD context cannot be split into two safe chunks: "
            f"squad_id={record.squad_id}."
        )

    return min(
        safe_boundaries,
        key=lambda boundary: (abs(boundary - target_boundary), boundary),
    )


def _split_squad_context(
    record: SquadContextRecord,
) -> tuple[SquadContextChunk, SquadContextChunk]:
    """把完整 SQuAD context 切成前后两个连续 Append chunk。

    两个 chunk 合起来覆盖完整官方 context，且边界不会切断官方答案。
    """
    boundary = _choose_context_split_boundary(record)
    answer_intervals = _answer_intervals(record)
    raw_segments = (
        (0, boundary, record.context[:boundary]),
        (boundary, len(record.context), record.context[boundary:]),
    )

    chunks: list[SquadContextChunk] = []
    for chunk_index, (char_start, char_end, text) in enumerate(raw_segments):
        contains_gold_answer = any(
            char_start <= answer_start and answer_end <= char_end
            for answer_start, answer_end in answer_intervals
        )
        chunks.append(
            SquadContextChunk(
                chunk_index=chunk_index,
                char_start=char_start,
                char_end=char_end,
                text=text,
                contains_gold_answer=contains_gold_answer,
            )
        )

    if len(chunks) != NUM_SQUAD_APPEND_CHUNKS:
        raise AssertionError("SQuAD split must produce exactly two chunks.")
    return (chunks[0], chunks[1])


def _format_squad_context_chunk(
    record: SquadContextRecord,
    chunk: SquadContextChunk,
) -> str:
    """把一个 SQuAD context 片段包装为独立 Append 文本。"""
    split = html.escape(record.split, quote=True)
    title = html.escape(record.title, quote=True)
    squad_id = html.escape(record.squad_id, quote=True)
    header = (
        f'<squad_context split="{split}" title="{title}" '
        f'squad_id="{squad_id}" segment="{chunk.chunk_index + 1}/'
        f'{NUM_SQUAD_APPEND_CHUNKS}">\n'
    )
    return f"{header}{chunk.text}\n</squad_context>\n"


def _parse_pipeline_pool(raw_pool: str) -> tuple[str, ...]:
    """解析 PipelinePool 的空池、单元素池和多元素池。"""
    normalized = raw_pool.strip()
    if normalized in ("", "()"):
        return ()

    # 先显式按逗号解析，避免 Python 把 (123) 当成整数。
    inner = normalized
    if normalized.startswith("(") and normalized.endswith(")"):
        inner = normalized[1:-1]
    parts = [part.strip() for part in inner.split(",") if part.strip()]
    if parts and all(re.fullmatch(r"[+-]?\d+", part) for part in parts):
        return tuple(str(int(part)) for part in parts)

    # 保留对未来 list/tuple 格式的兼容，但拒绝其它 Python 对象。
    try:
        parsed = ast.literal_eval(normalized)
    except (SyntaxError, ValueError) as exc:
        raise ValueError(f"Invalid PipelinePool value: {raw_pool!r}.") from exc
    if isinstance(parsed, int):
        return (str(parsed),)
    if not isinstance(parsed, (list, tuple)):
        raise ValueError(f"Invalid PipelinePool value: {raw_pool!r}.")
    return tuple(str(int(document_id)) for document_id in parsed)


def _load_trace_rows(anns_dir: Path, trace_file: str) -> list[AnnsTraceRow]:
    """读取一个问题的全部 ANNS Update 快照。"""
    trace_path = anns_dir / "res" / trace_file
    if not trace_path.is_file():
        raise FileNotFoundError(f"Missing ANNS trace: {trace_path}")

    rows: list[AnnsTraceRow] = []
    with trace_path.open(encoding="utf-8", newline="") as trace_stream:
        reader = csv.DictReader(trace_stream)
        required_columns = {
            "StartIteration",
            "EndIteration",
            "PipelinePool",
        }
        if reader.fieldnames is None or not required_columns.issubset(
            reader.fieldnames
        ):
            raise ValueError(f"ANNS trace has invalid columns: {trace_path}")
        for row_index, row in enumerate(reader):
            rows.append(
                AnnsTraceRow(
                    row_index=row_index,
                    start_iteration=int(row["StartIteration"]),
                    end_iteration=int(row["EndIteration"]),
                    document_ids=_parse_pipeline_pool(row["PipelinePool"]),
                )
            )
    return rows


def _find_corpus_shards(anns_dir: Path) -> list[Path]:
    """只选择两套重复 ANNS corpus shard 中的一套。"""
    shards = sorted(anns_dir.glob("retrieved_corpus_content.part.*.json"))
    if not shards:
        shards = sorted(
            anns_dir.glob("retrieved_corpus_content.[0-9]*.json")
        )
    if not shards:
        raise FileNotFoundError(f"No ANNS corpus shards found in {anns_dir}.")
    return shards


def _load_required_documents(
    anns_dir: Path,
    required_document_ids: set[str],
) -> tuple[dict[str, str], set[str]]:
    """从 corpus shard 中只保留本批 trace 实际引用的文档。

    Returns:
        已找到的文档映射，以及 corpus 中缺失的文档 id。
    """
    if not required_document_ids:
        return {}, set()

    remaining_ids = set(required_document_ids)
    documents: dict[str, str] = {}
    for shard_path in _find_corpus_shards(anns_dir):
        logger.info("Loading required documents from %s.", shard_path)
        shard = _load_json(shard_path)
        if not isinstance(shard, dict):
            raise ValueError(
                f"ANNS corpus shard is not an object: {shard_path}"
            )
        for document_id in remaining_ids.intersection(shard):
            documents[document_id] = str(shard[document_id])
        remaining_ids.difference_update(shard)
        if not remaining_ids:
            break

    if remaining_ids:
        example_ids = sorted(remaining_ids)[:10]
        logger.warning(
            "ANNS corpus is missing %d referenced document ids, for "
            "example %s. Affected queries will fail independently.",
            len(remaining_ids),
            example_ids,
        )
    return documents, remaining_ids


def _normalize_document_text(text: str) -> str:
    """压缩 ANNS 文档的行内空白，同时保留非空行边界。"""
    normalized_lines: list[str] = []
    for line in text.replace("\x00", " ").splitlines():
        normalized_line = re.sub(r"\s+", " ", line).strip()
        if normalized_line:
            normalized_lines.append(normalized_line)
    return "\n".join(normalized_lines)


def _truncate_text(text: str, max_chars: int) -> str:
    """按字符上限截断文本，并显式写入截断标记。"""
    if max_chars <= 0 or len(text) <= max_chars:
        return text
    marker = "\n[truncated]"
    content_length = max(0, max_chars - len(marker))
    return f"{text[:content_length].rstrip()}{marker}"


def _load_tokenizer(tokenizer_model: str | None) -> Any | None:
    """按需加载 tokenizer，只用于 ANNS 内容预算。"""
    if tokenizer_model is None:
        return None

    # 未启用 token 预算时生成器只依赖 Python 标准库。
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(
        tokenizer_model,
        trust_remote_code=True,
        local_files_only=True,
    )


def _encode_text(tokenizer: Any, text: str) -> list[int]:
    """计算不含 special token 的文本 token ids。"""
    return list(tokenizer.encode(text, add_special_tokens=False))


def _decode_token_ids(tokenizer: Any, token_ids: list[int]) -> str:
    """把截断后的 token ids 稳定还原为文本。"""
    return str(
        tokenizer.decode(
            token_ids,
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )
    )


def _truncate_text_by_tokens(
    text: str,
    max_tokens: int,
    tokenizer: Any | None,
) -> str:
    """按 tokenizer token 上限裁剪文本并保留截断标记。"""
    if max_tokens <= 0:
        return text
    if tokenizer is None:
        raise ValueError("--tokenizer-model is required for token budgets.")

    token_ids = _encode_text(tokenizer, text)
    if len(token_ids) <= max_tokens:
        return text

    marker = "\n[truncated]"
    marker_ids = _encode_text(tokenizer, marker)
    if len(marker_ids) >= max_tokens:
        return _decode_token_ids(
            tokenizer,
            marker_ids[:max_tokens],
        ).rstrip()
    content_budget = max_tokens - len(marker_ids)
    truncated_ids = token_ids[:content_budget] + marker_ids
    return _decode_token_ids(tokenizer, truncated_ids).rstrip()


def _truncate_wrapped_body_by_tokens(
    header: str,
    body: str,
    footer: str,
    max_tokens: int,
    tokenizer: Any | None,
) -> str:
    """在固定包装文本内裁剪正文，使整个文档满足 token 预算。"""
    if max_tokens <= 0:
        return f"{header}{body}{footer}"
    if tokenizer is None:
        raise ValueError("--tokenizer-model is required for token budgets.")

    wrapper_tokens = len(_encode_text(tokenizer, f"{header}{footer}"))
    body_budget = max(0, max_tokens - wrapper_tokens)
    truncated_body = (
        _truncate_text_by_tokens(body, body_budget, tokenizer)
        if body_budget
        else ""
    )
    return f"{header}{truncated_body}{footer}"


def _limit_joined_text_by_tokens(
    parts: list[str],
    separator: str,
    max_tokens: int,
    tokenizer: Any | None,
) -> str:
    """按 token 预算拼接 ANNS 文档，优先保留完整前序文档。"""
    if not parts:
        return ""
    if max_tokens <= 0:
        return separator.join(parts)
    if tokenizer is None:
        raise ValueError("--tokenizer-model is required for token budgets.")

    selected_parts: list[str] = []
    used_tokens = 0
    separator_tokens = len(_encode_text(tokenizer, separator))
    for part in parts:
        part_tokens = len(_encode_text(tokenizer, part))
        extra_separator_tokens = separator_tokens if selected_parts else 0
        required_tokens = extra_separator_tokens + part_tokens
        if used_tokens + required_tokens <= max_tokens:
            selected_parts.append(part)
            used_tokens += required_tokens
            continue

        remaining_tokens = max_tokens - used_tokens - extra_separator_tokens
        if remaining_tokens > 0:
            selected_parts.append(
                _truncate_text_by_tokens(
                    part,
                    remaining_tokens,
                    tokenizer,
                )
            )
        break

    if not selected_parts:
        return _truncate_text_by_tokens(parts[0], max_tokens, tokenizer)
    return separator.join(selected_parts)


def _format_anns_document(
    document_id: str,
    text: str,
    max_document_chars: int,
    max_document_tokens: int,
    tokenizer: Any | None,
) -> str:
    """把单个 ANNS 文档格式化为跨 Update 版本稳定的片段。"""
    normalized_text = _truncate_text(
        _normalize_document_text(text),
        max_document_chars,
    )
    header = f'<retrieved_document id="{document_id}">\n'
    footer = "\n</retrieved_document>"
    return _truncate_wrapped_body_by_tokens(
        header=header,
        body=normalized_text,
        footer=footer,
        max_tokens=max_document_tokens,
        tokenizer=tokenizer,
    )


def _materialize_anns_snapshot(
    row: AnnsTraceRow,
    documents: dict[str, str],
    max_document_chars: int,
    max_document_tokens: int,
    max_snapshot_tokens: int,
    tokenizer: Any | None,
) -> str:
    """按 PipelinePool 原顺序物化一份完整 Update 快照。

    空 PipelinePool 返回空字符串，表示合法的清空 Update source 事件。
    """
    formatted_documents = [
        _format_anns_document(
            document_id,
            documents[document_id],
            max_document_chars,
            max_document_tokens,
            tokenizer,
        )
        for document_id in row.document_ids
    ]
    return _limit_joined_text_by_tokens(
        parts=formatted_documents,
        separator="\n\n",
        max_tokens=max_snapshot_tokens,
        tokenizer=tokenizer,
    )


def _build_source_finished_event(
    source_id: str,
    source_type: SourceType,
    sequence_no: int,
    version: int,
) -> dict[str, Any]:
    """构造与 prompt-stream RESTFUL 协议同构的来源 barrier。"""
    return {
        "event": {
            "source_id": source_id,
            "event_type": "source_finished",
            "source_type": source_type,
            "chunk_id": f"{source_id}-finished",
            "version": version,
            "sequence_no": sequence_no,
            "text": None,
        },
        "provenance": {
            "kind": "source_barrier",
        },
    }


def _build_sample(
    query_id: str,
    question: str,
    trace_file: str,
    trace_rows: list[AnnsTraceRow],
    documents: dict[str, str],
    squad_match: SquadContextMatch,
    args: argparse.Namespace,
    tokenizer: Any | None,
) -> dict[str, Any]:
    """构造一个 SQuAD Append + ANNS Update 混合来源请求。

    events[].event 可以直接作为 /v1/prompt_stream/event 的内层 event。
    列表按来源分组只为稳定落盘；跨来源到达顺序仍由 replay 决定。
    """
    sample_id = (
        f"anns-query-{int(query_id):06d}"
        if query_id.isdigit()
        else f"anns-query-{query_id}"
    )
    squad_source_id = f"squad:{query_id}"
    anns_source_id = f"anns:{query_id}"
    squad_record = squad_match.record
    squad_chunks = _split_squad_context(squad_record)
    context_sha256 = hashlib.sha256(
        squad_record.context.encode("utf-8")
    ).hexdigest()

    squad_events: list[dict[str, Any]] = []
    for chunk in squad_chunks:
        squad_events.append(
            {
                "event": {
                    "source_id": squad_source_id,
                    "event_type": "append",
                    "source_type": SQUAD_SOURCE_TYPE,
                    "chunk_id": (
                        f"{squad_source_id}-context-{chunk.chunk_index}"
                    ),
                    "version": chunk.chunk_index,
                    "sequence_no": chunk.chunk_index,
                    "text": _format_squad_context_chunk(
                        squad_record,
                        chunk,
                    ),
                },
                "provenance": {
                    "kind": "squad_context_segment",
                    "split": squad_record.split,
                    "title": squad_record.title,
                    "paragraph_index": squad_record.paragraph_index,
                    "squad_id": squad_record.squad_id,
                    "match_method": squad_match.method,
                    "context_sha256": context_sha256,
                    "segment_index": chunk.chunk_index,
                    "num_segments": NUM_SQUAD_APPEND_CHUNKS,
                    "char_start": chunk.char_start,
                    "char_end": chunk.char_end,
                    "contains_gold_answer": (
                        chunk.contains_gold_answer
                    ),
                },
            }
        )
    squad_events.append(
        _build_source_finished_event(
            source_id=squad_source_id,
            source_type=SQUAD_SOURCE_TYPE,
            sequence_no=NUM_SQUAD_APPEND_CHUNKS,
            version=NUM_SQUAD_APPEND_CHUNKS - 1,
        )
    )

    anns_events: list[dict[str, Any]] = []
    for sequence_no, row in enumerate(trace_rows):
        snapshot_text = _materialize_anns_snapshot(
            row,
            documents,
            args.max_anns_document_chars,
            args.max_anns_document_tokens,
            args.max_update_snapshot_tokens,
            tokenizer,
        )
        anns_events.append(
            {
                "event": {
                    "source_id": anns_source_id,
                    "event_type": "update",
                    "source_type": ANNS_SOURCE_TYPE,
                    "chunk_id": (
                        f"{anns_source_id}-snapshot-{sequence_no}"
                    ),
                    "version": sequence_no,
                    "sequence_no": sequence_no,
                    "text": snapshot_text,
                },
                "provenance": {
                    "kind": "anns_snapshot",
                    "trace_file": trace_file,
                    "trace_row": row.row_index,
                    "start_iteration": row.start_iteration,
                    "end_iteration": row.end_iteration,
                    "document_ids": list(row.document_ids),
                },
            }
        )
    anns_events.append(
        _build_source_finished_event(
            source_id=anns_source_id,
            source_type=ANNS_SOURCE_TYPE,
            sequence_no=len(trace_rows),
            version=max(0, len(trace_rows) - 1),
        )
    )

    gold_answers = list(
        dict.fromkeys(answer.text for answer in squad_record.answers)
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "data_revision": DATA_REVISION,
        "sample_id": sample_id,
        "query_id": query_id,
        "question": question,
        "status": "complete",
        "start": {
            "api": "completions",
            "prompt": args.prefix,
            "stream": False,
            "max_tokens": args.max_tokens,
            "temperature": args.temperature,
        },
        "sources": [
            {
                "source_id": squad_source_id,
                "source_type": SQUAD_SOURCE_TYPE,
                "semantics": "monotonic_append",
            },
            {
                "source_id": anns_source_id,
                "source_type": ANNS_SOURCE_TYPE,
                "semantics": "full_snapshot",
            },
        ],
        "event_order": {
            "storage_layout": "grouped_by_source",
            "within_source": "sequence_no",
            "across_sources": "unspecified",
        },
        "events": squad_events + anns_events,
        "finish": {
            "after_source_ids": [squad_source_id, anns_source_id],
            "event": {
                "source_id": "final_question",
                "event_type": "prompt_finished",
                "source_type": None,
                "chunk_id": "final-question",
                "version": 0,
                "sequence_no": 0,
                "text": f"\nUser question: {question}\nAnswer:",
            },
        },
        "squad_context": {
            "split": squad_record.split,
            "title": squad_record.title,
            "paragraph_index": squad_record.paragraph_index,
            "squad_id": squad_record.squad_id,
            "match_method": squad_match.method,
            "is_impossible": squad_record.is_impossible,
            "gold_answers": gold_answers,
            "context_sha256": context_sha256,
            "num_append_chunks": NUM_SQUAD_APPEND_CHUNKS,
            "context_chars": len(squad_record.context),
        },
        "provenance": {
            "base_dataset": "stream2llm-data/anns",
            "base_query_id": query_id,
            "trace_file": trace_file,
            "append_dataset": "squad_v2",
        },
    }


def _validate_no_timing_fields(value: Any, path: str = "$") -> None:
    """递归拒绝数据构建或旧 schema 遗留的真实时间字段。"""
    forbidden_keys = {
        "ready_offset_ms",
        "fetch_duration_ms",
        "search_duration_ms",
        "collection_duration_ms",
        "start_time_us",
        "end_time_us",
        "collected_at",
        "generated_at",
        "updated_at",
    }
    if isinstance(value, dict):
        for key, child in value.items():
            if key in forbidden_keys:
                raise ValueError(f"Timing field {path}.{key} is not allowed.")
            _validate_no_timing_fields(child, f"{path}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _validate_no_timing_fields(child, f"{path}[{index}]")


def _validate_sample(sample: dict[str, Any]) -> None:
    """检查现有回放器依赖的 schema、来源和顺序约束。"""
    if sample.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("Unexpected mixed-source schema version.")
    if sample.get("data_revision") != DATA_REVISION:
        raise ValueError("Unexpected mixed-source data revision.")
    if not sample.get("start", {}).get("prompt"):
        raise ValueError("Sample start.prompt must be non-empty.")
    if not sample.get("finish", {}).get("event", {}).get("text"):
        raise ValueError("Sample finish question must be non-empty.")
    if sample.get("event_order", {}).get(
        "across_sources"
    ) != "unspecified":
        raise ValueError("Cross-source event order must remain unspecified.")
    _validate_no_timing_fields(sample)

    source_definitions = sample.get("sources")
    if not isinstance(source_definitions, list) or len(
        source_definitions
    ) != 2:
        raise ValueError("Sample must declare one Append and one Update source.")
    source_types = {
        source.get("source_type")
        for source in source_definitions
        if isinstance(source, dict)
    }
    if source_types != {SQUAD_SOURCE_TYPE, ANNS_SOURCE_TYPE}:
        raise ValueError(f"Unexpected source types: {source_types}.")

    seen_chunk_ids: set[str] = set()
    source_events: dict[str, list[dict[str, Any]]] = defaultdict(list)
    num_append_events = 0
    for record in sample.get("events", []):
        event = record.get("event") if isinstance(record, dict) else None
        if not isinstance(event, dict):
            raise ValueError("Every event record must contain an event object.")
        source_id = event.get("source_id")
        chunk_id = event.get("chunk_id")
        if not isinstance(source_id, str) or not isinstance(chunk_id, str):
            raise ValueError("Every source event requires string ids.")
        if chunk_id in seen_chunk_ids:
            raise ValueError(f"Duplicate chunk_id={chunk_id}.")
        seen_chunk_ids.add(chunk_id)
        source_events[source_id].append(event)

        event_type = event.get("event_type")
        if event_type == "append":
            num_append_events += 1
            if not event.get("text"):
                raise ValueError(
                    f"Append chunk {chunk_id} must be non-empty."
                )
        # 空字符串是合法的清空 Update snapshot，只有 None 才是坏数据。
        if event_type == "update" and event.get("text") is None:
            raise ValueError(
                f"Update snapshot {chunk_id} must carry text."
            )

    if num_append_events != NUM_SQUAD_APPEND_CHUNKS:
        raise ValueError(
            "SQuAD Append source must contain exactly "
            f"{NUM_SQUAD_APPEND_CHUNKS} data events."
        )

    declared_source_ids = {
        source["source_id"] for source in source_definitions
    }
    if set(source_events) != declared_source_ids:
        raise ValueError("Declared sources and event source ids do not match.")
    for source_id, events in source_events.items():
        sequences = [int(event["sequence_no"]) for event in events]
        if sequences != list(range(len(events))):
            raise ValueError(
                f"Non-contiguous sequence numbers for {source_id}: "
                f"{sequences}."
            )
        if events[-1].get("event_type") != "source_finished":
            raise ValueError(
                f"Source {source_id} must end with source_finished."
            )
        versions = [
            int(event["version"])
            for event in events
            if event.get("event_type") in ("append", "update")
        ]
        if versions != list(range(len(versions))):
            raise ValueError(
                f"Non-contiguous data versions for {source_id}: {versions}."
            )

    finish_source_ids = set(
        sample["finish"].get("after_source_ids", [])
    )
    if finish_source_ids != declared_source_ids:
        raise ValueError(
            "finish.after_source_ids must contain both sources."
        )


def _request_path(output_dir: Path, query_id: str) -> Path:
    """返回一个 query 的稳定请求文件路径。"""
    filename = (
        f"anns-query-{int(query_id):06d}.json"
        if query_id.isdigit()
        else f"anns-query-{query_id}.json"
    )
    return output_dir / "requests" / filename


def _is_current_squad_sample(sample: Any) -> bool:
    """判断已有请求是否已经由当前 SQuAD Append 生成器构造。"""
    return (
        isinstance(sample, dict)
        and sample.get("schema_version") == SCHEMA_VERSION
        and sample.get("data_revision") == DATA_REVISION
        and sample.get("provenance", {}).get("append_dataset") == "squad_v2"
    )


def _rebuild_manifest(output_dir: Path) -> tuple[int, Counter[str]]:
    """从全部当前版本请求确定性重建 manifest.jsonl。

    Returns:
        manifest 样本数及全部样本的 SQuAD 匹配方式计数。
    """
    entries: list[dict[str, Any]] = []
    legacy_paths: list[Path] = []
    match_method_counts: Counter[str] = Counter()
    request_directory = output_dir / "requests"
    request_directory.mkdir(parents=True, exist_ok=True)

    for request_path in sorted(request_directory.glob("*.json")):
        sample = _load_json(request_path)
        if not _is_current_squad_sample(sample):
            legacy_paths.append(request_path)
            continue
        _validate_sample(sample)
        match_method = sample["squad_context"]["match_method"]
        match_method_counts[match_method] += 1
        num_append_events = sum(
            record["event"]["event_type"] == "append"
            for record in sample["events"]
        )
        num_update_events = sum(
            record["event"]["event_type"] == "update"
            for record in sample["events"]
        )
        entries.append(
            {
                "sample_id": sample["sample_id"],
                "query_id": sample["query_id"],
                "path": request_path.relative_to(output_dir).as_posix(),
                "status": sample["status"],
                "question": sample["question"],
                "num_append_events": num_append_events,
                "num_update_events": num_update_events,
                "squad_id": sample["squad_context"]["squad_id"],
                "squad_title": sample["squad_context"]["title"],
            }
        )

    if legacy_paths:
        logger.warning(
            "Ignoring %d legacy non-SQuAD request files while rebuilding "
            "manifest; run the full --overwrite command before replay.",
            len(legacy_paths),
        )
    entries.sort(key=lambda entry: _numeric_query_sort_key(entry["query_id"]))
    manifest_text = "".join(
        f"{json.dumps(entry, ensure_ascii=False)}\n" for entry in entries
    )
    _atomic_write_text(output_dir / "manifest.jsonl", manifest_text)
    return len(entries), match_method_counts


def _write_dataset_metadata(
    output_dir: Path,
    args: argparse.Namespace,
    num_samples: int,
    num_failed: int,
    match_method_counts: Counter[str],
) -> None:
    """写入 SQuAD Append 数据集级元信息。"""
    metadata = {
        "schema_version": SCHEMA_VERSION,
        "data_revision": DATA_REVISION,
        "description": (
            "Mixed prompt-stream dataset using two deterministic segments "
            "of the matched SQuAD v2 context as Append events and ANNS "
            "PipelinePool snapshots as Update events."
        ),
        "num_samples": num_samples,
        "num_failed_in_last_build": num_failed,
        "squad_match_method_counts": dict(
            sorted(match_method_counts.items())
        ),
        "event_order": {
            "within_source": "sequence_no",
            "across_sources": "controlled_by_replay",
        },
        "layout": {
            "manifest": "manifest.jsonl",
            "requests": "requests/*.json",
        },
        "append_source": {
            "dataset": "squad_v2",
            "num_chunks_per_request": NUM_SQUAD_APPEND_CHUNKS,
            "split_strategy": "balanced_sentence_boundary",
            "preserve_complete_context": True,
            "preserve_answer_spans": True,
        },
        "update_source": {
            "dataset": "stream2llm-data/anns",
            "semantics": "full_snapshot",
        },
        "generation_defaults": {
            "max_anns_document_chars": args.max_anns_document_chars,
            "max_anns_document_tokens": args.max_anns_document_tokens,
            "max_update_snapshot_tokens": args.max_update_snapshot_tokens,
            "tokenizer_model": args.tokenizer_model,
            "max_tokens": args.max_tokens,
            "temperature": args.temperature,
        },
    }
    _atomic_write_json(output_dir / "dataset.json", metadata)


def _validate_args(args: argparse.Namespace) -> None:
    """在读取大文件前校验参数范围与依赖关系。"""
    if args.max_tokens <= 0:
        raise ValueError("--max-tokens must be positive.")
    if args.temperature < 0.0:
        raise ValueError("--temperature must be non-negative.")
    non_negative_fields = (
        "max_anns_document_chars",
        "max_anns_document_tokens",
        "max_update_snapshot_tokens",
    )
    for field_name in non_negative_fields:
        if getattr(args, field_name) < 0:
            option_name = field_name.replace("_", "-")
            raise ValueError(f"--{option_name} must be non-negative.")

    token_budget_fields = (
        "max_anns_document_tokens",
        "max_update_snapshot_tokens",
    )
    if (
        any(getattr(args, name) > 0 for name in token_budget_fields)
        and args.tokenizer_model is None
    ):
        raise ValueError(
            "--tokenizer-model is required when token-level ANNS budgets "
            "are enabled."
        )
    if not args.prefix:
        raise ValueError("--prefix must be non-empty.")


def run(args: argparse.Namespace) -> None:
    """执行一次确定性、可恢复的混合来源数据生成。

    该函数会写入选中请求、重建只包含当前 SQuAD 版本请求的 manifest，并更新
    dataset 元信息。它不会访问网络，也不会删除旧网页缓存。
    """
    _validate_args(args)
    tokenizer = _load_tokenizer(args.tokenizer_model)

    query_map_path = args.anns_dir / "query_trace_map_5k.json"
    query_map = _load_json(query_map_path)
    if not isinstance(query_map, dict):
        raise ValueError(
            f"ANNS query map is not an object: {query_map_path}"
        )
    query_ids = _select_query_ids(query_map, args)
    if not query_ids:
        raise ValueError("No ANNS queries selected.")

    target_questions = {
        str(query_map[query_id]["query"]) for query_id in query_ids
    }
    squad_index = _load_squad_context_index(
        args.squad_dir,
        target_questions,
    )
    logger.info(
        "Indexed %d candidate SQuAD records for %d selected ANNS queries.",
        squad_index.num_indexed_records,
        len(query_ids),
    )

    # 当前版本请求默认可续跑；旧网页请求必须自动进入重建集合。
    query_ids_to_build: list[str] = []
    num_skipped = 0
    for query_id in query_ids:
        request_path = _request_path(args.output_dir, query_id)
        if request_path.exists() and not args.overwrite:
            existing_sample = _load_json(request_path)
            if _is_current_squad_sample(existing_sample):
                logger.info(
                    "Skipping current SQuAD request %s; use --overwrite "
                    "to rebuild it.",
                    request_path,
                )
                num_skipped += 1
                continue
            logger.info("Rebuilding legacy web request %s.", request_path)
        query_ids_to_build.append(query_id)

    # 只为确实需要重建的 query 解析 trace 和加载 corpus。
    traces: dict[str, list[AnnsTraceRow]] = {}
    required_document_ids: set[str] = set()
    num_failed = 0
    for query_id in query_ids_to_build:
        trace_file = str(query_map[query_id]["trace_file"])
        try:
            trace_rows = _load_trace_rows(args.anns_dir, trace_file)
        except Exception:
            num_failed += 1
            logger.exception(
                "Failed to load ANNS trace for query %s.",
                query_id,
            )
            if args.fail_fast:
                raise
            continue
        traces[query_id] = trace_rows
        for row in trace_rows:
            required_document_ids.update(row.document_ids)
    documents, _ = _load_required_documents(
        args.anns_dir,
        required_document_ids,
    )

    num_succeeded = num_skipped
    for query_index, query_id in enumerate(query_ids_to_build):
        trace_rows = traces.get(query_id)
        if trace_rows is None:
            continue
        question = str(query_map[query_id]["query"])
        trace_file = str(query_map[query_id]["trace_file"])
        request_path = _request_path(args.output_dir, query_id)
        logger.info(
            "Building query %s (%d/%d): %s",
            query_id,
            query_index + 1,
            len(query_ids_to_build),
            question,
        )
        try:
            squad_match = _resolve_squad_context(question, squad_index)
            sample = _build_sample(
                query_id=query_id,
                question=question,
                trace_file=trace_file,
                trace_rows=trace_rows,
                documents=documents,
                squad_match=squad_match,
                args=args,
                tokenizer=tokenizer,
            )
            _validate_sample(sample)
            _atomic_write_json(request_path, sample)
            num_succeeded += 1
        except Exception:
            num_failed += 1
            logger.exception("Failed to build query %s.", query_id)
            if args.fail_fast:
                raise

    (
        num_manifest_samples,
        manifest_match_method_counts,
    ) = _rebuild_manifest(args.output_dir)
    _write_dataset_metadata(
        output_dir=args.output_dir,
        args=args,
        num_samples=num_manifest_samples,
        num_failed=num_failed,
        match_method_counts=manifest_match_method_counts,
    )
    logger.info(
        "Build finished: selected=%d, built_or_skipped=%d, failed=%d, "
        "manifest_samples=%d.",
        len(query_ids),
        num_succeeded,
        num_failed,
        num_manifest_samples,
    )

    if num_failed:
        raise RuntimeError(
            f"Failed to build {num_failed} of "
            f"{len(query_ids)} selected queries."
        )


def main() -> None:
    """配置日志并运行数据生成入口。"""
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
    run(args)


if __name__ == "__main__":
    main()
