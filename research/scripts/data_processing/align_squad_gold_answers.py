# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""把混合来源请求中的 ANNS 问题与 SQuAD v2 标准答案对齐。

脚本不会改写 ``mix_source/requests``。它会读取每个请求的 ``query_id``、
``question`` 和构建阶段冻结的 ``squad_context.squad_id``，先用 ANNS query
map 校验问题来源，再从 SQuAD v2 train/dev 中恢复实际 Append context 对应的
标准答案，最终写出一份请求级 JSONL manifest 和一份覆盖率摘要。
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

logger = logging.getLogger(__name__)

SCHEMA_VERSION = "squad_gold_alignment/v1"
DEFAULT_SQUAD_DIR = Path("research/data/squad_v2")
DEFAULT_ANNS_QUERY_MAP = Path("research/data/anns/query_trace_map_5k.json")
DEFAULT_REQUESTS_DIR = Path("research/data/mix_source/requests")
DEFAULT_OUTPUT = Path(
    "research/data/mix_source/quality_eval/squad_gold_answers.jsonl"
)
DEFAULT_SUMMARY_OUTPUT = Path(
    "research/data/mix_source/quality_eval/squad_alignment_summary.json"
)
SQUAD_SPLIT_FILES = {
    "train": "train-v2.0.json",
    "dev": "dev-v2.0.json",
}

MatchMethod = Literal["exact", "normalized"]
MatchStatus = Literal[
    "matched",
    "matched_equivalent",
    "ambiguous",
    "unmatched",
]
SquadIdValidation = Literal["matched", "missing", "mismatch"]


@dataclass(frozen=True)
class SquadAnswer:
    """SQuAD 中的一条标准答案标注。"""

    text: str
    answer_start: int


@dataclass(frozen=True)
class SquadRecord:
    """一条 SQuAD 问题及其标准答案。"""

    split: str
    squad_id: str
    title: str
    question: str
    is_impossible: bool
    answers: tuple[SquadAnswer, ...]


@dataclass(frozen=True)
class SquadIndex:
    """同时支持精确问题和归一化问题查找的 SQuAD 索引。"""

    exact_question_to_records: dict[str, tuple[SquadRecord, ...]]
    normalized_question_to_records: dict[str, tuple[SquadRecord, ...]]
    num_records: int


@dataclass(frozen=True)
class MatchResult:
    """一个请求问题与 SQuAD 的对齐结果。"""

    status: MatchStatus
    method: MatchMethod | None
    records: tuple[SquadRecord, ...]


def parse_args() -> argparse.Namespace:
    """解析 SQuAD 标准答案对齐参数。

    Returns:
        命令行参数命名空间。
    """
    parser = argparse.ArgumentParser(
        description=(
            "Align mixed-source ANNS questions with SQuAD v2 gold answers."
        )
    )
    parser.add_argument(
        "--squad-dir",
        type=Path,
        default=DEFAULT_SQUAD_DIR,
        help="Directory containing train-v2.0.json and dev-v2.0.json.",
    )
    parser.add_argument(
        "--anns-query-map",
        type=Path,
        default=DEFAULT_ANNS_QUERY_MAP,
        help="Path to ANNS query_trace_map_5k.json.",
    )
    parser.add_argument(
        "--requests-dir",
        type=Path,
        default=DEFAULT_REQUESTS_DIR,
        help="Directory containing mixed-source request JSON files.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Output request-level JSONL manifest.",
    )
    parser.add_argument(
        "--summary-output",
        type=Path,
        default=DEFAULT_SUMMARY_OUTPUT,
        help="Output aggregate alignment summary JSON.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help=(
            "Return a non-zero exit code when any request is unmatched, "
            "ambiguous, or inconsistent with the ANNS query map."
        ),
    )
    parser.add_argument(
        "--log-level",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
        default="INFO",
        help="Logging level; defaults to INFO.",
    )
    return parser.parse_args()


def normalize_question(question: str) -> str:
    """把问题转换成用于安全回退匹配的稳定键。

    归一化会消除大小写、重音、空白和标点差异，但不会修改原始输出问题。只有当
    归一化键在 SQuAD 中唯一或所有候选答案等价时，脚本才接受该匹配。

    Args:
        question: 原始问题文本。

    Returns:
        仅包含小写字母、数字和单空格的匹配键。
    """
    # NFKD 把重音拆成组合字符，随后删除组合字符以兼容 Beyonce/Beyoncé。
    decomposed = unicodedata.normalize("NFKD", question).casefold()
    without_marks = "".join(
        character
        for character in decomposed
        if not unicodedata.combining(character)
    )
    # 标点只承担自然语言格式作用；统一替换为空格，避免问号和引号差异阻断匹配。
    alphanumeric_only = re.sub(r"[^\w]+", " ", without_marks, flags=re.UNICODE)
    return " ".join(alphanumeric_only.split())


def _load_json_object(path: Path, *, description: str) -> dict[str, Any]:
    """读取并校验一个 JSON object 文件。

    Args:
        path: 输入 JSON 文件路径。
        description: 用于异常信息的文件用途说明。

    Returns:
        解析后的顶层字典。

    Raises:
        FileNotFoundError: 输入文件不存在。
        ValueError: 顶层 JSON 不是 object。
    """
    if not path.is_file():
        raise FileNotFoundError(f"Missing {description}: {path}")
    with path.open("r", encoding="utf-8") as input_file:
        payload = json.load(input_file)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object in {description}: {path}")
    return payload


def _load_squad_records(path: Path, split: str) -> list[SquadRecord]:
    """读取一个 SQuAD v2 split 并展开为问题记录。

    Args:
        path: SQuAD v2 train 或 dev JSON 路径。
        split: 写入 provenance 的 split 名称。

    Returns:
        按原始文件顺序展开的问题记录。

    Raises:
        ValueError: 文件 schema 缺少必要字段或字段类型错误。
    """
    payload = _load_json_object(path, description=f"SQuAD {split} split")
    articles = payload.get("data")
    if not isinstance(articles, list):
        raise ValueError(f"SQuAD split has no list-valued data field: {path}")

    records: list[SquadRecord] = []
    # SQuAD 按 article -> paragraph -> question 三层嵌套存储，逐层展开便于索引。
    for article in articles:
        if not isinstance(article, dict):
            raise ValueError(f"Invalid SQuAD article in {path}")
        title = article.get("title")
        paragraphs = article.get("paragraphs")
        if not isinstance(title, str) or not isinstance(paragraphs, list):
            raise ValueError(f"Invalid SQuAD article fields in {path}")
        for paragraph in paragraphs:
            if not isinstance(paragraph, dict):
                raise ValueError(f"Invalid SQuAD paragraph in {path}")
            questions = paragraph.get("qas")
            if not isinstance(questions, list):
                raise ValueError(f"SQuAD paragraph has no qas list in {path}")
            for question_payload in questions:
                if not isinstance(question_payload, dict):
                    raise ValueError(f"Invalid SQuAD question in {path}")
                squad_id = question_payload.get("id")
                question = question_payload.get("question")
                is_impossible = question_payload.get("is_impossible", False)
                raw_answers = question_payload.get("answers")
                if (
                    not isinstance(squad_id, str)
                    or not isinstance(question, str)
                    or not isinstance(is_impossible, bool)
                    or not isinstance(raw_answers, list)
                ):
                    raise ValueError(f"Invalid SQuAD question fields in {path}")

                answers: list[SquadAnswer] = []
                for raw_answer in raw_answers:
                    if not isinstance(raw_answer, dict):
                        raise ValueError(f"Invalid SQuAD answer in {path}")
                    answer_text = raw_answer.get("text")
                    answer_start = raw_answer.get("answer_start")
                    if not isinstance(answer_text, str) or not isinstance(
                        answer_start, int
                    ):
                        raise ValueError(f"Invalid SQuAD answer fields in {path}")
                    answers.append(
                        SquadAnswer(
                            text=answer_text,
                            answer_start=answer_start,
                        )
                    )
                records.append(
                    SquadRecord(
                        split=split,
                        squad_id=squad_id,
                        title=title,
                        question=question,
                        is_impossible=is_impossible,
                        answers=tuple(answers),
                    )
                )
    return records


def load_squad_index(squad_dir: Path) -> SquadIndex:
    """加载 SQuAD v2 train/dev 并构造两级问题索引。

    Args:
        squad_dir: 包含两个 SQuAD v2 原始 JSON 的目录。

    Returns:
        支持精确和归一化查找的不可变索引。
    """
    records: list[SquadRecord] = []
    for split, file_name in SQUAD_SPLIT_FILES.items():
        records.extend(_load_squad_records(squad_dir / file_name, split))

    exact_index: defaultdict[str, list[SquadRecord]] = defaultdict(list)
    normalized_index: defaultdict[str, list[SquadRecord]] = defaultdict(list)
    for record in records:
        exact_index[record.question].append(record)
        normalized_index[normalize_question(record.question)].append(record)

    return SquadIndex(
        exact_question_to_records={
            question: tuple(question_records)
            for question, question_records in exact_index.items()
        },
        normalized_question_to_records={
            question: tuple(question_records)
            for question, question_records in normalized_index.items()
        },
        num_records=len(records),
    )


def _record_answer_signature(record: SquadRecord) -> tuple[bool, tuple[str, ...]]:
    """生成用于判断重复问题候选是否答案等价的签名。"""
    normalized_answers = tuple(
        sorted({normalize_question(answer.text) for answer in record.answers})
    )
    return record.is_impossible, normalized_answers


def match_squad_question(question: str, squad_index: SquadIndex) -> MatchResult:
    """按精确优先、归一化回退的顺序匹配一个问题。

    Args:
        question: 混合来源请求中的原始问题。
        squad_index: 完整 SQuAD v2 问题索引。

    Returns:
        匹配状态、方法和候选记录。多个候选只有在答案签名完全相同时才视为
        ``matched_equivalent``；冲突候选会明确标记为 ``ambiguous``。
    """
    exact_records = squad_index.exact_question_to_records.get(question, ())
    method: MatchMethod = "exact"
    records = exact_records
    if not records:
        method = "normalized"
        normalized_question = normalize_question(question)
        records = squad_index.normalized_question_to_records.get(
            normalized_question, ()
        )
    if not records:
        return MatchResult(status="unmatched", method=None, records=())
    if len(records) == 1:
        return MatchResult(status="matched", method=method, records=records)

    answer_signatures = {_record_answer_signature(record) for record in records}
    if len(answer_signatures) == 1:
        return MatchResult(
            status="matched_equivalent",
            method=method,
            records=records,
        )
    return MatchResult(status="ambiguous", method=method, records=records)


def _resolve_request_squad_id(
    request_payload: dict[str, Any],
    match_result: MatchResult,
) -> tuple[MatchResult, str | None, SquadIdValidation]:
    """用请求冻结的 SQuAD id 选择实际提供给模型的 context 标注。

    同一个问题可能在 SQuAD 中出现多次，而且答案粒度可能不同。数据构建器
    已经在请求中保存实际 Append context 的 ``squad_id``，质量评测必须优先
    使用该记录，不能重新按问题文本任意聚合候选答案。

    Args:
        request_payload: 当前混合来源请求 JSON。
        match_result: 按问题文本得到的初始 SQuAD 匹配结果。

    Returns:
        解析后的匹配结果、请求声明的 ``squad_id``，以及该 id 的校验状态。
        旧请求没有 ``squad_id`` 时保留原匹配行为并返回 ``missing``。
    """
    squad_context = request_payload.get("squad_context")
    if not isinstance(squad_context, dict):
        return match_result, None, "missing"

    request_squad_id = squad_context.get("squad_id")
    if not isinstance(request_squad_id, str):
        return match_result, None, "missing"

    selected_records = tuple(
        record
        for record in match_result.records
        if record.squad_id == request_squad_id
    )
    if len(selected_records) != 1:
        return match_result, request_squad_id, "mismatch"

    # squad_id 唯一确定了实际 Append context，冲突候选不再参与 gold 聚合。
    resolved_result = MatchResult(
        status="matched",
        method=match_result.method,
        records=selected_records,
    )
    return resolved_result, request_squad_id, "matched"


def _load_anns_query_map(path: Path) -> dict[str, str]:
    """读取 ANNS query id 到原始问题的映射。"""
    payload = _load_json_object(path, description="ANNS query map")
    query_id_to_question: dict[str, str] = {}
    for query_id, query_payload in payload.items():
        if not isinstance(query_id, str) or not isinstance(query_payload, dict):
            raise ValueError(f"Invalid ANNS query entry in {path}")
        question = query_payload.get("query")
        if not isinstance(question, str):
            raise ValueError(f"ANNS query {query_id} has no string query")
        query_id_to_question[query_id] = question
    return query_id_to_question


def _validate_anns_question(
    query_id: str,
    question: str,
    query_id_to_question: dict[str, str],
) -> str:
    """校验请求问题是否仍与 ANNS query map 一致。"""
    anns_question = query_id_to_question.get(query_id)
    if anns_question is None:
        return "missing_query_id"
    if question == anns_question:
        return "exact"
    if normalize_question(question) == normalize_question(anns_question):
        return "normalized"
    return "mismatch"


def _deduplicate_strings(values: list[str]) -> list[str]:
    """保持首次出现顺序地去重字符串。"""
    seen: set[str] = set()
    deduplicated: list[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        deduplicated.append(value)
    return deduplicated


def _serialize_candidates(records: tuple[SquadRecord, ...]) -> list[dict[str, Any]]:
    """把 SQuAD 候选压缩成适合审计的 JSON 结构。"""
    candidates: list[dict[str, Any]] = []
    for record in records:
        candidates.append(
            {
                "squad_id": record.squad_id,
                "split": record.split,
                "title": record.title,
                "question": record.question,
                "is_impossible": record.is_impossible,
                "answers": [
                    {
                        "text": answer.text,
                        "answer_start": answer.answer_start,
                    }
                    for answer in record.answers
                ],
            }
        )
    return candidates


def build_alignment_record(
    request_path: Path,
    requests_dir: Path,
    query_id_to_question: dict[str, str],
    squad_index: SquadIndex,
) -> dict[str, Any]:
    """为一个混合来源请求构造 gold-answer 对齐记录。

    Args:
        request_path: 单个 ``requests/*.json`` 路径。
        requests_dir: 用于生成稳定相对路径的请求根目录。
        query_id_to_question: ANNS query id 到原始问题的映射。
        squad_index: SQuAD v2 问题索引。

    Returns:
        可直接写入 JSONL 的请求级对齐对象。

    Raises:
        ValueError: 请求 JSON 缺少 ``sample_id``、``query_id`` 或 ``question``。
    """
    request_payload = _load_json_object(
        request_path,
        description="mixed-source request",
    )
    sample_id = request_payload.get("sample_id")
    query_id = request_payload.get("query_id")
    question = request_payload.get("question")
    sample_status = request_payload.get("status")
    if (
        not isinstance(sample_id, str)
        or not isinstance(query_id, str)
        or not isinstance(question, str)
    ):
        raise ValueError(
            "Mixed-source request must contain string sample_id, query_id, "
            f"and question fields: {request_path}"
        )

    anns_validation = _validate_anns_question(
        query_id,
        question,
        query_id_to_question,
    )
    match_result = match_squad_question(question, squad_index)
    (
        match_result,
        request_squad_id,
        request_squad_id_validation,
    ) = _resolve_request_squad_id(request_payload, match_result)
    is_usable_match = match_result.status in {
        "matched",
        "matched_equivalent",
    }

    gold_answers: list[str] = []
    is_impossible: bool | None = None
    if is_usable_match:
        gold_answers = _deduplicate_strings(
            [
                answer.text
                for record in match_result.records
                for answer in record.answers
            ]
        )
        is_impossible = match_result.records[0].is_impossible

    try:
        relative_request_path = request_path.relative_to(requests_dir.parent)
    except ValueError:
        relative_request_path = request_path

    return {
        "schema_version": SCHEMA_VERSION,
        "sample_id": sample_id,
        "query_id": query_id,
        "question": question,
        "sample_status": sample_status,
        "request_path": str(relative_request_path),
        "anns_question_validation": anns_validation,
        "request_squad_id": request_squad_id,
        "request_squad_id_validation": request_squad_id_validation,
        "match_status": match_result.status,
        "match_method": match_result.method,
        "is_impossible": is_impossible,
        "gold_answers": gold_answers,
        "squad_candidates": _serialize_candidates(match_result.records),
    }


def _write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    """原子写入请求级 JSONL manifest。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(f".{path.name}.tmp")
    with temporary_path.open("w", encoding="utf-8") as output_file:
        for record in records:
            output_file.write(
                json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n"
            )
    temporary_path.replace(path)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    """原子写入带缩进的摘要 JSON。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(f".{path.name}.tmp")
    with temporary_path.open("w", encoding="utf-8") as output_file:
        json.dump(
            payload,
            output_file,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        output_file.write("\n")
    temporary_path.replace(path)


def build_summary(
    records: list[dict[str, Any]],
    args: argparse.Namespace,
    squad_index: SquadIndex,
) -> dict[str, Any]:
    """汇总匹配覆盖率、答案类型和 ANNS 来源校验结果。"""
    match_status_counts = Counter(record["match_status"] for record in records)
    match_method_counts = Counter(
        record["match_method"]
        for record in records
        if record["match_method"] is not None
    )
    anns_validation_counts = Counter(
        record["anns_question_validation"] for record in records
    )
    request_squad_id_validation_counts = Counter(
        record["request_squad_id_validation"] for record in records
    )
    num_matched = match_status_counts["matched"] + match_status_counts[
        "matched_equivalent"
    ]
    num_answerable = sum(
        record["is_impossible"] is False
        for record in records
        if record["match_status"] in {"matched", "matched_equivalent"}
    )
    num_impossible = sum(record["is_impossible"] is True for record in records)
    num_requests = len(records)

    return {
        "schema_version": SCHEMA_VERSION,
        "inputs": {
            "squad_dir": str(args.squad_dir),
            "anns_query_map": str(args.anns_query_map),
            "requests_dir": str(args.requests_dir),
        },
        "outputs": {
            "manifest": str(args.output),
            "summary": str(args.summary_output),
        },
        "num_squad_records": squad_index.num_records,
        "num_requests": num_requests,
        "num_matched": num_matched,
        "match_coverage": num_matched / num_requests if num_requests else 0.0,
        "num_answerable": num_answerable,
        "num_impossible": num_impossible,
        "match_status_counts": dict(sorted(match_status_counts.items())),
        "match_method_counts": dict(sorted(match_method_counts.items())),
        "anns_question_validation_counts": dict(
            sorted(anns_validation_counts.items())
        ),
        "request_squad_id_validation_counts": dict(
            sorted(request_squad_id_validation_counts.items())
        ),
    }


def main() -> None:
    """执行混合来源问题与 SQuAD v2 标准答案的完整对齐。"""
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s | %(levelname)s | %(message)s",
    )

    squad_index = load_squad_index(args.squad_dir)
    query_id_to_question = _load_anns_query_map(args.anns_query_map)
    request_paths = sorted(args.requests_dir.glob("*.json"))
    if not request_paths:
        raise ValueError(f"No request JSON files found in {args.requests_dir}")

    alignment_records = [
        build_alignment_record(
            request_path,
            args.requests_dir,
            query_id_to_question,
            squad_index,
        )
        for request_path in request_paths
    ]
    summary = build_summary(alignment_records, args, squad_index)
    _write_jsonl(args.output, alignment_records)
    _write_json(args.summary_output, summary)

    logger.info(
        "Aligned %d/%d mixed-source requests with SQuAD v2 gold answers "
        "(coverage %.2f%%).",
        summary["num_matched"],
        summary["num_requests"],
        summary["match_coverage"] * 100.0,
    )
    logger.info("Wrote gold-answer manifest to %s.", args.output)
    logger.info("Wrote alignment summary to %s.", args.summary_output)

    has_alignment_failures = (
        summary["match_status_counts"].get("unmatched", 0) > 0
        or summary["match_status_counts"].get("ambiguous", 0) > 0
    )
    has_anns_mismatches = any(
        status not in {"exact", "normalized"} and count > 0
        for status, count in summary["anns_question_validation_counts"].items()
    )
    has_squad_id_mismatches = (
        summary["request_squad_id_validation_counts"].get("mismatch", 0) > 0
    )
    if args.strict and (
        has_alignment_failures
        or has_anns_mismatches
        or has_squad_id_mismatches
    ):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
