# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""以低并发方式回放 mixed-source 样本并保存完整生成答案。

该脚本提供 ``normal`` 和 ``segmented`` 两种模式。两种模式使用完全相同的
质量评测提示词、样本集合与采样参数，区别仅在于 prompt 是一次性提交，还是
通过 mixed-source prompt-stream RESTFUL API 分阶段提交。
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import logging
import time
import uuid
from collections import defaultdict
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

import httpx
from transformers import AutoTokenizer, PreTrainedTokenizerBase

logger = logging.getLogger(__name__)

MIX_SOURCE_SCHEMA_VERSION = "prompt_stream_mixed/v2"
QUALITY_REPLAY_SCHEMA_VERSION = "generation_quality_replay/v2"
DEFAULT_DATA_DIR = Path("research/data/mix_source")
DEFAULT_OUTPUT_DIR = Path(
    "research/scripts/replay/generation_quality/results"
)
DEFAULT_PROXY_URL = "http://127.0.0.1:8000"
DEFAULT_ANSWER_INSTRUCTION = (
    "Return only the shortest answer span. Do not repeat the question and do "
    "not provide an explanation."
)
CHAT_TEMPLATE_CONTENT_MARKER = "VLLM_PROMPT_STREAM_CONTENT_7CDA3F6B"

ReplayMode = Literal["normal", "segmented"]
SourceType = Literal["append", "update"]


@dataclass(frozen=True)
class GoldAnswerRecord:
    """一个 mixed-source 样本对应的 SQuAD 官方答案。"""

    sample_id: str
    query_id: str
    question: str
    request_path: str
    sample_status: str
    gold_answers: tuple[str, ...]
    is_impossible: bool
    match_status: str


@dataclass(frozen=True)
class SourceEventStream:
    """同一来源内按 ``sequence_no`` 排序的事件序列。"""

    source_id: str
    source_type: SourceType
    event_records: tuple[dict[str, Any], ...]


@dataclass(frozen=True)
class QualitySample:
    """一次质量回放所需的请求、事件与官方答案。"""

    sample_id: str
    query_id: str
    question: str
    status: str
    start_request: dict[str, Any]
    source_streams: tuple[SourceEventStream, ...]
    finish_event: dict[str, Any]
    gold: GoldAnswerRecord
    raw_sample: dict[str, Any]


@dataclass(frozen=True)
class RenderedQualityPrompt:
    """模型 chat template 渲染后的文本和分段 token 序列。"""

    start_token_ids: tuple[int, ...]
    finish_text: str
    finish_token_ids: tuple[int, ...]
    logical_prompt_token_ids: tuple[int, ...]
    logical_prompt_text: str


@dataclass(frozen=True)
class ChatTemplateRenderer:
    """使用目标模型 tokenizer 构造可分段的 chat prompt。"""

    tokenizer: PreTrainedTokenizerBase
    tokenizer_model: str
    chat_template_sha256: str

    @classmethod
    def from_pretrained(
        cls,
        tokenizer_model: str,
        trust_remote_code: bool,
    ) -> ChatTemplateRenderer:
        """加载 tokenizer，并固定本轮实验使用的 chat template。

        Args:
            tokenizer_model: Hugging Face 模型标识或本地 tokenizer 目录。
            trust_remote_code: 是否允许 tokenizer 执行仓库自定义代码。

        Returns:
            一个可重复使用的 chat template renderer。

        Raises:
            ValueError: tokenizer 没有可用的 chat template。
        """
        tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_model,
            trust_remote_code=trust_remote_code,
        )
        try:
            chat_template = tokenizer.get_chat_template()
        except ValueError as exc:
            raise ValueError(
                f"Tokenizer {tokenizer_model!r} has no usable chat template."
            ) from exc
        if not isinstance(chat_template, str) or not chat_template:
            raise ValueError(
                f"Tokenizer {tokenizer_model!r} returned an invalid chat template."
            )
        template_hash = hashlib.sha256(
            chat_template.encode("utf-8")
        ).hexdigest()
        return cls(
            tokenizer=tokenizer,
            tokenizer_model=tokenizer_model,
            chat_template_sha256=template_hash,
        )

    def encode_content(self, text: str) -> tuple[int, ...]:
        """编码一个 prompt 分段，不重复添加 BOS/EOS。"""
        return tuple(
            int(token_id)
            for token_id in self.tokenizer.encode(
                text,
                add_special_tokens=False,
            )
        )

    def render_chat_prompt(
        self,
        system_message: str,
        user_content: str,
    ) -> str:
        """把 system 和 user message 渲染为模型原生生成 Prompt。"""
        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_content},
        ]
        rendered = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        if not isinstance(rendered, str):
            raise ValueError("apply_chat_template(tokenize=False) must return text.")
        return rendered

    def render_envelope(self, system_message: str) -> tuple[str, str]:
        """提取 user 内容前后的模型模板文本。

        模板先用唯一 marker 作为 user 内容渲染，再从 marker 位置拆成固定
        prefix 和 suffix。动态 Append、Update 与最终问题随后填入二者之间。

        Args:
            system_message: 数据集通用模板，对应 chat 的 system message。

        Returns:
            ``(template_prefix, template_suffix)`` 文本二元组。

        Raises:
            ValueError: 模板改变、删除或重复插入内容 marker。
        """
        rendered = self.render_chat_prompt(
            system_message,
            CHAT_TEMPLATE_CONTENT_MARKER,
        )
        if rendered.count(CHAT_TEMPLATE_CONTENT_MARKER) != 1:
            raise ValueError(
                "Chat template must preserve the prompt-stream content marker "
                "exactly once."
            )
        prefix, suffix = rendered.split(
            CHAT_TEMPLATE_CONTENT_MARKER,
            maxsplit=1,
        )
        return prefix, suffix


@dataclass(frozen=True)
class SseCompletion:
    """完整消费一次 OpenAI completions SSE 后得到的生成结果。"""

    text: str
    finish_reason: str | None
    num_chunks: int
    num_token_ids: int


@dataclass
class GenerationResult:
    """一个样本在一种回放模式下的完整生成记录。"""

    schema_version: str
    run_id: str
    mode: ReplayMode
    sample_id: str
    query_id: str
    question: str
    gold_answers: list[str]
    is_impossible: bool
    match_status: str
    sample_status: str
    result_status: str
    response_text: str
    finish_reason: str | None
    request_id: str | None
    request_latency_ms: float | None
    num_completion_chunks: int
    num_completion_token_ids: int
    num_event_acks: int
    logical_prompt_chars: int
    logical_prompt_sha256: str
    logical_prompt_tokens: int
    logical_prompt_token_sha256: str
    error_stage: str | None = None
    error: str | None = None


def _utc_timestamp() -> str:
    """返回适合结果目录命名的 UTC 时间。"""
    return datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")


def _load_json(path: Path) -> dict[str, Any]:
    """读取一个 JSON 对象，并为格式错误补充文件路径。"""
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Cannot read JSON object from {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"JSON file {path} must contain an object.")
    return value


def _load_gold_manifest(path: Path) -> dict[str, GoldAnswerRecord]:
    """读取 SQuAD 对齐清单，只保留可用于官方评分的记录。

    ``matched`` 和 ``matched_equivalent`` 均保留。未匹配问题没有可靠的官方
    答案，因此不会进入质量回放，也不会被错误地按零分处理。
    """
    records: dict[str, GoldAnswerRecord] = {}
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise ValueError(f"Cannot read gold manifest {path}: {exc}") from exc

    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            raw_record = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"Invalid JSON at {path}:{line_number}: {exc}"
            ) from exc
        if not isinstance(raw_record, dict):
            raise ValueError(f"Gold record at {path}:{line_number} is not an object.")

        match_status = str(raw_record.get("match_status", "unmatched"))
        if match_status not in ("matched", "matched_equivalent"):
            continue
        sample_id = raw_record.get("sample_id")
        query_id = raw_record.get("query_id")
        question = raw_record.get("question")
        request_path = raw_record.get("request_path")
        sample_status = raw_record.get("sample_status")
        raw_answers = raw_record.get("gold_answers")
        is_impossible = bool(raw_record.get("is_impossible", False))
        identity_values = (
            sample_id,
            query_id,
            question,
            request_path,
            sample_status,
        )
        if not all(isinstance(value, str) for value in identity_values):
            raise ValueError(
                f"Gold record at {path}:{line_number} has invalid request "
                "metadata."
            )
        if not isinstance(raw_answers, list) or not all(
            isinstance(answer, str) for answer in raw_answers
        ):
            raise ValueError(
                f"Gold record {sample_id!r} has invalid gold_answers."
            )
        if not is_impossible and not raw_answers:
            raise ValueError(
                f"Answerable gold record {sample_id!r} has no official answer."
            )
        assert isinstance(sample_id, str)
        assert isinstance(query_id, str)
        assert isinstance(question, str)
        assert isinstance(request_path, str)
        assert isinstance(sample_status, str)
        if sample_id in records:
            raise ValueError(f"Duplicate sample_id={sample_id!r} in {path}.")
        records[sample_id] = GoldAnswerRecord(
            sample_id=sample_id,
            query_id=query_id,
            question=question,
            request_path=request_path,
            sample_status=sample_status,
            gold_answers=tuple(raw_answers),
            is_impossible=is_impossible,
            match_status=match_status,
        )

    if not records:
        raise ValueError(f"No matched SQuAD records found in {path}.")
    return records


def _extract_source_streams(
    sample: dict[str, Any],
) -> tuple[SourceEventStream, ...]:
    """从 mixed-source v2 的平铺事件恢复 source-local 有序流。"""
    source_definitions = sample.get("sources")
    event_records = sample.get("events")
    if not isinstance(source_definitions, list) or not isinstance(
        event_records, list
    ):
        raise ValueError("Sample requires list-valued sources and events.")

    records_by_source: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in event_records:
        event = record.get("event") if isinstance(record, dict) else None
        if not isinstance(event, dict) or not isinstance(
            event.get("source_id"), str
        ):
            raise ValueError("Every event record requires an event and source_id.")
        records_by_source[event["source_id"]].append(record)

    source_streams: list[SourceEventStream] = []
    for source_definition in source_definitions:
        if not isinstance(source_definition, dict):
            raise ValueError("Every source definition must be an object.")
        source_id = source_definition.get("source_id")
        source_type = source_definition.get("source_type")
        if not isinstance(source_id, str) or source_type not in (
            "append",
            "update",
        ):
            raise ValueError(f"Invalid source definition: {source_definition!r}.")

        source_records = records_by_source.pop(source_id, [])
        source_records.sort(key=lambda item: int(item["event"]["sequence_no"]))
        actual_sequence_numbers = [
            int(item["event"]["sequence_no"]) for item in source_records
        ]
        if actual_sequence_numbers != list(range(len(source_records))):
            raise ValueError(
                f"Source {source_id!r} has non-contiguous sequence numbers."
            )
        if (
            not source_records
            or source_records[-1]["event"].get("event_type")
            != "source_finished"
        ):
            raise ValueError(f"Source {source_id!r} must end with source_finished.")
        source_streams.append(
            SourceEventStream(
                source_id=source_id,
                source_type=source_type,
                event_records=tuple(source_records),
            )
        )

    if records_by_source:
        raise ValueError(
            f"Events refer to undeclared sources: {sorted(records_by_source)}."
        )
    return tuple(source_streams)


def _parse_sample(
    path: Path,
    gold_records: dict[str, GoldAnswerRecord],
) -> QualitySample | None:
    """解析一个请求文件；没有官方答案时返回 ``None``。"""
    sample = _load_json(path)
    if sample.get("schema_version") != MIX_SOURCE_SCHEMA_VERSION:
        raise ValueError(
            f"Sample {path} has unsupported schema={sample.get('schema_version')!r}."
        )
    sample_id = sample.get("sample_id")
    query_id = sample.get("query_id")
    question = sample.get("question")
    if not all(isinstance(value, str) for value in (sample_id, query_id, question)):
        raise ValueError(f"Sample {path} has invalid identity fields.")
    assert isinstance(sample_id, str)
    assert isinstance(query_id, str)
    assert isinstance(question, str)

    gold = gold_records.get(sample_id)
    if gold is None:
        return None
    if query_id != gold.query_id or question != gold.question:
        raise ValueError(
            f"Sample {sample_id!r} does not match its SQuAD alignment record."
        )

    start_request = sample.get("start")
    finish = sample.get("finish")
    if not isinstance(start_request, dict) or not isinstance(
        start_request.get("prompt"), str
    ):
        raise ValueError(f"Sample {sample_id!r} has no valid start prompt.")
    if not isinstance(finish, dict) or not isinstance(finish.get("event"), dict):
        raise ValueError(f"Sample {sample_id!r} has no valid finish event.")
    if sample.get("event_order", {}).get("across_sources") != "unspecified":
        raise ValueError(f"Sample {sample_id!r} fixes an across-source order.")

    return QualitySample(
        sample_id=sample_id,
        query_id=query_id,
        question=question,
        status=str(sample.get("status", "unknown")),
        start_request=dict(start_request),
        source_streams=_extract_source_streams(sample),
        finish_event=dict(finish["event"]),
        gold=gold,
        raw_sample=sample,
    )


def _numeric_query_sort_key(query_id: str) -> tuple[int, int | str]:
    """优先按数值 query id 排序，非数值 id 稳定排在其后。"""
    try:
        return (0, int(query_id))
    except ValueError:
        return (1, query_id)


def _select_gold_candidates(
    args: argparse.Namespace,
    gold_records: dict[str, GoldAnswerRecord],
) -> list[GoldAnswerRecord]:
    """只依据轻量 gold manifest 完成请求范围选择。

    该阶段不能打开体积较大的 request JSON。这样 ``--start-index``、
    ``--max-requests`` 和显式 query id 能在 NFS 数据读取前生效。

    Args:
        args: generation-quality 回放参数。
        gold_records: 已通过 SQuAD 对齐的请求级记录。

    Returns:
        按数值 query id 排序、完成状态与范围过滤后的候选记录。

    Raises:
        ValueError: 显式指定的 query id 不在可用 gold 记录中。
    """
    candidates = [
        record
        for record in gold_records.values()
        if args.include_partial or record.sample_status == "complete"
    ]
    candidates.sort(key=lambda record: _numeric_query_sort_key(record.query_id))

    if args.query_ids:
        requested_query_ids = set(args.query_ids)
        candidates = [
            record
            for record in candidates
            if record.query_id in requested_query_ids
        ]
        missing_query_ids = requested_query_ids - {
            record.query_id for record in candidates
        }
        if missing_query_ids:
            raise ValueError(
                "Requested query ids are missing, incomplete, or have no "
                f"aligned SQuAD answer: {sorted(missing_query_ids)}."
            )
        return candidates

    return candidates[args.start_index :]


def _resolve_gold_request_path(
    args: argparse.Namespace,
    gold_record: GoldAnswerRecord,
) -> Path:
    """把 gold manifest 中的相对路径解析到 requests 目录内。

    Args:
        args: 包含 mixed-source 数据根目录的回放参数。
        gold_record: 当前候选请求的 gold 对齐记录。

    Returns:
        经过目录边界校验的 request JSON 绝对路径。

    Raises:
        ValueError: manifest 路径为绝对路径或越出 requests 目录。
    """
    relative_path = Path(gold_record.request_path)
    if relative_path.is_absolute():
        raise ValueError(
            f"Gold record {gold_record.sample_id!r} uses an absolute "
            f"request_path={gold_record.request_path!r}."
        )

    request_directory = (args.data_dir / "requests").resolve()
    request_path = (args.data_dir / relative_path).resolve()
    try:
        request_path.relative_to(request_directory)
    except ValueError as exc:
        raise ValueError(
            f"Gold record {gold_record.sample_id!r} points outside "
            f"{request_directory}: {request_path}."
        ) from exc
    return request_path


def _select_samples(
    args: argparse.Namespace,
    gold_records: dict[str, GoldAnswerRecord],
) -> list[QualitySample]:
    """先按 gold manifest 截取范围，再解析实际需要的 request JSON。"""
    candidates = _select_gold_candidates(args, gold_records)
    if not candidates:
        raise ValueError("No eligible SQuAD-aligned samples selected.")

    # 显式 query id 数量通常很小，需要全部校验；普通范围模式只读到目标数量。
    num_target_samples = (
        len(candidates)
        if args.max_requests == 0
        else min(args.max_requests, len(candidates))
    )
    num_candidates_to_parse = (
        len(candidates) if args.query_ids else num_target_samples
    )
    logger.info(
        "Preselected %d eligible gold records; parsing at most %d request "
        "JSON files.",
        len(candidates),
        num_candidates_to_parse,
    )

    samples: list[QualitySample] = []
    for gold_record in candidates:
        if not args.query_ids and len(samples) >= num_target_samples:
            break
        request_path = _resolve_gold_request_path(args, gold_record)
        try:
            sample = _parse_sample(request_path, gold_records)
        except ValueError as exc:
            logger.warning("Skipping invalid sample %s: %s", request_path, exc)
            continue
        if sample is None:
            continue
        if sample.status != "complete" and not args.include_partial:
            continue
        samples.append(sample)

    if args.query_ids:
        requested_query_ids = set(args.query_ids)
        missing_query_ids = requested_query_ids - {
            sample.query_id for sample in samples
        }
        if missing_query_ids:
            raise ValueError(
                "Requested query ids are missing, incomplete, or have no aligned "
                f"SQuAD answer: {sorted(missing_query_ids)}."
            )
        if args.max_requests > 0:
            samples = samples[: args.max_requests]
    if not samples:
        raise ValueError("No eligible SQuAD-aligned samples selected.")
    return samples


def _quality_system_message(sample: QualitySample) -> str:
    """把数据集通用模板转换为 chat template 的 system message。"""
    return str(sample.start_request["prompt"]).strip()


def _quality_finish_text(
    sample: QualitySample,
    args: argparse.Namespace,
) -> str:
    """把短答案约束插入最终用户问题与 ``Answer:`` 之间。

    输出格式指令必须靠近生成位置，避免它位于长 RAG Prompt 的开头时被后续
    检索内容稀释。Normal 与 Segmented 均调用本函数，从而维持相同逻辑 Prompt。

    Raises:
        ValueError: 数据集 finish 文本不以 ``Answer:`` 结尾时抛出。
    """
    stored_finish_text = str(sample.finish_event.get("text") or "")
    answer_marker = "Answer:"
    if not stored_finish_text.rstrip().endswith(answer_marker):
        raise ValueError(
            f"Sample {sample.sample_id!r} finish text must end with "
            f"{answer_marker!r}."
        )
    question_text = stored_finish_text.rstrip()[: -len(answer_marker)].rstrip()
    answer_instruction = args.answer_instruction.strip()
    return f"{question_text}\n{answer_instruction}\n{answer_marker}"


def _data_records(source_stream: SourceEventStream) -> list[dict[str, Any]]:
    """返回来源中的 Append/Update 数据事件，排除 barrier。"""
    return [
        record
        for record in source_stream.event_records
        if record["event"].get("event_type") in ("append", "update")
    ]


def _final_source_texts(
    sample: QualitySample,
) -> tuple[list[str], list[str]]:
    """按物理布局返回全部 Append 文本和最终 Update 快照。"""
    append_parts: list[str] = []
    final_update_parts: list[str] = []
    for source_stream in sample.source_streams:
        records = _data_records(source_stream)
        if source_stream.source_type == "append":
            append_parts.extend(
                str(record["event"].get("text") or "")
                for record in records
            )
        elif records:
            final_update_parts.append(
                str(records[-1]["event"].get("text") or "")
            )
    return append_parts, final_update_parts


def _build_rendered_quality_prompt(
    sample: QualitySample,
    args: argparse.Namespace,
    renderer: ChatTemplateRenderer,
) -> RenderedQualityPrompt:
    """构造两种模式共享的模型原生 chat prompt。

    数据集通用模板作为 system message。Append、最终 Update 快照、用户问题和
    短答案约束共同组成 user message。模板生成的 user 前缀放入 start，user
    结束标记与 assistant generation header 放入 finish。

    每个动态分段独立执行 ``add_special_tokens=False`` 编码。Normal 直接发送
    拼接后的完整 token ids；Segmented 则发送同一组 start/event/finish token
    ids，避免 RESTFUL 层再次添加 BOS 或产生不同的 BPE 边界。
    """
    system_message = _quality_system_message(sample)
    append_parts, final_update_parts = _final_source_texts(sample)
    finish_user_text = _quality_finish_text(sample, args)
    user_content = "".join(
        (*append_parts, *final_update_parts, finish_user_text)
    )

    template_prefix, template_suffix = renderer.render_envelope(system_message)
    logical_prompt_text = "".join(
        (template_prefix, user_content, template_suffix)
    )
    fully_rendered_prompt = renderer.render_chat_prompt(
        system_message,
        user_content,
    )
    if logical_prompt_text != fully_rendered_prompt:
        raise ValueError(
            f"Chat template transforms user content for sample "
            f"{sample.sample_id!r}; it cannot be split losslessly."
        )

    start_token_ids = renderer.encode_content(template_prefix)
    suffix_token_ids = renderer.encode_content(template_suffix)
    finish_token_ids = (
        *renderer.encode_content(finish_user_text),
        *suffix_token_ids,
    )
    logical_prompt_token_ids: list[int] = list(start_token_ids)
    for prompt_part in (*append_parts, *final_update_parts):
        logical_prompt_token_ids.extend(renderer.encode_content(prompt_part))
    logical_prompt_token_ids.extend(finish_token_ids)

    if not start_token_ids or not finish_token_ids:
        raise ValueError(
            f"Chat template produced an empty prompt boundary for "
            f"sample={sample.sample_id!r}."
        )
    return RenderedQualityPrompt(
        start_token_ids=start_token_ids,
        finish_text=f"{finish_user_text}{template_suffix}",
        finish_token_ids=tuple(finish_token_ids),
        logical_prompt_token_ids=tuple(logical_prompt_token_ids),
        logical_prompt_text=logical_prompt_text,
    )


def _apply_generation_params(
    payload: dict[str, Any],
    args: argparse.Namespace,
) -> None:
    """把两种模式共享的确定性生成参数写入请求。"""
    payload["model"] = args.model
    payload["stream"] = True
    payload["max_tokens"] = args.max_tokens
    payload["temperature"] = args.temperature
    payload["seed"] = args.seed
    if args.stop_at_newline:
        # SQuAD 评测期待短答案片段；两种模式必须使用完全相同的停止条件。
        payload["stop"] = ["\n"]


def _normalize_v1_url(base_url: str, endpoint: str) -> str:
    """拼接可能已经带有 ``/v1`` 的 proxy URL。"""
    normalized_base = base_url.rstrip("/")
    if normalized_base.endswith("/v1") and endpoint.startswith("/v1/"):
        endpoint = endpoint.removeprefix("/v1")
    return f"{normalized_base}{endpoint}"


async def _post_json(
    client: httpx.AsyncClient,
    url: str,
    payload: dict[str, Any],
) -> dict[str, Any]:
    """发送 JSON POST 并返回对象形式响应。"""
    response = await client.post(url, json=payload)
    if response.is_error:
        raise RuntimeError(
            f"HTTP {response.status_code} from {url}: {response.text}"
        )
    try:
        response_payload = response.json()
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Non-JSON response from {url}: {response.text}") from exc
    if not isinstance(response_payload, dict):
        raise RuntimeError(f"JSON response from {url} is not an object.")
    return response_payload


async def _consume_sse(response: httpx.Response) -> SseCompletion:
    """消费完整 SSE，将 P 首 token 与 D 后续 token 文本拼接成答案。"""
    content_type = response.headers.get("content-type", "")
    if "text/event-stream" not in content_type:
        body = (await response.aread()).decode("utf-8", errors="replace")
        raise RuntimeError(
            f"Expected text/event-stream, got {content_type!r}: {body}"
        )

    text_parts: list[str] = []
    finish_reason: str | None = None
    num_chunks = 0
    num_token_ids = 0
    saw_done = False
    async for line in response.aiter_lines():
        if not line.startswith("data:"):
            continue
        data = line.removeprefix("data:").strip()
        if not data:
            continue
        if data == "[DONE]":
            saw_done = True
            break
        try:
            payload = json.loads(data)
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"Invalid SSE JSON payload: {data!r}.") from exc
        if not isinstance(payload, dict):
            raise RuntimeError("SSE payload must be a JSON object.")
        if "error" in payload:
            raise RuntimeError(f"Upstream SSE error: {payload['error']!r}.")

        choices = payload.get("choices")
        if not isinstance(choices, list):
            continue
        for choice in choices:
            if not isinstance(choice, dict) or choice.get("index", 0) != 0:
                continue
            text = choice.get("text")
            if not isinstance(text, str):
                delta = choice.get("delta")
                text = (
                    delta.get("content", "")
                    if isinstance(delta, dict)
                    else ""
                )
            if text:
                text_parts.append(text)
                num_chunks += 1
            token_ids = choice.get("token_ids")
            if isinstance(token_ids, list):
                num_token_ids += len(token_ids)
            if choice.get("finish_reason") is not None:
                finish_reason = str(choice["finish_reason"])

    if not saw_done:
        raise RuntimeError("SSE response closed before the [DONE] frame.")
    return SseCompletion(
        text="".join(text_parts),
        finish_reason=finish_reason,
        num_chunks=num_chunks,
        num_token_ids=num_token_ids,
    )


async def _stream_completion_request(
    client: httpx.AsyncClient,
    url: str,
    payload: dict[str, Any],
    sse_timeout: float,
) -> SseCompletion:
    """提交一个 completions 请求并在整体超时内消费完整 SSE。"""
    async with client.stream("POST", url, json=payload) as response:
        if response.is_error:
            body = await response.aread()
            raise RuntimeError(
                f"HTTP {response.status_code} from {url}: "
                f"{body.decode('utf-8', errors='replace')}"
            )
        try:
            async with asyncio.timeout(sse_timeout):
                return await _consume_sse(response)
        except TimeoutError as exc:
            raise RuntimeError(
                f"Timed out after {sse_timeout:.3f}s while consuming {url}."
            ) from exc


async def _send_source_stream(
    client: httpx.AsyncClient,
    args: argparse.Namespace,
    renderer: ChatTemplateRenderer,
    request_id: str,
    source_stream: SourceEventStream,
) -> int:
    """按 source-local 顺序发送预分词事件并等待 ACK。"""
    if source_stream.source_type == "append":
        initial_delay = args.append_initial_delay
        chunk_interval = args.append_chunk_interval
    else:
        initial_delay = args.update_initial_delay
        chunk_interval = args.update_chunk_interval

    num_data_events = 0
    num_acks = 0
    for record in source_stream.event_records:
        stored_event = record["event"]
        event_type = stored_event.get("event_type")
        if event_type in ("append", "update"):
            delay = initial_delay if num_data_events == 0 else chunk_interval
            num_data_events += 1
        else:
            delay = args.source_finish_delay
        if delay > 0:
            await asyncio.sleep(delay)

        outbound_event = dict(stored_event)
        event_text = outbound_event.get("text")
        if event_type in ("append", "update"):
            if not isinstance(event_text, str):
                raise ValueError(
                    f"Data event {source_stream.source_id!r}/"
                    f"{outbound_event.get('sequence_no')!r} has no text."
                )
            # P 节点直接采用该 token 序列，避免不同 REST 路径重复添加 BOS。
            outbound_event["token_ids"] = list(
                renderer.encode_content(event_text)
            )
        outbound_event["arrival_time"] = time.time()
        await _post_json(
            client,
            _normalize_v1_url(args.proxy_url, "/v1/prompt_stream/event"),
            {"request_id": request_id, "event": outbound_event},
        )
        num_acks += 1
    return num_acks


async def _abort_prompt_stream(
    client: httpx.AsyncClient,
    proxy_url: str,
    request_id: str,
) -> None:
    """尽力清理由失败回放留下的 P 节点长期请求。"""
    try:
        await _post_json(
            client,
            _normalize_v1_url(proxy_url, "/v1/prompt_stream/abort"),
            {"request_id": request_id},
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("Failed to abort request_id=%s: %s", request_id, exc)


def _empty_result(
    sample: QualitySample,
    args: argparse.Namespace,
    run_id: str,
    rendered_prompt: RenderedQualityPrompt,
) -> GenerationResult:
    """创建一个尚未完成的逐请求结果对象。"""
    prompt_bytes = rendered_prompt.logical_prompt_text.encode("utf-8")
    token_bytes = ",".join(
        str(token_id)
        for token_id in rendered_prompt.logical_prompt_token_ids
    ).encode("ascii")
    return GenerationResult(
        schema_version=QUALITY_REPLAY_SCHEMA_VERSION,
        run_id=run_id,
        mode=args.mode,
        sample_id=sample.sample_id,
        query_id=sample.query_id,
        question=sample.question,
        gold_answers=list(sample.gold.gold_answers),
        is_impossible=sample.gold.is_impossible,
        match_status=sample.gold.match_status,
        sample_status=sample.status,
        result_status="pending",
        response_text="",
        finish_reason=None,
        request_id=None,
        request_latency_ms=None,
        num_completion_chunks=0,
        num_completion_token_ids=0,
        num_event_acks=0,
        logical_prompt_chars=len(rendered_prompt.logical_prompt_text),
        logical_prompt_sha256=hashlib.sha256(prompt_bytes).hexdigest(),
        logical_prompt_tokens=len(rendered_prompt.logical_prompt_token_ids),
        logical_prompt_token_sha256=hashlib.sha256(token_bytes).hexdigest(),
    )


async def _replay_normal_sample(
    client: httpx.AsyncClient,
    sample: QualitySample,
    args: argparse.Namespace,
    run_id: str,
    rendered_prompt: RenderedQualityPrompt,
) -> GenerationResult:
    """把最终逻辑 prompt 一次性提交给普通 P→D completions 路径。"""
    result = _empty_result(sample, args, run_id, rendered_prompt)
    result.request_id = f"quality-normal-{run_id}-{sample.sample_id}"
    payload: dict[str, Any] = {
        "prompt": list(rendered_prompt.logical_prompt_token_ids),
        "request_id": result.request_id,
    }
    _apply_generation_params(payload, args)

    request_started_at = time.perf_counter()
    try:
        completion = await _stream_completion_request(
            client,
            _normalize_v1_url(args.proxy_url, "/v1/completions"),
            payload,
            args.sse_timeout,
        )
        result.result_status = "ok"
        result.response_text = completion.text
        result.finish_reason = completion.finish_reason
        result.num_completion_chunks = completion.num_chunks
        result.num_completion_token_ids = completion.num_token_ids
    except Exception as exc:  # noqa: BLE001
        result.result_status = "error"
        result.error_stage = "normal_completion"
        result.error = str(exc)
    result.request_latency_ms = (time.perf_counter() - request_started_at) * 1000
    return result


async def _replay_segmented_sample(
    client: httpx.AsyncClient,
    sample: QualitySample,
    args: argparse.Namespace,
    run_id: str,
    renderer: ChatTemplateRenderer,
    rendered_prompt: RenderedQualityPrompt,
) -> GenerationResult:
    """通过 prompt-stream RESTFUL API 提交 mixed-source 请求并完整生成。"""
    result = _empty_result(sample, args, run_id, rendered_prompt)
    external_request_id = (
        f"quality-stream-{run_id}-{sample.sample_id}-{uuid.uuid4().hex[:8]}"
    )
    start_payload = dict(sample.start_request)
    start_payload["prompt"] = list(rendered_prompt.start_token_ids)
    start_payload["request_id"] = external_request_id
    start_payload["api"] = "completions"
    _apply_generation_params(start_payload, args)

    request_started_at = time.perf_counter()
    request_id: str | None = None
    stage = "start"
    try:
        start_response = await _post_json(
            client,
            _normalize_v1_url(args.proxy_url, "/v1/prompt_stream/start"),
            start_payload,
        )
        request_id_value = start_response.get("request_id")
        if not isinstance(request_id_value, str):
            raise RuntimeError("Prompt-stream start response has no request_id.")
        request_id = request_id_value
        result.request_id = request_id

        stage = "source_events"
        source_tasks = [
            asyncio.create_task(
                _send_source_stream(
                    client,
                    args,
                    renderer,
                    request_id,
                    source_stream,
                )
            )
            for source_stream in sample.source_streams
        ]
        try:
            source_ack_counts = await asyncio.gather(*source_tasks)
        except BaseException:
            for source_task in source_tasks:
                if not source_task.done():
                    source_task.cancel()
            await asyncio.gather(*source_tasks, return_exceptions=True)
            raise
        result.num_event_acks = sum(source_ack_counts)

        stage = "finish_completion"
        finish_event = dict(sample.finish_event)
        finish_event["text"] = rendered_prompt.finish_text
        finish_event["token_ids"] = list(rendered_prompt.finish_token_ids)
        finish_event["arrival_time"] = time.time()
        completion = await _stream_completion_request(
            client,
            _normalize_v1_url(args.proxy_url, "/v1/prompt_stream/finish"),
            {"request_id": request_id, "event": finish_event},
            args.sse_timeout,
        )
        result.result_status = "ok"
        result.response_text = completion.text
        result.finish_reason = completion.finish_reason
        result.num_completion_chunks = completion.num_chunks
        result.num_completion_token_ids = completion.num_token_ids
    except Exception as exc:  # noqa: BLE001
        result.result_status = "error"
        result.error_stage = stage
        result.error = str(exc)
        if request_id is not None and stage != "finish_completion":
            await _abort_prompt_stream(client, args.proxy_url, request_id)
    result.request_latency_ms = (time.perf_counter() - request_started_at) * 1000
    return result


def _write_json(path: Path, value: Any) -> None:
    """以稳定、易读格式写入 JSON 文件。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            value,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            default=str,
        )
        + "\n",
        encoding="utf-8",
    )


def _make_run_directory(args: argparse.Namespace) -> tuple[str, Path]:
    """创建一个不可覆盖的独立结果目录。"""
    run_id = args.run_name or f"{_utc_timestamp()}-{args.mode}"
    run_directory = args.output_dir / run_id
    if run_directory.exists():
        raise FileExistsError(
            f"Output directory already exists: {run_directory}. "
            "Choose another --run-name."
        )
    run_directory.mkdir(parents=True)
    return run_id, run_directory


def _validate_args(args: argparse.Namespace) -> None:
    """在读取数据和发请求前检查 CLI 边界。"""
    if args.start_index < 0 or args.max_requests < 0:
        raise ValueError("--start-index and --max-requests must be non-negative.")
    if args.max_tokens <= 0:
        raise ValueError("--max-tokens must be positive.")
    if args.temperature < 0:
        raise ValueError("--temperature must be non-negative.")
    if not args.tokenizer_model.strip():
        raise ValueError("--tokenizer-model must not be empty.")
    if args.http_timeout <= 0 or args.sse_timeout <= 0:
        raise ValueError("HTTP and SSE timeouts must be positive.")
    if args.inter_request_delay < 0:
        raise ValueError("--inter-request-delay must be non-negative.")
    timing_values = (
        args.append_initial_delay,
        args.update_initial_delay,
        args.append_chunk_interval,
        args.update_chunk_interval,
        args.source_finish_delay,
    )
    if any(value < 0 for value in timing_values):
        raise ValueError("Prompt-stream timing values must be non-negative.")


def parse_args() -> argparse.Namespace:
    """解析 generation quality 回放参数。"""
    parser = argparse.ArgumentParser(
        description=(
            "Replay SQuAD-aligned mixed-source samples for answer-quality "
            "evaluation."
        )
    )
    parser.add_argument("--mode", choices=("normal", "segmented"), required=True)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument(
        "--gold-manifest",
        type=Path,
        default=None,
        help=(
            "SQuAD gold JSONL. Defaults to "
            "<data-dir>/quality_eval/squad_gold_answers.jsonl."
        ),
    )
    parser.add_argument(
        "--query-id",
        action="append",
        dest="query_ids",
        default=None,
        help="Select one query id; repeat this option to select several.",
    )
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument(
        "--max-requests",
        type=int,
        default=100,
        help="Maximum selected requests; 0 selects every eligible request.",
    )
    parser.add_argument("--include-partial", action="store_true")
    parser.add_argument("--proxy-url", default=DEFAULT_PROXY_URL)
    parser.add_argument("--model", default="qwen2.5-3b")
    parser.add_argument(
        "--tokenizer-model",
        required=True,
        help=(
            "Hugging Face tokenizer repository or local model directory used "
            "by apply_chat_template()."
        ),
    )
    parser.add_argument(
        "--tokenizer-trust-remote-code",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Allow custom tokenizer code while loading --tokenizer-model.",
    )
    parser.add_argument("--max-tokens", type=int, default=32)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument(
        "--answer-instruction",
        default=DEFAULT_ANSWER_INSTRUCTION,
        help="Instruction inserted immediately before the final Answer marker.",
    )
    parser.add_argument(
        "--stop-at-newline",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use a newline stop string so the model returns a short answer span.",
    )
    parser.add_argument("--append-initial-delay", type=float, default=0.0)
    parser.add_argument("--update-initial-delay", type=float, default=0.0)
    parser.add_argument("--append-chunk-interval", type=float, default=0.05)
    parser.add_argument("--update-chunk-interval", type=float, default=0.20)
    parser.add_argument("--source-finish-delay", type=float, default=0.0)
    parser.add_argument(
        "--inter-request-delay",
        type=float,
        default=0.20,
        help="Idle seconds after one complete request before the next starts.",
    )
    parser.add_argument("--http-timeout", type=float, default=600.0)
    parser.add_argument("--sse-timeout", type=float, default=300.0)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--log-level",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
        default="INFO",
    )
    return parser.parse_args()


async def run(args: argparse.Namespace) -> Path:
    """顺序回放选定样本，并在每个请求结束后立即持久化结果。"""
    _validate_args(args)
    renderer = ChatTemplateRenderer.from_pretrained(
        args.tokenizer_model,
        args.tokenizer_trust_remote_code,
    )
    logger.info(
        "Loaded chat template: tokenizer_model=%s template_sha256=%s",
        renderer.tokenizer_model,
        renderer.chat_template_sha256,
    )
    gold_manifest = args.gold_manifest
    if gold_manifest is None:
        gold_manifest = (
            args.data_dir / "quality_eval" / "squad_gold_answers.jsonl"
        )
    gold_records = _load_gold_manifest(gold_manifest)
    samples = _select_samples(args, gold_records)
    run_id, run_directory = _make_run_directory(args)

    selected_samples = [
        {
            "sample_id": sample.sample_id,
            "query_id": sample.query_id,
            "request_sha256": hashlib.sha256(
                json.dumps(
                    sample.raw_sample,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode("utf-8")
            ).hexdigest(),
        }
        for sample in samples
    ]
    _write_json(
        run_directory / "config.json",
        {
            "schema_version": QUALITY_REPLAY_SCHEMA_VERSION,
            "args": {**vars(args), "gold_manifest": str(gold_manifest)},
            "prompt_format": {
                "name": "model_chat_template_system_user/v1",
                "tokenizer_model": renderer.tokenizer_model,
                "chat_template_sha256": renderer.chat_template_sha256,
                "transport": "pretokenized_segments",
            },
            "num_selected_samples": len(samples),
            "selected_samples": selected_samples,
        },
    )

    num_ok = 0
    num_error = 0
    generations_path = run_directory / "generations.jsonl"
    async with httpx.AsyncClient(timeout=args.http_timeout) as client:
        with generations_path.open("w", encoding="utf-8") as output_file:
            for sample_index, sample in enumerate(samples):
                rendered_prompt = _build_rendered_quality_prompt(
                    sample,
                    args,
                    renderer,
                )
                logger.info(
                    "Quality replay START: mode=%s sample=%s query_id=%s "
                    "progress=%d/%d",
                    args.mode,
                    sample.sample_id,
                    sample.query_id,
                    sample_index + 1,
                    len(samples),
                )
                if args.dry_run:
                    result = _empty_result(
                        sample,
                        args,
                        run_id,
                        rendered_prompt,
                    )
                    result.result_status = "dry_run"
                elif args.mode == "normal":
                    result = await _replay_normal_sample(
                        client,
                        sample,
                        args,
                        run_id,
                        rendered_prompt,
                    )
                else:
                    result = await _replay_segmented_sample(
                        client,
                        sample,
                        args,
                        run_id,
                        renderer,
                        rendered_prompt,
                    )

                output_file.write(
                    json.dumps(asdict(result), ensure_ascii=False, sort_keys=True)
                    + "\n"
                )
                # 长时间实验即使被外部取消，也保留所有已完成请求。
                output_file.flush()
                if result.result_status in ("ok", "dry_run"):
                    num_ok += 1
                else:
                    num_error += 1
                logger.info(
                    "Quality replay DONE: mode=%s sample=%s status=%s "
                    "latency_ms=%s answer=%r",
                    args.mode,
                    sample.sample_id,
                    result.result_status,
                    (
                        f"{result.request_latency_ms:.3f}"
                        if result.request_latency_ms is not None
                        else None
                    ),
                    result.response_text,
                )
                if args.fail_fast and result.result_status == "error":
                    raise RuntimeError(result.error or "Quality replay failed.")
                if (
                    sample_index + 1 < len(samples)
                    and args.inter_request_delay > 0
                ):
                    await asyncio.sleep(args.inter_request_delay)

    _write_json(
        run_directory / "replay_summary.json",
        {
            "schema_version": QUALITY_REPLAY_SCHEMA_VERSION,
            "mode": args.mode,
            "num_selected_samples": len(samples),
            "num_successful_requests": num_ok,
            "num_failed_requests": num_error,
            "generation_file": str(generations_path),
        },
    )
    return run_directory


def main() -> None:
    """运行 generation quality 回放命令行入口。"""
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
    run_directory = asyncio.run(run(args))
    logger.info("Generation quality results written to %s.", run_directory)


if __name__ == "__main__":
    main()
