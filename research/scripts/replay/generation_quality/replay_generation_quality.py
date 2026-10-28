# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""以低并发方式回放 mixed-source 样本并保存完整生成答案。

该脚本提供 ``normal``、``segmented`` 和 ``sparse`` 三种模式。各模式使用相同的
质量评测提示词、样本集合与采样参数，区别仅在于 prompt 是一次性提交，还是
通过 mixed-source prompt-stream RESTFUL API 分阶段提交。
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import logging
import math
import sys
import time
import uuid
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Literal

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

from research.scripts.replay.common.config import (
    add_shared_arguments,
    parse_configured_args,
)
from research.scripts.replay.common.dataset import (
    numeric_query_sort_key as _numeric_query_sort_key,
)
from research.scripts.replay.common.dataset import parse_sample as parse_replay_sample
from research.scripts.replay.common.dataset import (
    select_samples as select_replay_samples,
)
from research.scripts.replay.common.dispatch import (
    abort_prompt_stream as _abort_prompt_stream,
)
from research.scripts.replay.common.dispatch import send_source_events
from research.scripts.replay.common.models import (
    RenderedReplayPrompt as RenderedQualityPrompt,
)
from research.scripts.replay.common.models import SourceEventStream
from research.scripts.replay.common.probe import validate_probe
from research.scripts.replay.common.prompt import (
    ChatTemplateRenderer,
    render_sample_prompt,
)
from research.scripts.replay.common.results import (
    make_run_directory as _make_run_directory,
)
from research.scripts.replay.common.results import write_json as _write_json
from research.scripts.replay.common.results import write_prompt_previews
from research.scripts.replay.common.runner import execute_workload
from research.scripts.replay.common.transport import (
    create_http_client,
    iter_sse_payloads,
)
from research.scripts.replay.common.transport import (
    normalize_v1_url as _normalize_v1_url,
)
from research.scripts.replay.common.transport import post_json as _post_json
from research.scripts.replay.generation_quality.evaluators import references

logger = logging.getLogger(__name__)

MIX_SOURCE_SCHEMA_VERSION = "prompt_stream_mixed/v2"
QUALITY_REPLAY_SCHEMA_VERSION = "generation_quality_replay/v4"
DEFAULT_DATA_DIR = Path("research/data/mix_source")
DEFAULT_OUTPUT_DIR = Path("research/scripts/replay/generation_quality/results")
DEFAULT_PROXY_URL = "http://127.0.0.1:8000"
DEFAULT_ANSWER_INSTRUCTION = (
    "Return only the shortest answer span. Do not repeat the question and do "
    "not provide an explanation."
)
CHAT_TEMPLATE_CONTENT_MARKER = "VLLM_PROMPT_STREAM_CONTENT_7CDA3F6B"

ReplayMode = Literal["normal", "segmented", "sparse"]
SourceType = Literal["append", "update"]


@dataclass(frozen=True)
class GoldAnswerRecord:
    """一个样本对应的官方答案及其数据集评分协议。"""

    sample_id: str
    query_id: str
    question: str
    request_path: str
    sample_status: str
    gold_answers: tuple[str, ...]
    is_impossible: bool
    match_status: str
    evaluation_protocol: str = "squad_v2"


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
class SseCompletion:
    """完整消费一次 OpenAI completions SSE 后得到的生成结果。"""

    text: str
    finish_reason: str | None
    num_chunks: int
    num_token_ids: int
    prompt_stream_stats: dict[str, Any] | None = None  # P 的实际执行和 probe 统计。
    masked_padding: dict[str, Any] | None = None  # P/D 共同使用的物理 padding 区间。
    num_prefill_token_ids: int = 0  # 兼容旧结果；当前回答全部来自 D，此项为零。
    num_decode_token_ids: int = 0


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
    logical_prompt_tokens: int | None
    logical_prompt_token_sha256: str | None
    error_stage: str | None = None
    error: str | None = None
    evaluation_protocol: str = "squad_v2"
    input_transport: str = "text"
    probe_keep_ratio: float | None = None  # 仅 sparse 模式注册 probe。
    prompt_stream_stats: dict[str, Any] | None = None
    masked_padding: dict[str, Any] | None = None
    num_prefill_token_ids: int = 0
    num_decode_token_ids: int = 0
    source_events_sha256: str = ""  # 配对时同时校验历史快照和来源时间配置。
    generation_settings: dict[str, Any] | None = None  # 固定模型、模板和采样口径。


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
            raise ValueError(f"Invalid JSON at {path}:{line_number}: {exc}") from exc
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
                f"Gold record at {path}:{line_number} has invalid request metadata."
            )
        if not isinstance(raw_answers, list) or not all(
            isinstance(answer, str) for answer in raw_answers
        ):
            raise ValueError(f"Gold record {sample_id!r} has invalid gold_answers.")
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


def _parse_sample(
    path: Path,
    gold_records: dict[str, GoldAnswerRecord],
) -> QualitySample | None:
    """解析一个请求文件；没有官方答案时返回 ``None``。"""
    parsed = parse_replay_sample(path)
    gold = gold_records.get(parsed.sample_id)
    if gold is None:
        return None
    if parsed.query_id != gold.query_id or parsed.question != gold.question:
        raise ValueError(
            f"Sample {parsed.sample_id!r} does not match its SQuAD alignment record."
        )
    return QualitySample(
        sample_id=parsed.sample_id,
        query_id=parsed.query_id,
        question=parsed.question,
        status=parsed.status,
        start_request=parsed.start_request,
        source_streams=parsed.source_streams,
        finish_event=parsed.finish_event,
        gold=gold,
        raw_sample=parsed.raw_sample,
    )


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

    requested_query_ids = set(args.query_ids or [])
    requested_sample_ids = set(getattr(args, "sample_ids", None) or [])
    if requested_query_ids or requested_sample_ids:
        candidates = [
            record
            for record in candidates
            if (not requested_query_ids or record.query_id in requested_query_ids)
            and (not requested_sample_ids or record.sample_id in requested_sample_ids)
        ]
        missing = requested_query_ids - {record.query_id for record in candidates}
        missing |= requested_sample_ids - {record.sample_id for record in candidates}
        if missing:
            raise ValueError(
                f"指定 ID 缺少完整且已对齐的 SQuAD 答案：{sorted(missing)}"
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
    if getattr(args, "dataset", None) or getattr(args, "questions_file", None):
        samples = []
        for case in select_replay_samples(args):
            protocol = case.evaluation["protocol"]
            gold = GoldAnswerRecord(
                sample_id=case.sample_id,
                query_id=case.query_id,
                question=case.question,
                request_path="",
                sample_status=case.status,
                gold_answers=tuple(references(case.evaluation)),
                is_impossible=False,
                match_status="matched",
                evaluation_protocol=protocol,
            )
            samples.append(
                QualitySample(
                    sample_id=case.sample_id,
                    query_id=case.query_id,
                    question=case.question,
                    status=case.status,
                    start_request=case.start_request,
                    source_streams=case.source_streams,
                    finish_event=case.finish_event,
                    gold=gold,
                    raw_sample=case.raw_sample,
                )
            )
        return samples
    candidates = _select_gold_candidates(args, gold_records)
    if not candidates:
        raise ValueError("No eligible SQuAD-aligned samples selected.")

    # 显式 query id 数量通常很小，需要全部校验；普通范围模式只读到目标数量。
    explicit_ids = bool(args.query_ids or getattr(args, "sample_ids", None))
    num_target_samples = (
        len(candidates)
        if args.max_requests == 0 or explicit_ids
        else min(args.max_requests, len(candidates))
    )
    num_candidates_to_parse = len(candidates) if explicit_ids else num_target_samples
    logger.info(
        "Preselected %d eligible gold records; parsing at most %d request JSON files.",
        len(candidates),
        num_candidates_to_parse,
    )

    samples: list[QualitySample] = []
    for gold_record in candidates:
        if not explicit_ids and len(samples) >= num_target_samples:
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

    if explicit_ids:
        missing = set(args.query_ids or []) - {sample.query_id for sample in samples}
        missing |= set(getattr(args, "sample_ids", None) or []) - {
            sample.sample_id for sample in samples
        }
        if missing:
            raise ValueError(f"指定 ID 没有可回放的 SQuAD 样本：{sorted(missing)}")
    if not samples:
        raise ValueError("No eligible SQuAD-aligned samples selected.")
    return samples


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
            f"Sample {sample.sample_id!r} finish text must end with {answer_marker!r}."
        )
    question_text = stored_finish_text.rstrip()[: -len(answer_marker)].rstrip()
    answer_instruction = args.answer_instruction.strip()
    return f"{question_text}\n{answer_instruction}\n{answer_marker}"


def _build_rendered_quality_prompt(
    sample: QualitySample,
    args: argparse.Namespace,
    renderer: ChatTemplateRenderer,
) -> RenderedQualityPrompt:
    """共享模板渲染，仅为质量实验添加靠近最终问题的短答案约束。"""
    finish = {**sample.finish_event, "text": _quality_finish_text(sample, args)}
    return render_sample_prompt(
        replace(sample, finish_event=finish),
        renderer,
        input_transport=args.input_transport,
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
    payload["return_token_ids"] = True  # 完整 SSE 精确统计 D 的输出 token。
    if args.stop_at_newline:
        # SQuAD 评测期待短答案片段；两种模式必须使用完全相同的停止条件。
        payload["stop"] = ["\n"]


async def _consume_sse(response: httpx.Response) -> SseCompletion:
    """消费完整 SSE，保留 P 控制元数据并汇合 D 的全部回答。"""
    content_type = response.headers.get("content-type", "")
    if "text/event-stream" not in content_type:
        body = (await response.aread()).decode("utf-8", errors="replace")
        raise RuntimeError(f"Expected text/event-stream, got {content_type!r}: {body}")

    text_parts: list[str] = []
    finish_reason: str | None = None
    num_chunks = 0
    num_token_ids = 0
    num_prefill_token_ids = num_decode_token_ids = 0
    stats = padding = None
    async for payload in iter_sse_payloads(response):
        # 元数据通常在 P 就绪事件；后续 D 帧没有这些字段，不能覆盖已保存值。
        if "prompt_stream_stats" in payload:
            stats = payload["prompt_stream_stats"]
        if "prompt_stream_masked_padding" in payload:
            padding = payload["prompt_stream_masked_padding"]
        choices = payload.get("choices")
        if not isinstance(choices, list):
            continue
        for choice in choices:
            if not isinstance(choice, dict) or choice.get("index", 0) != 0:
                continue
            text = choice.get("text")
            if not isinstance(text, str):
                delta = choice.get("delta")
                text = delta.get("content", "") if isinstance(delta, dict) else ""
            if text:
                text_parts.append(text)
                num_chunks += 1
            token_ids = choice.get("token_ids")
            if isinstance(token_ids, list):
                # 当前协议所有回答均由 D 生成；历史显式来源仍可识别。
                origin = payload.get("token_source") or "decode"
                num_token_ids += len(token_ids)
                if origin == "prefill":
                    num_prefill_token_ids += len(token_ids)
                elif origin == "decode":
                    num_decode_token_ids += len(token_ids)
            if choice.get("finish_reason") is not None:
                finish_reason = str(choice["finish_reason"])
                if finish_reason not in ("stop", "length"):
                    raise RuntimeError(
                        f"Generation failed: finish_reason={finish_reason}"
                    )

    # D 加载失败可能只留下 P 的完成通知和 [DONE]，不能作为完整答案评分。
    if finish_reason is None:
        raise RuntimeError("Incomplete generation: SSE has no terminal finish_reason.")

    return SseCompletion(
        text="".join(text_parts),
        finish_reason=finish_reason,
        num_chunks=num_chunks,
        num_token_ids=num_token_ids,
        prompt_stream_stats=stats,
        masked_padding=padding,
        num_prefill_token_ids=num_prefill_token_ids,
        num_decode_token_ids=num_decode_token_ids,
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
    sample: QualitySample,
    dispatch_start: float,
) -> int:
    """复用固定来源时钟和串行 ACK；只返回质量输出需要的事件计数。"""
    events = await send_source_events(
        client,
        args.proxy_url,
        request_id,
        sample,
        renderer,
        source_stream,
        dispatch_start,
        args,
        args.run_name or "quality",
    )
    return len(events)


def _empty_result(
    sample: QualitySample,
    args: argparse.Namespace,
    run_id: str,
    rendered_prompt: RenderedQualityPrompt,
) -> GenerationResult:
    """创建一个尚未完成的逐请求结果对象。"""
    prompt_bytes = rendered_prompt.logical_prompt_text.encode("utf-8")
    token_bytes = ",".join(
        str(token_id) for token_id in rendered_prompt.logical_prompt_token_ids
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
        logical_prompt_tokens=(
            len(rendered_prompt.logical_prompt_token_ids)
            if args.input_transport == "token_ids"
            else None
        ),
        logical_prompt_token_sha256=(
            hashlib.sha256(token_bytes).hexdigest()
            if args.input_transport == "token_ids"
            else None
        ),
        evaluation_protocol=sample.gold.evaluation_protocol,
        input_transport=args.input_transport,
        probe_keep_ratio=args.probe_keep_ratio if args.mode == "sparse" else None,
        source_events_sha256=hashlib.sha256(
            json.dumps(
                {
                    "sources": [asdict(source) for source in sample.source_streams],
                    "timing": {
                        key: getattr(args, key)
                        for key in (
                            "append_initial_delay",
                            "update_initial_delay",
                            "append_chunk_interval",
                            "update_chunk_interval",
                            "source_finish_delay",
                        )
                    },
                },
                sort_keys=True,
                ensure_ascii=False,
            ).encode("utf-8")
        ).hexdigest(),
        generation_settings={
            "model": args.model,
            "tokenizer_model": args.tokenizer_model,
            "chat_template_sha256": getattr(args, "chat_template_sha256", None),
            "max_tokens": args.max_tokens,
            "temperature": args.temperature,
            "seed": args.seed,
            "stop": ["\n"] if args.stop_at_newline else [],
            "answer_instruction": args.answer_instruction,
        },
    )


def _record_completion(result: GenerationResult, completion: SseCompletion) -> None:
    """保存完整 P/D 输出及原始统计；执行校验通过后才标记请求成功。"""
    result.response_text = completion.text
    result.finish_reason = completion.finish_reason
    result.num_completion_chunks = completion.num_chunks
    result.num_completion_token_ids = completion.num_token_ids
    result.num_prefill_token_ids = completion.num_prefill_token_ids
    result.num_decode_token_ids = completion.num_decode_token_ids
    result.prompt_stream_stats = completion.prompt_stream_stats
    result.masked_padding = completion.masked_padding


def _validate_stream_execution(
    sample: QualitySample, args: argparse.Namespace, completion: SseCompletion
) -> None:
    """确认完整回答来自所声明的分段策略，不能把未生效的 sparse 当成实验。"""
    stats = completion.prompt_stream_stats
    if stats is None or stats.get("coalescing_enabled") is not False:
        raise ValueError("质量对照要求 P 返回执行统计，并关闭 Update 合并")
    if completion.masked_padding is None:
        raise ValueError("分段质量对照要求 P/D 开启 masked padding")
    num_update = sum(
        record["event"]["event_type"] == "update"
        for source in sample.source_streams
        for record in source.event_records
    )
    if stats.get("update_events_accepted") != num_update:
        raise ValueError("P 接收的 Update 数与回放输入不一致")
    if args.mode == "sparse":
        num_append = sum(
            record["event"]["event_type"] == "append"
            and bool(record["event"].get("text") or record["event"].get("token_ids"))
            for source in sample.source_streams
            for record in source.event_records
        )
        if "query_probe" not in stats:
            raise ValueError("sparse 请求未返回 Query probe 统计")
        if not math.isclose(stats["query_probe"]["keep_ratio"], args.probe_keep_ratio):
            raise ValueError("P 实际稀疏比例与配置不一致")
        validate_probe(stats, num_append, args.probe_keep_ratio)
    elif "query_probe" in stats:
        raise ValueError("segmented dense 基线不应执行 probe")


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
        "prompt": (
            rendered_prompt.logical_prompt_text
            if args.input_transport == "text"
            else list(rendered_prompt.logical_prompt_token_ids)
        ),
        "add_special_tokens": False,
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
        _record_completion(result, completion)
        result.result_status = "ok"
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
    start_payload["prompt"] = (
        rendered_prompt.start_text
        if args.input_transport == "text"
        else list(rendered_prompt.start_token_ids)
    )
    start_payload["add_special_tokens"] = False
    start_payload["request_id"] = external_request_id
    start_payload["api"] = "completions"
    # 各模式的最终问题和 chat suffix 完全一致；probe 不读取官方答案。
    start_payload.pop("probe_query", None)
    start_payload.pop("probe_keep_ratio", None)
    if args.mode == "sparse":
        start_payload["probe_query"] = rendered_prompt.finish_text
        start_payload["probe_keep_ratio"] = args.probe_keep_ratio
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
        dispatch_start = time.perf_counter()
        source_tasks = [
            asyncio.create_task(
                _send_source_stream(
                    client,
                    args,
                    renderer,
                    request_id,
                    source_stream,
                    sample,
                    dispatch_start,
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
        finish_event.pop("token_ids", None)
        if args.input_transport == "token_ids":
            finish_event["token_ids"] = list(rendered_prompt.finish_token_ids)
        finish_event["arrival_time"] = time.time()
        completion = await _stream_completion_request(
            client,
            _normalize_v1_url(args.proxy_url, "/v1/prompt_stream/finish"),
            {"request_id": request_id, "event": finish_event},
            args.sse_timeout,
        )
        _record_completion(result, completion)
        stage = "validate_execution"
        _validate_stream_execution(sample, args, completion)
        result.result_status = "ok"
    except asyncio.CancelledError:
        if request_id is not None:
            await asyncio.shield(
                _abort_prompt_stream(client, args.proxy_url, request_id)
            )
        raise
    except Exception as exc:  # noqa: BLE001
        result.result_status = "error"
        result.error_stage = stage
        result.error = str(exc)
        if request_id is not None:
            await _abort_prompt_stream(client, args.proxy_url, request_id)
    result.request_latency_ms = (time.perf_counter() - request_started_at) * 1000
    return result


def _validate_args(args: argparse.Namespace) -> None:
    """在读取数据和发请求前检查 CLI 边界。"""
    if args.start_index < 0 or args.max_requests < 0:
        raise ValueError("--start-index and --max-requests must be non-negative.")
    if args.max_tokens <= 0:
        raise ValueError("--max-tokens must be positive.")
    if not math.isfinite(args.probe_keep_ratio) or not 0 <= args.probe_keep_ratio <= 1:
        raise ValueError("--probe-keep-ratio 必须在 [0, 1] 内")
    if not math.isfinite(args.temperature) or args.temperature < 0:
        raise ValueError("--temperature must be non-negative.")
    if not args.tokenizer_model.strip():
        raise ValueError("--tokenizer-model must not be empty.")
    if args.http_timeout <= 0 or args.sse_timeout <= 0:
        raise ValueError("HTTP and SSE timeouts must be positive.")
    if not math.isfinite(args.qps) or args.qps < 0 or args.max_concurrency < 1:
        raise ValueError("QPS 必须有限且非负，并发上限必须为正整数")
    if not math.isfinite(args.inter_request_delay) or args.inter_request_delay < 0:
        raise ValueError("--inter-request-delay must be non-negative.")
    timing_values = (
        args.append_initial_delay,
        args.update_initial_delay,
        args.append_chunk_interval,
        args.update_chunk_interval,
        args.source_finish_delay,
    )
    if any(not math.isfinite(value) or value < 0 for value in timing_values):
        raise ValueError("Prompt-stream timing values must be non-negative.")


def parse_args() -> argparse.Namespace:
    """解析 generation quality 回放参数。"""
    parser = argparse.ArgumentParser(
        description=("回放独立数据或 SQuAD 对齐样本，保存完整生成答案。")
    )
    parser.add_argument(
        "--mode", choices=("normal", "segmented", "sparse"), default="segmented"
    )
    parser.add_argument(
        "--probe-keep-ratio",
        type=float,
        default=0.5,
        help="sparse 模式保留的完整独占 KV block 比例",
    )
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
        default=str(Path.home() / "Models/Qwen2.5-3B"),
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
    parser.add_argument("--max-tokens", type=int, default=64)
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
        help="QPS=0 时每个客户端槽位完成请求后的空闲秒数。",
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
    add_shared_arguments(parser)
    args = parse_configured_args(parser)
    if (
        args.dataset is None
        and args.questions_file is None
        and args.data_dir == DEFAULT_DATA_DIR
        and args.gold_manifest is None
    ):
        args.dataset = "triviaqa"  # 无显式旧数据输入时，使用当前独立数据集。
    args.experiment_suite = "generation_quality"
    args.replay_mode = args.mode
    args.event_progress_log_interval = 0
    return args


async def run(args: argparse.Namespace) -> Path:
    """按公共请求时钟回放，并在每个请求结束后立即持久化结果。"""
    _validate_args(args)
    renderer = ChatTemplateRenderer.from_pretrained(
        args.tokenizer_model,
        args.tokenizer_trust_remote_code,
    )
    args.chat_template_sha256 = renderer.chat_template_sha256
    logger.info(
        "Loaded chat template: tokenizer_model=%s template_sha256=%s",
        renderer.tokenizer_model,
        renderer.chat_template_sha256,
    )
    gold_manifest = args.gold_manifest
    if gold_manifest is None and not (args.dataset or args.questions_file):
        gold_manifest = args.data_dir / "quality_eval" / "squad_gold_answers.jsonl"
    gold_records = (
        {}
        if args.dataset or args.questions_file
        else _load_gold_manifest(gold_manifest)
    )
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
            "question": sample.question,
            "gold_answers": list(sample.gold.gold_answers),
            "evaluation_protocol": sample.gold.evaluation_protocol,
        }
        for sample in samples
    ]
    _write_json(
        run_directory / "config.json",
        {
            "schema_version": QUALITY_REPLAY_SCHEMA_VERSION,
            "args": {
                **vars(args),
                "gold_manifest": str(gold_manifest) if gold_manifest else None,
            },
            "prompt_format": {
                "name": "model_chat_template_system_user/v1",
                "tokenizer_model": renderer.tokenizer_model,
                "chat_template_sha256": renderer.chat_template_sha256,
                "transport": args.input_transport,
            },
            "num_selected_samples": len(samples),
            "selected_samples": selected_samples,
        },
    )

    if args.dry_run:
        previews = [
            _build_rendered_quality_prompt(sample, args, renderer) for sample in samples
        ]
        write_prompt_previews(run_directory, samples, previews)

    num_ok = 0
    num_error = 0
    generations_path = run_directory / "generations.jsonl"
    async with create_http_client(args) as client:
        with generations_path.open("w", encoding="utf-8") as output_file:

            async def execute(
                index: int, offset: float, queue_delay: float
            ) -> GenerationResult:
                sample = samples[index]
                rendered = _build_rendered_quality_prompt(sample, args, renderer)
                if args.dry_run:
                    result = _empty_result(sample, args, run_id, rendered)
                    result.result_status = "dry_run"
                    return result
                if args.mode == "normal":
                    return await _replay_normal_sample(
                        client, sample, args, run_id, rendered
                    )
                return await _replay_segmented_sample(
                    client, sample, args, run_id, renderer, rendered
                )

            def record(result: GenerationResult) -> None:
                nonlocal num_ok, num_error
                output_file.write(
                    json.dumps(asdict(result), ensure_ascii=False, sort_keys=True)
                    + "\n"
                )
                output_file.flush()
                if result.result_status in ("ok", "dry_run"):
                    num_ok += 1
                else:
                    num_error += 1
                logger.info(
                    "Quality replay DONE: mode=%s sample=%s status=%s",
                    args.mode,
                    result.sample_id,
                    result.result_status,
                )

            def failure(result: GenerationResult) -> str | None:
                return (
                    result.error
                    if args.fail_fast and result.result_status == "error"
                    else None
                )

            await execute_workload(
                len(samples),
                args,
                execute,
                on_result=record,
                fail_fast_error=failure,
                closed_loop=True,
            )

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
