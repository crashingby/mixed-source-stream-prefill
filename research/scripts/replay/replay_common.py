# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""mixed-source REST 回放的共享实现。

三个对外入口 replay_prompt_stream.py、replay_merged_update.py 和
replay_normal.py 共用本模块的数据解析、SSE TTFT 计时和结果写入能力。
前两个使用 prompt-stream API，最后一个把完整逻辑 prompt 交给同一个 proxy
的普通 /v1/completions 路径。
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import logging
import math
import random
import statistics
import time
from collections import defaultdict
from collections.abc import Awaitable, Callable, Iterable, Sequence
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

import httpx

logger = logging.getLogger(__name__)

MIX_SOURCE_SCHEMA_VERSION = "prompt_stream_mixed/v2"
DEFAULT_DATA_DIR = Path("research/data/mix_source")
DEFAULT_OUTPUT_DIR = Path("research/scripts/replay/results")
DEFAULT_PROXY_URL = "http://127.0.0.1:8000"

ReplayMode = Literal["stream", "normal", "merged_update"]
ArrivalProcess = Literal["fixed", "poisson"]
STREAM_TIMING_MODES = frozenset(("stream", "merged_update"))

REQUEST_METRIC_FIELDS = (
    "run_id",
    "mode",
    "sample_id",
    "query_id",
    "sample_status",
    "result_status",
    "error",
    "planned_arrival_offset_ms",
    "queue_delay_ms",
    "request_start_wall_time",
    "start_ack_ms",
    "source_dispatch_ms",
    "finish_to_first_token_ms",
    "request_to_first_token_ms",
    "finish_to_done_ms",
    "request_to_done_ms",
    "ttft_origin",
    "num_append_chunks",
    "num_update_snapshots",
    "num_event_acks",
    "append_chars",
    "final_update_chars",
    "normal_prompt_chars",
    "completion_chars",
    "completion_chunks",
)

EVENT_METRIC_FIELDS = (
    "run_id",
    "mode",
    "sample_id",
    "query_id",
    "source_id",
    "source_type",
    "event_type",
    "chunk_id",
    "sequence_no",
    "version",
    "scheduled_offset_ms",
    "actual_send_offset_ms",
    "ack_latency_ms",
    "result_status",
    "error",
)


@dataclass(frozen=True)
class SourceEventStream:
    """一个来源的有序 prompt-stream RESTFUL 事件。

    Args:
        source_id: 数据集内稳定的来源 id。
        source_type: ``append`` 或 ``update``。
        event_records: 按 source 内 ``sequence_no`` 排列的完整事件记录。
    """

    source_id: str
    source_type: Literal["append", "update"]
    event_records: tuple[dict[str, Any], ...]


@dataclass(frozen=True)
class SampleReplayInput:
    """回放一个 mixed-source 样本所需的已验证状态。"""

    sample_id: str
    query_id: str
    status: str
    start_request: dict[str, Any]
    source_streams: tuple[SourceEventStream, ...]
    finish_event: dict[str, Any]
    raw_sample: dict[str, Any]


@dataclass
class EventMetric:
    """一次 prompt-stream event REST 调用的发送与 ack 指标。"""

    run_id: str
    mode: str
    sample_id: str
    query_id: str
    source_id: str
    source_type: str
    event_type: str
    chunk_id: str
    sequence_no: int
    version: int
    scheduled_offset_ms: float
    actual_send_offset_ms: float
    ack_latency_ms: float | None
    result_status: str
    error: str | None = None


@dataclass
class RequestMetric:
    """一个样本的一次 stream 或 normal 回放结果。

    ``finish_to_first_token_ms`` 是本实验的主指标。prompt-stream 类模式从
    finish HTTP 请求提交到 proxy SSE 的首 token；normal 模式从普通
    completions 请求提交到其首个 SSE token。
    """

    run_id: str
    mode: str
    sample_id: str
    query_id: str
    sample_status: str
    result_status: str
    planned_arrival_offset_ms: float
    queue_delay_ms: float
    request_start_wall_time: float
    start_ack_ms: float | None = None
    source_dispatch_ms: float | None = None
    finish_to_first_token_ms: float | None = None
    request_to_first_token_ms: float | None = None
    finish_to_done_ms: float | None = None
    request_to_done_ms: float | None = None
    ttft_origin: str | None = None
    num_append_chunks: int = 0
    num_update_snapshots: int = 0
    num_event_acks: int = 0
    append_chars: int = 0
    final_update_chars: int = 0
    normal_prompt_chars: int = 0
    completion_chars: int = 0
    completion_chunks: int = 0
    error: str | None = None
    event_metrics: list[EventMetric] = field(default_factory=list)
    response_payloads: list[dict[str, Any]] = field(default_factory=list)


@dataclass(frozen=True)
class SseMeasurement:
    """一次 OpenAI completions SSE 响应的首 token 与结束状态。"""

    first_token_time: float | None
    first_token_origin: str | None
    completed_time: float
    completion_chars: int
    completion_chunks: int
    payloads: tuple[dict[str, Any], ...]


ReplaySample = Callable[
    [
        httpx.AsyncClient,
        SampleReplayInput,
        argparse.Namespace,
        str,
        float,
        float,
    ],
    Awaitable[RequestMetric],
]


class ReplayHttpError(RuntimeError):
    """包装 HTTP/SSE 上游错误，并保留有助定位的响应正文。"""


def parse_args(mode: ReplayMode) -> argparse.Namespace:
    """解析一个指定模式的 mixed-source REST 回放参数。

    Args:
        mode: 回放模式，决定暴露的 CLI 参数及请求路径。

    Returns:
        命令行参数命名空间。
    """
    parser = argparse.ArgumentParser(
        description=(
            f"Replay mixed-source samples through the selected REST mode: {mode}."
        )
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DEFAULT_DATA_DIR,
        help="Directory containing requests/*.json generated by mix_source.",
    )
    parser.add_argument(
        "--query-id",
        action="append",
        dest="query_ids",
        default=None,
        help="Query id to replay. Repeat to select several samples.",
    )
    parser.add_argument(
        "--start-index",
        type=int,
        default=0,
        help="Start position in numeric query-id order when --query-id is absent.",
    )
    parser.add_argument(
        "--max-requests",
        type=int,
        default=1,
        help="Maximum selected requests; 0 means all selected requests.",
    )
    parser.add_argument(
        "--include-partial",
        action="store_true",
        help="Include samples whose data-generation status is partial.",
    )
    parser.add_argument(
        "--proxy-url",
        type=str,
        default=DEFAULT_PROXY_URL,
        help="Proxy base URL; may end in /v1 or omit it.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="qwen2.5-3b",
        help="Served model name sent through the proxy.",
    )
    parser.add_argument(
        "--qps",
        type=float,
        default=1.0,
        help=(
            "Logical request start rate. Set 0 for closed-loop sequential execution."
        ),
    )
    parser.add_argument(
        "--arrival-process",
        choices=("fixed", "poisson"),
        default="fixed",
        help="Inter-request arrival process when --qps is positive.",
    )
    parser.add_argument(
        "--max-concurrency",
        type=int,
        default=1,
        help="Maximum in-flight logical requests within one replay mode.",
    )
    if mode in STREAM_TIMING_MODES:
        parser.add_argument(
            "--append-initial-delay",
            type=float,
            default=0.0,
            help="Seconds from start ack to the first append chunk send.",
        )
        parser.add_argument(
            "--update-initial-delay",
            type=float,
            default=0.0,
            help="Seconds from start ack to the first update snapshot send.",
        )
        parser.add_argument(
            "--append-chunk-interval",
            type=float,
            default=0.0,
            help="Target seconds between append data chunks in the same source.",
        )
        parser.add_argument(
            "--update-chunk-interval",
            type=float,
            default=0.0,
            help="Target seconds between update snapshots in the same source.",
        )
        parser.add_argument(
            "--source-finish-delay",
            type=float,
            default=0.0,
            help="Seconds after a source's last data event before source_finished.",
        )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=None,
        help="Override each sample start.max_tokens for this replay.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=None,
        help="Override each sample start.temperature for this replay.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional vLLM sampling seed forwarded through the proxy.",
    )
    parser.add_argument(
        "--http-timeout",
        type=float,
        default=600.0,
        help="Per HTTP request timeout in seconds.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory under which this replay creates a timestamped run folder.",
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default=None,
        help="Optional stable run folder suffix; defaults to timestamp + mode.",
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=0,
        help="Seed for reproducible Poisson inter-arrival draws only.",
    )
    parser.add_argument(
        "--save-responses",
        action="store_true",
        help="Save parsed SSE response payloads per request for debugging.",
    )
    parser.add_argument(
        "--ttft-only",
        action="store_true",
        help=(
            "Stop consuming each SSE response immediately after the first "
            "completion token. This records TTFT only and avoids waiting for "
            "the full decode stream."
        ),
    )
    parser.add_argument(
        "--sse-timeout",
        type=float,
        default=None,
        help=(
            "Optional maximum seconds spent consuming one successful SSE "
            "response. The regular --http-timeout still controls connection "
            "and request-level timeouts."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate and summarize selected data without issuing HTTP requests.",
    )
    parser.add_argument(
        "--fail-fast",
        action="store_true",
        help="Stop after the first request error instead of recording and continuing.",
    )
    parser.add_argument(
        "--log-level",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
        default="INFO",
        help="Logging verbosity.",
    )
    args = parser.parse_args()
    args.replay_mode = mode
    return args


def _utc_timestamp() -> str:
    """返回用于输出目录名称的 UTC 时间戳。"""
    return datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")


def _json_default(value: Any) -> Any:
    """把 argparse Path 等少量对象转换为 JSON 可序列化值。"""
    if isinstance(value, Path):
        return value.as_posix()
    raise TypeError(f"Unsupported JSON value: {type(value)!r}.")


def _write_json(path: Path, value: Any) -> None:
    """写入格式稳定的 UTF-8 JSON 文件。

    Args:
        path: 输出文件路径。
        value: 可 JSON 序列化的数据。
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        f"{json.dumps(value, ensure_ascii=False, indent=2, default=_json_default)}\n",
        encoding="utf-8",
    )


def _write_csv(
    path: Path, field_names: Sequence[str], rows: Iterable[dict[str, Any]]
) -> None:
    """以固定列顺序写入 CSV 指标文件。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=field_names, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _numeric_query_sort_key(query_id: str) -> tuple[int, int | str]:
    """把数字 query id 按数值排序，其它 id 回退到字符串排序。"""
    if query_id.isdigit():
        return (0, int(query_id))
    return (1, query_id)


def _normalize_v1_url(base_url: str, endpoint: str) -> str:
    """把服务 base URL 和 `/v1/...` endpoint 规范化地拼接。

    Args:
        base_url: 允许带或不带尾部 `/v1` 的服务地址。
        endpoint: 以 `/v1/` 开头的 OpenAI 或 prompt-stream 路径。

    Returns:
        可直接传给 httpx 的完整 URL。
    """
    normalized_base_url = base_url.rstrip("/")
    normalized_endpoint = endpoint if endpoint.startswith("/") else f"/{endpoint}"
    if normalized_base_url.endswith("/v1") and normalized_endpoint.startswith("/v1/"):
        normalized_endpoint = normalized_endpoint[3:]
    return f"{normalized_base_url}{normalized_endpoint}"


def _prompt_stream_finish_endpoint(args: argparse.Namespace) -> str:
    """返回当前 prompt-stream replay 应使用的 finish 路径。

    TTFT-only 实验只需要 P 节点 finish 后的首 token。此时走 proxy 的
    `/v1/prompt_stream/finish_ttft`，避免 proxy 再启动 D 节点 decode 并触发
    不必要的 KV transfer。完整生成或非 TTFT-only 调试仍走原 `/finish`。
    """
    if getattr(args, "ttft_only", False):
        return "/v1/prompt_stream/finish_ttft"
    return "/v1/prompt_stream/finish"


def _load_json(path: Path) -> dict[str, Any]:
    """读取单个样本 JSON，并将格式错误转成易定位的 ValueError。"""
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Cannot read sample {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"Sample {path} must contain a JSON object.")
    return value


def _extract_source_streams(sample: dict[str, Any]) -> tuple[SourceEventStream, ...]:
    """从 v2 flat events 恢复按来源分组的有序事件流。

    Args:
        sample: 一个 ``prompt_stream_mixed/v2`` 样本对象。

    Returns:
        依照样本 `sources` 声明排列的来源事件流。

    Raises:
        ValueError: source/event 布局不满足当前数据格式不变量。
    """
    source_definitions = sample.get("sources")
    event_records = sample.get("events")
    if not isinstance(source_definitions, list) or not isinstance(event_records, list):
        raise ValueError("Sample requires list-valued sources and events.")

    records_by_source: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in event_records:
        event = record.get("event") if isinstance(record, dict) else None
        if not isinstance(event, dict):
            raise ValueError("Every sample event record must contain an event object.")
        source_id = event.get("source_id")
        if not isinstance(source_id, str):
            raise ValueError("Every sample event requires string source_id.")
        records_by_source[source_id].append(record)

    source_streams: list[SourceEventStream] = []
    for source_definition in source_definitions:
        if not isinstance(source_definition, dict):
            raise ValueError("Sample source definition must be an object.")
        source_id = source_definition.get("source_id")
        source_type = source_definition.get("source_type")
        if not isinstance(source_id, str) or source_type not in ("append", "update"):
            raise ValueError(
                f"Invalid sample source definition: {source_definition!r}."
            )
        source_records = records_by_source.pop(source_id, [])
        source_records.sort(key=lambda record: int(record["event"]["sequence_no"]))
        expected_sequence_numbers = list(range(len(source_records)))
        actual_sequence_numbers = [
            int(record["event"]["sequence_no"]) for record in source_records
        ]
        if actual_sequence_numbers != expected_sequence_numbers:
            raise ValueError(
                f"Source {source_id} has non-contiguous sequence numbers "
                f"{actual_sequence_numbers}."
            )
        if (
            not source_records
            or source_records[-1]["event"].get("event_type") != "source_finished"
        ):
            raise ValueError(f"Source {source_id} must end with source_finished.")
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


def _parse_sample(path: Path) -> SampleReplayInput:
    """读取并校验一个 mixed-source v2 样本。"""
    sample = _load_json(path)
    if sample.get("schema_version") != MIX_SOURCE_SCHEMA_VERSION:
        raise ValueError(
            f"Sample {path} has schema={sample.get('schema_version')!r}; "
            f"expected {MIX_SOURCE_SCHEMA_VERSION!r}."
        )
    if sample.get("start", {}).get("api") != "completions":
        raise ValueError(f"Sample {path} does not use the completions API.")
    finish = sample.get("finish")
    if not isinstance(finish, dict) or not isinstance(finish.get("event"), dict):
        raise ValueError(f"Sample {path} has no valid finish event.")
    event_order = sample.get("event_order", {})
    if event_order.get("across_sources") != "unspecified":
        raise ValueError(
            f"Sample {path} incorrectly fixes an across-source event order."
        )

    sample_id = sample.get("sample_id")
    query_id = sample.get("query_id")
    if not isinstance(sample_id, str) or not isinstance(query_id, str):
        raise ValueError(f"Sample {path} requires string sample_id and query_id.")
    start_request = sample.get("start")
    if not isinstance(start_request, dict) or not isinstance(
        start_request.get("prompt"), str
    ):
        raise ValueError(f"Sample {path} has no text start.prompt.")

    return SampleReplayInput(
        sample_id=sample_id,
        query_id=query_id,
        status=str(sample.get("status", "unknown")),
        start_request=dict(start_request),
        source_streams=_extract_source_streams(sample),
        finish_event=dict(finish["event"]),
        raw_sample=sample,
    )


def _select_samples(args: argparse.Namespace) -> list[SampleReplayInput]:
    """扫描已落盘 request 文件并按 CLI 条件选择可回放样本。

    不依赖 manifest，因为生成器可能仍在写入请求文件、尚未重建最终 manifest。
    """
    request_directory = args.data_dir / "requests"
    request_paths = sorted(request_directory.glob("*.json"))
    if not request_paths:
        raise FileNotFoundError(f"No request JSON files found in {request_directory}.")

    samples: list[SampleReplayInput] = []
    for request_path in request_paths:
        try:
            sample = _parse_sample(request_path)
        except ValueError as exc:
            logger.warning("Skipping invalid sample %s: %s", request_path, exc)
            continue
        if sample.status != "complete" and not args.include_partial:
            continue
        samples.append(sample)

    samples.sort(key=lambda sample: _numeric_query_sort_key(sample.query_id))
    if args.query_ids:
        requested_query_ids = set(args.query_ids)
        samples = [
            sample for sample in samples if sample.query_id in requested_query_ids
        ]
        found_query_ids = {sample.query_id for sample in samples}
        missing_query_ids = sorted(requested_query_ids - found_query_ids)
        if missing_query_ids:
            raise ValueError(
                f"Requested query ids were not selectable: {missing_query_ids}."
            )
    else:
        if args.start_index < 0:
            raise ValueError("--start-index must be non-negative.")
        samples = samples[args.start_index :]

    if not samples:
        raise ValueError("No eligible mixed-source samples selected.")
    return samples


def _get_data_events(source_stream: SourceEventStream) -> list[dict[str, Any]]:
    """返回一个来源的 append/update 数据事件，排除 source_finished。"""
    return [
        record
        for record in source_stream.event_records
        if record["event"].get("event_type") in ("append", "update")
    ]


def _build_normal_prompt(sample: SampleReplayInput) -> tuple[str, int, int]:
    """构造普通模式使用的最终逻辑 prompt。

    排布严格遵循实验定义：固定模板 + 所有 Append chunk + 每个 Update 来源的
    最终快照 + 最后用户问题。不会包含 source_finished 控制事件。

    Args:
        sample: 已验证的 mixed-source 样本。

    Returns:
        三元组 ``(prompt, append_chars, final_update_chars)``。
    """
    append_text_parts: list[str] = []
    final_update_text_parts: list[str] = []
    for source_stream in sample.source_streams:
        data_records = _get_data_events(source_stream)
        if source_stream.source_type == "append":
            append_text_parts.extend(
                str(record["event"].get("text") or "") for record in data_records
            )
        elif data_records:
            # Update 语义是完整快照，因此普通模式仅保留该来源最终版本。
            final_update_text_parts.append(
                str(data_records[-1]["event"].get("text") or "")
            )

    append_text = "".join(append_text_parts)
    final_update_text = "".join(final_update_text_parts)
    final_question = str(sample.finish_event.get("text") or "")
    prompt = (
        f"{sample.start_request['prompt']}{append_text}{final_update_text}"
        f"{final_question}"
    )
    return prompt, len(append_text), len(final_update_text)


def _override_sampling_params(
    request: dict[str, Any],
    args: argparse.Namespace,
) -> None:
    """把 CLI 覆盖项写入一个 start 或 normal 请求体。"""
    if args.max_tokens is not None:
        request["max_tokens"] = args.max_tokens
    if args.temperature is not None:
        request["temperature"] = args.temperature
    if args.seed is not None:
        request["seed"] = args.seed


async def _sleep_until(target_time: float) -> None:
    """在 monotonic 时钟下等待到指定目标时间，避免累计 sleep 漂移。"""
    remaining_seconds = target_time - time.perf_counter()
    if remaining_seconds > 0:
        await asyncio.sleep(remaining_seconds)


async def _post_json(
    client: httpx.AsyncClient,
    url: str,
    payload: dict[str, Any],
) -> dict[str, Any]:
    """发送 JSON POST，并在错误时保留可读的上游正文。"""
    try:
        response = await client.post(url, json=payload)
    except httpx.RequestError as exc:
        raise ReplayHttpError(f"HTTP request failed for {url}: {exc}") from exc
    if response.is_error:
        raise ReplayHttpError(
            f"HTTP {response.status_code} from {url}: {response.text}"
        )
    try:
        response_json = response.json()
    except ValueError as exc:
        raise ReplayHttpError(f"Non-JSON response from {url}: {response.text}") from exc
    if not isinstance(response_json, dict):
        raise ReplayHttpError(f"JSON response from {url} must be an object.")
    return response_json


def _request_limit(args: argparse.Namespace) -> int | None:
    """返回本次应保留的样本上限。

    显式 ``--query-id`` 延续原有语义：回放所有指定 id，不受
    ``--max-requests`` 默认值影响。批量回放时，0 表示不设上限。
    """
    if args.query_ids or args.max_requests == 0:
        return None
    return args.max_requests


def _limit_selected_samples(
    samples: Sequence[SampleReplayInput],
    args: argparse.Namespace,
) -> list[SampleReplayInput]:
    """按本地选择参数保留候选样本，不提前查看完整请求长度。"""
    request_limit = _request_limit(args)
    selected_samples = list(samples)
    if request_limit is not None:
        selected_samples = selected_samples[:request_limit]
    return selected_samples


def _is_first_token_choice(choice: dict[str, Any]) -> bool:
    """判断一个 OpenAI SSE choice 是否实际代表一个生成 token。"""
    if "token_ids" in choice and isinstance(choice["token_ids"], list):
        return True
    if "text" in choice:
        # 仅有 final finish_reason 的空 completion chunk 不应伪装成首 token。
        return choice.get("finish_reason") is None or bool(choice.get("text"))
    delta = choice.get("delta")
    if isinstance(delta, dict) and "content" in delta:
        return delta.get("content") is not None
    return False


def _chunk_text(choice: dict[str, Any]) -> str:
    """从 completions/chat-completions 兼容 choice 提取文本。"""
    text = choice.get("text")
    if isinstance(text, str):
        return text
    delta = choice.get("delta")
    if isinstance(delta, dict) and isinstance(delta.get("content"), str):
        return delta["content"]
    return ""


async def _consume_sse_response(
    response: httpx.Response,
    first_token_start_time: float,
    save_payloads: bool,
    ttft_only: bool,
) -> SseMeasurement:
    """读取一个 OpenAI SSE 响应并定位第一个生成 token。

    Args:
        response: 已打开且 HTTP status 成功的流式 HTTP 响应。
        first_token_start_time: TTFT 的起点；stream 模式为 finish 发送时刻。
        save_payloads: 是否把每个 JSON payload 保存到结果目录。
        ttft_only: 首 token 出现后是否立即停止消费 SSE body。

    Returns:
        首 token、结束时间和输出规模统计。

    Raises:
        ReplayHttpError: 上游返回非 SSE、错误 payload 或无有效首 token。
    """
    content_type = response.headers.get("content-type", "")
    if "text/event-stream" not in content_type:
        body = (await response.aread()).decode("utf-8", errors="replace")
        raise ReplayHttpError(
            "Expected text/event-stream for TTFT measurement, got "
            f"{content_type!r}: {body}"
        )

    first_token_time: float | None = None
    first_token_origin: str | None = None
    completion_chars = 0
    completion_chunks = 0
    payloads: list[dict[str, Any]] = []
    async for line in response.aiter_lines():
        if not line.startswith("data:"):
            continue
        data = line.removeprefix("data:").strip()
        if not data:
            continue
        if data == "[DONE]":
            break
        try:
            payload = json.loads(data)
        except json.JSONDecodeError as exc:
            raise ReplayHttpError(f"Invalid SSE JSON payload: {data!r}") from exc
        if not isinstance(payload, dict):
            raise ReplayHttpError(f"SSE payload must be an object: {payload!r}.")
        if save_payloads:
            payloads.append(payload)
        if "error" in payload:
            raise ReplayHttpError(f"Upstream SSE error: {payload['error']!r}")

        choices = payload.get("choices")
        if not isinstance(choices, list):
            continue
        for choice in choices:
            if not isinstance(choice, dict):
                continue
            if _is_first_token_choice(choice):
                if first_token_time is None:
                    first_token_time = time.perf_counter()
                    first_token_origin = (
                        "prefill" if "token_ids" in choice else "decode"
                    )
                completion_chunks += 1
                completion_chars += len(_chunk_text(choice))
                if ttft_only:
                    completed_time = time.perf_counter()
                    return SseMeasurement(
                        first_token_time=first_token_time,
                        first_token_origin=first_token_origin,
                        completed_time=completed_time,
                        completion_chars=completion_chars,
                        completion_chunks=completion_chunks,
                        payloads=tuple(payloads),
                    )

    completed_time = time.perf_counter()
    if first_token_time is None:
        raise ReplayHttpError("SSE response ended before emitting a completion token.")
    if first_token_time < first_token_start_time:
        raise AssertionError("First token time cannot precede TTFT start time.")
    return SseMeasurement(
        first_token_time=first_token_time,
        first_token_origin=first_token_origin,
        completed_time=completed_time,
        completion_chars=completion_chars,
        completion_chunks=completion_chunks,
        payloads=tuple(payloads),
    )


async def consume_sse_response(
    response: httpx.Response,
    first_token_start_time: float,
    save_payloads: bool,
    ttft_only: bool,
    timeout_seconds: float | None,
) -> SseMeasurement:
    """带可选整体超时的 SSE 消费包装。"""
    try:
        if timeout_seconds is None:
            return await _consume_sse_response(
                response,
                first_token_start_time,
                save_payloads,
                ttft_only,
            )
        async with asyncio.timeout(timeout_seconds):
            return await _consume_sse_response(
                response,
                first_token_start_time,
                save_payloads,
                ttft_only,
            )
    except TimeoutError as exc:
        raise ReplayHttpError(
            f"Timed out after {timeout_seconds:.3f}s while consuming SSE response."
        ) from exc


def _planned_arrival_offsets(
    num_requests: int,
    qps: float,
    arrival_process: ArrivalProcess,
    random_seed: int,
) -> list[float]:
    """生成一个 mode 内每个请求相对启动点的计划到达时间秒。"""
    if qps == 0:
        return [0.0] * num_requests
    random_generator = random.Random(random_seed)
    offsets = [0.0]
    for _ in range(1, num_requests):
        if arrival_process == "fixed":
            inter_arrival_seconds = 1.0 / qps
        else:
            inter_arrival_seconds = random_generator.expovariate(qps)
        offsets.append(offsets[-1] + inter_arrival_seconds)
    return offsets


async def _replay_requests(
    samples: Sequence[SampleReplayInput],
    args: argparse.Namespace,
    run_id: str,
    replay_sample: ReplaySample,
) -> list[RequestMetric]:
    """依据 QPS 和并发限制回放一个脚本的全部样本。

    Args:
        samples: 已验证的 mixed-source 样本。
        args: 当前脚本的 CLI 参数。
        run_id: 本次实验的稳定标识。
        replay_sample: 当前模式负责单请求 REST 链路的异步函数。

    Returns:
        与输入样本顺序一致的请求指标。
    """
    semaphore = asyncio.Semaphore(args.max_concurrency)
    planned_offsets = _planned_arrival_offsets(
        len(samples),
        args.qps,
        args.arrival_process,
        args.random_seed,
    )
    mode_start_time = time.perf_counter()

    async with httpx.AsyncClient(timeout=args.http_timeout) as client:

        async def run_one(
            sample: SampleReplayInput,
            planned_offset: float,
        ) -> RequestMetric:
            await _sleep_until(mode_start_time + planned_offset)
            semaphore_wait_start_time = time.perf_counter()
            async with semaphore:
                queue_delay_ms = (
                    time.perf_counter() - semaphore_wait_start_time
                ) * 1000
                return await replay_sample(
                    client,
                    sample,
                    args,
                    run_id,
                    planned_offset * 1000,
                    queue_delay_ms,
                )

        if args.qps == 0:
            results: list[RequestMetric] = []
            for sample in samples:
                result = await run_one(sample, 0.0)
                results.append(result)
                if args.fail_fast and result.result_status != "ok":
                    raise ReplayHttpError(result.error or "Replay request failed.")
            return results

        if not args.fail_fast:
            tasks = [
                asyncio.create_task(run_one(sample, planned_offset))
                for sample, planned_offset in zip(samples, planned_offsets)
            ]
            return await asyncio.gather(*tasks)

        async def run_indexed(
            index: int,
            sample: SampleReplayInput,
            planned_offset: float,
        ) -> tuple[int, RequestMetric]:
            return index, await run_one(sample, planned_offset)

        tasks = [
            asyncio.create_task(run_indexed(index, sample, planned_offset))
            for index, (sample, planned_offset) in enumerate(
                zip(samples, planned_offsets)
            )
        ]
        ordered_results: list[RequestMetric | None] = [None] * len(tasks)
        try:
            for completed_task in asyncio.as_completed(tasks):
                index, result = await completed_task
                ordered_results[index] = result
                if result.result_status != "ok":
                    raise ReplayHttpError(result.error or "Replay request failed.")
        except BaseException:
            for task in tasks:
                if not task.done():
                    task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            raise

        assert all(result is not None for result in ordered_results)
        return [result for result in ordered_results if result is not None]


def _percentile(values: Sequence[float], percentile: float) -> float | None:
    """计算线性插值百分位数；空序列返回 None。"""
    if not values:
        return None
    ordered_values = sorted(values)
    if len(ordered_values) == 1:
        return ordered_values[0]
    position = (len(ordered_values) - 1) * percentile / 100.0
    lower_index = math.floor(position)
    upper_index = math.ceil(position)
    lower_value = ordered_values[lower_index]
    upper_value = ordered_values[upper_index]
    interpolation = position - lower_index
    return lower_value + (upper_value - lower_value) * interpolation


def _summarize_metrics(metrics: Sequence[RequestMetric]) -> dict[str, Any]:
    """汇总各模式成功请求的 TTFT 分布与错误数量。"""
    summary_by_mode: dict[str, dict[str, Any]] = {}
    metrics_by_mode: dict[str, list[RequestMetric]] = defaultdict(list)
    for metric in metrics:
        metrics_by_mode[metric.mode].append(metric)

    for mode, mode_metrics in metrics_by_mode.items():
        successful_ttfts = [
            metric.finish_to_first_token_ms
            for metric in mode_metrics
            if metric.result_status == "ok"
            and metric.finish_to_first_token_ms is not None
        ]
        numeric_ttfts = [float(ttft) for ttft in successful_ttfts]
        summary_by_mode[mode] = {
            "num_requests": len(mode_metrics),
            "num_success": sum(metric.result_status == "ok" for metric in mode_metrics),
            "num_error": sum(metric.result_status != "ok" for metric in mode_metrics),
            "ttft_ms": {
                "mean": statistics.fmean(numeric_ttfts) if numeric_ttfts else None,
                "min": min(numeric_ttfts) if numeric_ttfts else None,
                "p50": _percentile(numeric_ttfts, 50),
                "p90": _percentile(numeric_ttfts, 90),
                "p95": _percentile(numeric_ttfts, 95),
                "max": max(numeric_ttfts) if numeric_ttfts else None,
            },
        }
    return summary_by_mode


def _request_metric_row(metric: RequestMetric) -> dict[str, Any]:
    """将 RequestMetric 扁平化为 request_metrics.csv 的一行。"""
    row = asdict(metric)
    row.pop("event_metrics", None)
    row.pop("response_payloads", None)
    return row


def _write_run_outputs(
    run_directory: Path,
    args: argparse.Namespace,
    samples: Sequence[SampleReplayInput],
    metrics: Sequence[RequestMetric],
) -> None:
    """写入可复现实验的配置、逐请求指标、逐事件指标和摘要。"""
    request_rows = [_request_metric_row(metric) for metric in metrics]
    event_rows = [
        asdict(event_metric)
        for metric in metrics
        for event_metric in metric.event_metrics
    ]
    _write_csv(
        run_directory / "request_metrics.csv", REQUEST_METRIC_FIELDS, request_rows
    )
    _write_csv(run_directory / "event_metrics.csv", EVENT_METRIC_FIELDS, event_rows)
    _write_json(
        run_directory / "config.json",
        {
            "args": vars(args),
            "num_selected_samples": len(samples),
            "selected_samples": [
                {
                    "sample_id": sample.sample_id,
                    "query_id": sample.query_id,
                    "status": sample.status,
                }
                for sample in samples
            ],
        },
    )
    _write_json(
        run_directory / "summary.json",
        {
            "replay_mode": args.replay_mode,
            "ttft_definition": (
                "Milliseconds from POST /v1/prompt_stream/finish to the first "
                "proxy SSE completion token."
                if args.replay_mode in STREAM_TIMING_MODES
                else (
                    "Milliseconds from POST /v1/completions to the first "
                    "proxy-forwarded D-node SSE completion token."
                )
            ),
            "summary_by_mode": _summarize_metrics(metrics),
        },
    )
    if args.save_responses:
        response_directory = run_directory / "responses"
        for metric in metrics:
            if not metric.response_payloads:
                continue
            _write_json(
                response_directory / f"{metric.mode}-{metric.sample_id}.json",
                metric.response_payloads,
            )


def _validate_args(args: argparse.Namespace) -> None:
    """在读取数据或发 HTTP 前检查数值与模式参数。"""
    if args.start_index < 0:
        raise ValueError("--start-index must be non-negative.")
    if args.max_requests < 0:
        raise ValueError("--max-requests must be non-negative.")
    if args.qps < 0:
        raise ValueError("--qps must be non-negative.")
    if args.max_concurrency <= 0:
        raise ValueError("--max-concurrency must be positive.")
    if args.http_timeout <= 0:
        raise ValueError("--http-timeout must be positive.")
    if args.sse_timeout is not None and args.sse_timeout <= 0:
        raise ValueError("--sse-timeout must be positive when specified.")
    if args.max_tokens is not None and args.max_tokens <= 0:
        raise ValueError("--max-tokens must be positive when specified.")
    timing_field_names = (
        (
            "append_initial_delay",
            "update_initial_delay",
            "append_chunk_interval",
            "update_chunk_interval",
            "source_finish_delay",
        )
        if args.replay_mode in STREAM_TIMING_MODES
        else ()
    )
    for field_name in timing_field_names:
        if getattr(args, field_name) < 0:
            raise ValueError(f"--{field_name.replace('_', '-')} must be non-negative.")


def _make_run_directory(args: argparse.Namespace) -> tuple[str, Path]:
    """创建本次回放的独立结果目录。"""
    run_id = args.run_name or f"{_utc_timestamp()}-{args.replay_mode}"
    run_directory = args.output_dir / run_id
    if run_directory.exists():
        raise FileExistsError(
            f"Replay output directory already exists: {run_directory}. "
            "Choose a new --run-name."
        )
    run_directory.mkdir(parents=True)
    return run_id, run_directory


async def run(
    args: argparse.Namespace,
    replay_sample: ReplaySample,
) -> Path:
    """选择样本、按模式回放，并写出完整实验指标。

    Args:
        args: 已解析的命令行参数。
        replay_sample: 当前脚本的单请求回放函数。

    Returns:
        本次结果目录。
    """
    _validate_args(args)
    candidate_samples = _select_samples(args)
    samples = _limit_selected_samples(candidate_samples, args)
    run_id, run_directory = _make_run_directory(args)
    logger.info(
        "Selected %d samples; mode=%s qps=%.3f max_concurrency=%d.",
        len(samples),
        args.replay_mode,
        args.qps,
        args.max_concurrency,
    )

    if args.dry_run:
        dry_metrics: list[RequestMetric] = []
        for sample in samples:
            normal_prompt, append_chars, final_update_chars = _build_normal_prompt(
                sample
            )
            dry_metrics.append(
                RequestMetric(
                    run_id=run_id,
                    mode="dry_run",
                    sample_id=sample.sample_id,
                    query_id=sample.query_id,
                    sample_status=sample.status,
                    result_status="ok",
                    planned_arrival_offset_ms=0.0,
                    queue_delay_ms=0.0,
                    request_start_wall_time=time.time(),
                    num_append_chunks=sum(
                        len(_get_data_events(source_stream))
                        for source_stream in sample.source_streams
                        if source_stream.source_type == "append"
                    ),
                    num_update_snapshots=sum(
                        len(_get_data_events(source_stream))
                        for source_stream in sample.source_streams
                        if source_stream.source_type == "update"
                    ),
                    append_chars=append_chars,
                    final_update_chars=final_update_chars,
                    normal_prompt_chars=len(normal_prompt),
                )
            )
        _write_run_outputs(run_directory, args, samples, dry_metrics)
        return run_directory

    metrics = await _replay_requests(
        samples,
        args,
        run_id,
        replay_sample,
    )
    _write_run_outputs(run_directory, args, samples, metrics)
    return run_directory


def main(
    mode: ReplayMode,
    replay_sample: ReplaySample,
) -> None:
    """运行一个指定模式的回放 CLI。

    Args:
        mode: 回放模式，用于选择 CLI 参数及结果标签。
        replay_sample: 由对应入口提供的单请求 REST 回放函数。
    """
    args = parse_args(mode)
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
    run_directory = asyncio.run(run(args, replay_sample))
    logger.info("Replay results written to %s.", run_directory)
