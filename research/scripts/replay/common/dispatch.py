# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""各实验共享的来源文本发送、ACK 计量与取消清理。"""

from __future__ import annotations

import argparse
import logging
import time

import httpx

from .models import EventMetric, SampleReplayInput, SourceEventStream
from .prompt import ChatTemplateRenderer, event_payload
from .timeline import should_log_event_progress, source_timing
from .timeline import sleep_until as _sleep_until
from .transport import ReplayHttpError
from .transport import normalize_v1_url as _normalize_v1_url
from .transport import post_json as _post_json

logger = logging.getLogger(__name__)


def longest_common_prefix_len(
    previous_token_ids: tuple[int, ...],
    current_token_ids: tuple[int, ...],
) -> int:
    """计算两个 source-local Update 快照的 token LCP 长度。

    Args:
        previous_token_ids: 同一 Update source 的上一代完整 token 快照。
        current_token_ids: 本次 ACK 返回的完整 token 快照。

    Returns:
        两代快照从首 token 开始连续相同的 token 数。
    """
    common_prefix_len = 0
    for previous_token_id, current_token_id in zip(
        previous_token_ids,
        current_token_ids,
    ):
        if previous_token_id != current_token_id:
            break
        common_prefix_len += 1
    return common_prefix_len


def metric_mode(args: argparse.Namespace) -> str:
    """按实验套件返回稳定的请求与事件指标标签。"""
    if args.experiment_suite == "update_coalescing":
        return "update_coalescing"
    return "stream"


async def send_source_events(
    client: httpx.AsyncClient,
    proxy_url: str,
    request_id: str,
    sample: SampleReplayInput,
    renderer: ChatTemplateRenderer,
    source_stream: SourceEventStream,
    source_dispatch_start_time: float,
    args: argparse.Namespace,
    run_id: str,
) -> list[EventMetric]:
    """按一个来源的 source-local 顺序发送所有事件并等待每次 ack。

    每个来源在独立 asyncio task 内运行。因此 Append 和 Update 可以并发抵达
    scheduler；同一来源严格串行，保证 Update 版本与 Append 序号不乱序。
    """
    initial_delay, chunk_interval = source_timing(source_stream.source_type, args)
    event_metrics: list[EventMetric] = []
    num_data_events_sent = 0
    last_data_ack_time: float | None = None
    previous_update_token_ids: tuple[int, ...] | None = None

    for event_index, record in enumerate(source_stream.event_records):
        stored_event = record["event"]
        event_type = str(stored_event["event_type"])
        chunk_id = str(stored_event.get("chunk_id") or "")
        sequence_no = int(stored_event.get("sequence_no", 0))
        version = int(stored_event.get("version", 0))
        text_chars = len(str(stored_event.get("text") or ""))
        should_log_event = should_log_event_progress(args, event_index)

        if event_type in ("append", "update"):
            scheduled_offset = initial_delay + num_data_events_sent * chunk_interval
            scheduled_target_time = source_dispatch_start_time + scheduled_offset
            num_data_events_sent += 1
        else:
            # SOURCE_FINISHED 严格从最后一条数据事件 ACK 完成时开始计时。
            barrier_base_time = last_data_ack_time
            if barrier_base_time is None:
                barrier_base_time = source_dispatch_start_time + initial_delay
            scheduled_target_time = barrier_base_time + args.source_finish_delay
            scheduled_offset = scheduled_target_time - source_dispatch_start_time
        if should_log_event:
            wait_ms = max(0.0, scheduled_target_time - time.perf_counter()) * 1000
            logger.info(
                "Replay stream WAIT event: run_id=%s request_id=%s "
                "sample_id=%s query_id=%s source_id=%s source_type=%s "
                "event_index=%d event_type=%s chunk_id=%s sequence_no=%d "
                "version=%d scheduled_offset_ms=%.3f wait_ms=%.3f "
                "text_chars=%d",
                run_id,
                request_id,
                sample.sample_id,
                sample.query_id,
                source_stream.source_id,
                source_stream.source_type,
                event_index,
                event_type,
                chunk_id,
                sequence_no,
                version,
                scheduled_offset * 1000,
                wait_ms,
                text_chars,
            )

        await _sleep_until(scheduled_target_time)
        actual_send_time = time.perf_counter()
        actual_send_offset_ms = (actual_send_time - source_dispatch_start_time) * 1000
        outbound_event = event_payload(
            stored_event, renderer, getattr(args, "input_transport", "text")
        )
        # 数据集没有实验时间；每次真正发 REST 请求时才附加 arrival_time。
        outbound_event["arrival_time"] = time.time()
        payload = {
            "request_id": request_id,
            "event": outbound_event,
        }
        if should_log_event:
            logger.info(
                "Replay stream SEND event: run_id=%s request_id=%s "
                "sample_id=%s query_id=%s source_id=%s source_type=%s "
                "event_index=%d event_type=%s chunk_id=%s sequence_no=%d "
                "version=%d actual_send_offset_ms=%.3f text_chars=%d",
                run_id,
                request_id,
                sample.sample_id,
                sample.query_id,
                source_stream.source_id,
                source_stream.source_type,
                event_index,
                event_type,
                chunk_id,
                sequence_no,
                version,
                actual_send_offset_ms,
                text_chars,
            )

        ack_start_time = time.perf_counter()
        try:
            ack_response = await _post_json(
                client,
                _normalize_v1_url(proxy_url, "/v1/prompt_stream/event"),
                payload,
            )
        except Exception as exc:
            if should_log_event:
                logger.warning(
                    "Replay stream ERROR event: run_id=%s request_id=%s "
                    "sample_id=%s query_id=%s source_id=%s source_type=%s "
                    "event_index=%d event_type=%s chunk_id=%s sequence_no=%d "
                    "version=%d error=%s",
                    run_id,
                    request_id,
                    sample.sample_id,
                    sample.query_id,
                    source_stream.source_id,
                    source_stream.source_type,
                    event_index,
                    event_type,
                    chunk_id,
                    sequence_no,
                    version,
                    exc,
                )
            event_metrics.append(
                EventMetric(
                    run_id=run_id,
                    mode=metric_mode(args),
                    sample_id=sample.sample_id,
                    query_id=sample.query_id,
                    source_id=source_stream.source_id,
                    source_type=source_stream.source_type,
                    event_type=event_type,
                    chunk_id=chunk_id,
                    sequence_no=sequence_no,
                    version=version,
                    scheduled_offset_ms=scheduled_offset * 1000,
                    actual_send_offset_ms=actual_send_offset_ms,
                    ack_latency_ms=None,
                    result_status="error",
                    error=str(exc),
                )
            )
            raise

        ack_completed_time = time.perf_counter()
        if event_type in ("append", "update"):
            last_data_ack_time = ack_completed_time

        num_event_tokens: int | None = None
        update_lcp_tokens: int | None = None
        update_lcp_ratio: float | None = None
        raw_token_ids = ack_response.get("token_ids")
        if (
            event_type == "update"
            and args.experiment_suite == "update_coalescing"
            and raw_token_ids is None
        ):
            raise ReplayHttpError(
                "Formal Update coalescing replay requires token_ids in every "
                "Update event ACK."
            )
        if raw_token_ids is not None:
            if not isinstance(raw_token_ids, list) or not all(
                type(token_id) is int for token_id in raw_token_ids
            ):
                raise ReplayHttpError(
                    "Prompt-stream event token_ids must be a list of integers."
                )
            event_token_ids = tuple(raw_token_ids)
            num_event_tokens = len(event_token_ids)
            if event_type == "update":
                if previous_update_token_ids is not None:
                    update_lcp_tokens = longest_common_prefix_len(
                        previous_update_token_ids,
                        event_token_ids,
                    )
                    update_lcp_ratio = update_lcp_tokens / max(1, num_event_tokens)
                # 下一代只与同一 Update source 的本代完整 token 快照比较。
                previous_update_token_ids = event_token_ids

        ack_latency_ms = (ack_completed_time - ack_start_time) * 1000
        if should_log_event:
            logger.info(
                "Replay stream ACK event: run_id=%s request_id=%s "
                "sample_id=%s query_id=%s source_id=%s source_type=%s "
                "event_index=%d event_type=%s chunk_id=%s sequence_no=%d "
                "version=%d ack_latency_ms=%.3f",
                run_id,
                request_id,
                sample.sample_id,
                sample.query_id,
                source_stream.source_id,
                source_stream.source_type,
                event_index,
                event_type,
                chunk_id,
                sequence_no,
                version,
                ack_latency_ms,
            )

        event_metrics.append(
            EventMetric(
                run_id=run_id,
                mode=metric_mode(args),
                sample_id=sample.sample_id,
                query_id=sample.query_id,
                source_id=source_stream.source_id,
                source_type=source_stream.source_type,
                event_type=event_type,
                chunk_id=chunk_id,
                sequence_no=sequence_no,
                version=version,
                scheduled_offset_ms=scheduled_offset * 1000,
                actual_send_offset_ms=actual_send_offset_ms,
                ack_latency_ms=ack_latency_ms,
                result_status="ok",
                num_event_tokens=num_event_tokens,
                update_lcp_tokens=update_lcp_tokens,
                update_lcp_ratio=update_lcp_ratio,
            )
        )
    return event_metrics


async def abort_prompt_stream(
    client: httpx.AsyncClient,
    proxy_url: str,
    request_id: str,
) -> None:
    """尽力清理发生中途错误的 P 节点长期请求。"""
    try:
        await _post_json(
            client,
            _normalize_v1_url(proxy_url, "/v1/prompt_stream/abort"),
            {"request_id": request_id},
        )
    except Exception as exc:  # pragma: no cover - 仅清理路径。
        logger.warning("Failed to abort prompt-stream request %s: %s", request_id, exc)
