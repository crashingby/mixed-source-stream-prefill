# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""通过 prompt-stream RESTFUL API 回放合并 Update 快照 baseline。

这个入口模拟一个朴素客户端：客户端本地维护 Append store 和 Update store，
但每当任意来源有新内容到达时，都会把当前
``Append store + Update store`` 合并成一个完整 Update 快照发给 P 节点。
服务端因此只能按一个 Update 来源做 LCP，而不能利用我们分段 KV 管理里的
Append/Update 独立缓存优势。
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import time
import uuid
from dataclasses import dataclass
from typing import Any, Literal

import httpx
from replay_common import (
    EventMetric,
    ReplayHttpError,
    RequestMetric,
    SampleReplayInput,
    _get_data_events,
    _normalize_v1_url,
    _override_sampling_params,
    _post_json,
    _prompt_stream_finish_endpoint,
    _sleep_until,
    consume_sse_response,
    main,
    should_log_event_progress,
)

logger = logging.getLogger(__name__)

MERGED_UPDATE_SOURCE_ID = "merged_update_source"
TIME_GROUP_EPSILON = 1e-9


@dataclass(frozen=True)
class ScheduledSourceEvent:
    """一个原始来源事件在客户端侧的计划到达时间。

    Args:
        scheduled_offset: 相对 start ack 的计划发送时间，单位秒。
        source_priority: 同一时刻内的处理优先级；Append 先于 Update。
        source_id: 数据集中原始来源 id，仅用于日志和稳定排序。
        source_type: 原始来源类型，取值为 ``append`` 或 ``update``。
        event: 数据集里保存的原始 append/update 事件体。
    """

    scheduled_offset: float
    source_priority: int
    source_id: str
    source_type: Literal["append", "update"]
    event: dict[str, Any]


def _make_merged_update_start_payload(
    sample: SampleReplayInput,
    args: argparse.Namespace,
    external_request_id: str,
) -> dict[str, Any]:
    """构造 proxy `/v1/prompt_stream/start` 请求。

    merged-update baseline 仍然走 prompt-stream 链路，所以 start 与真正流式
    prefill 相同：先创建 P 节点长期请求，finish 阶段用 SSE 测量 TTFT。
    """
    payload = dict(sample.start_request)
    payload.update(
        {
            "api": "completions",
            "model": args.model,
            "stream": True,
            "request_id": external_request_id,
        }
    )
    _override_sampling_params(payload, args)
    return payload


def _source_timing(
    source_type: str,
    args: argparse.Namespace,
) -> tuple[float, float]:
    """返回一个原始来源的首 chunk 延迟与后续数据 chunk 间隔。"""
    if source_type == "append":
        return args.append_initial_delay, args.append_chunk_interval
    if source_type == "update":
        return args.update_initial_delay, args.update_chunk_interval
    raise ValueError(f"Unsupported source_type={source_type!r}.")


def _source_priority(source_type: str) -> int:
    """返回同一计划时刻内的客户端状态更新顺序。"""
    if source_type == "append":
        return 0
    if source_type == "update":
        return 1
    raise ValueError(f"Unsupported source_type={source_type!r}.")


def _collect_scheduled_data_events(
    sample: SampleReplayInput,
    args: argparse.Namespace,
) -> list[ScheduledSourceEvent]:
    """把所有原始来源的数据事件转换为一条全局计划时间线。

    同一来源仍保持 source-local 顺序；不同来源只由 CLI 配置的 interval
    决定谁先到达。这个 baseline 不向服务端暴露原始来源，而是在发送前把
    已到达状态合并成单个 Update 快照。
    """
    scheduled_events: list[ScheduledSourceEvent] = []
    for source_stream in sample.source_streams:
        initial_delay, chunk_interval = _source_timing(
            source_stream.source_type,
            args,
        )
        for event_index, record in enumerate(_get_data_events(source_stream)):
            event = dict(record["event"])
            scheduled_events.append(
                ScheduledSourceEvent(
                    scheduled_offset=initial_delay + event_index * chunk_interval,
                    source_priority=_source_priority(source_stream.source_type),
                    source_id=source_stream.source_id,
                    source_type=source_stream.source_type,
                    event=event,
                )
            )

    scheduled_events.sort(
        key=lambda event: (
            round(event.scheduled_offset, 9),
            event.source_priority,
            event.source_id,
            int(event.event.get("sequence_no", 0)),
        )
    )
    return scheduled_events


def _iter_scheduled_groups(
    scheduled_events: list[ScheduledSourceEvent],
) -> list[tuple[float, tuple[ScheduledSourceEvent, ...]]]:
    """把同一计划时刻的原始事件合并成一个客户端 tick。

    当 Append 和 Update 的 interval 恰好同时触发时，只发送一次 merged
    Update 快照；这比背靠背发两个快照更公平，也更接近“某一时刻已有状态”
    的基线定义。
    """
    groups: list[tuple[float, tuple[ScheduledSourceEvent, ...]]] = []
    current_offset: float | None = None
    current_events: list[ScheduledSourceEvent] = []

    for scheduled_event in scheduled_events:
        if current_offset is None:
            current_offset = scheduled_event.scheduled_offset
        is_same_tick = (
            abs(scheduled_event.scheduled_offset - current_offset)
            <= TIME_GROUP_EPSILON
        )
        if not is_same_tick:
            groups.append((current_offset, tuple(current_events)))
            current_offset = scheduled_event.scheduled_offset
            current_events = []
        current_events.append(scheduled_event)

    if current_offset is not None:
        groups.append((current_offset, tuple(current_events)))
    return groups


def _format_trigger_events(
    trigger_events: tuple[ScheduledSourceEvent, ...],
) -> str:
    """把一次 merged snapshot 的原始触发事件压缩成日志字符串。"""
    parts: list[str] = []
    for trigger_event in trigger_events:
        event = trigger_event.event
        parts.append(
            f"{trigger_event.source_type}:{trigger_event.source_id}:"
            f"seq={event.get('sequence_no', 0)}:ver={event.get('version', 0)}"
        )
    return ",".join(parts)


def _merged_context_text(
    append_text_parts: list[str],
    update_text_parts: list[str],
) -> str:
    """按实验定义拼接当前客户端视角的检索上下文。

    Append 部分保持单调追加，Update 部分按来源保存最新快照。二者之间
    不额外插入分隔符，因为数据构建阶段已经在 chunk 文本里保留了换行
    和标签。
    """
    append_text = "".join(append_text_parts)
    update_text = "".join(update_text_parts)
    return f"{append_text}{update_text}"


def _make_merged_update_event(
    text: str,
    sequence_no: int,
) -> dict[str, Any]:
    """把当前完整上下文封装成一个 Update 事件。

    服务端看到的始终是同一个 Update 来源。这样 P 节点只能对
    ``Append+Update`` 整体快照做 LCP，与我们真正方案里的分段 KV 管理形成
    对照。
    """
    return {
        "source_id": MERGED_UPDATE_SOURCE_ID,
        "event_type": "update",
        "source_type": "update",
        "chunk_id": f"merged-update-{sequence_no}",
        "version": sequence_no,
        "sequence_no": sequence_no,
        "text": text,
        "arrival_time": time.time(),
    }


def _make_source_finished_event(sequence_no: int) -> dict[str, Any]:
    """构造 merged Update 来源的结束事件。"""
    return {
        "source_id": MERGED_UPDATE_SOURCE_ID,
        "event_type": "source_finished",
        "source_type": "update",
        "chunk_id": "merged-update-source-finished",
        "version": max(0, sequence_no - 1),
        "sequence_no": sequence_no,
        "arrival_time": time.time(),
    }


async def _send_prompt_stream_event(
    client: httpx.AsyncClient,
    proxy_url: str,
    request_id: str,
    outbound_event: dict[str, Any],
) -> None:
    """向 proxy 发送一个 prompt-stream event 并等待 ack。"""
    await _post_json(
        client,
        _normalize_v1_url(proxy_url, "/v1/prompt_stream/event"),
        {
            "request_id": request_id,
            "event": outbound_event,
        },
    )


async def _abort_prompt_stream(
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


async def _dispatch_merged_update_events(
    client: httpx.AsyncClient,
    sample: SampleReplayInput,
    args: argparse.Namespace,
    request_id: str,
    run_id: str,
    source_dispatch_start_time: float,
) -> tuple[list[EventMetric], int, int, int, int, int]:
    """按合并 Update baseline 的语义发送所有中间事件。

    Args:
        client: 共享 HTTP client。
        sample: 当前 mixed-source 样本。
        args: CLI 参数，提供两个来源的 interval。
        request_id: `/start` 返回的长期请求 id。
        run_id: 当前实验 run 标识。
        source_dispatch_start_time: 事件发送时间线的 monotonic 起点。

    Returns:
        ``(event_metrics, num_append_chunks, num_update_snapshots,
        append_chars, final_update_chars, normal_prompt_chars)``。其中
        ``num_update_snapshots`` 是实际发给服务端的 merged Update 数据事件数。
    """
    append_text_parts: list[str] = []
    update_source_order = [
        source_stream.source_id
        for source_stream in sample.source_streams
        if source_stream.source_type == "update"
    ]
    update_text_by_source = {source_id: "" for source_id in update_source_order}
    event_metrics: list[EventMetric] = []
    num_logical_append_chunks = 0
    num_merged_update_snapshots = 0

    scheduled_events = _collect_scheduled_data_events(sample, args)
    scheduled_groups = _iter_scheduled_groups(scheduled_events)
    sequence_no = 0
    last_scheduled_offset = 0.0

    for scheduled_offset, trigger_events in scheduled_groups:
        should_log_event = should_log_event_progress(args, sequence_no)
        scheduled_target_time = source_dispatch_start_time + scheduled_offset
        trigger_summary = _format_trigger_events(trigger_events)
        if should_log_event:
            wait_ms = max(0.0, scheduled_target_time - time.perf_counter()) * 1000
            logger.info(
                "Replay merged-update WAIT event: run_id=%s request_id=%s "
                "sample_id=%s query_id=%s event_index=%d sequence_no=%d "
                "scheduled_offset_ms=%.3f wait_ms=%.3f triggers=%s",
                run_id,
                request_id,
                sample.sample_id,
                sample.query_id,
                sequence_no,
                sequence_no,
                scheduled_offset * 1000,
                wait_ms,
                trigger_summary,
            )

        await _sleep_until(scheduled_target_time)
        last_scheduled_offset = scheduled_offset

        for trigger_event in trigger_events:
            event_text = str(trigger_event.event.get("text") or "")
            if trigger_event.source_type == "append":
                append_text_parts.append(event_text)
                num_logical_append_chunks += 1
            else:
                update_text_by_source[trigger_event.source_id] = event_text

        update_text_parts = [
            update_text_by_source[source_id]
            for source_id in update_source_order
        ]
        merged_text = _merged_context_text(append_text_parts, update_text_parts)
        outbound_event = _make_merged_update_event(merged_text, sequence_no)
        actual_send_time = time.perf_counter()
        actual_send_offset_ms = (
            actual_send_time - source_dispatch_start_time
        ) * 1000
        if should_log_event:
            logger.info(
                "Replay merged-update SEND event: run_id=%s request_id=%s "
                "sample_id=%s query_id=%s event_index=%d sequence_no=%d "
                "version=%d actual_send_offset_ms=%.3f merged_text_chars=%d "
                "logical_append_chunks=%d triggers=%s",
                run_id,
                request_id,
                sample.sample_id,
                sample.query_id,
                sequence_no,
                sequence_no,
                sequence_no,
                actual_send_offset_ms,
                len(merged_text),
                num_logical_append_chunks,
                trigger_summary,
            )

        ack_start_time = time.perf_counter()
        try:
            await _send_prompt_stream_event(
                client,
                args.proxy_url,
                request_id,
                outbound_event,
            )
        except Exception as exc:
            if should_log_event:
                logger.warning(
                    "Replay merged-update ERROR event: run_id=%s request_id=%s "
                    "sample_id=%s query_id=%s event_index=%d sequence_no=%d "
                    "version=%d triggers=%s error=%s",
                    run_id,
                    request_id,
                    sample.sample_id,
                    sample.query_id,
                    sequence_no,
                    sequence_no,
                    sequence_no,
                    trigger_summary,
                    exc,
                )
            event_metrics.append(
                EventMetric(
                    run_id=run_id,
                    mode="merged_update",
                    sample_id=sample.sample_id,
                    query_id=sample.query_id,
                    source_id=MERGED_UPDATE_SOURCE_ID,
                    source_type="update",
                    event_type="update",
                    chunk_id=str(outbound_event["chunk_id"]),
                    sequence_no=sequence_no,
                    version=sequence_no,
                    scheduled_offset_ms=scheduled_offset * 1000,
                    actual_send_offset_ms=actual_send_offset_ms,
                    ack_latency_ms=None,
                    result_status="error",
                    error=str(exc),
                )
            )
            raise

        ack_latency_ms = (time.perf_counter() - ack_start_time) * 1000
        if should_log_event:
            logger.info(
                "Replay merged-update ACK event: run_id=%s request_id=%s "
                "sample_id=%s query_id=%s event_index=%d sequence_no=%d "
                "version=%d ack_latency_ms=%.3f",
                run_id,
                request_id,
                sample.sample_id,
                sample.query_id,
                sequence_no,
                sequence_no,
                sequence_no,
                ack_latency_ms,
            )

        event_metrics.append(
            EventMetric(
                run_id=run_id,
                mode="merged_update",
                sample_id=sample.sample_id,
                query_id=sample.query_id,
                source_id=MERGED_UPDATE_SOURCE_ID,
                source_type="update",
                event_type="update",
                chunk_id=str(outbound_event["chunk_id"]),
                sequence_no=sequence_no,
                version=sequence_no,
                scheduled_offset_ms=scheduled_offset * 1000,
                actual_send_offset_ms=actual_send_offset_ms,
                ack_latency_ms=ack_latency_ms,
                result_status="ok",
            )
        )
        sequence_no += 1
        num_merged_update_snapshots += 1

    finished_offset = last_scheduled_offset + args.source_finish_delay
    source_finished_event = _make_source_finished_event(sequence_no)
    should_log_finished = should_log_event_progress(args, sequence_no)
    finished_target_time = source_dispatch_start_time + finished_offset
    if should_log_finished:
        wait_ms = max(0.0, finished_target_time - time.perf_counter()) * 1000
        logger.info(
            "Replay merged-update WAIT source_finished: run_id=%s "
            "request_id=%s sample_id=%s query_id=%s event_index=%d "
            "sequence_no=%d scheduled_offset_ms=%.3f wait_ms=%.3f",
            run_id,
            request_id,
            sample.sample_id,
            sample.query_id,
            sequence_no,
            sequence_no,
            finished_offset * 1000,
            wait_ms,
        )

    await _sleep_until(finished_target_time)
    actual_send_time = time.perf_counter()
    actual_send_offset_ms = (
        actual_send_time - source_dispatch_start_time
    ) * 1000
    if should_log_finished:
        logger.info(
            "Replay merged-update SEND source_finished: run_id=%s "
            "request_id=%s sample_id=%s query_id=%s event_index=%d "
            "sequence_no=%d actual_send_offset_ms=%.3f",
            run_id,
            request_id,
            sample.sample_id,
            sample.query_id,
            sequence_no,
            sequence_no,
            actual_send_offset_ms,
        )

    ack_start_time = time.perf_counter()
    try:
        await _send_prompt_stream_event(
            client,
            args.proxy_url,
            request_id,
            source_finished_event,
        )
    except Exception as exc:
        if should_log_finished:
            logger.warning(
                "Replay merged-update ERROR source_finished: run_id=%s "
                "request_id=%s sample_id=%s query_id=%s event_index=%d "
                "sequence_no=%d error=%s",
                run_id,
                request_id,
                sample.sample_id,
                sample.query_id,
                sequence_no,
                sequence_no,
                exc,
            )
        event_metrics.append(
            EventMetric(
                run_id=run_id,
                mode="merged_update",
                sample_id=sample.sample_id,
                query_id=sample.query_id,
                source_id=MERGED_UPDATE_SOURCE_ID,
                source_type="update",
                event_type="source_finished",
                chunk_id=str(source_finished_event["chunk_id"]),
                sequence_no=sequence_no,
                version=int(source_finished_event["version"]),
                scheduled_offset_ms=finished_offset * 1000,
                actual_send_offset_ms=actual_send_offset_ms,
                ack_latency_ms=None,
                result_status="error",
                error=str(exc),
            )
        )
        raise

    ack_latency_ms = (time.perf_counter() - ack_start_time) * 1000
    if should_log_finished:
        logger.info(
            "Replay merged-update ACK source_finished: run_id=%s "
            "request_id=%s sample_id=%s query_id=%s event_index=%d "
            "sequence_no=%d ack_latency_ms=%.3f",
            run_id,
            request_id,
            sample.sample_id,
            sample.query_id,
            sequence_no,
            sequence_no,
            ack_latency_ms,
        )

    event_metrics.append(
        EventMetric(
            run_id=run_id,
            mode="merged_update",
            sample_id=sample.sample_id,
            query_id=sample.query_id,
            source_id=MERGED_UPDATE_SOURCE_ID,
            source_type="update",
            event_type="source_finished",
            chunk_id=str(source_finished_event["chunk_id"]),
            sequence_no=sequence_no,
            version=int(source_finished_event["version"]),
            scheduled_offset_ms=finished_offset * 1000,
            actual_send_offset_ms=actual_send_offset_ms,
            ack_latency_ms=ack_latency_ms,
            result_status="ok",
        )
    )

    append_chars = len("".join(append_text_parts))
    final_update_chars = sum(len(text) for text in update_text_by_source.values())
    final_question = str(sample.finish_event.get("text") or "")
    normal_prompt_chars = (
        len(str(sample.start_request["prompt"]))
        + append_chars
        + final_update_chars
        + len(final_question)
    )
    return (
        event_metrics,
        num_logical_append_chunks,
        num_merged_update_snapshots,
        append_chars,
        final_update_chars,
        normal_prompt_chars,
    )


async def _replay_merged_update_sample(
    client: httpx.AsyncClient,
    sample: SampleReplayInput,
    args: argparse.Namespace,
    run_id: str,
    planned_arrival_offset_ms: float,
    queue_delay_ms: float,
) -> RequestMetric:
    """回放一个“Append+Update 合并成单 Update 源”的 baseline 请求。"""
    request_start_time = time.perf_counter()
    metric = RequestMetric(
        run_id=run_id,
        mode="merged_update",
        sample_id=sample.sample_id,
        query_id=sample.query_id,
        sample_status=sample.status,
        result_status="error",
        planned_arrival_offset_ms=planned_arrival_offset_ms,
        queue_delay_ms=queue_delay_ms,
        request_start_wall_time=time.time(),
    )
    request_id: str | None = None
    try:
        external_request_id = (
            f"replay-{run_id}-{sample.sample_id}-{uuid.uuid4().hex[:8]}"
        )
        start_payload = _make_merged_update_start_payload(
            sample,
            args,
            external_request_id,
        )
        start_ack_start_time = time.perf_counter()
        start_response = await _post_json(
            client,
            _normalize_v1_url(args.proxy_url, "/v1/prompt_stream/start"),
            start_payload,
        )
        metric.start_ack_ms = (time.perf_counter() - start_ack_start_time) * 1000
        request_id_value = start_response.get("request_id")
        if not isinstance(request_id_value, str):
            raise ReplayHttpError("Prompt-stream start response has no request_id.")
        request_id = request_id_value
        logger.info(
            "Replay merged-update START ack: run_id=%s request_id=%s "
            "sample_id=%s query_id=%s source_count=%d",
            run_id,
            request_id,
            sample.sample_id,
            sample.query_id,
            len(sample.source_streams),
        )

        source_dispatch_start_time = time.perf_counter()
        (
            event_metrics,
            num_append_chunks,
            num_update_snapshots,
            append_chars,
            final_update_chars,
            normal_prompt_chars,
        ) = await _dispatch_merged_update_events(
            client=client,
            sample=sample,
            args=args,
            request_id=request_id,
            run_id=run_id,
            source_dispatch_start_time=source_dispatch_start_time,
        )
        metric.source_dispatch_ms = (
            time.perf_counter() - source_dispatch_start_time
        ) * 1000
        metric.event_metrics = event_metrics
        metric.num_event_acks = len(event_metrics)
        metric.num_append_chunks = num_append_chunks
        metric.num_update_snapshots = num_update_snapshots
        metric.append_chars = append_chars
        metric.final_update_chars = final_update_chars
        metric.normal_prompt_chars = normal_prompt_chars

        finish_event = dict(sample.finish_event)
        finish_event["arrival_time"] = time.time()
        finish_payload = {
            "request_id": request_id,
            "event": finish_event,
        }
        logger.info(
            "Replay merged-update FINISH send: run_id=%s request_id=%s "
            "sample_id=%s query_id=%s num_event_acks=%d",
            run_id,
            request_id,
            sample.sample_id,
            sample.query_id,
            metric.num_event_acks,
        )
        finish_sent_time = time.perf_counter()
        async with client.stream(
            "POST",
            _normalize_v1_url(args.proxy_url, _prompt_stream_finish_endpoint(args)),
            json=finish_payload,
        ) as finish_response:
            if finish_response.is_error:
                error_body = await finish_response.aread()
                raise ReplayHttpError(
                    "HTTP "
                    f"{finish_response.status_code} from merged-update finish: "
                    f"{error_body.decode('utf-8', errors='replace')}"
                )
            sse_measurement = await consume_sse_response(
                finish_response,
                finish_sent_time,
                args.save_responses,
                args.ttft_only,
                args.sse_timeout,
            )

        metric.transport_status = "ok"
        metric.first_token_received = True
        metric.formal_result_status = "not_evaluated"
        metric.request_error_stage = None

        metric.finish_to_first_token_ms = (
            sse_measurement.first_token_time - finish_sent_time
        ) * 1000
        metric.request_to_first_token_ms = (
            sse_measurement.first_token_time - request_start_time
        ) * 1000
        metric.finish_to_done_ms = (
            sse_measurement.completed_time - finish_sent_time
        ) * 1000
        metric.request_to_done_ms = (
            sse_measurement.completed_time - request_start_time
        ) * 1000
        metric.ttft_origin = sse_measurement.first_token_origin
        metric.completion_chars = sse_measurement.completion_chars
        metric.completion_chunks = sse_measurement.completion_chunks
        metric.response_payloads = list(sse_measurement.payloads)
        metric.result_status = "ok"
        logger.info(
            "Replay merged-update FINISH ack: run_id=%s request_id=%s "
            "sample_id=%s query_id=%s ttft_ms=%.3f origin=%s",
            run_id,
            request_id,
            sample.sample_id,
            sample.query_id,
            metric.finish_to_first_token_ms,
            metric.ttft_origin,
        )
    except asyncio.CancelledError:
        if request_id is not None:
            await asyncio.shield(
                _abort_prompt_stream(client, args.proxy_url, request_id)
            )
        raise
    except Exception as exc:
        logger.warning(
            "Replay merged-update REQUEST error: run_id=%s request_id=%s "
            "sample_id=%s query_id=%s error=%s",
            run_id,
            request_id,
            sample.sample_id,
            sample.query_id,
            exc,
        )
        metric.transport_status = "error"
        metric.result_status = "error"
        metric.request_error_stage = "prompt_stream"
        metric.error = str(exc)
        if request_id is not None and not metric.first_token_received:
            await _abort_prompt_stream(client, args.proxy_url, request_id)
    return metric


if __name__ == "__main__":
    main("merged_update", _replay_merged_update_sample)
