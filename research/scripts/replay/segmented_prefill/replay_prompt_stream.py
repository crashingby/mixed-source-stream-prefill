# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""通过 prompt-stream RESTFUL API 回放 mixed-source 数据。"""

from __future__ import annotations

import argparse
import asyncio
import logging
import time
import uuid
from typing import Any

import httpx
from replay_common import (
    EventMetric,
    ReplayHttpError,
    RequestMetric,
    SampleReplayInput,
    SourceEventStream,
    apply_prompt_stream_stats,
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


def _make_stream_start_payload(
    sample: SampleReplayInput,
    args: argparse.Namespace,
    external_request_id: str,
) -> dict[str, Any]:
    """构造 proxy `/v1/prompt_stream/start` 请求。

    ``stream=true`` 要求 proxy 使用新增 SSE finish 分支：P 的首 token 会在
    D decode 完成前立即送回，因此能够精确测量 finish-to-first-token。
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
    """返回一个来源的首 chunk 延迟与后续数据 chunk 间隔。"""
    if source_type == "append":
        return args.append_initial_delay, args.append_chunk_interval
    if source_type == "update":
        return args.update_initial_delay, args.update_chunk_interval
    raise ValueError(f"Unsupported source_type={source_type!r}.")


def _longest_common_prefix_len(
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


def _metric_mode(args: argparse.Namespace) -> str:
    """按实验套件返回稳定的请求与事件指标标签。"""
    if args.experiment_suite == "update_coalescing":
        return "update_coalescing"
    return "stream"


async def _send_source_events(
    client: httpx.AsyncClient,
    proxy_url: str,
    request_id: str,
    sample: SampleReplayInput,
    source_stream: SourceEventStream,
    source_dispatch_start_time: float,
    args: argparse.Namespace,
    run_id: str,
) -> list[EventMetric]:
    """按一个来源的 source-local 顺序发送所有事件并等待每次 ack。

    每个来源在独立 asyncio task 内运行。因此 Append 和 Update 可以并发抵达
    scheduler；同一来源严格串行，保证 Update 版本与 Append 序号不乱序。
    """
    initial_delay, chunk_interval = _source_timing(source_stream.source_type, args)
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
        actual_send_offset_ms = (
            actual_send_time - source_dispatch_start_time
        ) * 1000
        outbound_event = dict(stored_event)
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
                    mode=_metric_mode(args),
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
                    update_lcp_tokens = _longest_common_prefix_len(
                        previous_update_token_ids,
                        event_token_ids,
                    )
                    update_lcp_ratio = update_lcp_tokens / max(
                        1, num_event_tokens
                    )
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
                mode=_metric_mode(args),
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


async def _replay_stream_sample(
    client: httpx.AsyncClient,
    sample: SampleReplayInput,
    args: argparse.Namespace,
    run_id: str,
    planned_arrival_offset_ms: float,
    queue_delay_ms: float,
) -> RequestMetric:
    """通过 proxy 完成一个 mixed-source stream 回放并计量 finish TTFT。"""
    request_start_time = time.perf_counter()
    metric = RequestMetric(
        run_id=run_id,
        mode=_metric_mode(args),
        sample_id=sample.sample_id,
        query_id=sample.query_id,
        sample_status=sample.status,
        result_status="error",
        planned_arrival_offset_ms=planned_arrival_offset_ms,
        queue_delay_ms=queue_delay_ms,
        request_start_wall_time=time.time(),
    )
    request_id: str | None = None
    finish_sent_time: float | None = None
    try:
        external_request_id = (
            f"replay-{run_id}-{sample.sample_id}-{uuid.uuid4().hex[:8]}"
        )
        start_payload = _make_stream_start_payload(sample, args, external_request_id)
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
            "Replay stream START ack: run_id=%s request_id=%s "
            "sample_id=%s query_id=%s source_count=%d",
            run_id,
            request_id,
            sample.sample_id,
            sample.query_id,
            len(sample.source_streams),
        )

        source_dispatch_start_time = time.perf_counter()
        source_tasks = [
            asyncio.create_task(
                _send_source_events(
                    client=client,
                    proxy_url=args.proxy_url,
                    request_id=request_id,
                    sample=sample,
                    source_stream=source_stream,
                    source_dispatch_start_time=source_dispatch_start_time,
                    args=args,
                    run_id=run_id,
                )
            )
            for source_stream in sample.source_streams
        ]
        try:
            source_event_results = await asyncio.gather(*source_tasks)
        except BaseException:
            # gather 默认不会取消兄弟 task。若一个来源失败而其它来源继续向
            # 已 abort 的长期 Request 发事件，会制造额外 404 和生命周期噪声。
            for source_task in source_tasks:
                if not source_task.done():
                    source_task.cancel()
            await asyncio.gather(*source_tasks, return_exceptions=True)
            raise
        metric.source_dispatch_ms = (
            time.perf_counter() - source_dispatch_start_time
        ) * 1000
        metric.event_metrics = [
            event_metric
            for source_result in source_event_results
            for event_metric in source_result
        ]
        metric.num_event_acks = len(metric.event_metrics)
        metric.num_append_chunks = sum(
            event.event_type == "append" for event in metric.event_metrics
        )
        metric.num_update_snapshots = sum(
            event.event_type == "update" for event in metric.event_metrics
        )
        update_lcp_ratios = [
            event.update_lcp_ratio
            for event in metric.event_metrics
            if event.update_lcp_ratio is not None
        ]
        if update_lcp_ratios:
            metric.mean_update_lcp_ratio = sum(update_lcp_ratios) / len(
                update_lcp_ratios
            )

        finish_event = dict(sample.finish_event)
        finish_event["arrival_time"] = time.time()
        finish_payload = {
            "request_id": request_id,
            "event": finish_event,
        }
        # TTFT 的时间零点必须紧邻 finish POST 提交动作。
        logger.info(
            "Replay stream FINISH send: run_id=%s request_id=%s "
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
                    f"HTTP {finish_response.status_code} from prompt-stream finish: "
                    f"{error_body.decode('utf-8', errors='replace')}"
                )
            sse_measurement = await consume_sse_response(
                finish_response,
                finish_sent_time,
                args.save_responses,
                args.ttft_only,
                args.sse_timeout,
            )

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
        apply_prompt_stream_stats(
            metric,
            sse_measurement.prompt_stream_stats,
            num_sent_update_events=metric.num_update_snapshots,
            require_stats=args.require_prompt_stream_stats,
            expected_coalescing=args.expected_coalescing,
            expected_config_json=args.p_node_coalescing_config_json,
        )
        metric.result_status = "ok"
        logger.info(
            "Replay stream FINISH ack: run_id=%s request_id=%s "
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
            # fail-fast 取消其它逻辑请求时，也要清理已经 start 的 P 长期请求。
            await asyncio.shield(
                _abort_prompt_stream(client, args.proxy_url, request_id)
            )
        raise
    except Exception as exc:
        logger.warning(
            "Replay stream REQUEST error: run_id=%s request_id=%s "
            "sample_id=%s query_id=%s error=%s",
            run_id,
            request_id,
            sample.sample_id,
            sample.query_id,
            exc,
        )
        metric.error = str(exc)
        error_completed_time = time.perf_counter()
        metric.request_to_done_ms = (
            error_completed_time - request_start_time
        ) * 1000
        if finish_sent_time is not None:
            metric.finish_to_done_ms = (
                error_completed_time - finish_sent_time
            ) * 1000
        if request_id is not None:
            await _abort_prompt_stream(client, args.proxy_url, request_id)
    return metric


if __name__ == "__main__":
    main("stream", _replay_stream_sample)
