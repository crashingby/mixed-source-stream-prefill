# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""通过 prompt-stream RESTFUL API 回放 mixed-source 数据。"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
import time
import uuid
from pathlib import Path
from typing import Any

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

from research.scripts.replay.common.dispatch import (
    abort_prompt_stream as _abort_prompt_stream,
)
from research.scripts.replay.common.dispatch import metric_mode as _metric_mode
from research.scripts.replay.common.dispatch import (
    send_source_events as _send_source_events,
)
from research.scripts.replay.common.metrics import apply_completion_measurement
from research.scripts.replay.common.metrics import (
    apply_prompt_stream_stats as apply_prompt_stream_stats,
)
from research.scripts.replay.common.models import EventMetric as EventMetric
from research.scripts.replay.common.models import (
    RenderedReplayPrompt as RenderedReplayPrompt,
)
from research.scripts.replay.common.models import RequestMetric as RequestMetric
from research.scripts.replay.common.models import SampleReplayInput as SampleReplayInput
from research.scripts.replay.common.models import SourceEventStream as SourceEventStream
from research.scripts.replay.common.models import SseMeasurement as SseMeasurement
from research.scripts.replay.common.prompt import (
    ChatTemplateRenderer as ChatTemplateRenderer,
)
from research.scripts.replay.common.prompt import (
    override_sampling_params as _override_sampling_params,
)
from research.scripts.replay.common.runner import main as main
from research.scripts.replay.common.timeline import (
    should_log_event_progress as should_log_event_progress,
)
from research.scripts.replay.common.transport import ReplayHttpError as ReplayHttpError
from research.scripts.replay.common.transport import (
    consume_sse_response as consume_sse_response,
)
from research.scripts.replay.common.transport import (
    normalize_v1_url as _normalize_v1_url,
)
from research.scripts.replay.common.transport import post_json as _post_json
from research.scripts.replay.common.transport import (
    prompt_stream_finish_endpoint as _prompt_stream_finish_endpoint,
)
from research.scripts.replay.common.transport import (
    prompt_stream_start_endpoint as _prompt_stream_start_endpoint,
)

logger = logging.getLogger(__name__)


def _make_stream_start_payload(
    sample: SampleReplayInput,
    rendered_prompt: RenderedReplayPrompt,
    args: argparse.Namespace,
    external_request_id: str,
) -> dict[str, Any]:
    """构造 proxy `/v1/prompt_stream/start` 请求。

    ``stream=true`` 要求 proxy 使用 SSE：先通知 P 已完成，再发送 D 回答。
    P-only 和完整生成分别以完成通知、D 首 token 作为计时终点。
    """
    payload = dict(sample.start_request)
    payload.update(
        {
            "api": "completions",
            "model": args.model,
            "prompt": (
                rendered_prompt.start_text
                if rendered_prompt.input_transport == "text"
                else list(rendered_prompt.start_token_ids)
            ),
            "stream": True,
            "add_special_tokens": False,
            "request_id": external_request_id,
        }
    )
    _override_sampling_params(payload, args)
    if args.return_token_ids:
        # D 节点必须返回每个 SSE chunk 的增量 token_ids，正式 TPS 才能
        # 使用真实 token 数，而不是把 chunk 数误当成 token 数。
        payload["return_token_ids"] = True
    return payload


def _apply_masked_padding_measurement(
    metric: RequestMetric,
    measurement: SseMeasurement,
    expected_mode: str,
) -> None:
    """记录并校验一次请求观察到的 masked-padding 模式。

    Args:
        metric: 当前请求的逐请求指标。
        measurement: SSE 消费阶段提取出的模式清单与 token 时刻。
        expected_mode: CLI 声明的 ``enabled``、``disabled`` 或 ``any``。

    Raises:
        ReplayHttpError: 实际 P 节点模式与实验标签不一致时抛出。
    """
    manifest = measurement.masked_padding_manifest
    observed_enabled = manifest is not None
    metric.masked_padding_expected = expected_mode
    metric.masked_padding_observed = observed_enabled
    if manifest is not None:
        metric.masked_padding_tokens = manifest["end"] - manifest["start"]

    if expected_mode == "any":
        metric.masked_padding_validation_status = "not_evaluated"
        return

    expected_enabled = expected_mode == "enabled"
    if observed_enabled != expected_enabled:
        metric.masked_padding_validation_status = "mismatch"
        raise ReplayHttpError(
            "P-node masked-padding mode does not match the experiment: "
            f"expected={expected_mode}, observed={observed_enabled}."
        )
    metric.masked_padding_validation_status = "valid"


async def _replay_stream_sample(
    client: httpx.AsyncClient,
    sample: SampleReplayInput,
    rendered_prompt: RenderedReplayPrompt,
    renderer: ChatTemplateRenderer,
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
        append_chars=rendered_prompt.append_chars,
        final_update_chars=rendered_prompt.final_update_chars,
        normal_prompt_chars=len(rendered_prompt.logical_prompt_text),
    )
    request_id: str | None = None
    finish_sent_time: float | None = None
    request_stage = "start"
    try:
        external_request_id = (
            f"replay-{run_id}-{sample.sample_id}-{uuid.uuid4().hex[:8]}"
        )
        start_payload = _make_stream_start_payload(
            sample,
            rendered_prompt,
            args,
            external_request_id,
        )
        start_ack_start_time = time.perf_counter()
        start_response = await _post_json(
            client,
            _normalize_v1_url(
                args.proxy_url,
                _prompt_stream_start_endpoint(args),
            ),
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

        request_stage = "source_dispatch"
        source_dispatch_start_time = time.perf_counter()
        source_tasks = [
            asyncio.create_task(
                _send_source_events(
                    client=client,
                    proxy_url=args.proxy_url,
                    request_id=request_id,
                    sample=sample,
                    renderer=renderer,
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
        finish_event["text"] = rendered_prompt.finish_text
        finish_event.pop("token_ids", None)
        if rendered_prompt.input_transport == "token_ids":
            finish_event["token_ids"] = list(rendered_prompt.finish_token_ids)
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
        request_stage = "finish_sse"
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

        _apply_masked_padding_measurement(
            metric,
            sse_measurement,
            args.expected_masked_padding,
        )

        # 首 token 已经到达后，后置 stats 校验不能把网络成功改写成失败。
        metric.transport_status = "ok"
        metric.first_token_received = True
        metric.result_status = "ok"
        metric.request_error_stage = None

        metric.finish_to_first_token_ms = (
            sse_measurement.ttft_time - finish_sent_time
        ) * 1000
        metric.request_to_first_token_ms = (
            sse_measurement.ttft_time - request_start_time
        ) * 1000
        metric.finish_to_done_ms = (
            sse_measurement.completed_time - finish_sent_time
        ) * 1000
        metric.request_to_done_ms = (
            sse_measurement.completed_time - request_start_time
        ) * 1000
        metric.ttft_origin = sse_measurement.ttft_origin
        apply_completion_measurement(metric, sse_measurement, finish_sent_time)
        apply_prompt_stream_stats(
            metric,
            sse_measurement.prompt_stream_stats,
            num_sent_update_events=metric.num_update_snapshots,
            require_stats=args.require_prompt_stream_stats,
            expected_coalescing=args.expected_coalescing,
            expected_config_json=args.p_node_coalescing_config_json,
        )
        if metric.stats_validation_status == "invalid":
            logger.warning(
                "Replay stream STATS invalid: run_id=%s request_id=%s "
                "sample_id=%s error=%s",
                run_id,
                request_id,
                sample.sample_id,
                metric.stats_validation_error,
            )
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
        metric.transport_status = "error"
        metric.result_status = "error"
        metric.request_error_stage = request_stage
        metric.error = str(exc)
        error_completed_time = time.perf_counter()
        metric.request_to_done_ms = (error_completed_time - request_start_time) * 1000
        if finish_sent_time is not None:
            metric.finish_to_done_ms = (error_completed_time - finish_sent_time) * 1000
        if request_id is not None and not metric.first_token_received:
            await _abort_prompt_stream(client, args.proxy_url, request_id)
    return metric


if __name__ == "__main__":
    main("stream", _replay_stream_sample)
