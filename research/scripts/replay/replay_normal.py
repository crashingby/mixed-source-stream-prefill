# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""通过 proxy 的普通 completions RESTFUL API 回放 mixed-source 数据。"""

from __future__ import annotations

import argparse
import time
from typing import Any

import httpx
from replay_common import (
    ReplayHttpError,
    RequestMetric,
    SampleReplayInput,
    _build_normal_prompt,
    _get_data_events,
    _normalize_v1_url,
    _override_sampling_params,
    consume_sse_response,
    main,
)


def _make_normal_payload(
    sample: SampleReplayInput,
    args: argparse.Namespace,
) -> tuple[dict[str, Any], int, int]:
    """构造 proxy 普通 `/v1/completions` SSE 请求与内容统计。"""
    prompt, append_chars, final_update_chars = _build_normal_prompt(sample)
    payload: dict[str, Any] = {
        "model": args.model,
        "prompt": prompt,
        "stream": True,
        "max_tokens": sample.start_request.get("max_tokens", 64),
        "temperature": sample.start_request.get("temperature", 0.0),
    }
    _override_sampling_params(payload, args)
    return payload, append_chars, final_update_chars


async def _replay_normal_sample(
    client: httpx.AsyncClient,
    sample: SampleReplayInput,
    args: argparse.Namespace,
    run_id: str,
    planned_arrival_offset_ms: float,
    queue_delay_ms: float,
) -> RequestMetric:
    """通过 proxy 普通 `/v1/completions` 回放最终逻辑 prompt 并计量 TTFT。"""
    request_start_time = time.perf_counter()
    normal_payload, append_chars, final_update_chars = _make_normal_payload(
        sample,
        args,
    )
    metric = RequestMetric(
        run_id=run_id,
        mode="normal",
        sample_id=sample.sample_id,
        query_id=sample.query_id,
        sample_status=sample.status,
        result_status="error",
        planned_arrival_offset_ms=planned_arrival_offset_ms,
        queue_delay_ms=queue_delay_ms,
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
        normal_prompt_chars=len(str(normal_payload["prompt"])),
    )
    try:
        # 普通模式的 TTFT 起点是整段 prompt 的 completions 请求提交时刻。
        async with client.stream(
            "POST",
            _normalize_v1_url(args.proxy_url, "/v1/completions"),
            json=normal_payload,
        ) as normal_response:
            if normal_response.is_error:
                error_body = await normal_response.aread()
                raise ReplayHttpError(
                    f"HTTP {normal_response.status_code} from normal: "
                    f"{error_body.decode('utf-8', errors='replace')}"
                )
            sse_measurement = await consume_sse_response(
                normal_response,
                request_start_time,
                args.save_responses,
                args.ttft_only,
                args.sse_timeout,
            )

        metric.finish_to_first_token_ms = (
            sse_measurement.first_token_time - request_start_time
        ) * 1000
        metric.request_to_first_token_ms = metric.finish_to_first_token_ms
        metric.finish_to_done_ms = (
            sse_measurement.completed_time - request_start_time
        ) * 1000
        metric.request_to_done_ms = metric.finish_to_done_ms
        metric.ttft_origin = "normal_decode"
        metric.completion_chars = sse_measurement.completion_chars
        metric.completion_chunks = sse_measurement.completion_chunks
        metric.response_payloads = list(sse_measurement.payloads)
        metric.result_status = "ok"
    except Exception as exc:
        metric.error = str(exc)
    return metric


if __name__ == "__main__":
    main("normal", _replay_normal_sample)
