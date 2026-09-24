# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""通过 proxy 的普通 completions RESTFUL API 回放 mixed-source 数据。"""

from __future__ import annotations

import argparse
import time
from typing import Any

import httpx
from replay_common import (
    ChatTemplateRenderer,
    RenderedReplayPrompt,
    ReplayHttpError,
    RequestMetric,
    SampleReplayInput,
    _get_data_events,
    _normalize_v1_url,
    _override_sampling_params,
    consume_sse_response,
    main,
)


def _make_normal_payload(
    sample: SampleReplayInput,
    rendered_prompt: RenderedReplayPrompt,
    args: argparse.Namespace,
) -> dict[str, Any]:
    """构造由分段 chat template token 组成的普通 SSE 请求。"""
    payload: dict[str, Any] = {
        "model": args.model,
        "prompt": list(rendered_prompt.logical_prompt_token_ids),
        "stream": True,
        "max_tokens": sample.start_request.get("max_tokens", 64),
        "temperature": sample.start_request.get("temperature", 0.0),
    }
    _override_sampling_params(payload, args)
    return payload


def _normal_completion_endpoint(args: argparse.Namespace) -> str:
    """返回普通 PD replay 当前应调用的 completions 路由。

    常规运行使用完整的 P→D 分离路径。TTFT-only 只需要 P 节点对完整 prompt
    做 prefill 并产生一个输出 token，因此调用 proxy 的专用路由，避免启动 D
    节点的 KV load 或后续 decode。

    Args:
        args: 已解析的 replay 命令行参数。

    Returns:
        proxy 上的普通 PD completions endpoint。
    """
    if args.ttft_only:
        return "/v1/completions_ttft"
    return "/v1/completions"


async def _replay_normal_sample(
    client: httpx.AsyncClient,
    sample: SampleReplayInput,
    rendered_prompt: RenderedReplayPrompt,
    _renderer: ChatTemplateRenderer,
    args: argparse.Namespace,
    run_id: str,
    planned_arrival_offset_ms: float,
    queue_delay_ms: float,
) -> RequestMetric:
    """通过 proxy 普通 P/D 分离回放最终逻辑 prompt 并计量 TTFT。"""
    request_start_time = time.perf_counter()
    normal_payload = _make_normal_payload(sample, rendered_prompt, args)
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
        append_chars=rendered_prompt.append_chars,
        final_update_chars=rendered_prompt.final_update_chars,
        normal_prompt_chars=len(rendered_prompt.logical_prompt_text),
    )
    try:
        # 普通模式的 TTFT 起点是整段 prompt 的 normal PD 请求提交时刻。
        # --ttft-only 对应 P-only 路由，首 token 不会再触发 D 的 KV load。
        async with client.stream(
            "POST",
            _normalize_v1_url(
                args.proxy_url,
                _normal_completion_endpoint(args),
            ),
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

        metric.transport_status = "ok"
        metric.first_token_received = True
        metric.formal_result_status = "not_evaluated"
        metric.request_error_stage = None

        metric.finish_to_first_token_ms = (
            sse_measurement.first_token_time - request_start_time
        ) * 1000
        metric.request_to_first_token_ms = metric.finish_to_first_token_ms
        metric.finish_to_done_ms = (
            sse_measurement.completed_time - request_start_time
        ) * 1000
        metric.request_to_done_ms = metric.finish_to_done_ms
        metric.ttft_origin = "normal_prefill"
        metric.completion_chars = sse_measurement.completion_chars
        metric.completion_chunks = sse_measurement.completion_chunks
        metric.response_payloads = list(sse_measurement.payloads)
        metric.result_status = "ok"
    except Exception as exc:
        metric.transport_status = "error"
        metric.result_status = "error"
        metric.request_error_stage = "completion_sse"
        metric.error = str(exc)
    return metric


if __name__ == "__main__":
    main("normal", _replay_normal_sample)
