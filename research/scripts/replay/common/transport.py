# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""HTTP 请求与完整 SSE 的解析。"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import time
from collections.abc import AsyncIterator
from typing import Any

import httpx

from .models import SseMeasurement

logger = logging.getLogger(__name__)


class ReplayHttpError(RuntimeError):
    """包装 HTTP/SSE 上游错误，并保留有助定位的响应正文。"""


class LegacyStatsSemanticsIncompatible(ReplayHttpError):
    """表示 schema v1 无法表达抢占后的 recovery lifecycle。"""


def normalize_v1_url(base_url: str, endpoint: str) -> str:
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


def prompt_stream_finish_endpoint(args: argparse.Namespace) -> str:
    """返回当前 prompt-stream replay 应使用的 finish 路径。

    TTFT-only 实验只需要 P 节点 finish 后的首 token。此时走 proxy 的
    `/v1/prompt_stream/finish_ttft`，避免 proxy 再启动 D 节点 decode 并触发
    不必要的 KV transfer。完整生成或非 TTFT-only 调试仍走原 `/finish`。
    """
    if getattr(args, "ttft_only", False):
        return "/v1/prompt_stream/finish_ttft"
    return "/v1/prompt_stream/finish"


def prompt_stream_start_endpoint(args: argparse.Namespace) -> str:
    """返回当前 prompt-stream replay 应使用的 start 路径。

    TTFT-only 必须从会话创建时就不携带 ``kv_transfer_params``；若仍使用
    普通 ``/start``，P 会错误地把该请求登记为未来由 D 拉取 KV 的 producer。

    Args:
        args: 回放命令行参数。

    Returns:
        proxy 的完整 P/D 或严格 P-only start endpoint。
    """
    if getattr(args, "ttft_only", False):
        return "/v1/prompt_stream/start_ttft"
    return "/v1/prompt_stream/start"


async def post_json(
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


def is_first_token_choice(choice: dict[str, Any]) -> bool:
    """判断一个 OpenAI SSE choice 是否实际代表一个生成 token。"""
    if (
        "token_ids" in choice
        and isinstance(choice["token_ids"], list)
        and choice["token_ids"]
    ):
        return True
    if "text" in choice:
        # 仅有 final finish_reason 的空 completion chunk 不应伪装成首 token。
        return choice.get("finish_reason") is None or bool(choice.get("text"))
    delta = choice.get("delta")
    if isinstance(delta, dict) and "content" in delta:
        return delta.get("content") is not None
    return False


def chunk_text(choice: dict[str, Any]) -> str:
    """从 completions/chat-completions 兼容 choice 提取文本。"""
    text = choice.get("text")
    if isinstance(text, str):
        return text
    delta = choice.get("delta")
    if isinstance(delta, dict) and isinstance(delta.get("content"), str):
        return delta["content"]
    return ""


async def consume_sse_body(
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
    completion_tokens = 0
    token_count_is_exact = True
    saw_token_id_metadata = False
    last_token_time: float | None = None
    payloads: list[dict[str, Any]] = []
    prompt_stream_stats: dict[str, Any] | None = None
    masked_padding_manifest: dict[str, int] | None = None
    async for payload in iter_sse_payloads(response):
        if save_payloads:
            payloads.append(payload)
        if "error" in payload:
            raise ReplayHttpError(f"Upstream SSE error: {payload['error']!r}")
        raw_stats = payload.get("prompt_stream_stats")
        if raw_stats is not None:
            if not isinstance(raw_stats, dict):
                raise ReplayHttpError(
                    "SSE prompt_stream_stats must be an object, "
                    f"but got {type(raw_stats).__name__}."
                )
            if prompt_stream_stats is not None and prompt_stream_stats != raw_stats:
                raise ReplayHttpError(
                    "One SSE response returned inconsistent prompt_stream_stats."
                )
            # 独立保存统计，避免 --ttft-only 且未保存 payload 时丢失首帧字段。
            prompt_stream_stats = dict(raw_stats)

        raw_padding = payload.get("prompt_stream_masked_padding")
        if raw_padding is not None:
            if (
                not isinstance(raw_padding, dict)
                or set(raw_padding) != {"version", "start", "end"}
                or any(
                    type(raw_padding[field_name]) is not int
                    for field_name in ("version", "start", "end")
                )
                or raw_padding["version"] != 1
                or not 0 <= raw_padding["start"] <= raw_padding["end"]
            ):
                raise ReplayHttpError(
                    "SSE prompt_stream_masked_padding must be a valid v1 range."
                )
            normalized_padding = {
                field_name: int(raw_padding[field_name])
                for field_name in ("version", "start", "end")
            }
            if (
                masked_padding_manifest is not None
                and masked_padding_manifest != normalized_padding
            ):
                raise ReplayHttpError(
                    "One SSE response returned inconsistent masked-padding manifests."
                )
            masked_padding_manifest = normalized_padding

        choices = payload.get("choices")
        if not isinstance(choices, list):
            continue
        for choice in choices:
            if not isinstance(choice, dict):
                continue
            if is_first_token_choice(choice):
                token_arrival_time = time.perf_counter()
                if first_token_time is None:
                    first_token_time = token_arrival_time
                    first_token_origin = payload.get("token_source") or (
                        "prefill" if "token_ids" in choice else "decode"
                    )
                completion_chunks += 1
                completion_chars += len(chunk_text(choice))
                raw_token_ids = choice.get("token_ids")
                if isinstance(raw_token_ids, list):
                    if not all(type(token_id) is int for token_id in raw_token_ids):
                        raise ReplayHttpError(
                            "Completion SSE token_ids must contain integers."
                        )
                    saw_token_id_metadata = True
                    completion_tokens += len(raw_token_ids)
                else:
                    # 未请求 token_ids 时只能以非空 SSE choice 近似一个 token；
                    # 该数值不得进入正式 TPS 结果。
                    token_count_is_exact = False
                    completion_tokens += 1
                last_token_time = token_arrival_time
                if ttft_only:
                    completed_time = time.perf_counter()
                    assert first_token_time is not None
                    assert last_token_time is not None
                    return SseMeasurement(
                        first_token_time=first_token_time,
                        first_token_origin=first_token_origin,
                        completed_time=completed_time,
                        completion_chars=completion_chars,
                        completion_chunks=completion_chunks,
                        completion_tokens=completion_tokens,
                        completion_token_count_exact=(
                            token_count_is_exact and saw_token_id_metadata
                        ),
                        last_token_time=last_token_time,
                        payloads=tuple(payloads),
                        prompt_stream_stats=prompt_stream_stats,
                        masked_padding_manifest=masked_padding_manifest,
                    )

    completed_time = time.perf_counter()
    if first_token_time is None:
        raise ReplayHttpError("SSE response ended before emitting a completion token.")
    if first_token_time < first_token_start_time:
        raise AssertionError("First token time cannot precede TTFT start time.")
    assert last_token_time is not None
    return SseMeasurement(
        first_token_time=first_token_time,
        first_token_origin=first_token_origin,
        completed_time=completed_time,
        completion_chars=completion_chars,
        completion_chunks=completion_chunks,
        completion_tokens=completion_tokens,
        completion_token_count_exact=(token_count_is_exact and saw_token_id_metadata),
        last_token_time=last_token_time,
        payloads=tuple(payloads),
        prompt_stream_stats=prompt_stream_stats,
        masked_padding_manifest=masked_padding_manifest,
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
            return await consume_sse_body(
                response,
                first_token_start_time,
                save_payloads,
                ttft_only,
            )
        async with asyncio.timeout(timeout_seconds):
            return await consume_sse_body(
                response,
                first_token_start_time,
                save_payloads,
                ttft_only,
            )
    except TimeoutError as exc:
        raise ReplayHttpError(
            f"Timed out after {timeout_seconds:.3f}s while consuming SSE response."
        ) from exc


def create_http_client(args: argparse.Namespace) -> httpx.AsyncClient:
    """为 PD 回放创建不继承外部代理、连接数足够的客户端。"""
    concurrency = getattr(args, "max_concurrency", 1)
    return httpx.AsyncClient(
        timeout=args.http_timeout,
        trust_env=False,
        limits=httpx.Limits(max_connections=max(8, concurrency * 3)),
    )


async def iter_sse_payloads(
    response: httpx.Response,
) -> AsyncIterator[dict[str, Any]]:
    """按 SSE 帧解析 JSON，支持多行 data；完整消费必须看到 DONE。

    首 token 模式由上层提前退出迭代。完整生成在 EOF 时必须拒绝断流，
    usage-only 帧与实验元数据保留原样，交给各实验观察器处理。
    """
    data_lines = []
    async for line in response.aiter_lines():
        if line.startswith("data:"):
            data_lines.append(line[5:].lstrip(" "))
        elif not line and data_lines:
            data = "\n".join(data_lines)
            data_lines.clear()
            if data.strip() == "[DONE]":
                return
            try:
                payload = json.loads(data)
            except json.JSONDecodeError as error:
                raise ReplayHttpError(f"Invalid SSE JSON payload: {data!r}") from error
            if not isinstance(payload, dict):
                raise ReplayHttpError("SSE data must encode a JSON object.")
            if payload.get("error"):
                raise ReplayHttpError(f"Upstream SSE error: {payload['error']!r}")
            yield payload
    raise ReplayHttpError("SSE response closed before [DONE].")
