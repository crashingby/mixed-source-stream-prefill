# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""最小前端共享的文本模板、来源屏障及 P-only/proxy 请求流程。"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import httpx

from research.scripts.replay.common.dispatch import abort_prompt_stream
from research.scripts.replay.common.prompt import ChatTemplateRenderer
from research.scripts.replay.common.transport import (
    chunk_text,
    iter_sse_payloads,
    normalize_v1_url,
    post_json,
)
from research.scripts.simple_test.prompt_stream_stats import (
    parse_prompt_stream_stats,
    print_prompt_stream_stats,
)

SourceType = Literal["append", "update"]


@dataclass(frozen=True)
class DemoEvent:
    """单次前端数据事件，Update 的 text 始终表示来源的完整当前快照。"""

    source_id: str
    source_type: SourceType
    text: str


def add_common_arguments(
    parser: argparse.ArgumentParser,
    prefix: str,
    final_question: str,
) -> None:
    """给三个最小前端添加一致参数，保留原有 --proxy-url 别名。"""
    parser.add_argument(
        "--server-url",
        "--proxy-url",
        dest="proxy_url",
        default="http://127.0.0.1:8000",
        help="P 或 proxy 地址，可带 /v1",
    )
    parser.add_argument("--target", choices=("p", "proxy"), default="proxy")
    parser.add_argument(
        "--ttft-only",
        action="store_true",
        help="proxy 只调用 P；直连 P 本来就只有首 token",
    )
    parser.add_argument("--model", default="qwen2.5-3b")
    parser.add_argument("--request-id")
    parser.add_argument(
        "--prefix", default=prefix, help="作为 system message 的固定说明"
    )
    parser.add_argument("--final-question", default=final_question)
    parser.add_argument(
        "--tokenizer-model", default=str(Path.home() / "Models/Qwen2.5-3B")
    )
    parser.add_argument(
        "--raw-prompt",
        action="store_true",
        help="直接使用原始 prefix/question，不应用 chat template",
    )
    parser.add_argument(
        "--max-tokens", type=int, default=64, help="完整 PD 的输出预算；P-only 强制为 1"
    )
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--event-delay", type=float, default=0.0)
    parser.add_argument("--http-timeout", type=float, default=120.0)
    parser.add_argument(
        "--expected-masked-padding",
        choices=("any", "enabled", "disabled"),
        default="any",
    )


def validate_arguments(args: argparse.Namespace, events: list[DemoEvent]) -> None:
    """在创建会话前拒绝非法参数、空数据和同 ID 的不同来源类型。"""
    if args.max_tokens < 1:
        raise ValueError("--max-tokens 必须为正整数")
    for name in ("event_delay", "temperature", "http_timeout"):
        value = getattr(args, name)
        if (
            not math.isfinite(value)
            or value < 0
            or (name == "http_timeout" and value == 0)
        ):
            raise ValueError(f"--{name.replace('_', '-')} 数值不合法：{value}")
    if not args.final_question.strip() or not events:
        raise ValueError("测试需要非空问题和至少一个数据事件")
    kinds: dict[str, SourceType] = {}
    for event in events:
        if not event.source_id or not event.text.strip():
            raise ValueError("来源 ID 与事件正文必须非空")
        if kinds.setdefault(event.source_id, event.source_type) != event.source_type:
            raise ValueError(f"Append 与 Update 不能共用来源 ID：{event.source_id}")


def print_json(title: str, payload: dict[str, Any]) -> None:
    """打印小规模测试的原始请求或响应，便于与服务日志对照。"""
    print(f"\n=== {title} ===")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


def data_envelope(
    request_id: str, event: DemoEvent, sequence_no: int
) -> dict[str, Any]:
    """为一个来源的数据事件附加连续序号及当前发送时间。"""
    return {
        "request_id": request_id,
        "event": {
            "source_id": event.source_id,
            "source_type": event.source_type,
            "event_type": event.source_type,
            "sequence_no": sequence_no,
            "version": sequence_no,
            "chunk_id": f"{event.source_id}-{sequence_no}",
            "text": event.text,
            "arrival_time": time.time(),
        },
    }


def ack_tokens(response: dict[str, Any]) -> list[int]:
    """读取服务端实际分词结果，客户端只校验和拼接 ACK，不调用 encode。"""
    tokens = response.get("token_ids")
    if not isinstance(tokens, list) or not all(type(token) is int for token in tokens):
        raise ValueError("start/数据 ACK 缺少整数 token_ids")
    return tokens


def padding_manifest(response: dict[str, Any], expected: str) -> dict[str, int] | None:
    """读取 P 或 proxy 的 mask 清单，并验证版本、范围和预期开关。"""
    params = response.get("kv_transfer_params") or {}
    padding = response.get("prompt_stream_masked_padding")
    nested = params.get("prompt_stream_masked_padding")
    if padding is not None and nested is not None and padding != nested:
        raise ValueError("顶层和 KV 元数据中的 padding 清单不一致")
    padding = padding if padding is not None else nested
    if padding is not None and (
        not isinstance(padding, dict)
        or set(padding) != {"version", "start", "end"}
        or any(type(value) is not int for value in padding.values())
        or padding["version"] != 1
        or not 0 <= padding["start"] <= padding["end"]
    ):
        raise ValueError(f"非法 padding 清单：{padding}")
    if expected == "enabled" and padding is None:
        raise ValueError("服务未返回预期的 masked-padding 清单")
    if expected == "disabled" and padding is not None:
        raise ValueError("服务返回了预期之外的 masked-padding 清单")
    return padding


def validate_prompt_tokens(
    response: dict[str, Any],
    prefix_tokens: list[int],
    source_tokens: dict[str, list[int]],
    source_types: dict[str, SourceType],
    padding: dict[str, int] | None,
) -> bool:
    """有实际 prompt IDs 时，校验最终输入仅含全部 Append 与最新 Update。"""
    physical = response.get("prompt_token_ids")
    if physical is None:
        return False  # 部分 proxy 响应不携带完整物理 token 视图。
    if not isinstance(physical, list) or not all(
        type(token) is int for token in physical
    ):
        raise ValueError("实际 prompt_token_ids 必须为整数数组")
    logical = list(physical)
    if padding is not None:
        if padding["end"] > len(physical):
            raise ValueError("padding 范围越出实际 prompt")
        logical = physical[: padding["start"]] + physical[padding["end"] :]
    expected = list(prefix_tokens)
    for kind in ("append", "update"):
        for source_id, tokens in source_tokens.items():
            if source_types[source_id] == kind:
                expected.extend(tokens)
    expected.extend(ack_tokens(response))
    if logical != expected:
        raise ValueError("实际 prompt 与 start/数据/finish ACK 不一致，或仍包含旧快照")
    return True


async def finish_request(
    client: httpx.AsyncClient,
    args: argparse.Namespace,
    path: str,
    payload: dict[str, Any],
) -> dict[str, Any]:
    """消费 P JSON、proxy JSON 或 proxy TTFT SSE，完整 SSE 必须有 DONE。"""
    async with client.stream(
        "POST", normalize_v1_url(args.proxy_url, path), json=payload
    ) as response:
        if response.is_error:
            body = (await response.aread()).decode(errors="replace")
            raise RuntimeError(f"finish HTTP {response.status_code}: {body}")
        if "text/event-stream" in response.headers.get("content-type", ""):
            # simple_test 完整读取控制流，断流即失败；P-only 也必须看到 DONE。
            frames = [frame async for frame in iter_sse_payloads(response)]
            result: dict[str, Any] = {
                "prefill_ready": any(
                    frame.get("event") == "prefill_ready" for frame in frames
                ),
                "output_text": "".join(
                    chunk_text(choice)
                    for frame in frames
                    for choice in frame.get("choices", [])
                ),
                "output_token_ids": [
                    token
                    for frame in frames
                    for choice in frame.get("choices", [])
                    for token in choice.get("token_ids", [])
                ],
            }
            for frame in frames:
                for name in ("prompt_stream_stats", "prompt_stream_masked_padding"):
                    if name in frame:
                        result[name] = frame[name]
            return result
        value = json.loads(await response.aread())
        if not isinstance(value, dict):
            raise ValueError("finish 响应必须是 JSON 对象")
        return value


async def run_demo(args: argparse.Namespace, events: list[DemoEvent]) -> dict[str, Any]:
    """运行并打印一次来源生命周期测试；异常和取消时中止已创建会话。

    Args:
        args: 公共 CLI 参数，target=p 直接调用用户现有 P 服务。
        events: 按发送顺序排列的数据；同源必须保持连续版本。

    Returns:
        已检查统计、mask 与可用 token 视图的 finish 响应。
    """
    validate_arguments(args, events)
    prefix, suffix = args.prefix, ""
    if not args.raw_prompt:
        renderer = ChatTemplateRenderer.from_pretrained(args.tokenizer_model)
        prefix, suffix = renderer.render_envelope(args.prefix.strip())
    p_only = args.target == "p" or args.ttft_only
    start_path = (
        "/v1/prompt_stream/start_ttft"
        if args.target == "proxy" and args.ttft_only
        else "/v1/prompt_stream/start"
    )
    finish_path = (
        "/v1/prompt_stream/finish_ttft"
        if args.target == "proxy" and args.ttft_only
        else "/v1/prompt_stream/finish"
    )
    start = {
        "api": "completions",
        "model": args.model,
        "prompt": prefix,
        "stream": False,
        "add_special_tokens": args.raw_prompt,
        "max_tokens": 1 if p_only else args.max_tokens,
        "temperature": args.temperature,
        "seed": args.seed,
    }
    if args.request_id is not None:
        start["request_id"] = args.request_id
    request_id: str | None = None
    completed = False
    counts: dict[str, int] = defaultdict(int)
    kinds: dict[str, SourceType] = {}
    tokens_by_source: dict[str, list[int]] = {}
    async with httpx.AsyncClient(timeout=args.http_timeout, trust_env=False) as client:
        try:
            print_json("start request", start)
            ack = await post_json(
                client, normalize_v1_url(args.proxy_url, start_path), start
            )
            print_json("start response", ack)
            request_id = ack.get("request_id")
            if not isinstance(request_id, str) or not request_id:
                raise ValueError("start 响应缺少非空 request_id")
            prefix_tokens = ack_tokens(ack)
            for index, event in enumerate(events):
                kinds[event.source_id] = event.source_type
                envelope = data_envelope(request_id, event, counts[event.source_id])
                print_json(f"data event {index}", envelope)
                ack = await post_json(
                    client,
                    normalize_v1_url(args.proxy_url, "/v1/prompt_stream/event"),
                    envelope,
                )
                print_json(f"data response {index}", ack)
                tokens = ack_tokens(ack)
                if event.source_type == "append":
                    tokens_by_source.setdefault(event.source_id, []).extend(tokens)
                else:
                    tokens_by_source[event.source_id] = tokens
                counts[event.source_id] += 1
                if args.event_delay > 0:
                    await asyncio.sleep(args.event_delay)
            # 每个来源独立封口；所有结束 ACK 完成后才允许提交最终问题。
            for source_id, count in counts.items():
                envelope = {
                    "request_id": request_id,
                    "event": {
                        "source_id": source_id,
                        "source_type": kinds[source_id],
                        "event_type": "source_finished",
                        "sequence_no": count,
                        "version": max(0, count - 1),
                        "arrival_time": time.time(),
                    },
                }
                print_json(f"source_finished {source_id}", envelope)
                ack = await post_json(
                    client,
                    normalize_v1_url(args.proxy_url, "/v1/prompt_stream/event"),
                    envelope,
                )
                print_json(f"source_finished response {source_id}", ack)
            finish = {
                "request_id": request_id,
                "event": {
                    "source_id": "final_question",
                    "event_type": "prompt_finished",
                    "text": args.final_question + suffix,
                    "arrival_time": time.time(),
                },
            }
            print_json("finish request", finish)
            finished = time.perf_counter()
            response = await finish_request(client, args, finish_path, finish)
            elapsed_ms = (time.perf_counter() - finished) * 1000
            print_json("finish response", response)
            stats = parse_prompt_stream_stats(
                response,
                num_sent_update_events=sum(
                    event.source_type == "update" for event in events
                ),
            )
            print_prompt_stream_stats(stats)
            if stats.update_events_accepted != sum(
                event.source_type == "update" for event in events
            ):
                raise ValueError("成功 finish 的 Update 接收数与发送数不一致")
            padding = padding_manifest(response, args.expected_masked_padding)
            checked = validate_prompt_tokens(
                response, prefix_tokens, tokens_by_source, kinds, padding
            )
            if p_only and not response.get("prefill_ready"):
                output_tokens = response.get("output_token_ids")
                if (
                    not isinstance(output_tokens, list)
                    or len(output_tokens) != 1
                    or type(output_tokens[0]) is not int
                ):
                    raise ValueError("P-only 测试必须返回恰好一个输出 token")
            print_json(
                "test summary",
                {
                    "target": args.target,
                    "p_only": p_only,
                    "request_id": request_id,
                    "num_data_events": len(events),
                    "num_finished_sources": len(counts),
                    "finish_response_ms": elapsed_ms,
                    "masked_padding": padding,
                    "prompt_tokens_checked": checked,
                    "output_text": response.get("output_text", response.get("choices")),
                    "status": "ok",
                },
            )
            completed = True
            return response
        finally:
            if request_id is not None and not completed:
                await asyncio.shield(
                    abort_prompt_stream(client, args.proxy_url, request_id)
                )
