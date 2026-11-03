# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""混合请求的固定分配、普通 prefill 和两种流式发送路径。"""

from __future__ import annotations

import argparse
import asyncio
import random
import time
import uuid
from dataclasses import dataclass
from typing import Any, Literal

import httpx

from research.scripts.replay.common.dispatch import abort_prompt_stream
from research.scripts.replay.common.models import (
    RenderedReplayPrompt,
    SampleReplayInput,
    SseMeasurement,
)
from research.scripts.replay.common.probe import validate_probe
from research.scripts.replay.common.prompt import (
    ChatTemplateRenderer,
    render_sample_prompt,
)
from research.scripts.replay.common.timeline import sleep_until, source_timing
from research.scripts.replay.common.transport import (
    consume_sse_response,
    normalize_v1_url,
    post_json,
)
from research.scripts.replay.token_invalidation.models import (
    WHOLE_SOURCE_ID,
    Layout,
    PlannedEvent,
)
from research.scripts.replay.token_invalidation.plan import (
    fingerprint,
    make_plan,
    materialize,
    plan_fingerprint,
)

RequestKind = Literal["regular", "stream"]


@dataclass(frozen=True)
class PreparedRequest:
    """负载开始前冻结输入；切换流式算法不改变类型分配与样本位置。"""

    sample: SampleReplayInput
    prompt: RenderedReplayPrompt
    events: tuple[PlannedEvent, ...]
    request_kind: RequestKind
    manifest: dict[str, Any]


def prepare_workload(
    samples: list[SampleReplayInput],
    renderer: ChatTemplateRenderer,
    args: argparse.Namespace,
) -> list[PreparedRequest]:
    """按固定比例随机分配类型，一条样本只占全局到达时间线中的一个位置。"""
    # 分配 RNG 与到达 RNG 分开，改变到达过程不会重新分配普通/流式样本。
    num_stream = int(len(samples) * args.stream_ratio + 0.5)
    stream_indices = set(
        random.Random(args.mix_seed).sample(range(len(samples)), num_stream)
    )
    prepared = []
    for index, sample in enumerate(samples):
        kind: RequestKind = "stream" if index in stream_indices else "regular"
        prompt = render_sample_prompt(sample, renderer)
        events = make_plan(sample, args)
        manifest = {
            "sample_index": index,
            "sample_id": sample.sample_id,
            "query_id": sample.query_id,
            "request_kind": kind,
            "prompt_sha256": fingerprint(prompt.logical_prompt_text),
            "source_plan_sha256": plan_fingerprint(events),
            "prompt_chars": len(prompt.logical_prompt_text),
            "append_chars": prompt.append_chars,
            "final_update_chars": prompt.final_update_chars,
            "num_append_events": sum(e.source_type == "append" for e in events),
            "num_update_events": sum(e.source_type == "update" for e in events),
        }
        prepared.append(PreparedRequest(sample, prompt, events, kind, manifest))
    return prepared


async def dispatch_stream(
    client: httpx.AsyncClient,
    item: PreparedRequest,
    args: argparse.Namespace,
    request_id: str,
    epoch: float,
    result: dict[str, Any],
) -> int:
    """两种算法使用同一原始事件时钟，返回 ACK 确认发生变化的 Update 数。

    同一请求内逐事件等待 ACK；到期积压保留原计划时间，不合并同时到达的事件。
    来源结束从最后一次 ACK 起算，空来源从 initial delay 起算。
    """
    layout: Layout = (
        "segmented" if args.stream_mode == "segmented_sparse" else "append_first"
    )
    events = materialize(item.events, layout)
    last_ack: dict[str, float] = {}
    previous_tokens: dict[str, list[int]] = {}
    changed_updates = 0

    async def send(event: dict[str, Any], target: float) -> dict[str, Any]:
        """发送一个事件并保留 ACK 延迟；失败事件也留在请求记录中。"""
        await sleep_until(target)
        payload = dict(event)
        payload.pop("token_ids", None)
        payload["arrival_time"] = time.time()
        sent = time.perf_counter()
        metric: dict[str, Any] = {
            "source_id": payload["source_id"],
            "event_type": payload["event_type"],
            "sequence_no": payload["sequence_no"],
            "planned_offset_ms": (target - epoch) * 1000,
            "sent_offset_ms": (sent - epoch) * 1000,
            "lateness_ms": max(0.0, sent - target) * 1000,
            "ack_latency_ms": None,
        }
        result["events"].append(metric)
        ack = await post_json(
            client,
            normalize_v1_url(args.proxy_url, "/v1/prompt_stream/event"),
            {"request_id": request_id, "event": payload},
        )
        metric["ack_latency_ms"] = (time.perf_counter() - sent) * 1000
        return ack

    for outbound in events:
        ack = await send(outbound.event, epoch + outbound.planned.offset_s)
        last_ack[outbound.planned.source_id] = time.perf_counter()
        if outbound.event["event_type"] == "update":
            # 使用服务端分词结果识别重复快照，客户端始终只发送原文。
            tokens = ack.get("token_ids")
            if not isinstance(tokens, list):
                raise ValueError("Update ACK 缺少 token_ids，无法校验有效更新数")
            source_id = outbound.event["source_id"]
            changed_updates += tokens != previous_tokens.get(source_id, [])
            previous_tokens[source_id] = tokens

    closing = [
        (
            last_ack.get(
                source.source_id, epoch + source_timing(source.source_type, args)[0]
            )
            + args.source_finish_delay,
            source,
        )
        for source in item.sample.source_streams
    ]
    if layout == "segmented":
        for target, source in sorted(
            closing, key=lambda pair: (pair[0], pair[1].source_id)
        ):
            await send(source.event_records[-1]["event"], target)
    else:
        await send(
            {
                "source_id": WHOLE_SOURCE_ID,
                "source_type": "update",
                "event_type": "source_finished",
                "sequence_no": len(events),
                "version": max(0, len(events) - 1),
            },
            max(
                (target for target, _ in closing),
                default=epoch + args.source_finish_delay,
            ),
        )
    return changed_updates


def validate_stream(
    measured: SseMeasurement,
    item: PreparedRequest,
    args: argparse.Namespace,
    changed_updates: int,
) -> None:
    """校验实际执行模式，防止未启用稀疏或意外启用合并被算成成功。"""
    stats = measured.prompt_stream_stats
    if stats is None or stats.get("coalescing_enabled") is not False:
        raise ValueError("流式响应必须包含统计，且 Update 合并必须关闭")
    if measured.masked_padding_manifest is None:
        raise ValueError("两组都要求 P 节点启用 masked padding")
    if stats.get("update_events_accepted") != changed_updates:
        raise ValueError("服务端 Update 计数与 ACK 确认的有效更新数不一致")
    if args.stream_mode == "segmented_sparse":
        validate_probe(stats, item.manifest["num_append_events"], args.keep_ratio)
    elif "query_probe" in stats:
        raise ValueError("纯 Update 基线不应执行 Query probe")


async def replay_request(
    client: httpx.AsyncClient,
    item: PreparedRequest,
    args: argparse.Namespace,
) -> dict[str, Any]:
    """将一个普通或流式请求发往同一 proxy/P；失败及取消时清理流式会话。"""
    result: dict[str, Any] = {
        **item.manifest,
        "stream_mode": args.stream_mode if item.request_kind == "stream" else None,
        "status": "error",
        "ttft_origin": None,
        "events": [],
    }
    started = time.perf_counter()
    request_id: str | None = None
    finished = False
    try:
        payload: dict[str, Any] = {
            "model": args.model,
            "max_tokens": 1,
            "temperature": 0.0,
            "seed": args.seed,
            "stream": True,
            "add_special_tokens": False,
            "return_token_ids": True,
        }
        changed_updates = 0
        if item.request_kind == "regular":
            endpoint = "/v1/completions_ttft"
            payload["prompt"] = item.prompt.logical_prompt_text
        else:
            # 客户端标识用于 start；后续事件与清理必须采用 ACK 中的内部 ID。
            payload.update(
                api="completions",
                prompt=item.prompt.start_text,
                request_id=f"mixed-{uuid.uuid4().hex}",
            )
            if args.stream_mode == "segmented_sparse":
                payload.update(
                    probe_query=item.prompt.finish_text,
                    probe_keep_ratio=args.keep_ratio,
                )
            ack = await post_json(
                client,
                normalize_v1_url(args.proxy_url, "/v1/prompt_stream/start_ttft"),
                payload,
            )
            request_id = ack["request_id"]
            epoch = time.perf_counter()
            result["start_ack_ms"] = (epoch - started) * 1000
            changed_updates = await dispatch_stream(
                client, item, args, request_id, epoch, result
            )
            result["source_dispatch_ms"] = (time.perf_counter() - epoch) * 1000
            finish = dict(item.sample.finish_event)
            finish.pop("token_ids", None)
            finish.update(text=item.prompt.finish_text, arrival_time=time.time())
            endpoint = "/v1/prompt_stream/finish_ttft"
            payload = {"request_id": request_id, "event": finish}
        submitted = time.perf_counter()
        async with client.stream(
            "POST", normalize_v1_url(args.proxy_url, endpoint), json=payload
        ) as response:
            if response.is_error:
                body = (await response.aread()).decode(errors="replace")
                raise ValueError(f"HTTP {response.status_code}: {body}")
            measured = await consume_sse_response(
                response, submitted, False, True, args.sse_timeout
            )
        finished = True
        result.update(
            request_to_first_token_ms=(measured.ttft_time - started) * 1000,
            finish_to_first_token_ms=(measured.ttft_time - submitted) * 1000,
            ttft_origin=measured.ttft_origin,
            prompt_stream_stats=measured.prompt_stream_stats,
            masked_padding_manifest=measured.masked_padding_manifest,
            changed_update_events=changed_updates,
        )
        if measured.prefill_ready_time is None or measured.completion_tokens:
            raise ValueError("P-only 必须返回 prefill_ready，不应生成回答 token")
        if item.request_kind == "stream":
            validate_stream(measured, item, args, changed_updates)
        result["status"] = "ok"
    except Exception as error:
        result["error"] = str(error)
    finally:
        if request_id is not None and not finished:
            await asyncio.shield(
                abort_prompt_stream(client, args.proxy_url, request_id)
            )
    return result
