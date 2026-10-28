# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""三个失效统计入口共享的参数、P-only 回放和增量结果保存。"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import logging
import math
import time
import uuid
from dataclasses import asdict
from pathlib import Path
from typing import Any

import httpx

from research.scripts.replay.common.config import (
    add_shared_arguments,
    parse_configured_args,
)
from research.scripts.replay.common.dataset import select_samples
from research.scripts.replay.common.dispatch import (
    abort_prompt_stream,
    longest_common_prefix_len,
)
from research.scripts.replay.common.metrics import (
    apply_completion_measurement,
    apply_prompt_stream_stats,
)
from research.scripts.replay.common.models import (
    EventMetric,
    RenderedReplayPrompt,
    SampleReplayInput,
    SseMeasurement,
)
from research.scripts.replay.common.prompt import (
    ChatTemplateRenderer,
    event_payload,
    override_sampling_params,
)
from research.scripts.replay.common.results import make_run_directory, write_json
from research.scripts.replay.common.runner import execute_workload
from research.scripts.replay.common.timeline import (
    should_log_event_progress,
    sleep_until,
    source_timing,
)
from research.scripts.replay.common.transport import (
    ReplayHttpError,
    consume_sse_response,
    create_http_client,
    normalize_v1_url,
    post_json,
)

from .metrics import write_outputs
from .models import WHOLE_SOURCE_ID, InvalidationMetric, Layout, OutboundEvent
from .plan import (
    final_evidence,
    fingerprint,
    make_plan,
    materialize,
    plan_fingerprint,
    render_prompt,
)

logger = logging.getLogger(__name__)
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "results"


def parse_args(layout: Layout) -> argparse.Namespace:
    """解析公共配置和控制时间，默认串行遍历 TriviaQA 的全部就绪请求。"""
    parser = argparse.ArgumentParser(
        description=f"通过 proxy TTFT-only 统计 token 失效：{layout}"
    )
    add_shared_arguments(parser)
    parser.add_argument(
        "--data-dir", type=Path, default=None, help="兼容旧 requests/*.json 数据"
    )
    parser.add_argument("--query-id", action="append", dest="query_ids")
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument(
        "--max-requests", type=int, default=0, help="0 遍历全部；CLI 优先于配置默认值"
    )
    parser.add_argument("--include-partial", action="store_true")
    parser.add_argument("--proxy-url", default="http://127.0.0.1:8000")
    parser.add_argument("--model", default="qwen2.5-3b")
    parser.add_argument(
        "--tokenizer-model", default=str(Path.home() / "Models/Qwen2.5-3B")
    )
    parser.add_argument(
        "--tokenizer-trust-remote-code",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    for option, default in (
        ("append-initial-delay", 0.0),
        ("update-initial-delay", 0.0),
        ("append-chunk-interval", 0.05),
        ("update-chunk-interval", 0.02),
        ("source-finish-delay", 0.0),
    ):
        parser.add_argument(f"--{option}", type=float, default=default)
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=1,
        help="兼容已有配置；实际 P-only 输出始终限制为 1",
    )
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--http-timeout", type=float, default=600.0)
    parser.add_argument("--sse-timeout", type=float, default=300.0)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--run-name")
    parser.add_argument("--ttft-only", action="store_true", default=True)
    parser.add_argument(
        "--expected-coalescing",
        choices=("disabled", "enabled", "any"),
        default="disabled",
    )
    parser.add_argument("--p-node-coalescing-config-json", default=None)
    parser.add_argument(
        "--expected-masked-padding",
        choices=("any", "enabled", "disabled"),
        default="any",
    )
    parser.add_argument("--event-progress-log-interval", type=int, default=0)
    parser.add_argument("--request-progress-log-interval", type=int, default=100)
    parser.add_argument("--experiment-profile", default="manual")
    parser.add_argument("--save-responses", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument("--formal-fail-fast", action="store_true")
    parser.add_argument(
        "--log-level", choices=("DEBUG", "INFO", "WARNING", "ERROR"), default="INFO"
    )
    args = parse_configured_args(parser)
    if args.dataset is None and args.questions_file is None and args.data_dir is None:
        args.dataset = "triviaqa"
    args.data_dir = args.data_dir or Path("research/data/mix_source")
    args.layout = layout
    args.replay_mode = layout
    args.experiment_suite = "token_invalidation"
    return args


def validate_args(args: argparse.Namespace) -> None:
    """检查时间、数量和采样参数，不允许此实验误走完整 D 生成路径。"""
    nonnegative = (
        "qps",
        "temperature",
        "append_initial_delay",
        "update_initial_delay",
        "append_chunk_interval",
        "update_chunk_interval",
        "source_finish_delay",
    )
    for name in nonnegative:
        if not math.isfinite(getattr(args, name)) or getattr(args, name) < 0:
            raise ValueError(f"--{name.replace('_', '-')} 必须有限且非负")
    for name in ("http_timeout", "sse_timeout"):
        if not math.isfinite(getattr(args, name)) or getattr(args, name) <= 0:
            raise ValueError(f"--{name.replace('_', '-')} 必须有限且为正数")
    if (
        args.start_index < 0
        or args.max_requests < 0
        or args.max_concurrency < 1
        or args.max_tokens < 1
    ):
        raise ValueError("请求数量、起点、并发上限或输出预算不合法")
    if args.event_progress_log_interval < 0 or args.request_progress_log_interval < 0:
        raise ValueError("日志间隔必须非负")
    if not args.ttft_only:
        raise ValueError("token_invalidation 只支持 proxy TTFT-only")
    if args.p_node_coalescing_config_json is not None:
        value = json.loads(args.p_node_coalescing_config_json)
        if not isinstance(value, dict):
            raise ValueError("P coalescing 配置必须为 JSON 对象")


def new_metric(
    sample: SampleReplayInput,
    prompt: RenderedReplayPrompt,
    args: argparse.Namespace,
    run_id: str,
    planned_offset_ms: float,
    queue_delay_ms: float,
) -> InvalidationMetric:
    """冻结原始输入、最终证据和共同事件计划，便于逐请求比较三种布局。"""
    plan = make_plan(sample, args)
    evidence = final_evidence(sample)
    return InvalidationMetric(
        run_id=run_id,
        mode=args.layout,
        sample_id=sample.sample_id,
        query_id=sample.query_id,
        sample_status=sample.status,
        result_status="error",
        planned_arrival_offset_ms=planned_offset_ms,
        queue_delay_ms=queue_delay_ms,
        request_start_wall_time=time.time(),
        input_sha256=fingerprint(sample.raw_sample),
        plan_sha256=plan_fingerprint(plan),
        evidence_sha256=fingerprint(evidence),
        logical_prompt_sha256=hashlib.sha256(
            prompt.logical_prompt_text.encode()
        ).hexdigest(),
        original_append_chunks=sum(item.source_type == "append" for item in plan),
        original_update_snapshots=sum(item.source_type == "update" for item in plan),
        append_chars=len(evidence["append"]),
        final_update_chars=len(evidence["update"]),
        normal_prompt_chars=len(prompt.logical_prompt_text),
    )


async def send_event(
    client: httpx.AsyncClient,
    outbound: dict[str, Any],
    source_id: str,
    source_type: str,
    target: float,
    epoch: float,
    renderer: ChatTemplateRenderer,
    args: argparse.Namespace,
    request_id: str,
    metric: InvalidationMetric,
    previous_tokens: dict[str, tuple[int, ...]],
) -> float:
    """在绝对时钟上发送并等 ACK，读取服务端分词仅作逻辑 LCP 诊断。

    Returns:
        本次 ACK 完成的 monotonic 时间；不是 GPU 计算完成时间。
    """
    await sleep_until(target)
    payload = event_payload(outbound, renderer, args.input_transport)
    payload["arrival_time"] = time.time()
    sent = time.perf_counter()
    event_metric = EventMetric(
        run_id=metric.run_id,
        mode=args.layout,
        sample_id=metric.sample_id,
        query_id=metric.query_id,
        source_id=source_id,
        source_type=source_type,
        event_type=payload["event_type"],
        chunk_id=str(payload.get("chunk_id") or ""),
        sequence_no=payload["sequence_no"],
        version=payload["version"],
        scheduled_offset_ms=(target - epoch) * 1000,
        actual_send_offset_ms=(sent - epoch) * 1000,
        ack_latency_ms=None,
        result_status="error",
    )
    metric.event_metrics.append(
        event_metric
    )  # 错误事件也必须保留，而非丢掉整条时间线。
    try:
        ack = await post_json(
            client,
            normalize_v1_url(args.proxy_url, "/v1/prompt_stream/event"),
            {"request_id": request_id, "event": payload},
        )
        completed = time.perf_counter()
        if payload["event_type"] in ("append", "update"):
            raw = ack.get("token_ids")
            if not isinstance(raw, list) or any(
                type(token) is not int for token in raw
            ):
                raise ReplayHttpError(
                    "数据 ACK 必须包含整数 token_ids，供诊断与接收数校验"
                )
            current = tuple(raw)
            event_metric.num_event_tokens = len(current)
            if payload["event_type"] == "update":
                previous = previous_tokens.get(source_id, ())
                lcp = longest_common_prefix_len(previous, current)
                event_metric.update_lcp_tokens = lcp
                event_metric.update_lcp_ratio = lcp / max(1, len(current))
                metric.potential_replaced_suffix_tokens += len(previous) - lcp
                metric.changed_update_events += current != previous
                previous_tokens[source_id] = current
                metric.outbound_update_events += 1
            else:
                metric.outbound_append_events += 1
        else:
            metric.source_finished_events += 1
        metric.num_event_acks += 1
        event_metric.ack_latency_ms = (completed - sent) * 1000
        event_metric.result_status = "ok"
        if should_log_event_progress(args, len(metric.event_metrics) - 1):
            logger.info(
                "ACK sample=%s source=%s seq=%s target_ms=%.3f actual_ms=%.3f",
                metric.sample_id,
                source_id,
                payload["sequence_no"],
                event_metric.scheduled_offset_ms,
                event_metric.actual_send_offset_ms,
            )
        return completed
    except Exception as exc:
        event_metric.error = str(exc)
        raise


async def dispatch_events(
    client: httpx.AsyncClient,
    sample: SampleReplayInput,
    events: tuple[OutboundEvent, ...],
    renderer: ChatTemplateRenderer,
    args: argparse.Namespace,
    request_id: str,
    metric: InvalidationMetric,
    epoch: float | None = None,
) -> None:
    """三种布局共享串行 ACK 和绝对时钟；全部数据后再封口各实际来源。"""
    epoch = time.perf_counter() if epoch is None else epoch
    last_ack: dict[str, float] = {}
    previous_tokens: dict[str, tuple[int, ...]] = {}
    for item in events:
        event = item.event
        ack_time = await send_event(
            client,
            event,
            event["source_id"],
            event["source_type"],
            epoch + item.planned.offset_s,
            epoch,
            renderer,
            args,
            request_id,
            metric,
            previous_tokens,
        )
        last_ack[item.planned.source_id] = ack_time
    closing = []
    for source in sample.source_streams:
        initial, _ = source_timing(source.source_type, args)
        target = (
            last_ack.get(source.source_id, epoch + initial) + args.source_finish_delay
        )
        closing.append((target, source))
    if args.layout == "segmented":
        for target, source in sorted(
            closing, key=lambda value: (value[0], value[1].source_id)
        ):
            # version 是来源快照版本，旧数据中不一定等于事件序号。
            finished = dict(source.event_records[-1]["event"])
            finished.pop("arrival_time", None)
            finished.pop("token_ids", None)
            await send_event(
                client,
                finished,
                source.source_id,
                source.source_type,
                target,
                epoch,
                renderer,
                args,
                request_id,
                metric,
                previous_tokens,
            )
    else:
        # 整体基线只有一个真实 Update 来源，须等所有原始来源都可封口。
        target = max(
            (value[0] for value in closing), default=epoch + args.source_finish_delay
        )
        finished = {
            "source_id": WHOLE_SOURCE_ID,
            "source_type": "update",
            "event_type": "source_finished",
            "sequence_no": len(events),
            "version": max(0, len(events) - 1),
        }
        await send_event(
            client,
            finished,
            WHOLE_SOURCE_ID,
            "update",
            target,
            epoch,
            renderer,
            args,
            request_id,
            metric,
            previous_tokens,
        )
    metric.source_dispatch_ms = (time.perf_counter() - epoch) * 1000
    metric.num_append_chunks = metric.outbound_append_events
    metric.num_update_snapshots = metric.outbound_update_events


def validate_measurement(
    metric: InvalidationMetric, measured: SseMeasurement, args: argparse.Namespace
) -> None:
    """校验原始 stats 后才赋主指标；抢占丢弃与逻辑替换潜力单独保存。"""
    apply_prompt_stream_stats(
        metric,
        measured.prompt_stream_stats,
        num_sent_update_events=metric.outbound_update_events,
        require_stats=True,
        expected_coalescing=args.expected_coalescing,
        expected_config_json=args.p_node_coalescing_config_json,
    )
    metric.masked_padding_observed = measured.masked_padding_manifest is not None
    metric.masked_padding_expected = args.expected_masked_padding
    metric.masked_padding_tokens = (
        measured.masked_padding_manifest["end"]
        - measured.masked_padding_manifest["start"]
        if measured.masked_padding_manifest
        else 0
    )
    expected = args.expected_masked_padding
    mask_valid = expected == "any" or metric.masked_padding_observed == (
        expected == "enabled"
    )
    metric.masked_padding_validation_status = "valid" if mask_valid else "invalid"
    errors = []
    if not mask_valid:
        errors.append("masked-padding 开关与预期不一致")
    if measured.prefill_ready_time is None or measured.completion_tokens != 0:
        errors.append("TTFT-only 必须返回 P 完成通知，且不包含回答 token")
    if metric.stats_validation_status != "valid":
        errors.append(metric.stats_validation_error or "主指标要求完整 schema-v2 统计")
    elif metric.update_events_accepted != metric.changed_update_events:
        errors.append("服务端 Update 接收数与 ACK 中发生变化的快照数不一致")
    if errors:
        metric.formal_result_status = "invalid"
        metric.stats_validation_error = "; ".join(errors)
        return
    metric.snapshot_invalidated_tokens = metric.update_tokens_invalidated_by_snapshot
    metric.preemption_discarded_tokens = sum(
        (
            metric.update_tokens_discarded_by_preemption,
            metric.prefix_append_tokens_discarded_by_preemption,
            metric.question_tokens_discarded_by_preemption,
        )
    )
    metric.formal_result_status = "valid"


async def replay_sample(
    client: httpx.AsyncClient,
    sample: SampleReplayInput,
    renderer: ChatTemplateRenderer,
    args: argparse.Namespace,
    run_id: str,
    planned_offset_ms: float,
    queue_delay_ms: float,
) -> InvalidationMetric:
    """经 proxy 的 P-only 路由执行一请求，错误与取消时清理已创建会话。"""
    prompt = render_prompt(sample, renderer, args.layout, args.input_transport)
    metric = new_metric(sample, prompt, args, run_id, planned_offset_ms, queue_delay_ms)
    started = time.perf_counter()
    request_id: str | None = None
    stage = "start"
    try:
        start = dict(sample.start_request)
        start.update(
            api="completions",
            model=args.model,
            stream=True,
            add_special_tokens=False,
            prompt=prompt.start_text
            if args.input_transport == "text"
            else list(prompt.start_token_ids),
            request_id=f"invalidation-{uuid.uuid4().hex}",
            return_token_ids=True,
        )
        override_sampling_params(start, args)
        start["max_tokens"] = 1  # 不依赖配置中旧的 decode budget，明确不启动 D。
        ack = await post_json(
            client,
            normalize_v1_url(args.proxy_url, "/v1/prompt_stream/start_ttft"),
            start,
        )
        acknowledged_at = time.perf_counter()  # 构造快照的 CPU 时间也计入事件迟发。
        request_id = ack.get("request_id")
        if not isinstance(request_id, str) or not request_id:
            raise ReplayHttpError("start 响应缺少 request_id")
        metric.start_ack_ms = (acknowledged_at - started) * 1000
        stage = "events"
        events = materialize(make_plan(sample, args), args.layout)
        await dispatch_events(
            client, sample, events, renderer, args, request_id, metric, acknowledged_at
        )
        finish = dict(sample.finish_event)
        finish["text"] = prompt.finish_text
        finish.pop("token_ids", None)
        if args.input_transport == "token_ids":
            finish["token_ids"] = list(prompt.finish_token_ids)
        finish["arrival_time"] = time.time()
        stage = "finish_sse"
        sent = time.perf_counter()
        async with client.stream(
            "POST",
            normalize_v1_url(args.proxy_url, "/v1/prompt_stream/finish_ttft"),
            json={"request_id": request_id, "event": finish},
        ) as response:
            if response.is_error:
                body = (await response.aread()).decode(errors="replace")
                raise ReplayHttpError(f"finish HTTP {response.status_code}: {body}")
            # 统计实验消费完整的 P-only SSE，确保末尾统计和 DONE 都已收到。
            measured = await consume_sse_response(
                response, sent, args.save_responses, True, args.sse_timeout
            )
        metric.transport_status = "ok"
        metric.first_token_received = True
        metric.result_status = "ok"
        metric.ttft_origin = measured.ttft_origin
        metric.finish_to_first_token_ms = (measured.ttft_time - sent) * 1000
        metric.request_to_first_token_ms = (measured.ttft_time - started) * 1000
        metric.finish_to_done_ms = (measured.completed_time - sent) * 1000
        metric.request_to_done_ms = (measured.completed_time - started) * 1000
        apply_completion_measurement(metric, measured, sent)
        validate_measurement(metric, measured, args)
    except asyncio.CancelledError:
        if request_id is not None:
            await asyncio.shield(
                abort_prompt_stream(client, args.proxy_url, request_id)
            )
        raise
    except Exception as exc:
        metric.transport_status = "error"
        metric.result_status = "error"
        metric.formal_result_status = "invalid"
        metric.request_error_stage = stage
        metric.error = str(exc)
        if request_id is not None:
            await abort_prompt_stream(client, args.proxy_url, request_id)
    return metric


async def run(args: argparse.Namespace) -> Path:
    """读取选定数据，逐请求构造布局和保存结果，不预建全数据集的合并快照。"""
    validate_args(args)
    samples = select_samples(args)
    renderer = ChatTemplateRenderer.from_pretrained(
        args.tokenizer_model, args.tokenizer_trust_remote_code
    )
    run_id, directory = make_run_directory(args)
    selected = []
    for sample in samples:
        selected.append(
            {
                "sample_id": sample.sample_id,
                "input_sha256": fingerprint(sample.raw_sample),
                "plan_sha256": plan_fingerprint(make_plan(sample, args)),
                "evidence_sha256": fingerprint(final_evidence(sample)),
            }
        )
    write_json(
        directory / "config.json",
        {
            "schema": "token_invalidation_replay/v1",
            "args": vars(args),
            "num_selected": len(samples),
            "selected_samples": selected,
            "selected_sha256": fingerprint(selected),
            "chat_template_sha256": renderer.chat_template_sha256,
            "effective_max_tokens": 1,
            "dispatch": "absolute_serial_ack_every_event",
            "source_finished_phase": "after_all_data_events",
        },
    )
    metrics: list[InvalidationMetric] = []
    logger.info(
        "Selected %d samples: layout=%s results=%s",
        len(samples),
        args.layout,
        directory,
    )
    try:
        if args.dry_run:
            with (directory / "prompt_preview.jsonl").open(
                "w", encoding="utf-8"
            ) as stream:
                for sample in samples:
                    plan = make_plan(sample, args)
                    prompt = render_prompt(
                        sample, renderer, args.layout, args.input_transport
                    )
                    stream.write(
                        json.dumps(
                            {
                                "sample_id": sample.sample_id,
                                "layout": args.layout,
                                "plan_sha256": plan_fingerprint(plan),
                                "logical_prompt_text": prompt.logical_prompt_text,
                                "start_text": prompt.start_text,
                                "finish_text": prompt.finish_text,
                                "num_original_data_events": len(plan),
                                "event_plan": [
                                    {
                                        "offset_s": item.planned.offset_s,
                                        "original_source_id": item.planned.source_id,
                                        "event": item.event,
                                    }
                                    for item in materialize(plan, args.layout)
                                ],
                            },
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
            return directory
        async with create_http_client(args) as client:
            with (directory / "request_metrics.jsonl").open(
                "w", encoding="utf-8"
            ) as stream:

                async def execute(
                    index: int, offset: float, queue_delay: float
                ) -> InvalidationMetric:
                    return await replay_sample(
                        client,
                        samples[index],
                        renderer,
                        args,
                        run_id,
                        offset,
                        queue_delay,
                    )

                def record(metric: InvalidationMetric) -> None:
                    metrics.append(metric)
                    stream.write(json.dumps(asdict(metric), ensure_ascii=False) + "\n")
                    stream.flush()
                    every = args.request_progress_log_interval
                    if metric.formal_result_status != "valid" or (
                        every and len(metrics) % every == 0
                    ):
                        logger.info(
                            "Completed %d/%d sample=%s status=%s "
                            "invalidated=%s error=%s",
                            len(metrics),
                            len(samples),
                            metric.sample_id,
                            metric.formal_result_status,
                            metric.snapshot_invalidated_tokens,
                            metric.error or metric.stats_validation_error,
                        )

                def failure(metric: InvalidationMetric) -> str | None:
                    if args.fail_fast and metric.transport_status != "ok":
                        return metric.error or "请求失败"
                    if args.formal_fail_fast and metric.formal_result_status != "valid":
                        return metric.stats_validation_error or "失效统计无效"
                    return None

                await execute_workload(
                    len(samples),
                    args,
                    execute,
                    on_result=record,
                    fail_fast_error=failure,
                )
    finally:
        write_outputs(directory, metrics, len(samples))
    return directory


def main(layout: Layout) -> None:
    """执行绑定布局的独立入口，不在 CLI 中悄悄改变实验模式。"""
    args = parse_args(layout)
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
    directory = asyncio.run(run(args))
    logger.info("Invalidation results written to %s", directory)
