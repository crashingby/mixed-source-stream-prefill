# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""使用独立 Append / PipelinePool 数据验证联合 Query probe 的完整功能。"""

import argparse
import asyncio
import hashlib
import json
import logging
import math
import sys
import time
import uuid
from pathlib import Path
from typing import Any

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

from research.scripts.replay.common.config import (
    add_shared_arguments,
    parse_configured_args,
)
from research.scripts.replay.common.dataset import select_samples
from research.scripts.replay.common.dispatch import abort_prompt_stream
from research.scripts.replay.common.models import SampleReplayInput
from research.scripts.replay.common.probe import validate_probe
from research.scripts.replay.common.prompt import (
    ChatTemplateRenderer,
    render_sample_prompt,
)
from research.scripts.replay.common.timeline import (
    planned_arrival_offsets,
    sleep_until,
    source_timing,
)
from research.scripts.replay.common.transport import (
    consume_sse_response,
    normalize_v1_url,
    post_json,
)
from research.scripts.replay.query_probe.performance_stats import summarize

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """解析数据、来源时间与功能配置；默认只回放三条 TriviaQA。"""
    parser = argparse.ArgumentParser(description=__doc__)
    add_shared_arguments(parser)
    parser.set_defaults(dataset="triviaqa")
    parser.add_argument("--proxy-url", default="http://127.0.0.1:8001")
    parser.add_argument("--model", default="qwen2.5-3b")
    parser.add_argument(
        "--tokenizer-model", default=str(Path.home() / "Models/Qwen2.5-3B")
    )
    parser.add_argument("--keep-ratio", type=float, default=0.5)
    parser.add_argument("--mode", choices=("dense", "sparse"), default="sparse")
    parser.add_argument("--warmup-requests", type=int, default=0)
    parser.add_argument("--run-name")
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--max-requests", type=int, default=3)
    parser.add_argument("--query-id", action="append", dest="query_ids")
    parser.add_argument("--include-partial", action="store_true")
    parser.add_argument("--append-initial-delay", type=float, default=0.0)
    parser.add_argument("--update-initial-delay", type=float, default=0.05)
    parser.add_argument("--append-chunk-interval", type=float, default=0.1)
    parser.add_argument("--update-chunk-interval", type=float, default=0.05)
    parser.add_argument("--source-finish-delay", type=float, default=0.0)
    parser.add_argument("--http-timeout", type=float, default=600.0)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--output-dir", type=Path, default=Path(__file__).parent / "results"
    )
    args = parse_configured_args(parser)
    if args.warmup_requests < 0:
        parser.error("warmup-requests 必须非负")
    if args.run_name and Path(args.run_name).name != args.run_name:
        parser.error("run-name 必须为单层目录名")
    if not 0 <= args.keep_ratio <= 1:
        parser.error("keep-ratio 必须在 [0, 1] 内")
    if args.input_transport != "text":
        parser.error("Query probe replay 使用 text，由 P 节点统一分词")
    if args.qps < 0 or args.max_concurrency < 1:
        parser.error("qps 必须非负，max-concurrency 必须为正数")
    for name in (
        "append_initial_delay",
        "update_initial_delay",
        "append_chunk_interval",
        "update_chunk_interval",
        "source_finish_delay",
    ):
        if not math.isfinite(getattr(args, name)) or getattr(args, name) < 0:
            parser.error(f"{name} 必须有限且非负")
    return args


def event_plan(
    sample: SampleReplayInput, args: argparse.Namespace
) -> list[tuple[float, dict[str, Any]]]:
    """按绝对时间排列两个来源，source_finished 保持在本来源最后。"""
    events = []
    for source in sample.source_streams:
        initial, interval = source_timing(source.source_type, args)
        for index, record in enumerate(source.event_records):
            event = dict(record["event"])
            if event["event_type"] == "source_finished":
                offset = (
                    initial + max(0, index - 1) * interval + args.source_finish_delay
                )
            else:
                offset = initial + index * interval
            event.pop("token_ids", None)
            events.append((offset, event))
    return sorted(events, key=lambda item: item[0])


async def replay_sample(
    client: httpx.AsyncClient,
    sample: SampleReplayInput,
    renderer: ChatTemplateRenderer,
    args: argparse.Namespace,
) -> dict[str, Any]:
    """经 proxy TTFT-only 回放一条请求；失败时中止会话，结果保留原始统计。"""
    prompt = render_sample_prompt(sample, renderer)
    plan = event_plan(sample, args)
    result: dict[str, Any] = {
        "sample_id": sample.sample_id,
        "query_id": sample.query_id,
        "keep_ratio": args.keep_ratio if args.mode == "sparse" else None,
        "mode": args.mode,
        "prompt_sha256": hashlib.sha256(
            prompt.logical_prompt_text.encode()
        ).hexdigest(),
        "events_sha256": hashlib.sha256(
            json.dumps(plan, sort_keys=True).encode()
        ).hexdigest(),
        "status": "dry_run" if args.dry_run else "pending",
        "events": [
            {"offset_s": offset, "type": event["event_type"]} for offset, event in plan
        ],
    }
    if args.dry_run:
        return result
    request_id = None
    completed = False
    request_start = time.perf_counter()
    lateness_ms: list[float] = []
    try:
        start = dict(sample.start_request)
        start.update(
            api="completions",
            model=args.model,
            prompt=prompt.start_text,
            add_special_tokens=False,
            max_tokens=1,
            temperature=0.0,
            stream=True,
            return_token_ids=True,
            request_id=f"probe-{uuid.uuid4().hex}",
        )
        if args.mode == "sparse":
            start.update(
                probe_query=prompt.finish_text, probe_keep_ratio=args.keep_ratio
            )
        ack = await post_json(
            client,
            normalize_v1_url(args.proxy_url, "/v1/prompt_stream/start_ttft"),
            start,
        )
        request_id = ack["request_id"]
        epoch = time.perf_counter()
        result["start_ack_ms"] = (epoch - request_start) * 1000
        for offset, event in plan:
            await sleep_until(epoch + offset)
            lateness_ms.append(max(0.0, (time.perf_counter() - epoch - offset) * 1000))
            await post_json(
                client,
                normalize_v1_url(args.proxy_url, "/v1/prompt_stream/event"),
                {"request_id": request_id, "event": event},
            )
        finish = dict(sample.finish_event)
        finish.pop("token_ids", None)
        finish["text"] = prompt.finish_text
        sent = time.perf_counter()
        result["source_dispatch_ms"] = (sent - epoch) * 1000
        result["event_lateness_mean_ms"] = sum(lateness_ms) / max(1, len(lateness_ms))
        result["event_lateness_max_ms"] = max(lateness_ms, default=0.0)
        async with client.stream(
            "POST",
            normalize_v1_url(args.proxy_url, "/v1/prompt_stream/finish_ttft"),
            json={"request_id": request_id, "event": finish},
        ) as response:
            if response.is_error:
                raise RuntimeError((await response.aread()).decode())
            measured = await consume_sse_response(
                response, sent, False, False, args.http_timeout
            )
        completed = True
        assert measured.first_token_time is not None
        result["request_to_first_token_ms"] = (
            measured.first_token_time - request_start
        ) * 1000
        result["finish_to_first_token_ms"] = (measured.first_token_time - sent) * 1000
        result["first_token_origin"] = measured.first_token_origin
        result["masked_padding_manifest"] = measured.masked_padding_manifest
        stats = measured.prompt_stream_stats
        result["prompt_stream_stats"] = stats
        if stats is None:
            raise ValueError("响应缺少 prompt stream 统计")
        if stats.get("coalescing_enabled") is not False:
            raise ValueError("本实验必须关闭 Update 合并")
        if measured.masked_padding_manifest is None:
            raise ValueError("本实验必须启用分段 masked padding")
        if stats["update_events_accepted"] != sum(
            event["event_type"] == "update" for _, event in plan
        ):
            raise ValueError("服务端接收的 Update 数与回放计划不一致")
        if args.mode == "sparse":
            validate_probe(
                stats,
                sum(event["event_type"] == "append" for _, event in plan),
                args.keep_ratio,
            )
        elif "query_probe" in stats:
            raise ValueError("dense 基线不应执行 Query probe")
        result["status"] = "ok"
        result["output_tokens"] = measured.completion_tokens
    except Exception as error:
        result["status"] = "error"
        result["error"] = str(error)
    finally:
        if request_id is not None and not completed:
            await asyncio.shield(
                abort_prompt_stream(client, args.proxy_url, request_id)
            )
    return result


async def run(args: argparse.Namespace) -> Path:
    """预热后执行同一批样本；closed-loop 保持指定并发，结果逐请求落盘。"""
    samples = select_samples(args)
    if (
        args.max_requests > 0
        and not args.query_ids
        and not args.sample_ids
        and len(samples) != args.max_requests
    ):
        raise ValueError(f"需要 {args.max_requests} 条样本，实际只有 {len(samples)} 条")
    renderer = ChatTemplateRenderer.from_pretrained(args.tokenizer_model)
    run_name = args.run_name or (
        f"run-{time.strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:6]}"
    )
    directory = args.output_dir / run_name
    directory.mkdir(parents=True)
    (directory / "config.json").write_text(
        json.dumps(vars(args), default=str, indent=2)
    )
    (directory / "samples.json").write_text(
        json.dumps([sample.sample_id for sample in samples], indent=2)
    )
    limits = httpx.Limits(
        max_connections=max(100, args.max_concurrency * 4),
        max_keepalive_connections=max(20, args.max_concurrency * 2),
    )
    async with httpx.AsyncClient(
        timeout=args.http_timeout, trust_env=False, limits=limits
    ) as client:

        async def phase(selected: list[SampleReplayInput], name: str) -> dict[str, Any]:
            """用固定 worker 数产生闭环负载；预热与正式计时独立。"""
            offsets = planned_arrival_offsets(
                len(selected), args.qps, args.arrival_process, args.random_seed
            )
            pending = iter(enumerate(selected))
            results: list[dict[str, Any]] = []
            epoch = time.perf_counter()
            with (directory / f"{name}.jsonl").open("w", encoding="utf-8") as output:

                async def worker() -> None:
                    """一个请求结束后立即取下一个，维持目标在途请求数。"""
                    for index, sample in pending:
                        if args.qps > 0:
                            await sleep_until(epoch + offsets[index])
                        admitted = time.perf_counter()
                        result = await replay_sample(client, sample, renderer, args)
                        result["sample_index"] = index
                        result["queue_delay_ms"] = (
                            max(0.0, admitted - epoch - offsets[index]) * 1000
                            if args.qps > 0
                            else 0.0
                        )
                        results.append(result)
                        output.write(json.dumps(result, ensure_ascii=False) + "\n")
                        output.flush()
                        if len(results) % 10 == 0 or result["status"] == "error":
                            logger.info(
                                "%s %s: %d/%d complete; %s %s",
                                args.mode,
                                name,
                                len(results),
                                len(selected),
                                sample.sample_id,
                                result["status"],
                            )
                        # 每条完成后刷新，长任务中途也能查看均值与失败数。
                        summary = summarize(results, time.perf_counter() - epoch)
                        (directory / f"{name}_summary.json").write_text(
                            json.dumps(summary, indent=2)
                        )

                await asyncio.gather(
                    *(worker() for _ in range(min(args.max_concurrency, len(selected))))
                )
            return summarize(results, time.perf_counter() - epoch)

        if args.warmup_requests and not args.dry_run:
            warmup = await phase(samples[: args.warmup_requests], "warmup")
            if warmup["errors"]:
                raise RuntimeError(f"预热失败，详见 {directory}")
        summary = await phase(samples, "requests")
    summary.update(
        mode=args.mode,
        concurrency=args.max_concurrency,
        load_model="closed_loop" if args.qps == 0 else "open_loop",
        warmup_requests=min(args.warmup_requests, len(samples)),
    )
    (directory / "summary.json").write_text(json.dumps(summary, indent=2))
    print(directory)
    if summary["errors"]:
        raise RuntimeError(f"{summary['errors']} 个请求失败，详见 {directory}")
    return directory


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    asyncio.run(run(parse_args()))
