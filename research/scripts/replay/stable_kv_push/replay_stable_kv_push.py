# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""通过专用 proxy 比较稳定 KV 提前 Push 与 finish 后统一 Push。"""

from __future__ import annotations

import argparse
import asyncio
import csv
import hashlib
import json
import logging
import math
import sys
import time
import uuid
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
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
from research.scripts.replay.common.dispatch import (
    send_source_events as _send_source_events,
)
from research.scripts.replay.common.load import mark_first_response, read_load_summary
from research.scripts.replay.common.models import (
    RenderedReplayPrompt as RenderedReplayPrompt,
)
from research.scripts.replay.common.models import SampleReplayInput as SampleReplayInput
from research.scripts.replay.common.prompt import (
    ChatTemplateRenderer as ChatTemplateRenderer,
)
from research.scripts.replay.common.prompt import (
    render_sample_prompt as _render_sample_prompt,
)
from research.scripts.replay.common.results import write_prompt_previews
from research.scripts.replay.common.runner import execute_workload
from research.scripts.replay.common.transport import (
    create_http_client,
    iter_sse_payloads,
)
from research.scripts.replay.common.transport import (
    normalize_v1_url as _normalize_v1_url,
)
from research.scripts.replay.common.transport import post_json as _post_json

logger = logging.getLogger(__name__)

# 时长均由 proxy/connector 在各自进程内计时；不相减不同主机的时钟。
PROXY_METRIC_FIELDS = (
    "p_finish_to_d_first_token_ms",
    "d_request_to_first_token_ms",
    "early_transferred_bytes",
    "final_transferred_bytes",
    "total_transferred_bytes",
    "early_transferred_blocks",
    "final_transferred_blocks",
    "final_transfer_ms",
)


@dataclass
class StreamObservation:
    """记录完整 SSE 输出，分别识别 P 完成通知与 D 回答。"""

    first_prefill_time: float | None = None  # 客户端收到 P 完成事件的时刻。
    first_decode_time: float | None = None  # 客户端看到 D 首个回答 token 的时刻。
    last_decode_time: float | None = None  # 最后一个实际 D token 的时刻。
    prefill_tokens: list[int] = field(default_factory=list)
    decode_tokens: list[int] = field(default_factory=list)
    token_ids_exact: bool = True  # 任一内容 chunk 缺 token_ids 时变为 False。
    completion_text: str = ""
    finish_reason: str | None = None
    stable_kv_metrics: dict[str, Any] = field(default_factory=dict)
    masked_padding: dict[str, Any] | None = None
    payloads: list[dict[str, Any]] = field(default_factory=list)

    def observe(self, payload: dict[str, Any], now: float) -> None:
        """合并一条 SSE 数据；服务端错误不能被后续 DONE 掩盖。"""
        if payload.get("error"):
            raise RuntimeError(f"Proxy SSE error: {payload['error']!r}")
        self.payloads.append(payload)
        metrics = payload.get("stable_kv_metrics")
        if metrics is not None:
            if not isinstance(metrics, dict):
                raise ValueError("stable_kv_metrics must be a JSON object.")
            self.stable_kv_metrics.update(metrics)
        manifest = payload.get("prompt_stream_masked_padding")
        if isinstance(manifest, dict):
            self.masked_padding = manifest

        if payload.get("event") == "prefill_ready":
            self.first_prefill_time = now  # 不增加 prefill_tokens 或回答文本。

        # usage/metadata-only SSE 不计为生成 token，也不重置 D 的首 token 时刻。
        for choice in payload.get("choices", []):
            text = choice.get("text") or ""
            token_ids = choice.get("token_ids")
            if choice.get("finish_reason") is not None:
                self.finish_reason = str(choice["finish_reason"])
            if not text and not token_ids:
                continue
            source = payload.get("token_source")
            if source not in ("prefill", "decode"):
                raise ValueError(f"Generated SSE is missing token_source: {payload!r}")
            valid_ids = isinstance(token_ids, list) and all(
                type(token) is int and token >= 0 for token in token_ids
            )
            if not valid_ids or (text and not token_ids):
                self.token_ids_exact = False
                token_ids = []
            self.completion_text += text
            if source == "prefill":
                if self.first_prefill_time is None:
                    self.first_prefill_time = now
                self.prefill_tokens.extend(token_ids)
            else:
                if self.first_decode_time is None:
                    self.first_decode_time = now
                self.last_decode_time = now
                self.decode_tokens.extend(token_ids)


async def consume_stable_sse(response: httpx.Response) -> StreamObservation:
    """共享完整 SSE 帧解析，保留本实验 P/D token 与传输指标观察器。"""
    observation = StreamObservation()
    async for payload in iter_sse_payloads(response):
        observation.observe(payload, time.perf_counter())
        if observation.first_decode_time is not None:
            mark_first_response("decode", observation.first_decode_time)
    return observation


def validate_transfer_metrics(
    metrics: dict[str, Any], early_enabled: bool, has_decode: bool
) -> list[str]:
    """检查实验开关、计数守恒与时间单位；缺字段不填造零值。"""
    errors: list[str] = []
    observed = metrics.get("early_stable_kv_push")
    if observed is not early_enabled:
        errors.append(
            f"early_stable_kv_push expected={early_enabled}, got={observed!r}"
        )
    for name in PROXY_METRIC_FIELDS:
        value = metrics.get(name)
        if not has_decode and name in (
            "p_finish_to_d_first_token_ms",
            "d_request_to_first_token_ms",
        ):
            continue
        if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
            errors.append(f"{name} is missing or invalid: {value!r}")
    early = metrics.get("early_transferred_bytes")
    final = metrics.get("final_transferred_bytes")
    total = metrics.get("total_transferred_bytes")
    if all(type(value) in (int, float) for value in (early, final, total)):
        if total != early + final:
            errors.append("total_transferred_bytes != early + final")
        if not early_enabled and early != 0:
            errors.append("Finish-only control unexpectedly transferred early bytes")
    return errors


def _select_samples(args: argparse.Namespace) -> list[SampleReplayInput]:
    """复用公共读取器；按 manifest ID 选择，不依赖文件名编号。"""
    return select_samples(args)


async def _replay_sample(
    client: httpx.AsyncClient,
    sample: SampleReplayInput,
    rendered: RenderedReplayPrompt,
    renderer: ChatTemplateRenderer,
    args: argparse.Namespace,
    queue_delay_ms: float,
) -> dict[str, Any]:
    """完整发送 start、两条来源、finish；出错时向 proxy 清理两端会话。"""
    started = time.perf_counter()
    request_id: str | None = None
    row: dict[str, Any] = {
        "run_id": args.run_name,
        "sample_id": sample.sample_id,
        "query_id": sample.query_id,
        "early_stable_kv_push": args.early_push,
        "transport_status": "error",
        "metrics_status": "not_evaluated",
        "queue_delay_ms": queue_delay_ms,
        "error": None,
        "metrics_error": None,
    }
    stage = "start"
    try:
        start = dict(sample.start_request)
        start.update(
            api="completions",
            model=args.model,
            prompt=(
                rendered.start_text
                if rendered.input_transport == "text"
                else list(rendered.start_token_ids)
            ),
            add_special_tokens=False,
            stream=True,
            request_id=f"stable-{uuid.uuid4().hex}",
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            seed=args.seed,
            early_stable_kv_push=args.early_push,
            return_token_ids=True,
        )
        ack = await _post_json(
            client, _normalize_v1_url(args.proxy_url, "/v1/prompt_stream/start"), start
        )
        request_id = ack.get("request_id")
        if not isinstance(request_id, str):
            raise ValueError("start response has no request_id.")
        row["request_id"] = request_id
        row["start_ack_ms"] = (time.perf_counter() - started) * 1000
        stage = "events"
        dispatch_start = time.perf_counter()
        source_tasks = [
            asyncio.create_task(
                _send_source_events(
                    client,
                    args.proxy_url,
                    request_id,
                    sample,
                    renderer,
                    source,
                    dispatch_start,
                    args,
                    args.run_name,
                )
            )
            for source in sample.source_streams
        ]
        try:
            events = await asyncio.gather(*source_tasks)
        except BaseException:
            for task in source_tasks:
                task.cancel()
            await asyncio.gather(*source_tasks, return_exceptions=True)
            raise
        row["event_metrics"] = [asdict(event) for source in events for event in source]
        row["source_dispatch_ms"] = (time.perf_counter() - dispatch_start) * 1000
        finish_event = dict(sample.finish_event)
        finish_event.update(
            text=rendered.finish_text,
            arrival_time=time.time(),
        )
        finish_event.pop("token_ids", None)
        if rendered.input_transport == "token_ids":
            finish_event["token_ids"] = list(rendered.finish_token_ids)
        stage = "finish"
        logger.info("FINISH sample=%s request_id=%s", sample.sample_id, request_id)
        finish_start = time.perf_counter()
        # 截止时间覆盖收到 HTTP headers 前的等待，也覆盖整个 SSE 生命周期。
        async with asyncio.timeout(args.finish_timeout):
            async with client.stream(
                "POST",
                _normalize_v1_url(args.proxy_url, "/v1/prompt_stream/finish"),
                json={"request_id": request_id, "event": finish_event},
            ) as response:
                if response.is_error:
                    body = (await response.aread()).decode("utf-8", errors="replace")
                    raise RuntimeError(f"finish HTTP {response.status_code}: {body}")
                observation = await consume_stable_sse(response)
        done = time.perf_counter()
        row.update(transport_status="ok", request_duration_ms=(done - started) * 1000)
        row["finish_to_done_ms"] = (done - finish_start) * 1000
        row["finish_reason"] = observation.finish_reason
        row["completion_text"] = observation.completion_text
        row["prefill_token_ids"] = observation.prefill_tokens
        row["decode_token_ids"] = observation.decode_tokens
        row["completion_token_count_exact"] = observation.token_ids_exact
        row["completion_tokens"] = (
            len(observation.prefill_tokens) + len(observation.decode_tokens)
            if observation.token_ids_exact
            else None
        )
        row["decode_tokens"] = (
            len(observation.decode_tokens) if observation.token_ids_exact else None
        )
        if observation.first_prefill_time is not None:
            row["finish_to_p_first_client_ms"] = (
                observation.first_prefill_time - finish_start
            ) * 1000
        has_decode = observation.first_decode_time is not None
        row["decode_observed"] = has_decode
        if has_decode:
            first_decode = observation.first_decode_time
            assert first_decode is not None
            row["finish_to_d_first_client_ms"] = (first_decode - finish_start) * 1000
            if observation.first_prefill_time is not None:
                row["p_first_to_d_first_client_ms"] = (
                    first_decode - observation.first_prefill_time
                ) * 1000
            elapsed = (observation.last_decode_time or first_decode) - first_decode
            # D 首 token 之前的迁移延迟不混入稳态 decode TPS。
            if observation.token_ids_exact and len(observation.decode_tokens) > 1:
                row["decode_tpot_ms"] = (
                    elapsed * 1000 / (len(observation.decode_tokens) - 1)
                )
                row["decode_tps"] = (
                    (len(observation.decode_tokens) - 1) / elapsed
                    if elapsed > 0
                    else None
                )
        metrics = observation.stable_kv_metrics
        row["stable_kv_metrics"] = metrics
        row.update({name: metrics.get(name) for name in PROXY_METRIC_FIELDS})
        # 可选诊断量仍保留原始字段，缺失时不把它们冒充零耗时。
        for name in (
            "finalize_wait_ms",
            "received_blocks_at_finalize",
            "completed_bytes_before_p_finish",
            "completed_blocks_before_p_finish",
            "final_completion_ms",
        ):
            row[name] = metrics.get(name)
        errors = validate_transfer_metrics(metrics, args.early_push, has_decode)
        if observation.masked_padding is None:
            errors.append(
                "The P response is missing the required masked-padding manifest"
            )
        row["masked_padding_enabled"] = observation.masked_padding is not None
        row["masked_padding_manifest"] = observation.masked_padding
        row["metrics_status"] = "valid" if not errors else "invalid"
        row["metrics_error"] = "; ".join(errors) if errors else None
        total_bytes = metrics.get("total_transferred_bytes")
        early_bytes = metrics.get("early_transferred_bytes")
        if (
            type(total_bytes) in (int, float)
            and total_bytes > 0
            and type(early_bytes) in (int, float)
        ):
            row["early_byte_fraction"] = early_bytes / total_bytes
        if args.save_responses:
            row["response_payloads"] = observation.payloads
        logger.info(
            "DONE sample=%s transport=ok metrics=%s D_first_ms=%s",
            sample.sample_id,
            row["metrics_status"],
            row.get("p_finish_to_d_first_token_ms"),
        )
    except asyncio.CancelledError:
        if request_id is not None:
            await asyncio.shield(
                abort_prompt_stream(client, args.proxy_url, request_id)
            )
        raise
    except Exception as exc:
        row.update(error=f"{type(exc).__name__}: {exc}", error_stage=stage)
        logger.error("FAILED sample=%s stage=%s: %s", sample.sample_id, stage, exc)
        if request_id is not None:
            try:
                async with asyncio.timeout(15):
                    await _post_json(
                        client,
                        _normalize_v1_url(args.proxy_url, "/v1/prompt_stream/abort"),
                        {"request_id": request_id},
                    )
            except Exception as abort_error:
                row["abort_error"] = str(abort_error)
    return row


async def run(args: argparse.Namespace) -> Path:
    """预渲染选中样本，按 QPS 和逻辑请求并发上限回放并及时落盘。"""
    samples = _select_samples(args)
    renderer = ChatTemplateRenderer.from_pretrained(
        args.tokenizer_model, args.tokenizer_trust_remote_code
    )
    prompts = [
        _render_sample_prompt(sample, renderer, input_transport=args.input_transport)
        for sample in samples
    ]
    output = args.output_dir / args.run_name
    output.mkdir(parents=True, exist_ok=False)  # 防止覆盖另一组对照结果。
    config = dict(vars(args))
    # 保存受控性能集的生成参数，后续移动数据目录也不丢失实验自变量。
    dataset_summary_path = args.data_dir / "dataset_summary.json"
    if not getattr(args, "dataset", None) and dataset_summary_path.is_file():
        dataset_summary = json.loads(dataset_summary_path.read_text(encoding="utf-8"))
        if not isinstance(dataset_summary, dict):
            raise ValueError(f"Invalid dataset summary: {dataset_summary_path}")
        config["controlled_dataset_metadata"] = dataset_summary
    config["sample_ids"] = [sample.sample_id for sample in samples]
    config["chat_template_sha256"] = renderer.chat_template_sha256
    config["dataset_sha256"] = hashlib.sha256(
        json.dumps(
            [sample.raw_sample for sample in samples],
            sort_keys=True,
            ensure_ascii=False,
        ).encode()
    ).hexdigest()
    (output / "config.json").write_text(
        json.dumps(config, default=str, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    logger.info("Selected %d samples; results=%s", len(samples), output)
    if args.dry_run:
        write_prompt_previews(output, samples, prompts)
        return output
    rows: list[dict[str, Any]] = []
    epoch = time.perf_counter()
    async with create_http_client(args) as client:
        with (output / "requests.jsonl").open("w", encoding="utf-8") as stream:

            async def execute(
                index: int, offset: float, queue_delay: float
            ) -> dict[str, Any]:
                row = await _replay_sample(
                    client, samples[index], prompts[index], renderer, args, queue_delay
                )
                row["planned_arrival_offset_ms"] = offset
                return row

            def record(row: dict[str, Any]) -> None:
                rows.append(row)
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")
                stream.flush()

            await execute_workload(
                len(samples), args, execute, on_result=record, output_dir=output
            )
    wall_seconds = time.perf_counter() - epoch
    scalar_keys = sorted(
        {
            key
            for row in rows
            for key, value in row.items()
            if value is None or isinstance(value, (str, int, float, bool))
        }
    )
    with (output / "request_metrics.csv").open(
        "w", encoding="utf-8", newline=""
    ) as stream:
        writer = csv.DictWriter(stream, scalar_keys, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    exact_tokens = sum(
        row.get("completion_tokens") or 0
        for row in rows
        if row["transport_status"] == "ok"
    )
    summary = {
        "load_summary": read_load_summary(output),
        "num_requests": len(rows),
        "num_transport_success": sum(row["transport_status"] == "ok" for row in rows),
        "num_metrics_valid": sum(row["metrics_status"] == "valid" for row in rows),
        "wall_seconds": wall_seconds,
        "aggregate_output_tokens_per_second": exact_tokens / wall_seconds,
        "aggregate_output_token_count_exact": all(
            row.get("completion_token_count_exact", False)
            for row in rows
            if row["transport_status"] == "ok"
        ),
    }
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    return output


def parse_args() -> argparse.Namespace:
    """解析实验参数；两种模式只改变 start 的提前传输布尔开关。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir", type=Path, default=Path("research/data/mix_source_perf_qwen")
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path(__file__).resolve().parent / "results"
    )
    parser.add_argument("--proxy-url", default="http://g2c14:8000")
    parser.add_argument("--model", default="qwen2.5-3b")
    parser.add_argument(
        "--tokenizer-model", default=str(Path.home() / "Models/Qwen2.5-3B")
    )
    parser.add_argument("--tokenizer-trust-remote-code", action="store_true")
    parser.add_argument(
        "--early-push", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--query-id", action="append")
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--max-requests", type=int, default=16)
    parser.add_argument("--qps", type=float, default=0.0)
    parser.add_argument(
        "--arrival-process", choices=("fixed", "poisson"), default="poisson"
    )
    parser.add_argument("--random-seed", type=int, default=2026)
    parser.add_argument("--max-concurrency", type=int, default=8)
    parser.add_argument("--append-initial-delay", type=float, default=0.0)
    parser.add_argument("--update-initial-delay", type=float, default=0.0)
    parser.add_argument("--append-chunk-interval", type=float, default=0.5)
    parser.add_argument("--update-chunk-interval", type=float, default=0.2)
    parser.add_argument("--source-finish-delay", type=float, default=0.0)
    parser.add_argument("--event-progress-log-interval", type=int, default=0)
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--http-timeout", type=float, default=120.0)
    parser.add_argument("--finish-timeout", type=float, default=180.0)
    parser.add_argument("--run-name")
    parser.add_argument("--save-responses", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    add_shared_arguments(parser)
    args = parse_configured_args(parser)
    nonnegative = (
        "qps",
        "append_initial_delay",
        "update_initial_delay",
        "append_chunk_interval",
        "update_chunk_interval",
        "source_finish_delay",
    )
    if any(
        not math.isfinite(getattr(args, key)) or getattr(args, key) < 0
        for key in nonnegative
    ):
        parser.error("QPS and event delays must be finite and non-negative.")
    if args.max_requests < 0 or args.start_index < 0 or args.max_concurrency < 1:
        parser.error("Invalid request count, start index or concurrency.")
    if args.max_tokens < 1:
        parser.error("--max-tokens must be >= 1 to observe D generation.")
    if args.http_timeout <= 0 or args.finish_timeout <= 0:
        parser.error("Timeouts must be positive.")
    args.experiment_suite = "stable_kv_push"  # 复用事件日志时保留独立套件身份。
    if args.run_name is None:
        mode = "early" if args.early_push else "finish-only"
        args.run_name = datetime.now(UTC).strftime(f"%Y%m%dT%H%M%SZ-{mode}")
    if Path(args.run_name).name != args.run_name or args.run_name in (".", ".."):
        parser.error("--run-name must be one directory name.")
    return args


def main() -> None:
    """运行回放并输出结果位置，失败请求保留在逐请求文件中。"""
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s"
    )
    output = asyncio.run(run(parse_args()))
    logger.info("Replay finished: %s", output)


if __name__ == "__main__":
    main()
