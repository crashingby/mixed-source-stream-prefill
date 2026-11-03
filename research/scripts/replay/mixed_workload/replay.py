# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""在一份到达时间线上混合普通完整 prefill 与流式 prefill，测量 P 侧 TTFT。"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import math
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

from research.scripts.replay.common.config import (
    add_shared_arguments,
    parse_configured_args,
)
from research.scripts.replay.common.dataset import select_samples
from research.scripts.replay.common.prompt import ChatTemplateRenderer
from research.scripts.replay.common.results import utc_timestamp, write_json
from research.scripts.replay.common.runner import execute_workload
from research.scripts.replay.common.timeline import planned_arrival_offsets
from research.scripts.replay.common.transport import create_http_client
from research.scripts.replay.mixed_workload.metrics import write_summary
from research.scripts.replay.mixed_workload.workload import (
    prepare_workload,
    replay_request,
)
from research.scripts.replay.token_invalidation.plan import fingerprint

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """读取共享负载参数与混合比例；只支持原文输入和 P-only 完成通知。"""
    parser = argparse.ArgumentParser(description=__doc__)
    add_shared_arguments(parser)
    parser.set_defaults(
        dataset="triviaqa",
        qps=1.0,
        arrival_process="poisson",
        random_seed=2026,
        max_concurrency=8,
    )
    parser.add_argument(
        "--stream-mode",
        choices=("normal", "segmented_sparse"),
        default="segmented_sparse",
    )
    parser.add_argument(
        "--stream-ratio",
        type=float,
        default=0.5,
        help="流式请求数量比例；0 为全普通，1 为全流式",
    )
    parser.add_argument(
        "--mix-seed",
        type=int,
        default=2026,
        help="类型分配种子，与到达 random_seed 独立",
    )
    parser.add_argument("--keep-ratio", type=float, default=0.5)
    parser.add_argument("--query-id", action="append", dest="query_ids")
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument(
        "--max-requests", type=int, default=1000, help="两类请求总数量；0 遍历全部"
    )
    parser.add_argument("--include-partial", action="store_true")
    parser.add_argument("--proxy-url", default="http://127.0.0.1:8000")
    parser.add_argument("--model", default="qwen2.5-3b")
    parser.add_argument(
        "--tokenizer-model", default=str(Path.home() / "Models/Qwen2.5-3B")
    )
    parser.add_argument("--tokenizer-trust-remote-code", action="store_true")
    parser.add_argument("--seed", type=int, default=2026)
    for option, default in (
        ("append-initial-delay", 0.0),
        ("update-initial-delay", 0.0),
        ("append-chunk-interval", 0.05),
        ("update-chunk-interval", 0.02),
        ("source-finish-delay", 0.0),
    ):
        parser.add_argument(f"--{option}", type=float, default=default)
    parser.add_argument("--http-timeout", type=float, default=600.0)
    parser.add_argument("--sse-timeout", type=float, default=600.0)
    parser.add_argument(
        "--experiment-profile",
        default="manual",
        help="标识相同 P 节点配置，写入配对指纹",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path(__file__).parent / "results"
    )
    parser.add_argument("--run-name")
    parser.add_argument("--request-progress-log-interval", type=int, default=25)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--fail-fast", action="store_true")
    args = parse_configured_args(parser)
    for field in ("stream_ratio", "keep_ratio"):
        if not 0 <= getattr(args, field) <= 1:
            parser.error(f"{field} 必须在 [0, 1] 内")
    for field in (
        "append_initial_delay",
        "update_initial_delay",
        "append_chunk_interval",
        "update_chunk_interval",
        "source_finish_delay",
    ):
        value = getattr(args, field)
        if not math.isfinite(value) or value < 0:
            parser.error(f"{field} 必须有限且非负")
    for field in ("http_timeout", "sse_timeout"):
        if not math.isfinite(getattr(args, field)) or getattr(args, field) <= 0:
            parser.error(f"{field} 必须有限且为正数")
    if min(args.start_index, args.max_requests, args.request_progress_log_interval) < 0:
        parser.error("请求数量、起点与日志间隔必须非负")
    if args.input_transport != "text":
        parser.error("混合负载只发送原文，由 P 统一分词")
    if args.run_name and (
        args.run_name in (".", "..") or Path(args.run_name).name != args.run_name
    ):
        parser.error("run-name 必须是单层目录名称")
    return args


async def run(args: argparse.Namespace) -> Path:
    """冻结混合清单后在共享 client、共享负载 runner 中发送，逐请求落盘。"""
    samples = select_samples(args)
    if not samples:
        raise ValueError("没有可回放的样本")
    renderer = ChatTemplateRenderer.from_pretrained(
        args.tokenizer_model, args.tokenizer_trust_remote_code
    )
    prepared = prepare_workload(samples, renderer, args)
    directory = args.output_dir.expanduser() / (
        args.run_name or f"{utc_timestamp()}-{args.stream_mode}"
    )
    directory.mkdir(parents=True, exist_ok=False)
    offsets = planned_arrival_offsets(
        len(prepared), args.qps, args.arrival_process, args.random_seed
    )
    plan = [
        {
            **item.manifest,
            "planned_arrival_offset_ms": offset * 1000
            if args.load_mode == "open_loop"
            else None,
        }
        for item, offset in zip(prepared, offsets)
    ]
    with (directory / "plan.jsonl").open("w", encoding="utf-8") as stream:
        for row in plan:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    # 算法与 keep_ratio 是对比变量；输入、到达过程、采样和服务配置标签必须一致。
    shared_settings = {
        key: getattr(args, key)
        for key in (
            "model",
            "seed",
            "experiment_profile",
            "load_mode",
            "qps",
            "arrival_process",
            "random_seed",
            "inter_request_delay",
            "source_finish_delay",
            "http_timeout",
            "sse_timeout",
        )
    }
    shared_settings["inflight_limit"] = (
        args.max_concurrency if args.load_mode == "closed_loop" else args.max_inflight
    )
    shared_settings["chat_template_sha256"] = renderer.chat_template_sha256
    shared_settings["tokenizer_model"] = renderer.tokenizer_model
    shared_settings["source_initial_delays"] = [
        args.append_initial_delay,
        args.update_initial_delay,
    ]
    config = {
        "schema": "mixed_workload/v1",
        "args": vars(args),
        "shared_settings": shared_settings,
        "workload_sha256": fingerprint({"settings": shared_settings, "requests": plan}),
    }
    write_json(directory / "config.json", config)
    logger.info(
        "Selected %d requests; regular=%d stream=%d; load=%s; output=%s",
        len(prepared),
        sum(item.request_kind == "regular" for item in prepared),
        sum(item.request_kind == "stream" for item in prepared),
        args.load_mode,
        directory,
    )
    completed = 0
    try:
        async with create_http_client(args) as client:
            with (directory / "requests.jsonl").open("w", encoding="utf-8") as output:

                async def execute(
                    index: int, offset: float, queue: float
                ) -> dict[str, Any]:
                    item = prepared[index]
                    row = (
                        {**item.manifest, "status": "dry_run"}
                        if args.dry_run
                        else await replay_request(client, item, args)
                    )
                    row.update(planned_arrival_offset_ms=offset, queue_delay_ms=queue)
                    return row

                def record(row: dict[str, Any]) -> None:
                    nonlocal completed
                    output.write(json.dumps(row, ensure_ascii=False) + "\n")
                    output.flush()
                    completed += 1
                    every = args.request_progress_log_interval
                    if row["status"] == "error" or (every and completed % every == 0):
                        logger.info(
                            "Completed %d/%d; kind=%s sample=%s status=%s error=%s",
                            completed,
                            len(prepared),
                            row["request_kind"],
                            row["sample_id"],
                            row["status"],
                            row.get("error"),
                        )

                await execute_workload(
                    len(prepared),
                    args,
                    execute,
                    on_result=record,
                    fail_fast_error=lambda row: (
                        row.get("error") if args.fail_fast else None
                    ),
                    output_dir=directory,
                )
    finally:
        # 中途取消也汇总已写入记录；未来未发送请求不会变成零延迟成功样本。
        write_summary(directory, config)
    return directory


def main() -> None:
    """启动单组混合回放；失败请求导致非零退出，但仍保留完整诊断文件。"""
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    args = parse_args()
    directory = asyncio.run(run(args))
    logger.info("Results: %s", directory)
    summary = json.loads((directory / "summary.json").read_text())
    group = summary["groups"]["all"]
    if not args.dry_run and group["successful_requests"] != group["selected_requests"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
