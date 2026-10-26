# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""公共请求到达、并发控制和结果持久化；单请求链路由实验提供。"""

from __future__ import annotations

import argparse
import asyncio
import logging
import time
from collections.abc import Awaitable, Callable, Mapping, Sequence
from pathlib import Path
from typing import TypeVar

from .config import parse_args, validate_args
from .dataset import get_data_events, select_samples
from .models import (
    RenderedReplayPrompt,
    ReplayMode,
    ReplaySample,
    RequestMetric,
    SampleReplayInput,
)
from .prompt import ChatTemplateRenderer, render_sample_prompt
from .results import (
    RequestJournal,
    make_run_directory,
    write_prompt_previews,
    write_run_outputs,
)
from .timeline import planned_arrival_offsets, sleep_until
from .transport import ReplayHttpError, create_http_client

logger = logging.getLogger(__name__)
ResultT = TypeVar("ResultT")


async def execute_workload(
    count: int,
    args: argparse.Namespace,
    execute_one: Callable[[int, float, float], Awaitable[ResultT]],
    *,
    on_result: Callable[[ResultT], None] | None = None,
    fail_fast_error: Callable[[ResultT], str | None] | None = None,
    closed_loop: bool = False,
) -> list[ResultT]:
    """共享请求时钟；串行和并发均先落盘，再判断实验失败。

    Args:
        count: 已选样本数。
        args: QPS、随机种子、并发及可选串行间隔。
        execute_one: 接收索引、计划偏移毫秒和客户端准入延迟毫秒。
        on_result: 每请求完成后的同步持久化回调。
        fail_fast_error: 返回错误文本时取消其它在途请求。
        closed_loop: QPS=0 时按并发上限补充请求；默认保留其他实验的串行语义。
    """
    qps = getattr(args, "qps", 0.0)
    offsets = planned_arrival_offsets(
        count,
        qps,
        getattr(args, "arrival_process", "fixed"),
        getattr(args, "random_seed", 0),
    )
    semaphore = asyncio.Semaphore(getattr(args, "max_concurrency", 1))
    epoch = time.perf_counter()

    async def run_one(index: int) -> tuple[int, ResultT]:
        target = epoch + offsets[index]
        if qps > 0:
            await sleep_until(target)
        async with semaphore:
            queue_delay = (
                max(0.0, time.perf_counter() - target) * 1000 if qps > 0 else 0.0
            )
            result = await execute_one(index, offsets[index] * 1000, queue_delay)
            if on_result is not None:
                on_result(result)
            if fail_fast_error is not None and (error := fail_fast_error(result)):
                raise ReplayHttpError(error)
            # 闭环模式的空闲间隔属于当前客户端槽位，不阻塞其他在途请求。
            delay = getattr(args, "inter_request_delay", 0.0)
            if qps == 0 and closed_loop and delay > 0:
                await asyncio.sleep(delay)
            return index, result

    if qps == 0 and not closed_loop:
        results = []
        for index in range(count):
            _, result = await run_one(index)
            results.append(result)
            delay = getattr(args, "inter_request_delay", 0.0)
            if delay > 0 and index + 1 < count:
                await asyncio.sleep(delay)
        return results
    tasks = [asyncio.create_task(run_one(index)) for index in range(count)]
    try:
        # 完成顺序只影响日志；返回顺序仍按问题清单对齐。
        pairs = await asyncio.gather(*tasks)
        return [result for _, result in sorted(pairs)]
    except BaseException:
        for task in tasks:
            if not task.done():
                task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        raise


async def replay_requests(
    samples: Sequence[SampleReplayInput],
    rendered_prompts: Mapping[str, RenderedReplayPrompt],
    renderer: ChatTemplateRenderer,
    args: argparse.Namespace,
    run_id: str,
    replay_sample: ReplaySample,
    *,
    on_result: Callable[[RequestMetric], None] | None = None,
) -> list[RequestMetric]:
    """为 normal/stream/merged 实验提供同一个执行时钟和 HTTP 客户端。"""
    async with create_http_client(args) as client:

        async def execute(
            index: int, offset: float, queue_delay: float
        ) -> RequestMetric:
            sample = samples[index]
            return await replay_sample(
                client,
                sample,
                rendered_prompts[sample.sample_id],
                renderer,
                args,
                run_id,
                offset,
                queue_delay,
            )

        def failure(result: RequestMetric) -> str | None:
            if args.fail_fast and result.result_status != "ok":
                return result.error or "Replay request failed."
            if args.formal_fail_fast and result.formal_result_status == "invalid":
                return result.stats_validation_error or "Invalid formal result."
            return None

        return await execute_workload(
            len(samples),
            args,
            execute,
            on_result=on_result,
            fail_fast_error=failure,
        )


async def run(args: argparse.Namespace, replay_sample: ReplaySample) -> Path:
    """读取选定输入，构造文本，执行并保留中断前已完成的请求。"""
    validate_args(args)
    samples = select_samples(args)
    renderer = ChatTemplateRenderer.from_pretrained(
        args.tokenizer_model, args.tokenizer_trust_remote_code
    )
    prompts = {
        sample.sample_id: render_sample_prompt(
            sample, renderer, input_transport=args.input_transport
        )
        for sample in samples
    }
    run_id, directory = make_run_directory(args)
    # 在联网之前保存解析后的配置、输入指纹和样本选择。
    write_run_outputs(directory, args, renderer, samples, [])
    logger.info(
        "Selected %d samples; mode=%s transport=%s.",
        len(samples),
        args.replay_mode,
        args.input_transport,
    )
    journal = RequestJournal(directory)
    try:
        if args.dry_run:
            write_prompt_previews(directory, samples, list(prompts.values()))
            for sample in samples:
                prompt = prompts[sample.sample_id]
                journal.record(
                    RequestMetric(
                        run_id=run_id,
                        mode="dry_run",
                        sample_id=sample.sample_id,
                        query_id=sample.query_id,
                        sample_status=sample.status,
                        result_status="ok",
                        planned_arrival_offset_ms=0.0,
                        queue_delay_ms=0.0,
                        request_start_wall_time=time.time(),
                        num_append_chunks=sum(
                            len(get_data_events(source))
                            for source in sample.source_streams
                            if source.source_type == "append"
                        ),
                        num_update_snapshots=sum(
                            len(get_data_events(source))
                            for source in sample.source_streams
                            if source.source_type == "update"
                        ),
                        append_chars=prompt.append_chars,
                        final_update_chars=prompt.final_update_chars,
                        normal_prompt_chars=len(prompt.logical_prompt_text),
                    )
                )
        else:
            await replay_requests(
                samples,
                prompts,
                renderer,
                args,
                run_id,
                replay_sample,
                on_result=journal.record,
            )
    finally:
        journal.close()
        write_run_outputs(directory, args, renderer, samples, journal.metrics)
    return directory


def main(mode: ReplayMode, replay_sample: ReplaySample) -> None:
    """运行一个实验入口；各实验只提供单请求实现。"""
    args = parse_args(mode)
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
    directory = asyncio.run(run(args, replay_sample))
    logger.info("Replay results written to %s.", directory)
