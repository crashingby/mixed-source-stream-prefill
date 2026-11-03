# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""验证闭环槽位、独立 QPS 到达和客户端等待的真实计量边界。"""

import argparse
import asyncio
import json
import sys
from pathlib import Path

import httpx
import pytest

from research.scripts.replay.common.load import (
    active_sample,
    mark_first_response,
    mark_request_sent,
    resolve_load_mode,
)
from research.scripts.replay.common.runner import execute_workload
from research.scripts.replay.common.timeline import planned_arrival_offsets
from research.scripts.replay.common.transport import (
    consume_sse_body,
    create_http_client,
)
from research.scripts.replay.query_probe.summarize_performance import compare


def arguments(**overrides: object) -> argparse.Namespace:
    """提供无事件间隔的最小配置，各实验仅改变负载自变量。"""
    values = dict(
        load_mode="auto",
        qps=0.0,
        max_concurrency=3,
        max_inflight=0,
        arrival_process="fixed",
        random_seed=2026,
        http_timeout=10,
    )
    values.update(overrides)
    return argparse.Namespace(**values)


@pytest.mark.parametrize("qps,expected_peak", [(0.0, 3), (200.0, 6)])
def test_load_separates_closed_slots_and_open_arrivals(
    tmp_path: Path, qps: float, expected_peak: int
) -> None:
    """服务端等待整批到达；开环必须能超出闭环槽位数，才会解除屏障。"""
    args = arguments(qps=qps)

    async def exercise() -> None:
        active = peak = 0
        barrier = asyncio.Event()

        async def serve(request: httpx.Request) -> httpx.Response:
            nonlocal active, peak
            if request.url.path.endswith("/event"):
                return httpx.Response(200, json={})
            active += 1
            peak = max(peak, active)
            if active == expected_peak:
                barrier.set()
            await barrier.wait()
            await asyncio.sleep(0.005)
            active -= 1
            return httpx.Response(200, json={})

        async with httpx.AsyncClient(
            transport=httpx.MockTransport(serve),
            event_hooks={"request": [mark_request_sent]},
        ) as client:

            async def execute(index: int, offset: float, queue: float) -> dict:
                await client.post("http://proxy/v1/prompt_stream/start_ttft")
                # 同一个逻辑请求的多个事件不能增加发送 QPS。
                for _ in range(3):
                    await client.post("http://proxy/v1/prompt_stream/event")
                mark_first_response("prefill_ready")
                return {"sample_id": str(index), "status": "ok"}

            async with asyncio.timeout(3):
                results = await execute_workload(6, args, execute, output_dir=tmp_path)
        assert peak == expected_peak
        assert [r["sample_id"] for r in results] == [str(i) for i in range(6)]
        assert all(r["load"]["ttft_origin"] == "prefill_ready" for r in results)
        assert active_sample.get() is None  # 请求上下文不能泄漏到下一轮。

    asyncio.run(exercise())
    summary = json.loads((tmp_path / "load_summary.json").read_text())
    assert summary["sent_requests"] == summary["successful_requests"] == 6
    assert summary["max_inflight"] == expected_peak
    assert summary["planned_qps"] == (qps or None)
    assert summary["actual_send_qps"] > 0
    rows = [
        json.loads(line)
        for line in (tmp_path / "load_requests.jsonl").read_text().splitlines()
    ]
    if qps:
        rows.sort(key=lambda row: row["sample_index"])
        assert [row["planned_arrival_offset_ms"] for row in rows] == pytest.approx(
            [i * 5 for i in range(6)]
        )
    else:
        assert all(row["queue_delay_ms"] == 0 for row in rows)


def test_open_loop_cap_records_backlog_and_full_wait(tmp_path: Path) -> None:
    """显式保护上限会积压，但不得重排计划时间或漏计客户端等待。"""
    args = arguments(qps=200.0, max_inflight=2, max_concurrency=99)

    async def exercise() -> None:
        started = 0
        first_pair = asyncio.Event()
        release = asyncio.Event()

        async def serve(request: httpx.Request) -> httpx.Response:
            nonlocal started
            started += 1
            if started == 2:
                first_pair.set()
            await release.wait()
            return httpx.Response(200, json={})

        async with httpx.AsyncClient(
            transport=httpx.MockTransport(serve),
            event_hooks={"request": [mark_request_sent]},
        ) as client:

            async def execute(index: int, offset: float, queue: float) -> dict:
                await asyncio.sleep(0.01)  # 模拟 HTTP 之前的客户端 prompt 构造。
                await client.post("http://proxy/v1/completions_ttft")
                mark_first_response("prefill_ready")
                return {"sample_id": str(index), "status": "ok"}

            task = asyncio.create_task(
                execute_workload(6, args, execute, output_dir=tmp_path)
            )
            async with asyncio.timeout(3):
                await first_pair.wait()
                await asyncio.sleep(0.05)
                assert started == 2
                release.set()
                results = await task
            for row in results:
                load = row["load"]
                assert load["client_preparation_ms"] >= 8
                assert load["arrival_to_first_token_ms"] == pytest.approx(
                    load["dispatch_lag_ms"] + load["sent_to_first_token_ms"]
                )

    asyncio.run(exercise())
    summary = json.loads((tmp_path / "load_summary.json").read_text())
    assert summary["max_inflight"] == 2
    assert summary["max_client_backlog"] >= 4
    assert summary["latency_ms"]["client_queue_ms"]["max"] > 20


@pytest.mark.parametrize("p_only", [False, True])
def test_load_clock_distinguishes_ready_from_answer(
    tmp_path: Path, p_only: bool
) -> None:
    """完整生成必须跳过 P 通知，P-only 才以它作为 TTFT 终点。"""

    async def exercise() -> None:
        async def serve(request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                200,
                headers={"content-type": "text/event-stream"},
                content='data: {"event":"prefill_ready","choices":[]}\n\n'
                'data: {"choices":[{"text":"A","token_ids":[1]}]}\n\n'
                "data: [DONE]\n\n",
            )

        async with httpx.AsyncClient(
            transport=httpx.MockTransport(serve),
            event_hooks={"request": [mark_request_sent]},
        ) as client:

            async def execute(index: int, offset: float, queue: float) -> dict:
                response = await client.post("http://proxy/v1/completions")
                await consume_sse_body(response, 0, False, p_only)
                return {"status": "ok"}

            rows = await execute_workload(1, arguments(), execute, output_dir=tmp_path)
        assert rows[0]["load"]["ttft_origin"] == (
            "prefill_ready" if p_only else "decode"
        )

    asyncio.run(exercise())


def test_open_loop_http_pool_has_no_hidden_cap() -> None:
    """开环不应在 HTTP 池中再次套用 max_concurrency。"""

    async def exercise() -> None:
        async with create_http_client(arguments(qps=1.0, max_concurrency=1)) as client:
            assert client._transport._pool._max_connections == sys.maxsize
        async with create_http_client(arguments(qps=1.0, max_inflight=4)) as client:
            assert client._transport._pool._max_connections == 16

    asyncio.run(exercise())


def test_poisson_arrivals_are_reproducible() -> None:
    """配对实验只共享到达种子，响应速度不参与到达计划。"""
    assert planned_arrival_offsets(20, 2, "poisson", 7) == planned_arrival_offsets(
        20, 2, "poisson", 7
    )
    assert planned_arrival_offsets(0, 2, "poisson", 7) == []


@pytest.mark.parametrize(
    "overrides",
    [
        {"load_mode": "closed_loop", "qps": 1},
        {"load_mode": "open_loop", "qps": 0},
        {"qps": float("nan")},
        {"max_inflight": -1},
        {"max_inflight": 4},
    ],
)
def test_invalid_load_configuration(overrides: dict) -> None:
    """拒绝互相矛盾的负载参数，避免实际执行与配置标签不一致。"""
    with pytest.raises(ValueError):
        resolve_load_mode(arguments(**overrides))


def test_cancel_retains_unsent_arrivals(tmp_path: Path) -> None:
    """外部取消时仍记录已发、排队和未来到达，不把未发送请求算成吞吐。"""

    async def exercise() -> None:
        started = asyncio.Event()

        async def execute(index: int, offset: float, queue: float) -> dict:
            await mark_request_sent(
                httpx.Request("POST", "http://proxy/v1/completions")
            )
            started.set()
            await asyncio.Event().wait()
            return {"status": "ok"}

        task = asyncio.create_task(
            execute_workload(
                3, arguments(qps=1.0, max_inflight=1), execute, output_dir=tmp_path
            )
        )
        await asyncio.wait_for(started.wait(), 3)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert active_sample.get() is None

    asyncio.run(exercise())
    summary = json.loads((tmp_path / "load_summary.json").read_text())
    assert summary["sent_requests"] == 1
    assert summary["not_sent_requests"] == 2
    assert summary["cancelled_requests"] == 3
    assert summary["completed_requests"] == summary["successful_requests"] == 0
    assert summary["client_backlog_at_last_arrival"] is None
    rows = [
        json.loads(line)
        for line in (tmp_path / "load_requests.jsonl").read_text().splitlines()
    ]
    assert len(rows) == 3
    assert all(row["arrival_to_done_ms"] is None for row in rows if row["sample_index"])


def test_dry_run_does_not_wait_for_qps(tmp_path: Path) -> None:
    """预览不应按低 QPS 等待，也不生成零吞吐的伪实验结果。"""

    async def exercise() -> None:
        async def execute(index: int, offset: float, queue: float) -> dict:
            return {"status": "dry_run"}

        async with asyncio.timeout(1):
            rows = await execute_workload(
                2, arguments(qps=0.001, dry_run=True), execute, output_dir=tmp_path
            )
        assert len(rows) == 2
        assert not (tmp_path / "load_summary.json").exists()

    asyncio.run(exercise())


def test_matrix_compares_qps_and_legacy_concurrency(tmp_path: Path) -> None:
    """新开环图使用含等待的延迟；旧闭环结果仍能汇总，但不能混淆到达计划。"""
    for label in ("c4", "qps1"):
        for mode, factor in (("dense", 2), ("sparse", 1)):
            directory = tmp_path / f"{mode}-{label}"
            directory.mkdir()
            load = {
                "load_mode": "open_loop",
                "planned_qps": 1,
                "inflight_limit": None,
                "arrival_process": "fixed",
                "actual_send_qps": 1,
            }
            (directory / "summary.json").write_text(
                json.dumps({"load_summary": load} if label.startswith("qps") else {})
            )
            row = {
                "sample_id": "one",
                "status": "ok",
                "prompt_sha256": "same",
                "events_sha256": "same",
                "load": {"planned_arrival_offset_ms": 0},
                "arrival_to_first_token_ms": factor * 100,
                "sent_to_first_token_ms": factor * 50,
                "request_to_first_token_ms": factor * 60,
                "finish_to_first_token_ms": factor * 20,
            }
            (directory / "requests.jsonl").write_text(json.dumps(row) + "\n")
    rows = compare(tmp_path)
    assert len(rows) == 5
    assert [row["load_mode"] for row in rows] == ["closed_loop"] * 2 + ["open_loop"] * 3
    assert all(row["mean_speedup"] == 2 for row in rows)
    assert (tmp_path / "comparison/qps_ttft_comparison.png").is_file()
    assert (tmp_path / "comparison/ttft_comparison.png").is_file()
    path = tmp_path / "sparse-qps1/requests.jsonl"
    row = json.loads(path.read_text())
    row["load"]["planned_arrival_offset_ms"] = 100
    path.write_text(json.dumps(row) + "\n")
    with pytest.raises(ValueError, match="请求到达计划不一致"):
        compare(tmp_path)
