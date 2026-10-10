# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""独立稳定 KV 代理的控制时序、首 token 与故障收尾测试。"""

import asyncio
import json
from types import SimpleNamespace
from typing import Any

import httpx
import pytest

from research.scripts.server import stable_kv_proxy as proxy


async def _exercise_proxy(
    *, early_push: bool, stream: bool, fail_transfer: bool = False
) -> tuple[httpx.Response, list[tuple[str, str, dict[str, Any]]]]:
    """用内存 HTTP transport 驱动真实路由，不启动服务或 GPU。"""
    calls: list[tuple[str, str, dict[str, Any]]] = []
    physical_tokens = list(range(36))  # PA=19，padding=13，UPDATE/Question=4。
    manifest = {
        "do_remote_prefill": True,
        "remote_num_tokens": 36,
        "remote_block_ids": [[11, 12, 13]],
        "prompt_stream_exact_transfer": True,
        "prompt_stream_physical_prompt_len": 36,
        "prompt_stream_logical_prompt_len": 23,
        "prompt_stream_remote_block_counts": [3],
        "prompt_stream_remote_block_sizes": [16],
        "prompt_stream_prefill_output_tokens": 1,
        "prompt_stream_masked_padding": {"version": 1, "start": 19, "end": 32},
    }

    async def producer(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        path = request.url.path
        calls.append(("P", path, body))
        if path.endswith("/describe"):
            result = {"enabled": True, "engine_id": "P", "host": "g2c9", "port": 5600}
        elif path.endswith("/start"):
            assert body["max_tokens"] == 1
            assert "kv_transfer_params" not in body
            result = {"request_id": "request-internal", "token_ids": [1, 2]}
        elif path.endswith("/event"):
            result = {"token_ids": [3], "event_type": "append"}
        elif path.endswith("/finish"):
            result = {
                "prompt_token_ids": physical_tokens,
                "output_token_ids": [99],
                "output_text": "P",
                "output_finish_reason": "length",
                "kv_transfer_params": manifest,
                "prompt_stream_stats": {"metrics_schema_version": 2},
            }
        elif path.endswith("/status"):
            result = {
                "metrics": {
                    "early_transferred_blocks": 1 if early_push else 0,
                    "final_transferred_blocks": 2 if early_push else 3,
                    "final_transfer_ms": 0.75,
                }
            }
        else:
            result = {"status": "ok"}
        return httpx.Response(200, json=result)

    async def consumer(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        path = request.url.path
        calls.append(("D", path, body))
        if path.endswith("/reserve"):
            assert body["early_push"] is early_push
            result = {"peer": {"engine_id": "D", "host": "g2c9", "port": 5601}}
        elif path.endswith("/finalize"):
            assert body["physical_prompt_token_ids"] == physical_tokens
            assert body["kv_transfer_params"] == manifest
            result = {
                "state": "finalizing",
                "ready": False,
                "metrics": {"received_blocks_at_finalize": 1 if early_push else 0},
            }
        elif path.endswith("/status"):
            result = {
                "state": "failed" if fail_transfer else "ready",
                "ready": not fail_transfer,
                "metrics": {"received_blocks_at_finalize": 1 if early_push else 0},
            }
        elif path.endswith("/completions"):
            assert body["prompt"] == physical_tokens + [99]
            assert body["max_tokens"] == 3
            assert body["stream"] is True
            assert body["return_token_ids"] is True
            assert body["kv_transfer_params"]["do_remote_prefill"] is False
            assert body["kv_transfer_params"]["do_remote_decode"] is False
            assert body["kv_transfer_params"]["stable_kv_session_id"]
            assert (
                body["kv_transfer_params"]["prompt_stream_masked_padding"]
                == (manifest["prompt_stream_masked_padding"])
            )
            chunks = [
                {"choices": [{"text": "", "token_ids": [77]}]},
                {
                    "choices": [
                        {
                            "text": "answer",
                            "token_ids": [78, 79],
                            "finish_reason": "length",
                        }
                    ],
                    "usage": {
                        "prompt_tokens": 37,
                        "completion_tokens": 3,
                        "total_tokens": 40,
                    },
                },
            ]
            content = b"".join(
                proxy.common._encode_sse_payload(item) for item in chunks
            )
            return httpx.Response(
                200,
                content=content + b"data: [DONE]\n\n",
                headers={"content-type": "text/event-stream"},
            )
        else:
            result = {"status": "ok"}
        return httpx.Response(200, json=result)

    application = proxy.app
    application.state.args = SimpleNamespace(
        early_stable_kv_push=True, finalize_timeout=1.0, status_poll_interval=0.001
    )
    application.state.sessions = {}
    async with (
        httpx.AsyncClient(
            transport=httpx.MockTransport(producer), base_url="http://P/v1"
        ) as p_client,
        httpx.AsyncClient(
            transport=httpx.MockTransport(consumer), base_url="http://D/v1"
        ) as d_client,
        httpx.AsyncClient(
            transport=httpx.ASGITransport(app=application), base_url="http://proxy"
        ) as client,
    ):
        application.state.prefiller = {"client": p_client, "host": "P", "port": 8100}
        application.state.decoder = {"client": d_client, "host": "D", "port": 8200}
        start = await client.post(
            "/v1/prompt_stream/start",
            json={
                "model": "qwen2.5-3b",
                "prompt": [1, 2],
                "max_tokens": 4,
                "stream": stream,
                "early_stable_kv_push": early_push,
            },
        )
        assert start.status_code == 200, start.text
        request_id = start.json()["request_id"]
        event = await client.post(
            "/v1/prompt_stream/event",
            json={"request_id": request_id, "event": {"event_type": "append"}},
        )
        assert event.status_code == 200
        response = await client.post(
            "/v1/prompt_stream/finish",
            json={"request_id": request_id, "event": {"event_type": "prompt_finished"}},
        )
        assert not application.state.sessions
    return response, calls


@pytest.mark.parametrize("early_push", [False, True])
@pytest.mark.parametrize("stream", [False, True])
def test_stable_proxy_preserves_handoff_and_metrics(
    early_push: bool, stream: bool
) -> None:
    """验证两种传输模式均保留 P token、padding 和最终 D 指标。"""
    response, calls = asyncio.run(_exercise_proxy(early_push=early_push, stream=stream))
    assert response.status_code == 200, response.text
    if stream:
        chunks = [
            json.loads(line[5:])
            for line in response.text.splitlines()
            if line.startswith("data:") and line != "data: [DONE]"
        ]
        assert chunks[0]["token_source"] == "prefill"
        assert chunks[1]["token_source"] == "decode"
        assert chunks[1]["choices"][0]["token_ids"] == [77]
        metrics = chunks[-1]["stable_kv_metrics"]
    else:
        payload = response.json()
        assert payload["choices"][0]["token_ids"] == [99, 77, 78, 79]
        assert payload["choices"][0]["text"] == "Panswer"
        assert payload["usage"]["prompt_tokens"] == 36
        assert payload["usage"]["completion_tokens"] == 4
        metrics = payload["stable_kv_metrics"]
    assert metrics["early_stable_kv_push"] is early_push
    assert metrics["early_transferred_blocks"] == (1 if early_push else 0)
    assert metrics["received_blocks_at_finalize"] == (1 if early_push else 0)
    assert metrics["p_finish_to_d_first_token_ms"] >= metrics["finalize_wait_ms"]
    assert metrics["d_request_to_first_token_ms"] is not None
    assert metrics["final_transfer_ms"] == 0.75
    paths = [(side, path) for side, path, _ in calls]
    assert ("D", "/v1/prompt_stream/event") not in paths
    assert paths.index(("D", "/v1/stable_kv/finalize")) < paths.index(
        ("D", "/v1/completions")
    )
    assert ("P", "/v1/stable_kv/abort") in paths
    assert ("D", "/v1/stable_kv/abort") in paths


def test_stable_proxy_never_decodes_failed_transfer() -> None:
    """最终接收失败仍可交付 P 首 token，但不得启动 D 生成。"""
    response, calls = asyncio.run(
        _exercise_proxy(early_push=True, stream=True, fail_transfer=True)
    )
    assert response.status_code == 200
    assert '"token_source": "prefill"' in response.text
    assert '"error"' in response.text
    assert all(not path.endswith("/completions") for _, path, _ in calls)
    assert any(side == "D" and path.endswith("/abort") for side, path, _ in calls)
