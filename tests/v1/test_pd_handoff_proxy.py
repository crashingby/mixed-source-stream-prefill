# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""验证 P 完成通知与 D 回答独立，包含 P 提前 stop 和单 token 预算。"""

import asyncio
import json
from types import SimpleNamespace
from typing import Any

import httpx
import pytest

from research.scripts.replay.common.transport import consume_sse_body
from research.scripts.server import proxy_server as proxy


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("budget", [1, 4])
@pytest.mark.parametrize("p_stop", [False, True])
def test_normal_pd_keeps_original_prompt_and_budget(
    monkeypatch: pytest.MonkeyPatch, stream: bool, budget: int, p_stop: bool
) -> None:
    """P 的采样结果和 stop 状态不得修改 D 的输入、预算或回答。"""
    prompt = list(range(32))
    original = {
        "model": "test",
        "prompt": prompt,
        "max_tokens": budget,
        "stream": stream,
    }
    prefill = {
        "choices": [
            {
                "prompt_token_ids": prompt,
                "token_ids": [999],
                "text": "P",
                "finish_reason": "stop" if p_stop else "length",
            }
        ],
        "kv_transfer_params": {"remote_num_tokens": 32, "remote_block_ids": [[1, 2]]},
    }
    answer = {
        "choices": [
            {"text": "D", "token_ids": [77] * budget, "finish_reason": "length"}
        ],
        "usage": {"prompt_tokens": 32, "completion_tokens": budget},
    }
    decode_calls = []

    def check_decode(body: dict[str, Any]) -> None:
        """检查真实发送边界，确认块对齐时没有额外输入尾 token。"""
        assert body["prompt"] == prompt
        assert body["max_tokens"] == budget
        assert body["add_special_tokens"] is False
        decode_calls.append(body)

    async def post_json(
        client: Any, endpoint: str, body: dict, *args: Any, **kwargs: Any
    ) -> dict:
        """替换 P 请求和非流式 D 请求，流式 D 仍经过 HTTP transport。"""
        if kwargs.get("service_name") == "decode":
            check_decode(body)
            return answer
        assert body["max_tokens"] == 1
        return prefill

    async def exercise() -> None:
        """通过实际 handler 检查完整响应和完成事件。"""

        async def request_json() -> dict:
            return original

        def decode(request: httpx.Request) -> httpx.Response:
            check_decode(json.loads(request.content))
            return httpx.Response(
                200,
                stream=httpx.ByteStream(
                    proxy._encode_sse_payload(answer) + b"data: [DONE]\n\n"
                ),
            )

        async with httpx.AsyncClient(
            transport=httpx.MockTransport(decode), base_url="http://D"
        ) as client:
            monkeypatch.setattr(
                proxy, "get_next_client", lambda app, role: {"client": client}
            )
            request = SimpleNamespace(json=request_json, headers={}, app=None)
            response = await proxy._handle_normal_pd_completions(
                request, ttft_only=False
            )
            if stream:
                body = b"".join([chunk async for chunk in response.body_iterator])
                frames = [
                    json.loads(frame[6:])
                    for frame in body.split(b"\n\n")
                    if frame.startswith(b"data: {")
                ]
                assert frames[0]["event"] == "prefill_ready"
                assert frames[0]["choices"] == []
                assert frames[1] == answer
            else:
                assert json.loads(response.body) == answer
        assert len(decode_calls) == 1

    monkeypatch.setattr(proxy, "_post_json", post_json)
    monkeypatch.setattr(
        proxy, "global_args", SimpleNamespace(disable_kv_transfer=False), raising=False
    )
    asyncio.run(exercise())


@pytest.mark.parametrize("p_only", [False, True])
def test_ready_event_is_not_an_answer_token(p_only: bool) -> None:
    """同一完成事件只完成 P-only 计时；完整 PD 必须等 D 的实际回答。"""
    ready = proxy._build_prefill_ready_event("request", "test", {})
    frames = [ready]
    if not p_only:
        frames.append(
            {
                "choices": [
                    {"text": "answer", "token_ids": [77], "finish_reason": "stop"}
                ]
            }
        )
    response = httpx.Response(
        200,
        headers={"content-type": "text/event-stream"},
        content=b"".join(proxy._encode_sse_payload(frame) for frame in frames)
        + b"data: [DONE]\n\n",
    )
    result = asyncio.run(consume_sse_body(response, 0, True, p_only))
    assert result.prefill_ready_time is not None
    assert result.completion_tokens == (0 if p_only else 1)
    assert result.ttft_origin == ("prefill_ready" if p_only else "decode")
    assert (result.first_token_time is None) is p_only
    assert (result.last_token_time is None) is p_only
