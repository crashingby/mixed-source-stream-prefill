# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""用内存 HTTP 替身验证 mask 在 P/proxy/D 和 TTFT-only 边界不丢失。"""

import asyncio
import copy
import json
from types import SimpleNamespace
from typing import Any

import pytest
from fastapi import HTTPException

from research.scripts.server import proxy_server as proxy


def _prefill_response() -> dict[str, Any]:
    """构造 block size 为 4、含一个填充槽位的 P 完成响应。"""
    return {
        "request_id": "masked-request",
        "prompt_token_ids": [1, 2, 3, 3, 7, 8, 9],
        "output_token_ids": [19],
        "output_text": "answer",
        "output_finish_reason": "length",
        "kv_transfer_params": {
            "do_remote_prefill": True,
            "remote_num_tokens": 7,
            "remote_block_ids": [[17, 20]],
            "prompt_stream_exact_transfer": True,
            "prompt_stream_physical_prompt_len": 7,
            "prompt_stream_logical_prompt_len": 6,
            "prompt_stream_prefill_output_tokens": 0,
            "prompt_stream_remote_block_counts": [2],
            "prompt_stream_remote_block_sizes": [4],
            "prompt_stream_masked_padding": {
                "version": 1,
                "start": 3,
                "end": 4,
            },
        },
        "prompt_stream_stats": {"metrics_schema_version": 2},
    }


def _session(*, p_only: bool = False) -> proxy.PromptStreamProxySession:
    """建立只在测试内存中使用的代理会话，不创建网络连接。"""
    return proxy.PromptStreamProxySession(
        request_id="masked-request",
        prefill_client_info={"id": "P"},
        decode_client_info=None if p_only else {"id": "D"},
        api_endpoint="/completions",
        base_request={"model": "test", "max_tokens": 4, "stream": False},
        decode_prompt_text=None,
        decode_prompt_token_ids=[1, 2, 3],
    )


def _request(session: proxy.PromptStreamProxySession) -> Any:
    """提供 route 所需的 json/app 接口；不会启动 ASGI 服务。"""

    async def request_json() -> dict[str, Any]:
        return {
            "request_id": session.request_id,
            "event": {"event_type": "prompt_finished", "token_ids": [9]},
        }

    return SimpleNamespace(
        json=request_json,
        app=SimpleNamespace(
            state=SimpleNamespace(prompt_stream_sessions={session.request_id: session})
        ),
    )


def test_masked_padding_transfer_preserves_original_physical_prompt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """D 接收未压紧的物理 prompt 和全部输出预算，P 采样结果不进入回答。"""
    prefill = _prefill_response()
    sent_requests: list[dict[str, Any]] = []

    async def post_json(*args: Any, **kwargs: Any) -> dict[str, Any]:
        if args[1] == "/prompt_stream/finish":
            return copy.deepcopy(prefill)
        sent_requests.append(args[2])
        return {
            "choices": [{"text": " tail", "token_ids": [20]}],
            "usage": {"prompt_tokens": 7, "completion_tokens": 1, "total_tokens": 8},
        }

    monkeypatch.setattr(proxy, "_post_json", post_json)
    monkeypatch.setattr(
        proxy, "global_args", SimpleNamespace(disable_kv_transfer=False), raising=False
    )
    result = asyncio.run(proxy.finish_prompt_stream(_request(_session())))
    assert len(sent_requests) == 1
    decode = sent_requests[0]
    assert decode["prompt"] == prefill["prompt_token_ids"]
    assert decode["kv_transfer_params"] == prefill["kv_transfer_params"]
    assert decode["max_tokens"] == 4
    payload = json.loads(result.body)
    assert payload["prompt_stream_masked_padding"] == {
        "version": 1,
        "start": 3,
        "end": 4,
    }
    assert payload["choices"][0]["text"] == " tail"
    assert payload["choices"][0]["token_ids"] == [20]
    assert payload["usage"]["prompt_tokens"] == 7


def test_ttft_without_connector_keeps_mask_in_first_sse_frame(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """P-only 输出只有语义清单时，首帧仍携带 mask 且从不请求 D。"""
    prefill = _prefill_response()
    prefill["kv_transfer_params"] = {
        name: value
        for name, value in prefill["kv_transfer_params"].items()
        if name
        in {
            "prompt_stream_masked_padding",
            "prompt_stream_physical_prompt_len",
            "prompt_stream_logical_prompt_len",
            "prompt_stream_prefill_output_tokens",
        }
    }
    endpoints: list[str] = []

    async def post_json(*args: Any, **kwargs: Any) -> dict[str, Any]:
        endpoints.append(args[1])
        return prefill

    monkeypatch.setattr(proxy, "_post_json", post_json)
    result = asyncio.run(
        proxy.finish_prompt_stream_ttft(_request(_session(p_only=True)))
    )
    assert endpoints == ["/prompt_stream/finish"]
    payload = json.loads(result.body.split(b"\n\n")[0].removeprefix(b"data: "))
    assert payload["event"] == "prefill_ready"
    assert payload["choices"] == []
    assert payload["prompt_stream_masked_padding"]["start"] == 3
    assert payload["prompt_stream_stats"] == prefill["prompt_stream_stats"]


def test_debug_recompute_cannot_drop_mask(monkeypatch: pytest.MonkeyPatch) -> None:
    """拒绝把含 mask 的物理 prompt 降级成 D 的普通本地重算。"""
    endpoints: list[str] = []

    async def post_json(*args: Any, **kwargs: Any) -> dict[str, Any]:
        endpoints.append(args[1])
        return _prefill_response()

    monkeypatch.setattr(proxy, "_post_json", post_json)
    monkeypatch.setattr(
        proxy, "global_args", SimpleNamespace(disable_kv_transfer=True), raising=False
    )
    with pytest.raises(HTTPException) as error:
        asyncio.run(proxy.finish_prompt_stream(_request(_session())))
    assert error.value.status_code == 409
    assert endpoints == ["/prompt_stream/finish"]


@pytest.mark.parametrize(
    "padding",
    [
        None,
        {"version": 2, "start": 3, "end": 4},
        {"version": True, "start": 3, "end": 4},
        {"version": 1, "start": -1, "end": 4},
        {"version": 1, "start": 4, "end": 3},
        {"version": 1, "start": 3, "end": 9},
    ],
)
def test_invalid_padding_manifest_rejected(padding: Any) -> None:
    """非法 mask 元数据必须在 D 网络请求之前失败。"""
    prefill = _prefill_response()
    prefill["kv_transfer_params"]["prompt_stream_masked_padding"] = padding
    with pytest.raises(HTTPException) as error:
        proxy._validate_prompt_stream_padding_manifest(prefill)
    assert error.value.status_code == 502


def test_padding_length_mismatch_rejected() -> None:
    """逻辑长度与物理长度之差必须等于中间填充长度。"""
    prefill = _prefill_response()
    prefill["kv_transfer_params"]["prompt_stream_logical_prompt_len"] = 7
    with pytest.raises(HTTPException):
        proxy._validate_prompt_stream_padding_manifest(prefill)


def test_empty_padding_range_remains_explicit() -> None:
    """恰好对齐的请求保留模式标记，不能按 token 值猜测是否开启。"""
    prefill = _prefill_response()
    params = prefill["kv_transfer_params"]
    params["prompt_stream_masked_padding"]["start"] = 4
    params["prompt_stream_logical_prompt_len"] = 7
    session = _session()
    session.decode_prompt_token_ids = prefill["prompt_token_ids"]
    proxy._validate_prompt_stream_transfer(session, prefill)
    assert proxy._validate_prompt_stream_padding_manifest(prefill) is not None
