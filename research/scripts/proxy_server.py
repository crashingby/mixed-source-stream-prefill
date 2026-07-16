# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import argparse
import itertools
import logging
import os
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any

import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


@dataclass
class PromptStreamProxySession:
    """代理服务器保存的 prompt stream 会话状态。"""

    request_id: str
    prefill_client_info: dict[str, Any]
    decode_client_info: dict[str, Any]
    api_endpoint: str
    base_request: dict[str, Any]
    decode_prompt_text: str | None
    decode_prompt_token_ids: list[int] | None
    decode_prompt_finished: bool = False


@asynccontextmanager
async def lifespan(app: FastAPI):
    """初始化 P/D 节点 HTTP client 池和 prompt stream 会话表。"""
    app.state.prefill_clients = []
    app.state.decode_clients = []
    app.state.prompt_stream_sessions = {}

    for i, (host, port) in enumerate(global_args.prefiller_instances):
        app.state.prefill_clients.append(
            {
                "client": httpx.AsyncClient(
                    timeout=None,
                    base_url=f"http://{host}:{port}/v1",
                    limits=httpx.Limits(
                        max_connections=None,
                        max_keepalive_connections=None,
                    ),
                ),
                "host": host,
                "port": port,
                "id": i,
            }
        )

    for i, (host, port) in enumerate(global_args.decoder_instances):
        app.state.decode_clients.append(
            {
                "client": httpx.AsyncClient(
                    timeout=None,
                    base_url=f"http://{host}:{port}/v1",
                    limits=httpx.Limits(
                        max_connections=None,
                        max_keepalive_connections=None,
                    ),
                ),
                "host": host,
                "port": port,
                "id": i,
            }
        )

    app.state.prefill_iterator = itertools.cycle(range(len(app.state.prefill_clients)))
    app.state.decode_iterator = itertools.cycle(range(len(app.state.decode_clients)))

    logger.info(
        "Initialized %s prefill clients and %s decode clients.",
        len(app.state.prefill_clients),
        len(app.state.decode_clients),
    )

    yield

    for client_info in app.state.prefill_clients:
        await client_info["client"].aclose()
    for client_info in app.state.decode_clients:
        await client_info["client"].aclose()


app = FastAPI(lifespan=lifespan)


def parse_args():
    """解析代理服务器启动参数。"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument(
        "--prefiller-hosts",
        "--prefiller-host",
        type=str,
        nargs="+",
        default=["localhost"],
    )
    parser.add_argument(
        "--prefiller-ports",
        "--prefiller-port",
        type=int,
        nargs="+",
        default=[8100],
    )
    parser.add_argument(
        "--decoder-hosts",
        "--decoder-host",
        type=str,
        nargs="+",
        default=["localhost"],
    )
    parser.add_argument(
        "--decoder-ports",
        "--decoder-port",
        type=int,
        nargs="+",
        default=[8200],
    )
    parser.add_argument(
        "--disable-kv-transfer",
        action="store_true",
        help=(
            "Do not forward kv_transfer_params to D. This is a debug mode "
            "for comparing remote KV decode against local D prefill."
        ),
    )
    args = parser.parse_args()

    if len(args.prefiller_hosts) != len(args.prefiller_ports):
        raise ValueError("Number of prefiller hosts must match ports")
    if len(args.decoder_hosts) != len(args.decoder_ports):
        raise ValueError("Number of decoder hosts must match ports")

    args.prefiller_instances = list(zip(args.prefiller_hosts, args.prefiller_ports))
    args.decoder_instances = list(zip(args.decoder_hosts, args.decoder_ports))
    return args


def get_next_client(app: FastAPI, service_type: str) -> dict[str, Any]:
    """按 round-robin 选择一个 P 或 D 节点。"""
    if service_type == "prefill":
        client_idx = next(app.state.prefill_iterator)
        return app.state.prefill_clients[client_idx]
    if service_type == "decode":
        client_idx = next(app.state.decode_iterator)
        return app.state.decode_clients[client_idx]
    raise ValueError(f"Unknown service type: {service_type}")


def _authorization_headers(request_id: str) -> dict[str, str]:
    """构造转发给 P/D 节点的认证和 request id 头。"""
    headers = {"X-Request-Id": request_id}
    if api_key := os.environ.get("OPENAI_API_KEY"):
        headers["Authorization"] = f"Bearer {api_key}"
    return headers


def _get_api_endpoint(req_data: dict[str, Any]) -> str:
    """把用户传入的 api 名称转换为 OpenAI endpoint。"""
    api = req_data.get("api", "completions")
    if api in ("completions", "/completions"):
        return "/completions"
    if api in ("chat/completions", "/chat/completions"):
        return "/chat/completions"
    raise HTTPException(status_code=400, detail=f"Unsupported api={api}")


def _build_decode_base_request(req_data: dict[str, Any]) -> dict[str, Any]:
    """从 start 请求中提取后续发给 D 节点的生成参数。"""
    base_request = dict(req_data)
    sampling_params = base_request.pop("sampling_params", {}) or {}
    base_request.pop("api", None)
    base_request.pop("request_id", None)
    base_request.update(sampling_params)
    return base_request


def _get_initial_decode_prompt_token_ids(
    req_data: dict[str, Any],
    response_json: dict[str, Any],
) -> list[int] | None:
    """提取 P 节点 start 响应里的实际 prefix token ids。

    Args:
        req_data: 用户发给 proxy `/start` 的原始请求体。
        response_json: P 节点 `/prompt_stream/start` 返回的 JSON。

    Returns:
        P 节点实际使用的 prefix token ids。若 P 节点未返回 token_ids，且
        start 请求本身就是 token id prompt，则回退使用请求里的 token ids。
    """
    token_ids = response_json.get("token_ids")
    if isinstance(token_ids, list):
        return [int(token_id) for token_id in token_ids]

    prompt = req_data.get("prompt")
    if isinstance(prompt, list) and all(isinstance(x, int) for x in prompt):
        return list(prompt)
    return None


def _get_initial_decode_prompt_text(req_data: dict[str, Any]) -> str | None:
    """提取 proxy 侧用于 D 节点 decode 的初始 prompt 文本。

    Args:
        req_data: 用户发给 proxy `/start` 的原始请求体。

    Returns:
        如果 start 使用文本 prompt，则返回该文本；如果使用 token_ids/list[int]
        形式，则返回 None，表示 proxy 不能在文本层可靠重建最终 prompt。
    """
    prompt = req_data.get("prompt")
    if isinstance(prompt, str):
        return prompt
    return None


def _append_decode_prompt_token_ids(
    session: PromptStreamProxySession,
    response_json: dict[str, Any],
) -> None:
    """把 P 节点 ACK 中的 append token ids 追加到 D 侧 prompt 镜像。

    Args:
        session: 当前 prompt stream 会话。
        response_json: P 节点 `/prompt_stream/event` 返回的 JSON。
    """
    if response_json.get("event_type") != "append":
        return

    token_ids = response_json.get("token_ids")
    if not isinstance(token_ids, list):
        session.decode_prompt_token_ids = None
        return

    if session.decode_prompt_token_ids is not None:
        session.decode_prompt_token_ids.extend(int(token_id) for token_id in token_ids)


def _append_decode_prompt_text(
    session: PromptStreamProxySession,
    event: dict[str, Any],
) -> None:
    """把 append 文本事件同步追加到 proxy 侧 decode prompt 镜像。

    Args:
        session: 当前 prompt stream 会话。
        event: RESTFUL event envelope 里的 `event` 字段。
    """
    # 当前 research/scripts 的前端只覆盖 append-only 文本模式；update 的
    # 最终 prompt 需要按 source 快照重建，不能简单尾部追加。
    if event.get("event_type") != "append":
        return

    # 如果上游传 token_ids 而不是 text，proxy 暂时不能保证和 P/D tokenizer
    # 结果完全一致，因此放弃文本镜像，避免构造错误 prompt。
    event_text = event.get("text")
    if not isinstance(event_text, str):
        session.decode_prompt_text = None
        return

    if session.decode_prompt_text is not None:
        session.decode_prompt_text += event_text


def _finish_decode_prompt_token_ids(
    session: PromptStreamProxySession,
    response_json: dict[str, Any],
) -> None:
    """把 P 节点 ACK 中的 finish token ids 追加到 D 侧 prompt 镜像。

    Args:
        session: 当前 prompt stream 会话。
        response_json: P 节点 `/prompt_stream/finish` 返回的 JSON。
    """
    # Segment 模式下 P 节点会返回包含 PREFIX_APPEND 尾部 padding 的物理
    # prompt。D 必须完整采用这份镜像，不能继续使用 proxy 自己的逻辑拼接。
    physical_prompt_token_ids = response_json.get("prompt_token_ids")
    if isinstance(physical_prompt_token_ids, list):
        session.decode_prompt_token_ids = [
            int(token_id) for token_id in physical_prompt_token_ids
        ]
        return

    token_ids = response_json.get("token_ids")
    if not isinstance(token_ids, list):
        session.decode_prompt_token_ids = None
        return

    if session.decode_prompt_token_ids is not None:
        session.decode_prompt_token_ids.extend(int(token_id) for token_id in token_ids)


def _finish_decode_prompt_text(
    session: PromptStreamProxySession,
    event: dict[str, Any],
) -> None:
    """把 prompt_finished 文本追加到 proxy 侧 decode prompt 镜像。

    Args:
        session: 当前 prompt stream 会话。
        event: RESTFUL finish envelope 里的 `event` 字段。
    """
    # finish 可能因为 D 节点失败而被手动重试；避免把 final question 追加两次。
    if session.decode_prompt_finished:
        return
    session.decode_prompt_finished = True

    event_text = event.get("text")
    if not isinstance(event_text, str):
        session.decode_prompt_text = None
        return

    if session.decode_prompt_text is not None:
        session.decode_prompt_text += event_text


def _apply_decode_prompt_to_request(
    session: PromptStreamProxySession,
    decode_request: dict[str, Any],
) -> None:
    """把 proxy 侧重建出的完整 prompt 写入 D 节点请求体。

    Args:
        session: 当前 prompt stream 会话。
        decode_request: 即将发送给 D 节点的 OpenAI 请求体。
    """
    if session.api_endpoint != "/completions":
        return
    if session.decode_prompt_token_ids is not None:
        decode_request["prompt"] = list(session.decode_prompt_token_ids)
        decode_request["add_special_tokens"] = False
        return
    if session.decode_prompt_text is None:
        return
    decode_request["prompt"] = session.decode_prompt_text


def _append_prefill_output_token_ids(
    session: PromptStreamProxySession,
    response_json: dict[str, Any],
) -> int:
    """把 P 节点首 token 追加到 D 侧 prompt token 镜像。

    Args:
        session: 当前 prompt stream 会话。
        response_json: P 节点 `/prompt_stream/finish` 返回的 JSON。

    Returns:
        实际追加的 P 侧输出 token 数。
    """
    output_token_ids = response_json.get("output_token_ids")
    if not isinstance(output_token_ids, list):
        return 0
    if session.decode_prompt_token_ids is None:
        return 0

    normalized_token_ids = [int(token_id) for token_id in output_token_ids]
    session.decode_prompt_token_ids.extend(normalized_token_ids)
    return len(normalized_token_ids)


def _shrink_decode_max_tokens(
    decode_request: dict[str, Any],
    num_prefill_output_tokens: int,
) -> None:
    """扣除 P 节点已经生成并交给 D 的首 token 数。

    Args:
        decode_request: 即将发给 D 节点的 OpenAI 请求体。
        num_prefill_output_tokens: P 节点已经生成的输出 token 数。
    """
    if num_prefill_output_tokens <= 0:
        return
    max_tokens = decode_request.get("max_tokens")
    if isinstance(max_tokens, int):
        decode_request["max_tokens"] = max(0, max_tokens - num_prefill_output_tokens)


def _validate_prompt_stream_transfer(
    session: PromptStreamProxySession,
    response_json: dict[str, Any],
    num_prefill_output_tokens: int,
) -> None:
    """在请求 D 之前验证 token view 与远端 KV 覆盖范围一致。"""
    physical_prompt_token_ids = response_json.get("prompt_token_ids")
    kv_transfer_params = response_json.get("kv_transfer_params")
    remote_num_tokens = (
        kv_transfer_params.get("remote_num_tokens")
        if isinstance(kv_transfer_params, dict)
        else None
    )
    if not isinstance(physical_prompt_token_ids, list):
        raise HTTPException(
            status_code=502,
            detail="P node did not return physical prompt token ids.",
        )
    if not isinstance(remote_num_tokens, int):
        raise HTTPException(
            status_code=502,
            detail="P node did not return an integer remote_num_tokens.",
        )
    physical_prompt_len = len(physical_prompt_token_ids)
    reported_physical_prompt_len = kv_transfer_params.get(
        "prompt_stream_physical_prompt_len"
    )
    if reported_physical_prompt_len != physical_prompt_len:
        raise HTTPException(
            status_code=502,
            detail=(
                "P node returned inconsistent physical prompt lengths: "
                f"tokens={physical_prompt_len}, "
                f"metadata={reported_physical_prompt_len}."
            ),
        )
    decode_prompt_len = physical_prompt_len + num_prefill_output_tokens
    if not physical_prompt_len <= remote_num_tokens <= decode_prompt_len:
        raise HTTPException(
            status_code=502,
            detail=(
                "Prompt stream KV/token mismatch: "
                f"physical_prompt_len={physical_prompt_len}, "
                f"remote_num_tokens={remote_num_tokens}, "
                f"decode_prompt_len={decode_prompt_len}."
            ),
        )
    if (
        session.decode_prompt_token_ids is None
        or len(session.decode_prompt_token_ids) != decode_prompt_len
    ):
        actual_decode_prompt_len = (
            None
            if session.decode_prompt_token_ids is None
            else len(session.decode_prompt_token_ids)
        )
        raise HTTPException(
            status_code=502,
            detail=(
                "Proxy failed to construct the exact D token view: "
                f"expected_len={decode_prompt_len}, "
                f"actual_len={actual_decode_prompt_len}."
            ),
        )


def _should_finish_without_decode(
    decode_request: dict[str, Any],
    response_json: dict[str, Any],
    num_prefill_output_tokens: int,
) -> bool:
    """判断 P 首 token 是否已经满足原始请求的停止条件。"""
    if response_json.get("output_finish_reason") == "stop":
        return True
    max_tokens = decode_request.get("max_tokens")
    return (
        isinstance(max_tokens, int)
        and max_tokens <= num_prefill_output_tokens
    )


def _build_prefill_only_response(
    session: PromptStreamProxySession,
    response_json: dict[str, Any],
) -> dict[str, Any]:
    """构造无需启动 D 时的 OpenAI completions 响应。"""
    prompt_token_ids = response_json.get("prompt_token_ids") or []
    output_token_ids = response_json.get("output_token_ids") or []
    completion_tokens = len(output_token_ids)
    prompt_tokens = len(prompt_token_ids)
    finish_reason = response_json.get("output_finish_reason")
    if finish_reason != "stop":
        finish_reason = "length"
    return {
        "id": f"cmpl-{session.request_id}",
        "object": "text_completion",
        "created": int(time.time()),
        "model": session.base_request.get("model"),
        "choices": [
            {
                "index": 0,
                "text": response_json.get("output_text", ""),
                "logprobs": None,
                "finish_reason": finish_reason,
                "stop_reason": response_json.get("output_stop_reason"),
                "token_ids": output_token_ids,
            }
        ],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
    }


def _merge_prefill_output(
    response_json: dict[str, Any],
    prefill_response: dict[str, Any],
    num_prefill_output_tokens: int,
) -> dict[str, Any]:
    """把 P 首 token 合并回 D 的非流式 completions 响应。"""
    if num_prefill_output_tokens <= 0:
        return response_json
    choices = response_json.get("choices")
    if not isinstance(choices, list) or len(choices) != 1:
        raise HTTPException(
            status_code=502,
            detail="D node must return exactly one completion choice.",
        )
    choice = choices[0]
    choice["text"] = prefill_response.get("output_text", "") + choice.get(
        "text", ""
    )
    token_ids = choice.get("token_ids")
    if isinstance(token_ids, list):
        choice["token_ids"] = list(
            prefill_response.get("output_token_ids") or []
        ) + token_ids

    usage = response_json.get("usage")
    if isinstance(usage, dict):
        prompt_tokens = usage.get("prompt_tokens")
        completion_tokens = usage.get("completion_tokens")
        if isinstance(prompt_tokens, int):
            usage["prompt_tokens"] = max(
                0,
                prompt_tokens - num_prefill_output_tokens,
            )
        if isinstance(completion_tokens, int):
            usage["completion_tokens"] = (
                completion_tokens + num_prefill_output_tokens
            )
    return response_json


def _build_prefill_start_request(req_data: dict[str, Any]) -> dict[str, Any]:
    """构造发给 P 节点的 start 请求。"""
    prefill_request = dict(req_data)
    prefill_request["kv_transfer_params"] = {
        "do_remote_decode": True,
        "do_remote_prefill": False,
        "remote_engine_id": None,
        "remote_block_ids": None,
        "remote_host": None,
        "remote_port": None,
    }
    prefill_request["stream"] = False
    prefill_request["max_tokens"] = 1
    prefill_request.pop("stream_options", None)
    prefill_request.pop("min_tokens", None)
    prefill_request.pop("min_completion_tokens", None)
    return prefill_request


def _format_upstream_error(
    service_name: str,
    endpoint: str,
    response: httpx.Response,
) -> dict[str, Any]:
    """把上游 P/D 节点错误整理成 proxy 可读的错误体。

    Args:
        service_name: 上游服务名称，例如 `prefill` 或 `decode`。
        endpoint: proxy 调用的上游 endpoint。
        response: 上游 HTTP 响应。

    Returns:
        可以直接放入 `HTTPException.detail` 的错误字典。
    """
    try:
        upstream_body: Any = response.json()
    except ValueError:
        upstream_body = response.text
    return {
        "service": service_name,
        "endpoint": endpoint,
        "status_code": response.status_code,
        "body": upstream_body,
    }


async def _post_json(
    client_info: dict[str, Any],
    endpoint: str,
    data: dict[str, Any],
    request_id: str,
    service_name: str = "prefill",
) -> dict[str, Any]:
    """向上游 P/D 节点发送 JSON 请求并返回 JSON 响应。"""
    try:
        response = await client_info["client"].post(
            endpoint,
            json=data,
            headers=_authorization_headers(request_id),
        )
    except httpx.RequestError as exc:
        upstream = f"{client_info['host']}:{client_info['port']}{endpoint}"
        raise HTTPException(
            status_code=502,
            detail={
                "service": service_name,
                "endpoint": endpoint,
                "upstream": upstream,
                "error": str(exc),
            },
        ) from exc
    if response.is_error:
        raise HTTPException(
            status_code=response.status_code,
            detail=_format_upstream_error(service_name, endpoint, response),
        )
    return response.json()


@app.post("/v1/prompt_stream/start")
async def start_prompt_stream(request: Request):
    """创建 P/D prompt stream 会话。"""
    req_data = await request.json()
    api_endpoint = _get_api_endpoint(req_data)
    if api_endpoint != "/completions":
        raise HTTPException(
            status_code=400,
            detail=(
                "Prompt stream segment KV currently supports only "
                "the completions API."
            ),
        )
    if req_data.get("stream", False):
        raise HTTPException(
            status_code=400,
            detail="Prompt stream proxy currently supports stream=false only.",
        )
    prefill_client_info = get_next_client(request.app, "prefill")
    decode_client_info = get_next_client(request.app, "decode")

    # P 节点会返回内部 request_id；后续事件必须使用这个 id。
    response_json = await _post_json(
        prefill_client_info,
        "/prompt_stream/start",
        _build_prefill_start_request(req_data),
        req_data.get("request_id", "prompt-stream-start"),
    )
    request_id = response_json["request_id"]

    request.app.state.prompt_stream_sessions[request_id] = PromptStreamProxySession(
        request_id=request_id,
        prefill_client_info=prefill_client_info,
        decode_client_info=decode_client_info,
        api_endpoint=api_endpoint,
        base_request=_build_decode_base_request(req_data),
        decode_prompt_text=_get_initial_decode_prompt_text(req_data),
        decode_prompt_token_ids=_get_initial_decode_prompt_token_ids(
            req_data,
            response_json,
        ),
    )
    return JSONResponse(content=response_json)


@app.post("/v1/prompt_stream/event")
async def add_prompt_stream_event(request: Request):
    """转发 Append / Update / Source Finished 事件到绑定的 P 节点。"""
    req_data = await request.json()
    request_id = req_data.get("request_id")
    session = request.app.state.prompt_stream_sessions.get(request_id)
    if session is None:
        raise HTTPException(
            status_code=404,
            detail=f"Unknown prompt stream request_id={request_id}",
        )

    req_data["request_id"] = session.request_id
    response_json = await _post_json(
        session.prefill_client_info,
        "/prompt_stream/event",
        req_data,
        session.request_id,
    )
    _append_decode_prompt_token_ids(session, response_json)
    _append_decode_prompt_text(session, req_data.get("event", {}))
    return JSONResponse(content=response_json)


@app.post("/v1/prompt_stream/finish")
async def finish_prompt_stream(request: Request):
    """完成 P 节点 prefill，并把 kv_transfer_params 交给 D 节点 decode。"""
    req_data = await request.json()
    request_id = req_data.get("request_id")
    session = request.app.state.prompt_stream_sessions.get(request_id)
    if session is None:
        raise HTTPException(
            status_code=404,
            detail=f"Unknown prompt stream request_id={request_id}",
        )

    req_data["request_id"] = session.request_id
    prefill_response = await _post_json(
        session.prefill_client_info,
        "/prompt_stream/finish",
        req_data,
        session.request_id,
    )
    # P finish 成功后长期 Request 与 P session 已经结束；proxy 也采用
    # at-most-once finish，后续 D/校验失败不再保留一个无法重试的空壳会话。
    request.app.state.prompt_stream_sessions.pop(request_id, None)
    _finish_decode_prompt_token_ids(session, prefill_response)
    _finish_decode_prompt_text(session, req_data.get("event", {}))
    num_prefill_output_tokens = 0
    if not global_args.disable_kv_transfer:
        num_prefill_output_tokens = _append_prefill_output_token_ids(
            session,
            prefill_response,
        )
        if num_prefill_output_tokens > 1:
            raise HTTPException(
                status_code=502,
                detail=(
                    "P node returned more than one prefill output token: "
                    f"{num_prefill_output_tokens}."
                ),
            )
        _validate_prompt_stream_transfer(
            session,
            prefill_response,
            num_prefill_output_tokens,
        )

    decode_request = dict(session.base_request)
    if not global_args.disable_kv_transfer and _should_finish_without_decode(
        decode_request,
        prefill_response,
        num_prefill_output_tokens,
    ):
        response_json = _build_prefill_only_response(session, prefill_response)
        return JSONResponse(
            content=response_json,
        )
    _shrink_decode_max_tokens(decode_request, num_prefill_output_tokens)
    _apply_decode_prompt_to_request(session, decode_request)
    if not global_args.disable_kv_transfer:
        decode_request["kv_transfer_params"] = prefill_response["kv_transfer_params"]
    prompt = decode_request.get("prompt")
    prompt_len = len(prompt) if isinstance(prompt, list) else None
    kv_transfer_params = decode_request.get("kv_transfer_params") or {}
    logger.info(
        "Proxy finish decode request: request_id=%s prompt_token_len=%s "
        "remote_num_tokens=%s remote_block_counts=%s "
        "num_prefill_output_tokens=%s disable_kv_transfer=%s",
        session.request_id,
        prompt_len,
        kv_transfer_params.get("remote_num_tokens"),
        [len(group) for group in kv_transfer_params.get("remote_block_ids", [])],
        num_prefill_output_tokens,
        global_args.disable_kv_transfer,
    )

    decode_response = await _post_json(
        session.decode_client_info,
        session.api_endpoint,
        decode_request,
        session.request_id,
        service_name="decode",
    )
    response_json = _merge_prefill_output(
        decode_response,
        prefill_response,
        num_prefill_output_tokens,
    )
    return JSONResponse(content=response_json)


@app.post("/v1/prompt_stream/abort")
async def abort_prompt_stream(request: Request):
    """取消 proxy、P 节点和 D 节点上的 prompt stream 会话。"""
    req_data = await request.json()
    request_id = req_data.get("request_id")
    session = request.app.state.prompt_stream_sessions.pop(request_id, None)
    if session is None:
        raise HTTPException(
            status_code=404,
            detail=f"Unknown prompt stream request_id={request_id}",
        )

    await _post_json(
        session.prefill_client_info,
        "/prompt_stream/abort",
        {"request_id": session.request_id},
        session.request_id,
    )
    return JSONResponse(content={"request_id": request_id, "status": "aborted"})


@app.get("/healthcheck")
async def healthcheck():
    """返回代理服务器健康状态。"""
    return {
        "status": "ok",
        "prefill_instances": len(app.state.prefill_clients),
        "decode_instances": len(app.state.decode_clients),
        "prompt_stream_sessions": len(app.state.prompt_stream_sessions),
    }


if __name__ == "__main__":
    global global_args
    global_args = parse_args()

    import uvicorn

    uvicorn.run(app, host=global_args.host, port=global_args.port)
