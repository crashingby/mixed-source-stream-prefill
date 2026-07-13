# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import argparse
import itertools
import logging
import os
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any

import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse

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
    if session.decode_prompt_text is None:
        return
    decode_request["prompt"] = session.decode_prompt_text


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
    response = await client_info["client"].post(
        endpoint,
        json=data,
        headers=_authorization_headers(request_id),
    )
    if response.is_error:
        raise HTTPException(
            status_code=response.status_code,
            detail=_format_upstream_error(service_name, endpoint, response),
        )
    return response.json()


async def _stream_decode_response(
    session: PromptStreamProxySession,
    req_data: dict[str, Any],
):
    """从 D 节点流式转发 decode 输出。"""
    async with session.decode_client_info["client"].stream(
        "POST",
        session.api_endpoint,
        json=req_data,
        headers=_authorization_headers(session.request_id),
    ) as response:
        response.raise_for_status()
        async for chunk in response.aiter_bytes():
            yield chunk


@app.post("/v1/prompt_stream/start")
async def start_prompt_stream(request: Request):
    """创建 P/D prompt stream 会话。"""
    req_data = await request.json()
    api_endpoint = _get_api_endpoint(req_data)
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
    _finish_decode_prompt_text(session, req_data.get("event", {}))

    decode_request = dict(session.base_request)
    _apply_decode_prompt_to_request(session, decode_request)
    decode_request["kv_transfer_params"] = prefill_response["kv_transfer_params"]

    if decode_request.get("stream", False):
        return StreamingResponse(
            _stream_decode_response(session, decode_request),
            media_type="application/json",
        )

    response = await session.decode_client_info["client"].post(
        session.api_endpoint,
        json=decode_request,
        headers=_authorization_headers(session.request_id),
    )
    if response.is_error:
        raise HTTPException(
            status_code=response.status_code,
            detail=_format_upstream_error("decode", session.api_endpoint, response),
        )
    return JSONResponse(content=response.json())


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
