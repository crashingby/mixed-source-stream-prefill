# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import argparse
import itertools
import json
import logging
import os
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any

import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


@dataclass
class PromptStreamProxySession:
    """代理服务器保存的 prompt stream 会话状态。"""

    request_id: str
    prefill_client_info: dict[str, Any]
    # TTFT-only 会话从创建时起就是 P-only，因此不会选择 D 节点客户端。
    decode_client_info: dict[str, Any] | None
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
            "for comparing remote KV decode against local D prefill. "
            "Masked-padding prompt streams require KV transfer and reject "
            "this debug mode."
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
    # probe 只属于 P 的内部执行，不能作为 D 的生成参数。
    base_request.pop("probe_query", None)
    base_request.pop("probe_keep_ratio", None)
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


def _validate_prompt_stream_padding_manifest(
    response_json: dict[str, Any],
) -> dict[str, int] | None:
    """校验 P 的 padding 语义，供 PD 转发与 P-only 首帧共同使用。

    padding 的 token id 可能与真实文本相同，因此只能使用 P 明确导出的
    物理区间。该验证不读取远端 block IDs，也适用于无 connector 的输出。

    Args:
        response_json: P 的 finish JSON，包含物理 prompt 和可选语义清单。

    Returns:
        已校验的原始区间字典；未开启 masked padding 时返回 None。

    Raises:
        HTTPException: P 返回的版本、区间或逻辑/物理长度不一致。
    """
    params = response_json.get("kv_transfer_params")
    if not isinstance(params, dict) or "prompt_stream_masked_padding" not in params:
        return None
    padding = params["prompt_stream_masked_padding"]
    physical_tokens = response_json.get("prompt_token_ids")
    physical_len = params.get("prompt_stream_physical_prompt_len")
    logical_len = params.get("prompt_stream_logical_prompt_len")
    if (
        not isinstance(padding, dict)
        or set(padding) != {"version", "start", "end"}
        or any(type(padding[key]) is not int for key in ("version", "start", "end"))
        or padding["version"] != 1
        or type(physical_len) is not int
        or type(logical_len) is not int
        or not isinstance(physical_tokens, list)
        or physical_len != len(physical_tokens)
        or not 0 <= padding["start"] <= padding["end"] < physical_len
        or logical_len != physical_len - padding["end"] + padding["start"]
    ):
        raise HTTPException(
            status_code=502,
            detail="P node returned an invalid masked-padding manifest.",
        )
    return padding


def _validate_prompt_stream_transfer(
    session: PromptStreamProxySession,
    response_json: dict[str, Any],
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
    if type(remote_num_tokens) is not int or remote_num_tokens < 0:
        raise HTTPException(
            status_code=502,
            detail=("P node did not return a non-negative integer remote_num_tokens."),
        )
    physical_prompt_len = len(physical_prompt_token_ids)
    padding = _validate_prompt_stream_padding_manifest(response_json)
    reported_physical_prompt_len = kv_transfer_params.get(
        "prompt_stream_physical_prompt_len"
    )
    if (
        type(reported_physical_prompt_len) is not int
        or reported_physical_prompt_len != physical_prompt_len
    ):
        raise HTTPException(
            status_code=502,
            detail=(
                "P node returned inconsistent physical prompt lengths: "
                f"tokens={physical_prompt_len}, "
                f"metadata={reported_physical_prompt_len}."
            ),
        )
    decode_prompt_len = physical_prompt_len
    if kv_transfer_params.get("prompt_stream_exact_transfer") is not True:
        raise HTTPException(
            status_code=502,
            detail="P node did not mark prompt stream KV as exact transfer.",
        )
    if remote_num_tokens != physical_prompt_len:
        raise HTTPException(
            status_code=502,
            detail=(
                "Prompt stream KV/token mismatch: "
                f"physical_prompt_len={physical_prompt_len}, "
                f"remote_num_tokens={remote_num_tokens}, "
                f"decode_prompt_len={decode_prompt_len}."
            ),
        )
    reported_output_tokens = kv_transfer_params.get(
        "prompt_stream_prefill_output_tokens"
    )
    if type(reported_output_tokens) is not int or reported_output_tokens != 0:
        raise HTTPException(
            status_code=502,
            detail=(
                "P node returned inconsistent prefill output-token metadata: "
                "expected=0, "
                f"manifest={reported_output_tokens}."
            ),
        )
    remote_block_ids = kv_transfer_params.get("remote_block_ids")
    manifest_block_counts = kv_transfer_params.get("prompt_stream_remote_block_counts")
    manifest_block_sizes = kv_transfer_params.get("prompt_stream_remote_block_sizes")
    if (
        not isinstance(remote_block_ids, (list, tuple))
        or not isinstance(manifest_block_counts, (list, tuple))
        or not isinstance(manifest_block_sizes, (list, tuple))
    ):
        raise HTTPException(
            status_code=502,
            detail="P node returned an incomplete prompt stream block manifest.",
        )
    if not (
        len(remote_block_ids) == len(manifest_block_counts) == len(manifest_block_sizes)
    ):
        raise HTTPException(
            status_code=502,
            detail=(
                "P node returned inconsistent prompt stream KV-group counts: "
                f"remote={len(remote_block_ids)}, "
                f"block_counts={len(manifest_block_counts)}, "
                f"block_sizes={len(manifest_block_sizes)}."
            ),
        )
    if not all(
        type(block_count) is int and block_count >= 0
        for block_count in manifest_block_counts
    ) or not all(
        type(block_size) is int and block_size > 0
        for block_size in manifest_block_sizes
    ):
        raise HTTPException(
            status_code=502,
            detail="P node returned invalid prompt stream block counts or sizes.",
        )
    # UPDATE 段从 block 边界开始；mask 只能遮住 PREFIX_APPEND 的尾部空位。
    if padding is not None and any(
        padding["end"] % block_size != 0
        or padding["end"] - padding["start"] >= block_size
        for block_size in manifest_block_sizes
    ):
        raise HTTPException(
            status_code=502,
            detail="P node masked-padding range is not a segment boundary.",
        )
    if not all(isinstance(group, (list, tuple)) for group in remote_block_ids):
        raise HTTPException(
            status_code=502,
            detail="P node returned invalid prompt stream remote block groups.",
        )
    if not all(
        type(block_id) is int and block_id >= 0
        for group in remote_block_ids
        for block_id in group
    ):
        raise HTTPException(
            status_code=502,
            detail="P node returned invalid prompt stream remote block IDs.",
        )
    remote_block_counts = [len(group) for group in remote_block_ids]
    if remote_block_counts != list(manifest_block_counts):
        raise HTTPException(
            status_code=502,
            detail=(
                "P node returned inconsistent prompt stream block counts: "
                f"remote={remote_block_counts}, "
                f"manifest={list(manifest_block_counts)}."
            ),
        )
    derived_block_counts = [
        (physical_prompt_len + block_size - 1) // block_size
        for block_size in manifest_block_sizes
    ]
    if list(manifest_block_counts) != derived_block_counts:
        raise HTTPException(
            status_code=502,
            detail=(
                "P node block manifest does not cover the physical prompt "
                f"exactly: manifest={list(manifest_block_counts)}, "
                f"derived={derived_block_counts}."
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


def _attach_prompt_stream_metadata(
    response_payload: dict[str, Any],
    prefill_response: dict[str, Any],
) -> dict[str, Any]:
    """把 P 节点统计和 padding 模式附加到客户端的 JSON/SSE 响应。

    统计描述的是 P 节点内部的 Prompt Stream work 与 KV 状态，proxy 只做
    透明透传，不能重新计算。该 helper 只作用于客户端响应 payload；调用方
    不得把结果复用为 D 节点请求，从而保证研究统计不会进入
    ``kv_transfer_params`` 或 decode request。

    Args:
        response_payload: 即将返回给客户端的 JSON 或 SSE payload。
        prefill_response: P 节点 ``/prompt_stream/finish`` 返回的 JSON。

    Returns:
        已附加可选统计及 masked-padding 区间的原 payload。

    Raises:
        HTTPException: P 节点返回了非对象类型的统计字段时抛出。
    """
    # 首 token 与模式元数据同时交付，TTFT-only 客户端无需等待结束帧。
    padding = _validate_prompt_stream_padding_manifest(prefill_response)
    if padding is not None:
        response_payload["prompt_stream_masked_padding"] = dict(padding)
    prompt_stream_stats = prefill_response.get("prompt_stream_stats")
    if prompt_stream_stats is None:
        # 兼容尚未导出统计的 P 节点；simple_test/replay 会对实验结果做
        # fail-closed 校验，proxy 本身仍可服务旧版本节点。
        return response_payload
    if not isinstance(prompt_stream_stats, dict):
        raise HTTPException(
            status_code=502,
            detail=(
                "P node returned non-object prompt_stream_stats: "
                f"{type(prompt_stream_stats).__name__}."
            ),
        )

    # 保持 P 节点的字段和值不变，避免 proxy 产生第二套统计口径。
    response_payload["prompt_stream_stats"] = prompt_stream_stats
    return response_payload


def _build_prefill_start_request(
    req_data: dict[str, Any], *, enable_kv_transfer: bool
) -> dict[str, Any]:
    """构造发给 P 节点的单 token prefill 请求。

    Args:
        req_data: 客户端发给 proxy 的原始请求体。
        enable_kv_transfer: 是否为后续 D 节点 decode 准备 NIXL KV transfer。
            TTFT-only 为 ``False``，P 节点不建立任何远端 KV 生命周期。

    Returns:
        发送给 P 节点的非流式、单 token prefill 请求。
    """
    prefill_request = dict(req_data)

    # proxy 决定 P/D 生命周期，不能让客户端残留的参数绕过 TTFT-only 隔离。
    prefill_request.pop("kv_transfer_params", None)
    if enable_kv_transfer:
        # 完整 P/D 请求由 P 节点持有 KV，等待 D 节点稍后按 block manifest 拉取。
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


def _build_normal_pd_prefill_request(
    req_data: dict[str, Any], *, enable_kv_transfer: bool
) -> dict[str, Any]:
    """构造普通 completions 的 P 节点首 token 请求。

    P 返回实际执行的 prompt token IDs。D 加载这些位置的 KV 后重新计算
    最后一个 prompt token 并采样回答；P 的采样结果仅用于确认 P 已完成。

    Args:
        req_data: 客户端发给 proxy 的原始 completions 请求体。
        enable_kv_transfer: 是否为后续 D decode 保留 P 节点 KV。

    Returns:
        发给 P 节点的非流式、单 token completions 请求体。

    Raises:
        HTTPException: 请求要求多个 choice，当前实验 proxy 无法一一绑定 P/D
            KV 生命周期时抛出。
    """
    num_choices = req_data.get("n", 1)
    if type(num_choices) is not int or num_choices != 1:
        raise HTTPException(
            status_code=400,
            detail="Normal PD completions currently requires n=1.",
        )

    # 复用 prompt-stream P 请求的 KV 配置、非流式 collector 和单 token
    # budget；额外要求返回 token ids，保证 D 的 token view 和 P 完全一致。
    prefill_request = _build_prefill_start_request(
        req_data,
        enable_kv_transfer=enable_kv_transfer,
    )
    prefill_request["return_token_ids"] = True
    return prefill_request


def _get_normal_pd_prefill_choice(
    prefill_response: dict[str, Any],
) -> dict[str, Any]:
    """校验并提取普通 PD P 响应中的唯一 choice。

    Args:
        prefill_response: P 节点 /completions 的 JSON 响应。

    Returns:
        已验证的唯一 choice；其中必须包含 prompt 与单个 output token ids。

    Raises:
        HTTPException: P 响应无法构成 D 节点所需的精确 token 序列时抛出。
    """
    choices = prefill_response.get("choices")
    if not isinstance(choices, list) or len(choices) != 1:
        raise HTTPException(
            status_code=502,
            detail="P node normal PD response must contain exactly one choice.",
        )
    choice = choices[0]
    if not isinstance(choice, dict):
        raise HTTPException(
            status_code=502,
            detail="P node normal PD response choice must be a JSON object.",
        )

    prompt_token_ids = choice.get("prompt_token_ids")
    output_token_ids = choice.get("token_ids")
    if (
        not isinstance(prompt_token_ids, list)
        or not all(type(token_id) is int for token_id in prompt_token_ids)
        or not isinstance(output_token_ids, list)
        or len(output_token_ids) != 1
        or not all(type(token_id) is int for token_id in output_token_ids)
    ):
        raise HTTPException(
            status_code=502,
            detail=(
                "P node normal PD response must include prompt_token_ids and "
                "exactly one output token id."
            ),
        )
    return choice


def _build_normal_pd_decode_request(
    req_data: dict[str, Any],
    prefill_response: dict[str, Any],
) -> dict[str, Any]:
    """把 P 实际执行的原始 prompt 与 KV 参数交给 D，由 D 生成全部回答。"""
    prefill_choice = _get_normal_pd_prefill_choice(prefill_response)
    decode_request = _build_decode_base_request(req_data)
    # 精确 token IDs 避免 D 再次分词；P 的采样结果不属于 D prompt。
    decode_request["prompt"] = list(prefill_choice["prompt_token_ids"])
    decode_request["add_special_tokens"] = False
    if not global_args.disable_kv_transfer:
        params = prefill_response.get("kv_transfer_params")
        if not isinstance(params, dict):
            raise HTTPException(
                status_code=502,
                detail="P node normal PD response is missing kv_transfer_params.",
            )
        decode_request["kv_transfer_params"] = params
    return decode_request


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


def _encode_sse_payload(payload: dict[str, Any]) -> bytes:
    """把一个 OpenAI 风格 JSON chunk 编码成 SSE data frame。

    Args:
        payload: 可 JSON 序列化的单个响应 chunk。

    Returns:
        以 UTF-8 编码的 ``data: ...\\n\\n`` SSE frame。
    """
    serialized_payload = json.dumps(payload, ensure_ascii=False)
    return f"data: {serialized_payload}\n\n".encode()


async def _forward_upstream_stream(
    response: httpx.Response,
) -> AsyncIterator[bytes]:
    """透明转发 D 节点的流式响应，并保证连接最终关闭。

    Args:
        response: 已建立且状态码成功的 D 节点 streaming response。

    Yields:
        D 节点返回的原始 SSE 字节。
    """
    try:
        async for chunk in response.aiter_raw():
            if chunk:
                yield chunk
    finally:
        await response.aclose()


async def _forward_normal_request(
    request: Request,
    endpoint: str,
) -> StreamingResponse | JSONResponse:
    """把普通 OpenAI REST 请求转发到一个 D 节点。

    普通模式不经过 P 节点，也不附加 KV transfer 参数。stream=true 时保留
    上游 SSE；stream=false 时返回普通 JSON。这个接口让实验客户端只需要
    配置一个 proxy 地址。

    Args:
        request: 客户端发送给 proxy 的 FastAPI 请求。
        endpoint: D 节点上的 OpenAI endpoint，例如 /completions。

    Returns:
        StreamingResponse 或 JSONResponse。
    """
    req_data = await request.json()
    decode_client_info = get_next_client(request.app, "decode")
    request_id = request.headers.get("X-Request-Id")
    if not request_id:
        request_id = f"normal-{time.time_ns()}"

    if not req_data.get("stream", False):
        response_json = await _post_json(
            decode_client_info,
            endpoint,
            req_data,
            request_id,
            service_name="decode",
        )
        return JSONResponse(content=response_json)

    upstream_request = decode_client_info["client"].build_request(
        "POST",
        endpoint,
        json=req_data,
        headers=_authorization_headers(request_id),
    )
    try:
        upstream_response = await decode_client_info["client"].send(
            upstream_request,
            stream=True,
        )
    except httpx.RequestError as exc:
        upstream = (
            f"{decode_client_info['host']}:{decode_client_info['port']}{endpoint}"
        )
        raise HTTPException(
            status_code=502,
            detail={
                "service": "decode",
                "endpoint": endpoint,
                "upstream": upstream,
                "error": str(exc),
            },
        ) from exc

    if upstream_response.is_error:
        await upstream_response.aread()
        detail = _format_upstream_error(
            "decode",
            endpoint,
            upstream_response,
        )
        await upstream_response.aclose()
        raise HTTPException(
            status_code=upstream_response.status_code,
            detail=detail,
        )

    return StreamingResponse(
        _forward_upstream_stream(upstream_response),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


def _build_prefill_ready_event(
    request_id: str,
    model: str | None,
    prefill_response: dict[str, Any],
) -> dict[str, Any]:
    """构造 P 首 token 就绪控制事件；不携带回答或占用生成预算。

    Args:
        request_id: 业务请求标识。
        model: 客户端请求的模型名称。
        prefill_response: P 完成响应，供透传 probe 统计和 padding 清单。

    Returns:
        choices 为空的控制事件，可用于 P-only 或完整 PD 的 SSE。
    """
    return _attach_prompt_stream_metadata(
        {
            "id": f"cmpl-{request_id}",
            "object": "prefill_ready",
            "event": "prefill_ready",
            "measurement_scope": "p_side_ttft",
            "created": int(time.time()),
            "model": model,
            "choices": [],
        },
        prefill_response,
    )


async def _stream_pd_completion(
    decode_client_info: dict[str, Any],
    api_endpoint: str,
    request_id: str,
    decode_request: dict[str, Any],
    ready_event: dict[str, Any],
) -> AsyncIterator[bytes]:
    """先通知 P 已完成，再转发 D 的完整回答；D 错误保留为 SSE 错误。"""
    yield _encode_sse_payload(ready_event)
    try:
        async with decode_client_info["client"].stream(
            "POST",
            api_endpoint,
            json=decode_request,
            headers=_authorization_headers(request_id),
        ) as response:
            if response.is_error:
                await response.aread()
                yield _encode_sse_payload(
                    {"error": _format_upstream_error("decode", api_endpoint, response)}
                )
                yield b"data: [DONE]\n\n"
                return
            async for chunk in response.aiter_raw():
                if chunk:
                    yield chunk
    except httpx.RequestError as exc:
        yield _encode_sse_payload({"error": {"service": "decode", "message": str(exc)}})
        yield b"data: [DONE]\n\n"


async def _handle_normal_pd_completions(
    request: Request,
    *,
    ttft_only: bool,
) -> StreamingResponse | JSONResponse | Response:
    """执行普通 PD；TTFT-only 返回 P 就绪事件，完整请求由 D 返回全部回答。"""
    req_data = await request.json()
    request_id = req_data.get("request_id")
    if not isinstance(request_id, str) or not request_id:
        request_id = request.headers.get("X-Request-Id")
    if not request_id:
        request_id = f"normal-pd-{time.time_ns()}"

    # 常规 PD 只处理一个 choice；多 choice 的 P/D KV 绑定不在本实验范围内。
    # TTFT-only 在 P 节点产出首 token 后立即结束，不能建立虚假的 D 侧 KV
    # transfer 生命周期；完整 PD 路径则保留既有的 producer 配置。
    prefill_request = _build_normal_pd_prefill_request(
        req_data,
        enable_kv_transfer=not ttft_only,
    )
    prefill_request["request_id"] = request_id
    prefill_client_info = get_next_client(request.app, "prefill")
    prefill_response = await _post_json(
        prefill_client_info,
        "/completions",
        prefill_request,
        request_id,
        service_name="prefill",
    )
    _get_normal_pd_prefill_choice(prefill_response)
    ready_event = _build_prefill_ready_event(
        request_id, req_data.get("model"), prefill_response
    )
    if ttft_only:
        # P-only 从创建时即禁用 KV producer，完成后不产生 D 生命周期。
        return Response(
            content=_encode_sse_payload(ready_event) + b"data: [DONE]\n\n",
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    decode_request = _build_normal_pd_decode_request(req_data, prefill_response)
    decode_client_info = get_next_client(request.app, "decode")
    if decode_request.get("stream", False):
        return StreamingResponse(
            _stream_pd_completion(
                decode_client_info,
                "/completions",
                request_id,
                decode_request,
                ready_event,
            ),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )
    decode_response = await _post_json(
        decode_client_info,
        "/completions",
        decode_request,
        request_id,
        service_name="decode",
    )
    return JSONResponse(content=decode_response)


@app.post("/v1/completions", response_model=None)
async def forward_normal_completions(
    request: Request,
) -> StreamingResponse | JSONResponse | Response:
    """通过标准 P→D 分离处理普通 completions 请求。"""
    return await _handle_normal_pd_completions(request, ttft_only=False)


@app.post("/v1/completions_ttft", response_model=None)
async def forward_normal_completions_ttft(
    request: Request,
) -> Response:
    """只执行普通 PD 的 P 侧采样，用完成通知测量 P 侧 TTFT。"""
    response = await _handle_normal_pd_completions(request, ttft_only=True)
    assert isinstance(response, Response)
    return response


@app.post("/v1/chat/completions", response_model=None)
async def forward_normal_chat_completions(
    request: Request,
) -> StreamingResponse | JSONResponse:
    """转发普通 chat completions 请求到 D 节点。"""
    return await _forward_normal_request(request, "/chat/completions")


@app.post("/v1/responses", response_model=None)
async def forward_normal_responses(
    request: Request,
) -> StreamingResponse | JSONResponse:
    """转发普通 Responses API 请求到 D 节点。"""
    return await _forward_normal_request(request, "/responses")


async def _start_prompt_stream(
    request: Request, *, enable_kv_transfer: bool
) -> JSONResponse:
    """创建完整 P/D 或 P-only 的 prompt-stream 会话。

    Args:
        request: 客户端的 prompt-stream start 请求。
        enable_kv_transfer: ``True`` 时为后续 D decode 建立 NIXL producer
            生命周期；``False`` 时创建严格 P-only 的 TTFT 会话。

    Returns:
        P 节点 start 响应，其中包含后续事件所需的内部 request id。
    """
    req_data = await request.json()
    api_endpoint = _get_api_endpoint(req_data)
    if api_endpoint != "/completions":
        raise HTTPException(
            status_code=400,
            detail=(
                "Prompt stream segment KV currently supports only the completions API."
            ),
        )
    # P 节点自身的 start 始终由 _build_prefill_start_request 强制使用非流式
    # collector；是否建立 D 侧 KV 生命周期由独立的 start 路由决定。
    prefill_client_info = get_next_client(request.app, "prefill")
    decode_client_info = (
        get_next_client(request.app, "decode") if enable_kv_transfer else None
    )

    # P 节点会返回内部 request_id；后续事件必须使用这个 id。
    response_json = await _post_json(
        prefill_client_info,
        "/prompt_stream/start",
        _build_prefill_start_request(
            req_data,
            enable_kv_transfer=enable_kv_transfer,
        ),
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


@app.post("/v1/prompt_stream/start")
async def start_prompt_stream(request: Request):
    """创建需要把 P 节点 KV 交给 D 节点的完整 P/D 会话。"""
    return await _start_prompt_stream(request, enable_kv_transfer=True)


@app.post("/v1/prompt_stream/start_ttft")
async def start_prompt_stream_ttft(request: Request):
    """创建严格 P-only 的 prompt-stream TTFT 会话。

    该路由不会向 P 传递 ``kv_transfer_params``。因此即使 P 已启用 NIXL，
    请求也不会注册为待 D 拉取的 producer request。
    """
    return await _start_prompt_stream(request, enable_kv_transfer=False)


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
    # 只有完整 P/D 会话需要维护 D 请求的精确 token/text 镜像。
    if session.decode_client_info is not None:
        _append_decode_prompt_token_ids(session, response_json)
        _append_decode_prompt_text(session, req_data.get("event", {}))
    return JSONResponse(content=response_json)


@app.post("/v1/prompt_stream/finish")
async def finish_prompt_stream(request: Request):
    """完成 P 节点 prefill，并把必需的 KV 参数交给 D 节点 decode。"""
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
    # TTFT-only 没有 D 请求，避免为无消费者的 decode 镜像继续做状态更新。
    if session.decode_client_info is not None:
        _finish_decode_prompt_token_ids(session, prefill_response)
        _finish_decode_prompt_text(session, req_data.get("event", {}))

    # 完整 `/finish` 一定会进入 D decode。即使 proxy 的 debug 选项关闭了
    # KV load，P 仍必须导出完整物理 prompt 视图，避免把仅适用于 TTFT-only
    # 的无 connector P 输出错误地用于完整 P/D 请求。
    kv_transfer_params = prefill_response.get("kv_transfer_params")
    if (
        not isinstance(kv_transfer_params, dict)
        or kv_transfer_params.get("do_remote_prefill") is not True
        or not isinstance(kv_transfer_params.get("remote_block_ids"), (list, tuple))
    ):
        raise HTTPException(
            status_code=502,
            detail=(
                "P node full prompt-stream finish requires "
                "kv_transfer_params; use /v1/prompt_stream/finish_ttft "
                "when KV transfer is disabled."
            ),
        )

    # 无 connector 时也可能有 padding 语义清单，但它不代表存在可拉取的 KV。
    # debug 重算若直接丢掉该清单，D 会重新让 padding 参与注意力与 RoPE。
    padding = _validate_prompt_stream_padding_manifest(prefill_response)
    if global_args.disable_kv_transfer and padding is not None:
        raise HTTPException(
            status_code=409,
            detail=(
                "--disable-kv-transfer cannot discard masked-padding semantics. "
                "Enable KV transfer, or use /v1/prompt_stream/finish_ttft."
            ),
        )

    if not global_args.disable_kv_transfer:
        _validate_prompt_stream_transfer(session, prefill_response)
    decode_request = dict(session.base_request)
    _apply_decode_prompt_to_request(session, decode_request)
    if not global_args.disable_kv_transfer:
        decode_request["kv_transfer_params"] = prefill_response["kv_transfer_params"]
    decode_client_info = session.decode_client_info
    if decode_client_info is None:
        raise HTTPException(
            status_code=409, detail="P-only sessions must use /finish_ttft."
        )
    if decode_request.get("stream", False):
        return StreamingResponse(
            _stream_pd_completion(
                decode_client_info,
                session.api_endpoint,
                session.request_id,
                decode_request,
                _build_prefill_ready_event(
                    session.request_id,
                    session.base_request.get("model"),
                    prefill_response,
                ),
            ),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )
    decode_response = await _post_json(
        decode_client_info,
        session.api_endpoint,
        decode_request,
        session.request_id,
        service_name="decode",
    )
    return JSONResponse(
        content=_attach_prompt_stream_metadata(decode_response, prefill_response)
    )


@app.post("/v1/prompt_stream/finish_ttft")
async def finish_prompt_stream_ttft(request: Request):
    """完成 P 侧采样并返回 prefill_ready 控制事件，不启动 D。"""
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
    # P finish 成功后长期 Request 已结束；TTFT-only 不会再访问 D 节点，
    # 因此 proxy 会话也可以立即释放。
    request.app.state.prompt_stream_sessions.pop(request_id, None)
    _finish_decode_prompt_token_ids(session, prefill_response)
    _finish_decode_prompt_text(session, req_data.get("event", {}))

    output_token_ids = prefill_response.get("output_token_ids")
    if not isinstance(output_token_ids, list) or len(output_token_ids) != 1:
        raise HTTPException(
            status_code=502,
            detail=(
                "P node TTFT finish must return exactly one output token: "
                f"output_token_ids={output_token_ids!r}."
            ),
        )

    # TTFT-only 的响应体只有两帧，且两帧均已在此时完全构造完毕。直接写入
    # Response 而不是交给 StreamingResponse 的异步 generator，可避免 HTTP
    # headers 已发出、但 generator 尚未获得调度时，replay 端无限等待 body。
    sse_body = (
        _encode_sse_payload(
            _build_prefill_ready_event(
                session.request_id,
                session.base_request.get("model"),
                prefill_response,
            )
        )
        + b"data: [DONE]\n\n"
    )
    logger.info(
        "Proxy TTFT-only finish completed: request_id=%s "
        "prompt_stream_blocks=%s output_tokens=%s sse_bytes=%s",
        session.request_id,
        [
            len(group)
            for group in (
                prefill_response.get("kv_transfer_params", {}).get(
                    "remote_block_ids",
                    [],
                )
            )
        ],
        len(output_token_ids),
        len(sse_body),
    )
    return Response(
        content=sse_body,
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


@app.post("/v1/prompt_stream/abort")
async def abort_prompt_stream(request: Request):
    """幂等地取消 proxy 和 P 节点上的 prompt stream 会话。

    P finish 成功后 proxy 会立即移除本地 session。此后即使 D decode 或
    NIXL 传输失败，replay 的 finally 清理仍应收到成功响应，而不是把已结束
    的会话误报成 REST 路由不存在。
    """
    req_data = await request.json()
    request_id = req_data.get("request_id")
    if not isinstance(request_id, str):
        raise HTTPException(
            status_code=400,
            detail="prompt_stream abort requires string request_id.",
        )
    session = request.app.state.prompt_stream_sessions.get(request_id)
    if session is None:
        return JSONResponse(
            content={
                "request_id": request_id,
                "status": "already_finished_or_aborted",
            }
        )

    await _post_json(
        session.prefill_client_info,
        "/prompt_stream/abort",
        {"request_id": session.request_id},
        session.request_id,
    )
    # Only forget the binding after P confirms the abort. On a network error,
    # keep it so the caller can retry instead of leaking an orphan P request.
    request.app.state.prompt_stream_sessions.pop(request_id, None)
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
