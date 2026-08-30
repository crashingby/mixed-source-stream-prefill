# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import argparse
import json
import time
from typing import Any

import httpx

from prompt_stream_stats import (
    parse_prompt_stream_stats,
    print_prompt_stream_stats,
)


DEFAULT_PREFIX = (
    "You are a helpful assistant. Read the latest streaming context snapshot "
    "first, then answer the final user question.\n\n"
)

DEFAULT_UPDATE_SNAPSHOTS = [
    "Document snapshot: The proxy receives stream-prefill events before "
    "decode.\n",
    "Document snapshot: The proxy receives stream-prefill events before it "
    "sends decode to the D node.\n",
    "Document snapshot: The proxy receives stream-prefill events before it "
    "sends the final decode request to the D node.\n",
]

DEFAULT_FINAL_QUESTION = (
    "\nUser question: According to the latest document snapshot, what does the "
    "proxy do before decode?\nAnswer:"
)


def parse_args() -> argparse.Namespace:
    """解析 update-only 前端示例脚本的启动参数。

    这个脚本模拟一个只有 update source 的 RAG 前端。每个 update 事件都表示
    该来源当前的完整快照，而不是相对于上一轮的增量 diff。P 节点会在来源内
    比较新旧快照的 LCP，并只重新 prefill 失效后的 update 后缀。

    Returns:
        命令行参数命名空间。
    """
    parser = argparse.ArgumentParser(
        description="Send an update-only prompt-stream demo to the proxy server."
    )
    parser.add_argument(
        "--proxy-url",
        type=str,
        default="http://127.0.0.1:8000",
        help="Prompt stream proxy base URL, without trailing /v1.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="qwen2.5-3b",
        help="Served model name forwarded to the P and D vLLM servers.",
    )
    parser.add_argument(
        "--request-id",
        type=str,
        default=None,
        help="Optional external request id for the initial start request.",
    )
    parser.add_argument(
        "--prefix",
        type=str,
        default=DEFAULT_PREFIX,
        help="Fixed prefix/template text sent in start.",
    )
    parser.add_argument(
        "--final-question",
        type=str,
        default=DEFAULT_FINAL_QUESTION,
        help="Final user question sent as the prompt_finished event.",
    )
    parser.add_argument(
        "--snapshot",
        action="append",
        dest="snapshots",
        default=None,
        help=(
            "Full update snapshot text. Can be specified multiple times. "
            "Each value replaces the previous snapshot for the same source."
        ),
    )
    parser.add_argument(
        "--source-id",
        type=str,
        default="demo_update_source",
        help="Logical update source id used by all demo snapshots.",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=64,
        help="Max decode tokens forwarded to the D node after finish.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Decode temperature forwarded to the D node after finish.",
    )
    parser.add_argument(
        "--event-delay",
        type=float,
        default=0.0,
        help="Sleep seconds between update events, useful for manual tracing.",
    )
    return parser.parse_args()


def _endpoint(proxy_url: str, path: str) -> str:
    """把 proxy base URL 和 RESTFUL path 拼成完整 URL。

    Args:
        proxy_url: 代理服务器 base URL，例如 `http://127.0.0.1:8000`。
        path: 以 `/v1/...` 开头的接口路径。

    Returns:
        完整 HTTP URL。
    """
    return f"{proxy_url.rstrip('/')}{path}"


def _print_json(title: str, payload: dict[str, Any]) -> None:
    """以稳定格式打印一次 JSON 请求或响应。

    Args:
        title: 打印段落标题。
        payload: 需要展示的 JSON 对象。
    """
    print(f"\n=== {title} ===")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


def _post_json(
    client: httpx.Client,
    proxy_url: str,
    path: str,
    payload: dict[str, Any],
) -> dict[str, Any]:
    """向 proxy 发送 JSON 请求并解析 JSON 响应。

    Args:
        client: 复用连接的同步 HTTP client。
        proxy_url: proxy base URL。
        path: `/v1/prompt_stream/...` 路径。
        payload: 请求体。

    Returns:
        proxy 返回的 JSON 对象。
    """
    response = client.post(_endpoint(proxy_url, path), json=payload)
    if response.is_error:
        print(f"\n=== error response from {path} ===")
        print(response.text)
        response.raise_for_status()
    return response.json()


def _build_start_payload(args: argparse.Namespace) -> dict[str, Any]:
    """构造发给 proxy `/start` 的请求体。

    Args:
        args: 命令行参数。

    Returns:
        与 `research/scripts/proxy_server.py` 对齐的 start JSON。
    """
    payload: dict[str, Any] = {
        "api": "completions",
        "model": args.model,
        "prompt": args.prefix,
        "stream": False,
        "max_tokens": args.max_tokens,
        "temperature": args.temperature,
    }

    # request_id 是可选外部 id；P 节点会返回真正用于后续事件的内部 id。
    if args.request_id is not None:
        payload["request_id"] = args.request_id
    return payload


def _build_update_event(
    request_id: str,
    source_id: str,
    sequence_no: int,
    snapshot_text: str,
) -> dict[str, Any]:
    """构造一个 update 文本快照事件 envelope。

    Args:
        request_id: proxy start 返回的内部 request id。
        source_id: update 来源 id。
        sequence_no: 来源内单调递增序号，同时作为 demo 版本号。
        snapshot_text: 当前来源的完整文本快照。

    Returns:
        与 proxy `/event` 接口对齐的 JSON envelope。
    """
    return {
        "request_id": request_id,
        "event": {
            "source_id": source_id,
            "event_type": "update",
            "source_type": "update",
            "chunk_id": f"{source_id}-snapshot-{sequence_no}",
            "version": sequence_no,
            "sequence_no": sequence_no,
            "text": snapshot_text,
            "arrival_time": time.time(),
        },
    }


def _build_finish_event(
    request_id: str,
    final_question: str,
) -> dict[str, Any]:
    """构造 prompt_finished 事件 envelope。

    Args:
        request_id: proxy start 返回的内部 request id。
        final_question: 最后的用户问题原始文本。

    Returns:
        与 proxy `/finish` 接口对齐的 JSON envelope。
    """
    return {
        "request_id": request_id,
        "event": {
            "source_id": "final_question",
            "event_type": "prompt_finished",
            "source_type": None,
            "chunk_id": "final-question",
            "version": 0,
            "sequence_no": 0,
            "text": final_question,
            "arrival_time": time.time(),
        },
    }


def run_update_demo(args: argparse.Namespace) -> None:
    """运行一次 update-only prompt stream 端到端示例。

    这个函数只负责前端侧协议验证：它不会自己 tokenize，也不会直接访问
    P/D 节点；所有请求都先发送给 proxy server。

    Args:
        args: 命令行参数。
    """
    snapshots = args.snapshots or DEFAULT_UPDATE_SNAPSHOTS

    with httpx.Client(timeout=None) as client:
        start_payload = _build_start_payload(args)
        _print_json("start request", start_payload)
        start_response = _post_json(
            client,
            args.proxy_url,
            "/v1/prompt_stream/start",
            start_payload,
        )
        _print_json("start response", start_response)

        # 后续事件必须使用 proxy/P 节点返回的 request_id。
        request_id = start_response["request_id"]

        # 逐个发送 update 快照，模拟同一个检索来源不断改写自身内容。
        for sequence_no, snapshot_text in enumerate(snapshots):
            event_payload = _build_update_event(
                request_id,
                args.source_id,
                sequence_no,
                snapshot_text,
            )
            _print_json(f"update event {sequence_no}", event_payload)
            event_response = _post_json(
                client,
                args.proxy_url,
                "/v1/prompt_stream/event",
                event_payload,
            )
            _print_json(f"update response {sequence_no}", event_response)

            # 可选 delay 让人能在 proxy/P 节点日志里观察事件间隔。
            if args.event_delay > 0:
                time.sleep(args.event_delay)

        finish_payload = _build_finish_event(request_id, args.final_question)
        _print_json("finish request", finish_payload)

        finish_response = _post_json(
            client,
            args.proxy_url,
            "/v1/prompt_stream/finish",
            finish_payload,
        )
        _print_json("finish response", finish_response)
        prompt_stream_stats = parse_prompt_stream_stats(
            finish_response,
            num_sent_update_events=len(snapshots),
        )
        print_prompt_stream_stats(prompt_stream_stats)


def main() -> None:
    """脚本入口。"""
    args = parse_args()
    run_update_demo(args)


if __name__ == "__main__":
    main()
