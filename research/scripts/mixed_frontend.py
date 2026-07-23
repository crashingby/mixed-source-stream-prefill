# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import argparse
import json
import time
from dataclasses import dataclass
from typing import Any, Literal, cast

import httpx


PromptStreamEventKind = Literal["append", "update"]

DEFAULT_PREFIX = (
    "You are a helpful assistant. Read the streaming context and the latest "
    "mutable snapshot first, then answer the final user question.\n\n"
)

DEFAULT_MIXED_EVENTS: list[tuple[PromptStreamEventKind, str]] = [
    (
        "append",
        "Stable document A: vLLM uses EngineCore to keep long-lived request "
        "state.\n",
    ),
    (
        "update",
        "Mutable search result v0: the proxy receives prompt stream events "
        "before decode.\n",
    ),
    (
        "append",
        "Stable document B: append chunks represent monotonic context growth.\n",
    ),
    (
        "update",
        "Mutable search result v1: the proxy receives prompt stream events, "
        "lets the P node prefill them, then sends the final decode request to "
        "the D node.\n",
    ),
    (
        "append",
        "Stable document C: P and D nodes communicate KV cache metadata through "
        "kv_transfer_params.\n",
    ),
    
]

DEFAULT_FINAL_QUESTION = (
    "\nUser question: Based on the stable documents and the latest mutable "
    "snapshot, what does the proxy do before decode?\nAnswer:"
)


@dataclass(frozen=True)
class MixedPromptStreamEvent:
    """描述混合模式脚本中的一个前端事件。

    Args:
        event_kind: 事件类型，只允许 `append` 或 `update`。
        text: 事件携带的原始文本。append 表示新增 chunk，update 表示
            同一个 update source 的完整当前快照。
    """

    event_kind: PromptStreamEventKind
    text: str


def parse_args() -> argparse.Namespace:
    """解析 mixed prompt stream 前端示例脚本的启动参数。

    这个脚本模拟一个同时包含 append 和 update 的 RAG 前端。默认事件序列
    会先发送 append，再发送 update，然后连续发送两个 append，最后再次
    update，以覆盖 PREFIX_APPEND 和 UPDATE 两个 KV segment 的交错调度。

    Returns:
        命令行参数命名空间。
    """
    parser = argparse.ArgumentParser(
        description="Send a mixed append/update prompt-stream demo to the proxy."
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
        "--event",
        action="append",
        dest="events",
        default=None,
        help=(
            "Mixed event in '<append|update>:text' format. Can be specified "
            "multiple times and preserves command-line order."
        ),
    )
    parser.add_argument(
        "--append-source-id",
        type=str,
        default="demo_append_source",
        help="Logical append source id used by append demo events.",
    )
    parser.add_argument(
        "--update-source-id",
        type=str,
        default="demo_update_source",
        help="Logical update source id used by update demo snapshots.",
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
        help="Sleep seconds between mixed events, useful for manual tracing.",
    )
    return parser.parse_args()


def _parse_mixed_event(raw_event: str) -> MixedPromptStreamEvent:
    """解析命令行传入的单个 mixed event。

    Args:
        raw_event: `--event` 的原始字符串，格式为 `append:text` 或
            `update:text`。

    Returns:
        结构化后的事件描述。

    Raises:
        ValueError: 当事件格式或事件类型非法时抛出。
    """
    event_kind, separator, event_text = raw_event.partition(":")
    if not separator:
        raise ValueError(
            "Mixed event must use '<append|update>:text' format, but got "
            f"{raw_event!r}."
        )

    normalized_kind = event_kind.strip().lower()
    if normalized_kind not in ("append", "update"):
        raise ValueError(
            "Mixed event kind must be 'append' or 'update', but got "
            f"{event_kind!r}."
        )
    if not event_text:
        raise ValueError(
            "Mixed event text must be non-empty, but got "
            f"{raw_event!r}."
        )

    return MixedPromptStreamEvent(
        event_kind=cast(PromptStreamEventKind, normalized_kind),
        text=event_text,
    )


def _get_mixed_events(args: argparse.Namespace) -> list[MixedPromptStreamEvent]:
    """返回本次测试要发送的混合事件序列。

    Args:
        args: 命令行参数。

    Returns:
        按发送顺序排列的事件列表。
    """
    if args.events is None:
        return [
            MixedPromptStreamEvent(event_kind=event_kind, text=text)
            for event_kind, text in DEFAULT_MIXED_EVENTS
        ]

    return [_parse_mixed_event(raw_event) for raw_event in args.events]


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


def _build_append_event(
    request_id: str,
    source_id: str,
    sequence_no: int,
    text: str,
) -> dict[str, Any]:
    """构造一个 append 文本事件 envelope。

    Args:
        request_id: proxy start 返回的内部 request id。
        source_id: append 来源 id。
        sequence_no: append source 内单调递增序号。
        text: 当前 append chunk 的原始文本。

    Returns:
        与 proxy `/event` 接口对齐的 JSON envelope。
    """
    return {
        "request_id": request_id,
        "event": {
            "source_id": source_id,
            "event_type": "append",
            "source_type": "append",
            "chunk_id": f"{source_id}-{sequence_no}",
            "version": sequence_no,
            "sequence_no": sequence_no,
            "text": text,
            "arrival_time": time.time(),
        },
    }


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
        sequence_no: update source 内单调递增序号，同时作为 demo 版本号。
        snapshot_text: 当前 update source 的完整文本快照。

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


def _build_mixed_event(
    request_id: str,
    args: argparse.Namespace,
    event: MixedPromptStreamEvent,
    append_sequence_no: int,
    update_sequence_no: int,
) -> dict[str, Any]:
    """根据混合事件类型构造 RESTFUL event envelope。

    Args:
        request_id: proxy start 返回的内部 request id。
        args: 命令行参数。
        event: 当前待发送的混合事件。
        append_sequence_no: append source 的当前序号。
        update_sequence_no: update source 的当前序号。

    Returns:
        与 proxy `/event` 接口对齐的 JSON envelope。
    """
    if event.event_kind == "append":
        return _build_append_event(
            request_id,
            args.append_source_id,
            append_sequence_no,
            event.text,
        )

    return _build_update_event(
        request_id,
        args.update_source_id,
        update_sequence_no,
        event.text,
    )


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


def run_mixed_demo(args: argparse.Namespace) -> None:
    """运行一次 append/update 混合 prompt stream 端到端示例。

    这个函数只负责前端侧协议验证：它不会自己 tokenize，也不会直接访问
    P/D 节点；所有请求都先发送给 proxy server。

    Args:
        args: 命令行参数。
    """
    mixed_events = _get_mixed_events(args)
    append_sequence_no = 0
    update_sequence_no = 0

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

        # 按用户指定或默认顺序发送 append/update 事件，模拟多源 RAG 交错到达。
        for event_index, event in enumerate(mixed_events):
            event_payload = _build_mixed_event(
                request_id,
                args,
                event,
                append_sequence_no,
                update_sequence_no,
            )
            _print_json(
                f"{event.event_kind} event {event_index}",
                event_payload,
            )
            event_response = _post_json(
                client,
                args.proxy_url,
                "/v1/prompt_stream/event",
                event_payload,
            )
            _print_json(
                f"{event.event_kind} response {event_index}",
                event_response,
            )

            # append 和 update 各自维护 source 内 sequence_no。
            if event.event_kind == "append":
                append_sequence_no += 1
            else:
                update_sequence_no += 1

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


def main() -> None:
    """脚本入口。"""
    args = parse_args()
    run_mixed_demo(args)


if __name__ == "__main__":
    main()
