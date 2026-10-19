# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""最小前端的 CPU 协议测试，不访问真实服务或加载模型。"""

from __future__ import annotations

import argparse
import ast
import asyncio
import json
import sys
import time
from dataclasses import fields
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import httpx
import pytest

from research.scripts.simple_test import (
    append_frontend,
    mixed_frontend,
    update_frontend,
)
from research.scripts.simple_test import frontend_common as common
from research.scripts.simple_test.prompt_stream_stats import PromptStreamStats


def valid_stats(updates: int) -> dict[str, Any]:
    """构造无 recovery、无 coalescing 的合法完成统计。"""
    stats = {field.name: 0 for field in fields(PromptStreamStats)}
    stats.update(metrics_schema_version=2, coalescing_enabled=False)
    for name in (
        "update_events_accepted",
        "update_works_admitted",
        "update_works_completed",
        "update_policy_works_admitted",
        "update_policy_works_completed",
        "update_execution_attempts_started",
        "update_execution_attempts_completed",
    ):
        stats[name] = updates
    for name in (
        "eager_update_prefill_tokens",
        "saved_update_prefill_tokens",
        "extra_update_prefill_tokens",
        "policy_saved_update_prefill_tokens",
        "policy_extra_update_prefill_tokens",
    ):
        stats[name] = None
    return stats


class FakeServer:
    """模拟直连 P 和 proxy，验证来源序号、结束屏障及最后快照。"""

    def __init__(self, *, fail: str | None = None) -> None:
        self.calls: list[tuple[str, dict[str, Any]]] = []
        self.fail = fail
        self.sources: dict[str, tuple[str, list[int]]] = {}
        self.counts: dict[str, int] = {}
        self.ended: set[str] = set()
        self.updates = 0
        self.prefix: list[int] = []

    def handle(self, request: httpx.Request) -> httpx.Response:
        """按服务端规则分词；请求中不得携带客户端输入 token IDs。"""
        path, body = request.url.path, json.loads(request.content)
        self.calls.append((path, body))
        if "/start" in path:
            assert isinstance(body["prompt"], str)
            self.prefix = list(body["prompt"].encode())
            return httpx.Response(
                200, json={"request_id": "internal", "token_ids": self.prefix}
            )
        if path.endswith("/abort"):
            return httpx.Response(200, json={})
        event = body["event"]
        assert "token_ids" not in event
        if path.endswith("/event"):
            if self.fail == "event":
                return httpx.Response(500, text="event failed")
            source_id = event["source_id"]
            assert event["sequence_no"] == self.counts.get(source_id, 0)
            self.counts[source_id] = event["sequence_no"] + 1
            if event["event_type"] == "source_finished":
                self.ended.add(source_id)
                tokens = None
            else:
                tokens = list(event["text"].encode())
                if event["event_type"] == "append":
                    old = self.sources.get(source_id, ("append", []))[1]
                    self.sources[source_id] = ("append", old + tokens)
                else:
                    self.sources[source_id] = ("update", tokens)
                    self.updates += 1
            return httpx.Response(200, json={"token_ids": tokens})
        assert self.ended == set(self.sources)
        append = [
            token
            for kind, tokens in self.sources.values()
            if kind == "append"
            for token in tokens
        ]
        update = [
            token
            for kind, tokens in self.sources.values()
            if kind == "update"
            for token in tokens
        ]
        final = list(event["text"].encode())
        padding = {
            "version": 1,
            "start": len(self.prefix + append),
            "end": len(self.prefix + append) + 2,
        }
        stats = valid_stats(self.updates)
        if path.endswith("finish_ttft"):
            frame = {
                "choices": [{"text": "A", "token_ids": [99]}],
                "prompt_stream_stats": stats,
                "prompt_stream_masked_padding": padding,
            }
            ending = "" if self.fail == "disconnect" else "data: [DONE]\n\n"
            return httpx.Response(
                200,
                headers={"content-type": "text/event-stream"},
                content="data: " + json.dumps(frame) + "\n\n" + ending,
            )
        if self.fail == "stats":
            stats["update_events_accepted"] += 1
        physical = self.prefix + append + [0, 0] + update + final
        if self.fail == "layout":
            physical.append(777)
        return httpx.Response(
            200,
            json={
                "prompt_token_ids": physical,
                "token_ids": final,
                "output_text": "A",
                "output_token_ids": [99],
                "prompt_stream_stats": stats,
                "kv_transfer_params": {"prompt_stream_masked_padding": padding},
            },
        )


@pytest.fixture
def setup(
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[argparse.Namespace, list[common.DemoEvent]]:
    """配置仅渲染文本的模板替身，任何客户端 encode 都会失败。"""
    renderer = SimpleNamespace(
        render_envelope=lambda prefix: (
            "<system>" + prefix + "<user>",
            "</user><assistant>",
        )
    )
    monkeypatch.setattr(
        common.ChatTemplateRenderer, "from_pretrained", lambda path: renderer
    )
    parser = argparse.ArgumentParser()
    common.add_common_arguments(parser, "system", "question")
    args = parser.parse_args(
        [
            "--target",
            "p",
            "--server-url",
            "http://test/v1",
            "--expected-masked-padding",
            "enabled",
        ]
    )
    events = [
        common.DemoEvent("a", "append", "stable1\n"),
        common.DemoEvent("u", "update", "old snapshot\n"),
        common.DemoEvent("a", "append", "stable2\n"),
        common.DemoEvent("u", "update", "new\n"),
    ]
    return args, events


def install_server(monkeypatch: pytest.MonkeyPatch, server: FakeServer) -> None:
    """只替换本测试的 HTTP transport，保留真正 AsyncClient 的生命周期。"""
    client_class = httpx.AsyncClient
    monkeypatch.setattr(
        common.httpx,
        "AsyncClient",
        lambda **kwargs: client_class(
            **kwargs, transport=httpx.MockTransport(server.handle)
        ),
    )


@pytest.mark.parametrize("mode", ["p", "proxy", "proxy-ttft"])
def test_complete_flow(
    monkeypatch: pytest.MonkeyPatch, setup: tuple, mode: str
) -> None:
    """验证 P JSON、PD JSON 和 P-only proxy SSE 三种输出协议。"""
    args, events = setup
    args.target = "p" if mode == "p" else "proxy"
    args.ttft_only = mode == "proxy-ttft"
    server = FakeServer()
    install_server(monkeypatch, server)
    result = asyncio.run(common.run_demo(args, events))
    assert result["output_text"] == "A"
    start = server.calls[0][1]
    assert start["add_special_tokens"] is False
    assert start["max_tokens"] == (64 if mode == "proxy" else 1)
    paths = [path for path, _ in server.calls]
    assert paths[0].endswith("start_ttft" if args.ttft_only else "start")
    assert paths[-1].endswith("finish_ttft" if args.ttft_only else "finish")
    assert server.calls[-1][1]["event"]["text"].endswith("</user><assistant>")
    ends = [
        body["event"]
        for path, body in server.calls
        if path.endswith("/event") and body["event"]["event_type"] == "source_finished"
    ]
    assert [event["sequence_no"] for event in ends] == [2, 2]
    assert all("/v1/v1/" not in path for path in paths)


@pytest.mark.parametrize("fault", ["event", "stats", "layout", "disconnect"])
def test_failure_cleans_session(
    monkeypatch: pytest.MonkeyPatch, setup: tuple, fault: str
) -> None:
    """HTTP、统计、布局和断流失败都要发 abort，不能留着 dormant 会话。"""
    args, events = setup
    if fault == "disconnect":
        args.target, args.ttft_only = "proxy", True
    server = FakeServer(fail=fault)
    install_server(monkeypatch, server)
    with pytest.raises((RuntimeError, ValueError)):
        asyncio.run(common.run_demo(args, events))
    assert server.calls[-1][0] == "/v1/prompt_stream/abort"


@pytest.mark.parametrize("module", [append_frontend, update_frontend, mixed_frontend])
def test_entry_cli(monkeypatch: pytest.MonkeyPatch, module: Any) -> None:
    """旧 URL 参数继续可用，三个入口使用相同传输参数。"""
    monkeypatch.setattr(
        sys, "argv", ["demo", "--proxy-url", "http://test:8100", "--target", "p"]
    )
    args = module.parse_args()
    assert args.proxy_url == "http://test:8100"
    assert args.target == "p"
    captured = []

    async def run(args: argparse.Namespace, events: list[common.DemoEvent]) -> None:
        captured.extend(events)

    monkeypatch.setattr(module, "run_demo", run)
    getattr(module, "run_" + module.__name__.split(".")[-1].split("_")[0] + "_demo")(
        args
    )
    assert captured and all(event.text for event in captured)


def test_source_id_collision(setup: tuple) -> None:
    """同一个来源不能同时代表 Append 和 Update。"""
    args, _ = setup
    with pytest.raises(ValueError, match="共用"):
        common.validate_arguments(
            args,
            [
                common.DemoEvent("same", "append", "a"),
                common.DemoEvent("same", "update", "u"),
            ],
        )


@pytest.mark.parametrize("special_tokens", [True, False])
def test_start_flag_reaches_engine(special_tokens: bool) -> None:
    """隔离运行实际 serving.start，保证分词标记同时用于镜像和实际 engine。"""
    root = Path(__file__).resolve().parents[4]
    tree = ast.parse(
        (root / "vllm/entrypoints/serve/prompt_stream/serving.py").read_text()
    )
    method = next(
        node
        for cls in tree.body
        if isinstance(cls, ast.ClassDef)
        for node in cls.body
        if isinstance(node, ast.AsyncFunctionDef) and node.name == "start"
    )
    method.returns = None
    for argument in method.args.args:
        argument.annotation = None
    namespace = {
        "time": time,
        "PromptStreamStartResponse": SimpleNamespace,
        "PromptStreamSession": SimpleNamespace,
    }
    exec(
        compile(
            ast.fix_missing_locations(ast.Module(body=[method], type_ignores=[])),
            "serving.py",
            "exec",
        ),
        namespace,
    )
    captured = []

    async def add_request(**kwargs: Any) -> SimpleNamespace:
        captured.append(kwargs["tokenization_kwargs"]["add_special_tokens"])
        return SimpleNamespace(request_id="internal")

    service = SimpleNamespace(
        engine_client=SimpleNamespace(add_prompt_stream_request=add_request),
        sessions={},
        _build_prefill_sampling_params=lambda params: params,
        _resolve_start_prompt_token_ids=lambda prompt, **kwargs: [1],
        _align_to_block_size=lambda size: size,
        _max_model_len=lambda: 8192,
        _build_length_request=lambda **kwargs: SimpleNamespace(),
    )
    request = SimpleNamespace(
        request_id=None,
        prompt="rendered",
        add_special_tokens=special_tokens,
        get_sampling_params_dict=lambda: {},
    )
    response = asyncio.run(namespace["start"](service, request))
    assert response.request_id == "internal" and captured == [special_tokens]
