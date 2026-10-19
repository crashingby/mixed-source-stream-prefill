# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""回放数据、传输和评价边界的 CPU 回归测试，不依赖模型或 P/D 服务。"""

from __future__ import annotations

import argparse
import ast
import asyncio
import hashlib
import json
import sys
import time
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import httpx
import pytest

from research.scripts.replay.common import config, dataset, runner
from research.scripts.replay.common.dispatch import send_source_events
from research.scripts.replay.common.prompt import (
    ChatTemplateRenderer,
    LocalTemplateTokenizer,
    render_sample_prompt,
)
from research.scripts.replay.common.transport import ReplayHttpError, consume_sse_body
from research.scripts.replay.generation_quality import evaluate_squad as evaluation
from research.scripts.replay.generation_quality import (
    replay_generation_quality as quality,
)
from research.scripts.replay.generation_quality.evaluators import (
    references,
    score_answer,
)
from research.scripts.replay.segmented_prefill import (
    replay_merged_update,
    replay_normal,
    replay_prompt_stream,
)
from research.scripts.replay.stable_kv_push import replay_stable_kv_push as stable


def write_json(path: Path, value: Any) -> None:
    """写入临时测试数据。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")


def write_jsonl(path: Path, values: list[dict[str, Any]]) -> None:
    """保留物理换行，测试正文中的 Unicode 换行。"""
    path.write_text(
        "".join(json.dumps(value, ensure_ascii=False) + "\n" for value in values),
        encoding="utf-8",
    )


def digest(text: str) -> str:
    """计算夹具正文哈希。"""
    return hashlib.sha256(text.encode()).hexdigest()


@pytest.fixture
def inputs(tmp_path: Path) -> dict[str, Path]:
    """构造与真实生成器同构的数据；清单、文件名与问题顺序刻意不同。"""
    append_dir, pool_dir = tmp_path / "append", tmp_path / "pools"
    append_dir.mkdir()
    pool_dir.mkdir()
    questions_file = tmp_path / "questions.jsonl"
    questions, append_rows, pool_rows = [], [], []
    for index, sample_id in enumerate(("q-z", "q-a")):
        questions.append(
            {
                "sample_id": sample_id,
                "question_id": str(index + 10),
                "dataset": "triviaqa",
                "question": f"Where is place {index}?",
                "evaluation": {
                    "protocol": "triviaqa_question_level_em_f1",
                    "answer": {
                        "NormalizedAliases": ["alpha city", "GOLD_NOT_FOR_PROMPT"],
                        "HumanAnswers": [],
                    },
                },
            }
        )
        chunks = (
            [
                {"chunk_id": "a0", "text": "APP_0\u2028continued\n"},
                {"chunk_id": "a1", "text": "APP_1\n"},
            ]
            if index == 0
            else []
        )
        relative = f"samples/{digest(sample_id)}.json"
        append = {
            "schema": "append_chunks/v1",
            "sample_id": sample_id,
            "status": "ready" if chunks else "empty",
            "chunks": chunks,
            "append_sha256": digest("".join(chunk["text"] for chunk in chunks)),
        }
        pool = {
            "schema": "pipeline_pools/v1",
            "sample_id": sample_id,
            "status": "ready",
            "chunks": [
                {"chunk_id": "old", "text": "UP_OLD\n"},
                {"chunk_id": "extra", "text": "UP_EXTRA\n"},
                {"chunk_id": "new", "text": "UP_NEW\n"},
            ],
            "snapshots": [
                {
                    "sequence_no": 0,
                    "event": "candidate",
                    "chunk_ids": ["old", "extra"],
                    "text_sha256": digest("UP_OLD\nUP_EXTRA\n"),
                },
                {
                    "sequence_no": 1,
                    "event": "final",
                    "chunk_ids": ["new"],
                    "text_sha256": digest("UP_NEW\n"),
                },
            ],
        }
        write_json(append_dir / relative, append)
        write_json(pool_dir / relative, pool)
        append_rows.append(
            {"sample_id": sample_id, "path": relative, "status": append["status"]}
        )
        pool_rows.append({"sample_id": sample_id, "path": relative, "status": "ready"})
    write_jsonl(questions_file, questions)
    question_sha = dataset.fingerprint_file(questions_file)
    write_json(append_dir / "metadata.json", {"questions_sha256": question_sha})
    write_json(
        pool_dir / "metadata.json",
        {"settings": {"questions": {"sha256": question_sha}}},
    )
    write_jsonl(append_dir / "manifest.jsonl", list(reversed(append_rows)))
    write_jsonl(pool_dir / "manifest.jsonl", pool_rows)
    model_dir = tmp_path / "model"
    write_json(
        model_dir / "tokenizer_config.json",
        {
            "bos_token": "<bos>",
            "chat_template": "{{ bos_token }}{% for message in messages %}"
            "<{{ message.role }}>{{ message.content }}</{{ message.role }}>"
            "{% endfor %}{% if add_generation_prompt %}<assistant>{% endif %}",
        },
    )
    cfg = tmp_path / "replay.json"
    write_json(
        cfg,
        {
            "dataset": "triviaqa",
            "questions_file": "questions.jsonl",
            "append_dir": "append",
            "pipeline_pools_dir": "pools",
            "tokenizer_model": "./model",
            "qps": 0,
            "arrival_process": "fixed",
            "random_seed": 2026,
            "max_concurrency": 2,
            "max_requests": 2,
            "max_tokens": 8,
            "temperature": 0,
            "seed": 2026,
            "append_initial_delay": 0,
            "update_initial_delay": 0,
            "append_chunk_interval": 0,
            "update_chunk_interval": 0,
            "source_finish_delay": 0,
        },
    )
    return {
        "config": cfg,
        "questions": questions_file,
        "append": append_dir,
        "pools": pool_dir,
        "model": model_dir,
        "output": tmp_path / "results",
    }


def arguments(
    monkeypatch: pytest.MonkeyPatch,
    inputs: dict[str, Path],
    entry: str = "stream",
    extra: tuple[str, ...] = (),
) -> argparse.Namespace:
    """通过正式 CLI 解析同一份配置，防止测试绕过配置转换。"""
    argv = [
        "replay",
        "--config",
        str(inputs["config"]),
        "--output-dir",
        str(inputs["output"]),
        "--run-name",
        entry,
        *extra,
    ]
    if entry.startswith("quality-"):
        argv.extend(
            ["--mode", entry.removeprefix("quality-"), "--inter-request-delay", "0"]
        )
    monkeypatch.setattr(sys, "argv", argv)
    if entry.startswith("quality-"):
        return quality.parse_args()
    if entry == "stable":
        return stable.parse_args()
    return config.parse_args(entry)


def test_id_join_and_empty_source(
    monkeypatch: pytest.MonkeyPatch, inputs: dict[str, Path]
) -> None:
    """按 QA 顺序和 ID 对齐；空 Append 仍有结束屏障，Update 只保留最终快照。"""
    args = arguments(monkeypatch, inputs)
    samples = dataset.select_samples(args)
    assert [sample.sample_id for sample in samples] == ["q-z", "q-a"]
    assert [
        record["event"]["text"]
        for record in dataset.get_data_events(samples[0].source_streams[1])
    ] == ["UP_OLD\nUP_EXTRA\n", "UP_NEW\n"]
    assert len(samples[1].source_streams[0].event_records) == 1
    assert (
        samples[1].source_streams[0].event_records[0]["event"]["event_type"]
        == "source_finished"
    )
    renderer = ChatTemplateRenderer.from_pretrained(str(inputs["model"]))
    rendered = render_sample_prompt(samples[0], renderer)
    assert rendered.logical_prompt_text.endswith("</user><assistant>")
    assert "UP_NEW" in rendered.logical_prompt_text
    assert "UP_OLD" not in rendered.logical_prompt_text
    assert "GOLD_NOT_FOR_PROMPT" not in rendered.logical_prompt_text
    assert not rendered.logical_prompt_token_ids


@pytest.mark.parametrize(
    "fault", ["fingerprint", "text_hash", "duplicate", "traversal", "symlink"]
)
def test_invalid_evidence_fails(
    monkeypatch: pytest.MonkeyPatch, inputs: dict[str, Path], fault: str
) -> None:
    """拒绝错版本、损坏正文、重复 ID 和路径逃逸，避免静默发送错误证据。"""
    args = arguments(monkeypatch, inputs)
    if fault == "fingerprint":
        write_json(inputs["append"] / "metadata.json", {"questions_sha256": "bad"})
    else:
        manifest = list(dataset.iter_jsonl(inputs["pools"] / "manifest.jsonl"))
        path = inputs["pools"] / manifest[0]["path"]
        if fault == "text_hash":
            pool = dataset.load_json(path)
            pool["snapshots"][0]["text_sha256"] = "bad"
            write_json(path, pool)
        elif fault == "symlink":
            other = inputs["output"] / "escape.json"
            write_json(other, dataset.load_json(path))
            path.unlink()
            path.symlink_to(other)
        elif fault == "duplicate":
            manifest.append(manifest[0])
        else:
            manifest[0]["path"] = "../escape.json"
        write_jsonl(inputs["pools"] / "manifest.jsonl", manifest)
    with pytest.raises(ValueError):
        dataset.select_samples(args)


def test_selection_is_lazy(
    monkeypatch: pytest.MonkeyPatch, inputs: dict[str, Path]
) -> None:
    """小规模回放不读取未选中的正文；ID 显式选择不受默认数量上限影响。"""
    rows = list(dataset.iter_jsonl(inputs["pools"] / "manifest.jsonl"))
    (inputs["pools"] / rows[1]["path"]).unlink()
    args = arguments(monkeypatch, inputs, extra=("--max-requests", "1"))
    assert len(dataset.select_samples(args)) == 1
    args = arguments(monkeypatch, inputs, extra=("--sample-id", "q-a"))
    with pytest.raises(ValueError, match="Cannot read"):
        dataset.select_samples(args)


@pytest.mark.parametrize(
    "entry",
    [
        "normal",
        "stream",
        "merged_update",
        "quality-normal",
        "quality-segmented",
        "stable",
    ],
)
def test_shared_config(
    monkeypatch: pytest.MonkeyPatch, inputs: dict[str, Path], entry: str
) -> None:
    """所有入口共享数据配置，CLI 显式值优先，配置路径相对 JSON 所在目录。"""
    args = arguments(monkeypatch, inputs, entry, ("--max-requests", "1"))
    assert args.max_requests == 1
    assert args.questions_file == inputs["questions"]
    assert args.input_transport == "text"
    assert args.tokenizer_model == str(inputs["model"])


@pytest.mark.parametrize("value", [True, 1.5, "1"])
def test_config_rejects_bad_integer(
    monkeypatch: pytest.MonkeyPatch, inputs: dict[str, Path], value: Any
) -> None:
    """配置中整数不能悄悄截断浮点数或把布尔值当整数。"""
    values = dataset.load_json(inputs["config"])
    values["max_concurrency"] = value
    write_json(inputs["config"], values)
    with pytest.raises(ValueError):
        arguments(monkeypatch, inputs)


class FakeProxy:
    """模拟 JSON ACK 和完整 P/D SSE；记录实际 HTTP 请求正文。"""

    def __init__(self) -> None:
        self.requests: list[tuple[str, dict[str, Any]]] = []
        self.sessions: dict[str, dict[str, Any]] = {}
        self.final_prompts: list[str] = []

    def handle(self, request: httpx.Request) -> httpx.Response:
        """按真实接口处理原文，Update 覆盖此前快照，来源结束后才允许 finish。"""
        path = request.url.path
        body = json.loads(request.content)
        self.requests.append((path, body))
        assert "GOLD_NOT_FOR_PROMPT" not in request.content.decode()
        if "/start" in path:
            assert isinstance(body["prompt"], str)
            assert body["add_special_tokens"] is False
            self.sessions[body["request_id"]] = {
                "prefix": body["prompt"],
                "start": body,
                "stores": defaultdict(str),
                "finished": set(),
                "sequences": {},
            }
            return httpx.Response(200, json={"request_id": body["request_id"]})
        if path.endswith("/event"):
            event = body["event"]
            assert "token_ids" not in event
            session = self.sessions[body["request_id"]]
            source = event["source_id"]
            assert event["sequence_no"] == session["sequences"].get(source, 0)
            session["sequences"][source] = event["sequence_no"] + 1
            if event["event_type"] == "append":
                session["stores"][source] += event["text"]
            elif event["event_type"] == "update":
                session["stores"][source] = event["text"]
            else:
                session["finished"].add(source)
            return httpx.Response(
                200, json={"token_ids": list(event.get("text", "").encode())}
            )
        if "/finish" in path:
            assert "token_ids" not in body["event"]
            session = self.sessions[body["request_id"]]
            assert session["finished"] == set(session["sequences"])
            stores = session["stores"]
            # 测试服务器采用服务端固定的 Append -> Update 逻辑顺序。
            content = stores.get("append_source", "") + stores.get("update_source", "")
            content += stores.get("merged_update_source", "")
            self.final_prompts.append(
                session["prefix"] + content + body["event"]["text"]
            )
            early = session["start"].get("early_stable_kv_push", False)
        elif "/completions" in path:
            assert isinstance(body["prompt"], str)
            assert body["add_special_tokens"] is False
            self.final_prompts.append(body["prompt"])
            early = False
        else:
            return httpx.Response(200, json={})
        metrics = {field: 1 for field in stable.PROXY_METRIC_FIELDS}
        metrics.update(
            early_stable_kv_push=early,
            early_transferred_bytes=int(early),
            final_transferred_bytes=1,
            total_transferred_bytes=1 + int(early),
        )
        frames = [
            {
                "token_source": "prefill",
                "choices": [{"text": "Alpha", "token_ids": [1]}],
                "prompt_stream_masked_padding": {"version": 1, "start": 0, "end": 0},
            },
            {
                "token_source": "decode",
                "choices": [{"text": " city", "token_ids": [2]}],
            },
            {"choices": [], "stable_kv_metrics": metrics},
        ]
        sse = "".join("data: " + json.dumps(frame) + "\n\n" for frame in frames)
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            content=sse + "data: [DONE]\n\n",
        )


@pytest.mark.parametrize(
    "entry",
    [
        "normal",
        "stream",
        "merged_update",
        "quality-normal",
        "quality-segmented",
        "stable",
    ],
)
def test_http_workflow(
    monkeypatch: pytest.MonkeyPatch, inputs: dict[str, Path], entry: str
) -> None:
    """跑正式入口的完整 HTTP 工作流；文本模式任何客户端 encode 都视为失败。"""
    args = arguments(monkeypatch, inputs, entry)
    proxy = FakeProxy()

    def no_encode(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("text transport must not tokenize in replay")

    def client_factory(args: argparse.Namespace) -> httpx.AsyncClient:
        return httpx.AsyncClient(transport=httpx.MockTransport(proxy.handle))

    monkeypatch.setattr(LocalTemplateTokenizer, "encode", no_encode)
    target = (
        quality
        if entry.startswith("quality-")
        else stable
        if entry == "stable"
        else runner
    )
    monkeypatch.setattr(target, "create_http_client", client_factory)
    if target is runner:
        callbacks = {
            "normal": replay_normal._replay_normal_sample,
            "stream": replay_prompt_stream._replay_stream_sample,
            "merged_update": replay_merged_update._replay_merged_update_sample,
        }
        output = asyncio.run(runner.run(args, callbacks[entry]))
        records = list(dataset.iter_jsonl(output / "request_metrics.jsonl"))
        assert all(record["result_status"] == "ok" for record in records)
        assert (
            all(record["completion_token_count_exact"] for record in records)
            if entry != "merged_update"
            else True
        )
    elif target is quality:
        output = asyncio.run(quality.run(args))
        records = list(dataset.iter_jsonl(output / "generations.jsonl"))
        assert all(record["result_status"] == "ok" for record in records)
        assert all(record["response_text"] == "Alpha city" for record in records)
        assert (
            evaluation._mode_summary({row["sample_id"]: row for row in records})[
                "exact_match"
            ]
            == 100
        )
    else:
        output = asyncio.run(stable.run(args))
        records = list(dataset.iter_jsonl(output / "requests.jsonl"))
        assert all(record["transport_status"] == "ok" for record in records)
        assert all(record["metrics_status"] == "valid" for record in records)
    assert len(proxy.final_prompts) == 2
    assert all(
        "UP_NEW" in prompt and "UP_OLD" not in prompt for prompt in proxy.final_prompts
    )
    assert all(
        prompt.count("<bos>") == 1 and prompt.endswith("<assistant>")
        for prompt in proxy.final_prompts
    )
    renderer = ChatTemplateRenderer.from_pretrained(args.tokenizer_model)
    cases = dataset.select_samples(args)
    if target is quality:
        quality_cases = quality._select_samples(args, {})
        expected = [
            quality._build_rendered_quality_prompt(
                case, args, renderer
            ).logical_prompt_text
            for case in quality_cases
        ]
    else:
        expected = [
            render_sample_prompt(case, renderer).logical_prompt_text for case in cases
        ]
    assert proxy.final_prompts == expected


@pytest.mark.parametrize("ttft_only", [True, False])
def test_sse_disconnect(ttft_only: bool) -> None:
    """TTFT-only 可在首 token 后停止；完整输出必须有 DONE。"""
    response = httpx.Response(
        200,
        headers={"content-type": "text/event-stream"},
        content='data: {"choices":[{"text":"A","token_ids":[1]}]}\n\n',
    )
    consume = consume_sse_body(response, time.perf_counter(), False, ttft_only)
    if ttft_only:
        assert asyncio.run(consume).completion_tokens == 1
    else:
        with pytest.raises(ReplayHttpError, match="before \\[DONE\\]"):
            asyncio.run(consume)


def test_absolute_source_timing(
    monkeypatch: pytest.MonkeyPatch, inputs: dict[str, Path]
) -> None:
    """ACK 较慢时不额外累加 chunk_interval，结束屏障仍从最后 ACK 起算。"""
    args = arguments(monkeypatch, inputs)
    args.append_chunk_interval, args.source_finish_delay = 0.02, 0.04
    sample = dataset.select_samples(args)[0]
    clock, waits = [100.0], []

    async def sleep_until(target: float) -> None:
        waits.append(target)
        clock[0] = max(clock[0], target)

    async def ack(*args: Any, **kwargs: Any) -> dict[str, Any]:
        clock[0] += 0.10
        return {"token_ids": [1]}

    from research.scripts.replay.common import dispatch

    monkeypatch.setattr(dispatch.time, "perf_counter", lambda: clock[0])
    monkeypatch.setattr(dispatch, "_sleep_until", sleep_until)
    monkeypatch.setattr(dispatch, "_post_json", ack)
    renderer = ChatTemplateRenderer.from_pretrained(args.tokenizer_model)
    asyncio.run(
        send_source_events(
            None,
            args.proxy_url,
            "id",
            sample,
            renderer,
            sample.source_streams[0],
            100.0,
            args,
            "run",
        )
    )
    assert waits == pytest.approx([100.0, 100.02, 100.24])


def test_workload_persists_before_fail_fast() -> None:
    """失败请求先记录，随后取消其它在途任务并等待其清理。"""
    persisted, cleaned = [], []

    async def execute(index: int, offset: float, queue_delay: float) -> str:
        if index == 0:
            await asyncio.sleep(0.01)
            return "failed"
        try:
            await asyncio.sleep(10)
        finally:
            cleaned.append(index)
        return "ok"

    args = argparse.Namespace(
        qps=10000, arrival_process="fixed", random_seed=0, max_concurrency=2
    )
    with pytest.raises(ReplayHttpError):
        asyncio.run(
            runner.execute_workload(
                2,
                args,
                execute,
                on_result=persisted.append,
                fail_fast_error=lambda value: value,
            )
        )
    assert persisted == ["failed"]
    assert cleaned == [1]


@pytest.mark.parametrize(
    "protocol,prediction,answer,expected",
    [
        ("triviaqa_question_level_em_f1", "New_York", "new york", (1.0, 1.0)),
        ("triviaqa_question_level_em_f1", "A-B", "a b", (1.0, 1.0)),
        ("squad_v2", "A-B", "ab", (1.0, 1.0)),
        ("squad_v2", "a\u0301 cat", "\u0301 cat", (1.0, 1.0)),
        ("squad_v2", "the\u200dcat", "\u200dcat", (1.0, 1.0)),
        ("triviaqa_question_level_em_f1", "a\u0301 cat", "\u0301 cat", (1.0, 1.0)),
        ("triviaqa_question_level_em_f1", "the\u200dcat", "\u200dcat", (1.0, 1.0)),
        ("hotpotqa_answer_em_f1", "yes extra", "yes", (0.0, 0.0)),
        ("squad_v2", "", "", (1.0, 1.0)),
        ("triviaqa_question_level_em_f1", "", "", (1.0, 0.0)),
    ],
)
def test_official_normalization(
    protocol: str, prediction: str, answer: str, expected: tuple[float, float]
) -> None:
    """数据集的标点、下划线、yes/no 和空答案规则不能混用。"""
    assert score_answer(prediction, [answer], protocol) == expected


def test_evaluation_failure_denominator() -> None:
    """旧成功样本指标保持兼容，全请求指标给失败请求零分。"""
    records = {
        "ok": {
            "sample_id": "ok",
            "result_status": "ok",
            "response_text": "city",
            "gold_answers": ["city"],
            "is_impossible": False,
        },
        "bad": {"sample_id": "bad", "result_status": "error"},
    }
    summary = evaluation._mode_summary(records)
    assert summary["exact_match"] == 100
    assert summary["exact_match_all_requests"] == 50
    assert references(
        {
            "protocol": "triviaqa_question_level_em_f1",
            "answer": {
                "NormalizedAliases": ["first"],
                "HumanAnswers": ["Second_Answer"],
            },
        }
    ) == ["first", "second answer"]


def test_server_start_special_token_flag() -> None:
    """隔离运行实际服务端方法，验证模板输入不重复加 BOS 且默认行为保留。"""
    root = Path(__file__).resolve().parents[4]
    module = ast.parse(
        (root / "vllm/entrypoints/serve/prompt_stream/serving.py").read_text()
    )
    method = next(
        node
        for cls in module.body
        if isinstance(cls, ast.ClassDef)
        for node in cls.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "_resolve_start_prompt_token_ids"
    )
    method.decorator_list = []
    namespace: dict[str, Any] = {}
    exec(
        compile(
            ast.fix_missing_locations(ast.Module(body=[method], type_ignores=[])),
            "serving.py",
            "exec",
        ),
        namespace,
    )
    observed = []

    def encode(text: str, *, add_special_tokens: bool) -> list[int]:
        observed.append(add_special_tokens)
        return ([0] if add_special_tokens else []) + [1]

    service = SimpleNamespace(
        engine_client=SimpleNamespace(
            input_processor=SimpleNamespace(
                get_tokenizer=lambda: SimpleNamespace(encode=encode)
            )
        )
    )
    resolve = namespace[method.name]
    assert resolve(service, "template", add_special_tokens=False) == [1]
    assert resolve(service, "raw") == [0, 1]
    assert resolve(service, [3, 4], add_special_tokens=False) == [3, 4]
    assert observed == [False, True]


@pytest.mark.parametrize("entry", ["normal", "stream", "merged_update"])
def test_p_only_routes(
    monkeypatch: pytest.MonkeyPatch, inputs: dict[str, Path], entry: str
) -> None:
    """TTFT-only 必须从 start 即走 P-only，首 token 后不创建 D 请求。"""
    args = arguments(monkeypatch, inputs, entry, ("--ttft-only", "--max-requests", "1"))
    proxy = FakeProxy()
    monkeypatch.setattr(
        runner,
        "create_http_client",
        lambda args: httpx.AsyncClient(transport=httpx.MockTransport(proxy.handle)),
    )
    callbacks = {
        "normal": replay_normal._replay_normal_sample,
        "stream": replay_prompt_stream._replay_stream_sample,
        "merged_update": replay_merged_update._replay_merged_update_sample,
    }
    output = asyncio.run(runner.run(args, callbacks[entry]))
    record = next(dataset.iter_jsonl(output / "request_metrics.jsonl"))
    assert record["result_status"] == "ok"
    assert record["completion_tokens"] == 1
    paths = [path for path, _ in proxy.requests]
    if entry == "normal":
        assert paths == ["/v1/completions_ttft"]
    else:
        assert paths[0] == "/v1/prompt_stream/start_ttft"
        assert paths[-1] == "/v1/prompt_stream/finish_ttft"


@pytest.mark.parametrize(
    "entry",
    [
        "normal",
        "stream",
        "merged_update",
        "quality-normal",
        "quality-segmented",
        "stable",
    ],
)
def test_dry_run_does_not_send(
    monkeypatch: pytest.MonkeyPatch, inputs: dict[str, Path], entry: str
) -> None:
    """各入口 dry-run 保存可审阅 prompt，HTTP 替身接到请求即失败。"""
    args = arguments(monkeypatch, inputs, entry, ("--dry-run",))

    def no_http(request: httpx.Request) -> httpx.Response:
        raise AssertionError("dry-run sent an HTTP request")

    target = (
        quality
        if entry.startswith("quality-")
        else stable
        if entry == "stable"
        else runner
    )
    monkeypatch.setattr(
        target,
        "create_http_client",
        lambda args: httpx.AsyncClient(transport=httpx.MockTransport(no_http)),
    )
    if target is runner:
        output = asyncio.run(runner.run(args, replay_normal._replay_normal_sample))
    else:
        output = asyncio.run(target.run(args))
    previews = list(dataset.iter_jsonl(output / "prompt_preview.jsonl"))
    assert [record["sample_id"] for record in previews] == ["q-z", "q-a"]
    assert "GOLD_NOT_FOR_PROMPT" not in (output / "prompt_preview.jsonl").read_text()


def test_hotpotqa_protocol(
    monkeypatch: pytest.MonkeyPatch, inputs: dict[str, Path]
) -> None:
    """HotpotQA 同样从独立问题清单读取答案，保留 supporting facts 为评价元数据。"""
    questions = list(dataset.iter_jsonl(inputs["questions"]))
    for question in questions:
        question["dataset"] = "hotpotqa"
        question["evaluation"] = {
            "protocol": "hotpotqa_answer_em_f1",
            "answer": "yes",
            "supporting_facts": [["ONLY_EVAL", 1]],
        }
    write_jsonl(inputs["questions"], questions)
    sha = dataset.fingerprint_file(inputs["questions"])
    write_json(inputs["append"] / "metadata.json", {"questions_sha256": sha})
    write_json(
        inputs["pools"] / "metadata.json", {"settings": {"questions": {"sha256": sha}}}
    )
    values = dataset.load_json(inputs["config"])
    values["dataset"] = "hotpotqa"
    write_json(inputs["config"], values)
    args = arguments(monkeypatch, inputs, "quality-normal")
    samples = quality._select_samples(args, {})
    assert samples[0].gold.gold_answers == ("yes",)
    renderer = ChatTemplateRenderer.from_pretrained(args.tokenizer_model)
    assert (
        "ONLY_EVAL"
        not in quality._build_rendered_quality_prompt(
            samples[0], args, renderer
        ).logical_prompt_text
    )


def test_fail_fast_keeps_output(
    monkeypatch: pytest.MonkeyPatch, inputs: dict[str, Path]
) -> None:
    """正式入口失败后仍保留失败请求、CSV 与完成数量，方便排查与续跑。"""
    args = arguments(monkeypatch, inputs, "normal", ("--fail-fast",))
    monkeypatch.setattr(
        runner,
        "create_http_client",
        lambda args: httpx.AsyncClient(
            transport=httpx.MockTransport(
                lambda request: httpx.Response(503, text="unavailable")
            )
        ),
    )
    with pytest.raises(ReplayHttpError):
        asyncio.run(runner.run(args, replay_normal._replay_normal_sample))
    output = inputs["output"] / "normal"
    records = list(dataset.iter_jsonl(output / "request_metrics.jsonl"))
    assert len(records) == 1 and records[0]["result_status"] == "error"
    assert (output / "request_metrics.csv").is_file()
    summary = dataset.load_json(output / "summary.json")
    assert summary["num_uncompleted_requests"] == 1


def test_compatibility_token_ids(
    monkeypatch: pytest.MonkeyPatch, inputs: dict[str, Path]
) -> None:
    """显式 token_ids 模式保留预分词输入，默认 text 的更改不删除历史入口。"""
    args = arguments(monkeypatch, inputs, extra=("--input-transport", "token_ids"))
    renderer = ChatTemplateRenderer.from_pretrained(args.tokenizer_model)
    monkeypatch.setattr(
        LocalTemplateTokenizer,
        "encode",
        lambda self, text, **kwargs: list(text.encode()),
    )
    sample = dataset.select_samples(args)[0]
    rendered = render_sample_prompt(
        sample, renderer, input_transport=args.input_transport
    )
    payload = replay_normal._make_normal_payload(sample, rendered, args)
    assert isinstance(payload["prompt"], list)
    assert bytes(payload["prompt"]).decode() == rendered.logical_prompt_text


def test_cli_selection_overrides_config(
    monkeypatch: pytest.MonkeyPatch, inputs: dict[str, Path]
) -> None:
    """CLI 的重复 ID 选项替换配置选择，不能意外追加配置中的另一批请求。"""
    values = dataset.load_json(inputs["config"])
    values["sample_ids"] = ["q-z"]
    write_json(inputs["config"], values)
    args = arguments(monkeypatch, inputs, extra=("--sample-id", "q-a"))
    assert [sample.sample_id for sample in dataset.select_samples(args)] == ["q-a"]


@pytest.mark.parametrize(
    "field,value",
    [
        ("logical_prompt_sha256", "different"),
        ("evaluation_protocol", "hotpotqa_answer_em_f1"),
        ("input_transport", "token_ids"),
    ],
)
def test_pairing_rejects_incomparable_inputs(field: str, value: str) -> None:
    """评价前拒绝不同文本、不同评分协议或不同输入传输方式。"""
    record = {
        "sample_id": "s",
        "query_id": "10",
        "is_impossible": False,
        "gold_answers": ["answer"],
        "result_status": "ok",
        "response_text": "answer",
        "logical_prompt_sha256": "same",
        "input_transport": "text",
        "evaluation_protocol": "triviaqa_question_level_em_f1",
    }
    with pytest.raises(ValueError, match="Modes"):
        evaluation._paired_rows_and_summary(
            {"s": record}, {"s": {**record, field: value}}
        )


def test_legacy_dataset_and_gold_selection(
    monkeypatch: pytest.MonkeyPatch, inputs: dict[str, Path]
) -> None:
    """旧 mixed-source 和 SQuAD gold 路径使用公共解析，仍保留数值顺序及全部显式 ID。"""
    args = arguments(monkeypatch, inputs)
    cases = dataset.select_samples(args)
    legacy_dir = inputs["output"] / "legacy"
    gold_records = {}
    for case, query_id in zip(cases, ("10", "2")):
        filename = f"requests/anns-query-{query_id}.json"
        raw = {
            "schema_version": "prompt_stream_mixed/v2",
            "sample_id": case.sample_id,
            "query_id": query_id,
            "question": case.question,
            "status": "complete",
            "start": case.start_request,
            "finish": {"event": case.finish_event},
            "event_order": {"across_sources": "unspecified"},
            "sources": [
                {"source_id": source.source_id, "source_type": source.source_type}
                for source in case.source_streams
            ],
            "events": [
                record
                for source in case.source_streams
                for record in source.event_records
            ],
        }
        write_json(legacy_dir / filename, raw)
        gold_records[case.sample_id] = quality.GoldAnswerRecord(
            sample_id=case.sample_id,
            query_id=query_id,
            question=case.question,
            request_path=filename,
            sample_status="complete",
            gold_answers=("answer",),
            is_impossible=False,
            match_status="matched",
        )
    args.dataset, args.questions_file, args.data_dir = None, None, legacy_dir
    assert [case.query_id for case in dataset.select_samples(args)] == ["2", "10"]
    args.query_ids, args.max_requests = ["10", "2"], 1
    assert len(quality._select_samples(args, gold_records)) == 2
    assert len(dataset.select_samples(args)) == 2


def test_quality_finish_disconnect_aborts(
    monkeypatch: pytest.MonkeyPatch, inputs: dict[str, Path]
) -> None:
    """质量路径断流不能保留未完成的 P/D 会话，即使已经收到一部分答案。"""
    args = arguments(monkeypatch, inputs, "quality-segmented", ("--max-requests", "1"))
    proxy = FakeProxy()

    def handle(request: httpx.Request) -> httpx.Response:
        response = proxy.handle(request)
        if request.url.path.endswith("/finish"):
            return httpx.Response(
                200,
                headers={"content-type": "text/event-stream"},
                content='data: {"choices":[{"text":"Alpha"}]}\n\n',
            )
        return response

    monkeypatch.setattr(
        quality,
        "create_http_client",
        lambda args: httpx.AsyncClient(transport=httpx.MockTransport(handle)),
    )
    output = asyncio.run(quality.run(args))
    row = next(dataset.iter_jsonl(output / "generations.jsonl"))
    assert row["result_status"] == "error"
    assert "before [DONE]" in row["error"]
    assert proxy.requests[-1][0] == "/v1/prompt_stream/abort"
