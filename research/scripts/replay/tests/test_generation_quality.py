# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""验证稀疏完整生成、官方答案评分和三模式配对的质量实验边界。"""

from __future__ import annotations

import asyncio
import json
import os
import shlex
import subprocess
import sys
from argparse import Namespace
from dataclasses import replace
from pathlib import Path
from typing import Any

import httpx
import pytest

from research.scripts.replay.common.dataset import iter_jsonl
from research.scripts.replay.common.prompt import (
    ChatTemplateRenderer,
    render_sample_prompt,
)
from research.scripts.replay.common.runner import execute_workload
from research.scripts.replay.generation_quality import evaluate_answers as evaluation
from research.scripts.replay.generation_quality import (
    replay_generation_quality as quality,
)
from research.scripts.replay.tests import test_replay
from research.scripts.replay.tests.test_replay import FakeProxy, arguments

inputs = test_replay.inputs  # 复用独立数据与本地 tokenizer 夹具。


def test_quality_closed_loop_concurrency() -> None:
    """QPS=0 时实际填满四个槽位，并保持上限及结果与输入顺序一致。"""

    async def run() -> None:
        active = peak = 0
        first_batch = asyncio.Event()
        args = Namespace(qps=0, max_concurrency=4, inter_request_delay=0)

        async def execute(index: int, offset: float, delay: float) -> int:
            nonlocal active, peak
            active += 1
            peak = max(peak, active)
            if active == 4:
                first_batch.set()
            await first_batch.wait()  # 串行误实现无法通过此屏障。
            await asyncio.sleep(0)
            active -= 1
            return index

        async with asyncio.timeout(5):
            results = await execute_workload(12, args, execute, closed_loop=True)
        assert peak == 4
        assert active == 0
        assert results == list(range(12))

    asyncio.run(run())


def test_trim_template_preserves_stream_text(
    monkeypatch: pytest.MonkeyPatch, inputs: dict[str, Path], tmp_path: Path
) -> None:
    """临时构造 Llama 风格 trim 模板，验证拆分保留事件原文的边缘空白。"""
    args = arguments(monkeypatch, inputs, "quality-segmented")
    model = tmp_path / "trim-model"
    model.mkdir()
    (model / "tokenizer_config.json").write_text(
        json.dumps(
            {
                "bos_token": "<bos>",
                "chat_template": "{{ bos_token }}{% for message in messages %}"
                "<{{ message.role }}>{{ message.content | trim }}</{{ message.role }}>"
                "{% endfor %}<assistant>",
            }
        )
    )
    sample = quality._select_samples(args, {})[0]
    finish = {**sample.finish_event, "text": sample.finish_event["text"] + "\n\n"}
    renderer = ChatTemplateRenderer.from_pretrained(str(model))
    rendered = render_sample_prompt(replace(sample, finish_event=finish), renderer)
    _, suffix = renderer.render_envelope(str(sample.start_request["prompt"]).strip())
    assert rendered.finish_text == finish["text"] + suffix
    assert rendered.logical_prompt_text.endswith(rendered.finish_text)


@pytest.mark.parametrize(
    "tail",
    [
        {"error": {"message": "KV transfer failed"}},
        {"choices": [{"index": 0, "text": "", "finish_reason": "error"}]},
        {"choices": [{"index": 0, "text": "", "finish_reason": "abort"}]},
        {},
    ],
)
def test_quality_rejects_incomplete_decode(tail: dict[str, Any]) -> None:
    """D 报错或缺少终止帧时，P 的首 token 不能冒充完整答案。"""
    frames = [
        {"choices": [{"index": 0, "text": "Alpha", "token_ids": [1]}]},
        tail,
    ]
    body = "".join("data: " + json.dumps(frame) + "\n\n" for frame in frames)
    response = httpx.Response(
        200,
        headers={"content-type": "text/event-stream"},
        content=(body + "data: [DONE]\n\n").encode(),
    )
    with pytest.raises(RuntimeError, match="SSE error|Generation failed|Incomplete"):
        asyncio.run(quality._consume_sse(response))


def write_run(
    root: Path, mode: str, records: list[dict[str, Any]], selected: list[str]
) -> Path:
    """写入预先冻结的选中清单，模拟失败和运行中断后缺失结果的情况。"""
    directory = root / mode
    directory.mkdir()
    (directory / "config.json").write_text(
        json.dumps(
            {
                "selected_samples": [
                    {"sample_id": sample_id, "request_sha256": sample_id}
                    for sample_id in selected
                ],
            }
        )
    )
    (directory / "generations.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in records)
    )
    return directory


def generation(
    sample_id: str, mode: str, prediction: str, status: str = "ok"
) -> dict[str, Any]:
    """创建可严格配对的 TriviaQA 结果。"""
    return {
        "sample_id": sample_id,
        "query_id": sample_id,
        "mode": mode,
        "question": "Where?",
        "gold_answers": ["alpha city"],
        "is_impossible": False,
        "evaluation_protocol": "triviaqa_question_level_em_f1",
        "result_status": status,
        "response_text": prediction,
        "logical_prompt_sha256": "same",
        "source_events_sha256": "same-events",
        "input_transport": "text",
        "logical_prompt_token_sha256": None,
        "generation_settings": {"model": "qwen", "seed": 2026},
    }


def test_three_modes_and_missing_denominator(tmp_path: Path) -> None:
    """失败和中断缺失都计零；sparse 相对 segmented 单独计算质量变化。"""
    selected = ["a", "b", "c", "d"]
    paths = {}
    for mode in ("normal", "segmented", "sparse"):
        predictions = (
            ("wrong", "Alpha_city") if mode == "sparse" else ("Alpha city", "wrong")
        )
        records = [
            generation("a", mode, predictions[0]),
            generation("b", mode, predictions[1]),
            generation("c", mode, "alpha city", "error"),
        ]
        paths[mode] = write_run(tmp_path, mode, records, selected)
    summary = evaluation.evaluate_runs(paths, tmp_path / "analysis")
    assert set(summary["comparisons"]) == {
        "segmented_vs_normal",
        "sparse_vs_normal",
        "sparse_vs_segmented",
    }
    for metrics in summary["modes"].values():
        assert metrics["num_selected"] == 4
        assert metrics["num_missing"] == metrics["num_failed"] == 1
        assert metrics["exact_match"] == metrics["f1"] == 25
        assert metrics["exact_match_successful_only"] == 50
    sparse = summary["comparisons"]["sparse_vs_segmented"]
    assert sparse["em_delta_percentage_points"] == 0
    assert sparse["regression_rate_given_reference_exact"] == 1
    assert sparse["num_both_successful"] == 2
    assert sparse["exact_outcome_counts"]["reference_only_exact"] == 1
    assert sparse["exact_outcome_counts"]["candidate_only_exact"] == 1
    assert sparse["mcnemar_exact_p_value"] == 1


@pytest.mark.parametrize(
    "field,value",
    [
        ("logical_prompt_sha256", "changed"),
        ("source_events_sha256", "changed"),
        ("generation_settings", {"model": "different"}),
        ("evaluation_protocol", "hotpotqa_answer_em_f1"),
        ("gold_answers", ["other"]),
    ],
)
def test_quality_pair_rejects_different_inputs(
    tmp_path: Path, field: str, value: Any
) -> None:
    """相同问题 ID 不足以配对，历史事件、模型和答案口径都必须一致。"""
    normal = write_run(
        tmp_path, "normal", [generation("a", "normal", "Alpha city")], ["a"]
    )
    sparse_record = {**generation("a", "sparse", "Alpha city"), field: value}
    sparse = write_run(tmp_path, "sparse", [sparse_record], ["a"])
    with pytest.raises(ValueError, match=field):
        evaluation.evaluate_runs(
            {"normal": normal, "sparse": sparse}, tmp_path / "analysis"
        )


def test_empty_partial_run_and_dry_run(tmp_path: Path) -> None:
    """刚开始就中断也保留完整分母；dry-run 不能伪装成零分质量实验。"""
    directory = write_run(tmp_path, "sparse", [], ["a", "b"])
    run = evaluation.load_run("sparse", directory)
    assert evaluation.mode_summary(run)["num_missing"] == 2
    assert evaluation.mode_summary(run)["exact_match"] == 0
    (directory / "generations.jsonl").write_text(
        json.dumps(generation("a", "sparse", "", "dry_run")) + "\n"
    )
    with pytest.raises(ValueError, match="dry-run"):
        evaluation.load_run("sparse", directory)


def test_sparse_full_generation(
    monkeypatch: pytest.MonkeyPatch, inputs: dict[str, Path]
) -> None:
    """注册实际 finish suffix，完整消费 P 和 D 的答案，包含空 Append 样本。"""
    args = arguments(
        monkeypatch, inputs, "quality-sparse", ("--probe-keep-ratio", "0.5")
    )
    proxy = FakeProxy()

    def handle(request: httpx.Request) -> httpx.Response:
        response = proxy.handle(request)
        if request.url.path.endswith("/finish"):
            # 实际 proxy 没有 token_source 字段；仍需正确识别首 token 与续写。
            content = response.content.replace(b'"token_source": "prefill", ', b"")
            content = content.replace(b'"token_source": "decode", ', b"")
            return httpx.Response(
                200, headers={"content-type": "text/event-stream"}, content=content
            )
        return response

    monkeypatch.setattr(
        quality,
        "create_http_client",
        lambda args: httpx.AsyncClient(transport=httpx.MockTransport(handle)),
    )
    output = asyncio.run(quality.run(args))
    rows = list(iter_jsonl(output / "generations.jsonl"))
    assert len(rows) == 2 and all(row["result_status"] == "ok" for row in rows)
    assert all(row["response_text"] == "Alpha city" for row in rows)
    assert all(
        row["num_prefill_token_ids"] == 0 and row["num_decode_token_ids"] == 2
        for row in rows
    )
    assert all(row["probe_keep_ratio"] == 0.5 for row in rows)
    starts = [body for path, body in proxy.requests if path.endswith("/start")]
    finishes = [body for path, body in proxy.requests if path.endswith("/finish")]
    assert len(starts) == len(finishes) == 2
    for start, finish in zip(starts, finishes):
        assert start["probe_query"] == finish["event"]["text"]
        assert "GOLD_NOT_FOR_PROMPT" not in start["probe_query"]
    assert not any("ttft" in path for path, _ in proxy.requests)


def test_missing_probe_is_failed_generation(
    monkeypatch: pytest.MonkeyPatch, inputs: dict[str, Path]
) -> None:
    """即使答案正确，未返回 probe 证据的请求也不能进入 sparse 成功结果。"""
    args = arguments(monkeypatch, inputs, "quality-sparse", ("--max-requests", "1"))
    proxy = FakeProxy()

    def handle(request: httpx.Request) -> httpx.Response:
        response = proxy.handle(request)
        if request.url.path.endswith("/finish"):
            return httpx.Response(
                200,
                headers={"content-type": "text/event-stream"},
                content=response.content.replace(
                    b'"query_probe":', b'"ignored_probe":'
                ),
            )
        return response

    monkeypatch.setattr(
        quality,
        "create_http_client",
        lambda args: httpx.AsyncClient(transport=httpx.MockTransport(handle)),
    )
    output = asyncio.run(quality.run(args))
    row = next(iter_jsonl(output / "generations.jsonl"))
    assert row["result_status"] == "error"
    assert row["response_text"] == "Alpha city"
    assert row["error_stage"] == "validate_execution"
    assert evaluation.score_record(row) is None
    assert proxy.requests[-1][0].endswith("/abort")


@pytest.mark.parametrize(
    "protocol,prediction,answers,expected",
    [
        (
            "triviaqa_question_level_em_f1",
            "Sir_Arthur Conan-Doyle",
            ["arthur conan doyle", "sir arthur conan doyle"],
            (1.0, 1.0),
        ),
        (
            "triviaqa_question_level_em_f1",
            "Alpha city",
            ["beta city", "alpha"],
            (0.0, 2 / 3),
        ),
        ("hotpotqa_answer_em_f1", "yes extra", ["yes"], (0.0, 0.0)),
        ("hotpotqa_answer_em_f1", "The Beatles", ["Beatles"], (1.0, 1.0)),
        ("hotpotqa_answer_em_f1", "New York", ["York"], (0.0, 2 / 3)),
    ],
)
def test_dataset_official_scoring(
    protocol: str, prediction: str, answers: list[str], expected: tuple[float, float]
) -> None:
    """官方别名取独立最高分，并保留 HotpotQA 的布尔答案特殊规则。"""
    row = {
        **generation("a", "normal", prediction),
        "evaluation_protocol": protocol,
        "gold_answers": answers,
    }
    assert evaluation.score_record(row) == pytest.approx(expected)


def test_batch_uses_shared_config_and_explicit_overrides(tmp_path: Path) -> None:
    """执行真实 Bash 调度脚本，检查 JSON 参数和两种模式的 CLI 覆盖一致。"""
    capture = tmp_path / "capture.py"
    commands = tmp_path / "commands.jsonl"
    capture.write_text(
        "import json, sys\n"
        f"with open({str(commands)!r}, 'a') as stream:\n"
        "    stream.write(json.dumps(sys.argv[1:]) + '\\n')\n"
    )
    executable = tmp_path / "capture.sh"
    executable.write_text(
        "#!/usr/bin/env bash\n"
        f'exec {shlex.quote(sys.executable)} {shlex.quote(str(capture))} "$@"\n'
    )
    executable.chmod(0o755)
    root = Path(__file__).resolve().parents[4]
    env = {
        **os.environ,
        "REPO_ROOT": str(root),
        "PYTHON_BIN": str(executable),
        "DATASET": "hotpotqa",
        "MODE_ORDER": "segmented sparse",
        "PROXY_URL": "http://quality-test:8000",
        "MAX_REQUESTS": "3",
        "PROBE_KEEP_RATIO": "0.25",
        "OUTPUT_DIR": str(tmp_path / "results"),
    }
    env.pop("REPLAY_CONFIG", None)
    subprocess.run(
        [
            "bash",
            "research/scripts/replay/generation_quality/run_generation_quality.sbatch",
        ],
        cwd=root,
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )
    calls = [json.loads(line) for line in commands.read_text().splitlines()]
    assert len(calls) == 3
    for mode, command in zip(("segmented", "sparse"), calls[:2]):
        assert command[command.index("--mode") + 1] == mode
        assert command[command.index("--config") + 1].endswith("configs/hotpotqa.json")
        assert command[command.index("--probe-keep-ratio") + 1] == "0.25"
        assert command[command.index("--max-requests") + 1] == "3"
        assert "--max-tokens" not in command  # 未设置的参数必须继续采用 JSON。
    assert calls[-1][0].endswith("evaluate_answers.py")
    assert sum(value == "--run" for value in calls[-1]) == 2
