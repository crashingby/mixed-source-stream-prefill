# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""受控稳定 KV 性能集的 CPU 长度、顺序和覆盖边界测试。"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import pytest

SCRIPT_DIR = Path(__file__).resolve().parents[2] / "research/scripts/data_processing"
sys.path.insert(0, str(SCRIPT_DIR))
from build_stable_kv_push_dataset import (  # noqa: E402
    DatasetConfig,
    build_sample,
    encode,
    lcp_length,
    run,
    validate_config,
)


class CharacterTokenizer:
    """以 Unicode 字符为 token，使 CPU 测试不依赖模型文件或下载。"""

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        """为每个字符分配确定性 id，模拟无 BOS/EOS 的事件编码。"""
        return [ord(character) for character in text]

    def decode(self, token_ids: list[int], **kwargs: Any) -> str:
        """无损还原字符，允许调用方传入标准 tokenizer decode 选项。"""
        return "".join(chr(token_id) for token_id in token_ids)

    def apply_chat_template(
        self,
        messages: list[dict[str, str]],
        tokenize: bool = False,
        add_generation_prompt: bool = True,
    ) -> str:
        """保留 user 内容标记，提供可分割的测试 chat template。"""
        return (
            "<system>" + messages[0]["content"] + "</system>"
            "<user>" + messages[1]["content"] + "</user><assistant>"
        )


def make_config(tmp_path: Path, **overrides: Any) -> DatasetConfig:
    """构建小预算测试配置，只写 pytest 分配的临时目录。"""
    settings = {
        "tokenizer_model": "test-character-tokenizer",
        "output_dir": tmp_path,
        "num_requests": 2,
        "knowledge_tokens": 128,
        "append_tokens": (24, 64, 104),
    }
    settings.update(overrides)
    return DatasetConfig(**settings)


def test_exact_event_lengths_and_source_local_lcp(tmp_path: Path) -> None:
    """Append 分块之和、各代 Update 长度和相邻 LCP 都必须精确。"""
    tokenizer = CharacterTokenizer()
    config = make_config(tmp_path)
    for append_budget in config.append_tokens:
        sample = build_sample(tokenizer, config, 0, append_budget)
        append = [
            r["event"]["text"]
            for r in sample["events"]
            if r["event"]["event_type"] == "append"
        ]
        updates = [
            encode(tokenizer, r["event"]["text"])
            for r in sample["events"]
            if r["event"]["event_type"] == "update"
        ]
        assert len(append) == config.append_chunks
        assert sum(len(encode(tokenizer, text)) for text in append) == append_budget
        assert len(encode(tokenizer, "".join(append))) == append_budget
        assert len(updates) == config.update_snapshots
        update_budget = config.knowledge_tokens - append_budget
        assert all(len(snapshot) == update_budget for snapshot in updates)
        assert all(
            lcp_length(a, b) == int(update_budget * config.update_lcp_ratio)
            for a, b in zip(updates, updates[1:])
        )
        metrics = sample["controlled_workload"]
        assert metrics["physical_prompt_tokens"] == (
            metrics["logical_prompt_tokens"] + metrics["padding_tokens"]
        )
        assert metrics["required_sequence_tokens"] <= config.max_model_len


def test_matching_request_identity_and_question_across_scenarios(
    tmp_path: Path,
) -> None:
    """跨占比同 query 必须同模板问题，不同请求前缀标识则必须不同。"""
    tokenizer = CharacterTokenizer()
    config = make_config(tmp_path)
    scenarios = [build_sample(tokenizer, config, 0, n) for n in config.append_tokens]
    assert len({sample["start"]["prompt"] for sample in scenarios}) == 1
    assert len({sample["question"] for sample in scenarios}) == 1
    other = build_sample(tokenizer, config, 1, config.append_tokens[0])
    assert other["start"]["prompt"] != scenarios[0]["start"]["prompt"]
    assert scenarios[0] == build_sample(tokenizer, config, 0, config.append_tokens[0])


def test_source_sequences_finish_barrier_and_text_mode(tmp_path: Path) -> None:
    """保留 replay v2 来源序号和结束屏障，不把事件时间写进数据。"""
    sample = build_sample(CharacterTokenizer(), make_config(tmp_path), 0, 24)
    assert sample["schema_version"] == "prompt_stream_mixed/v2"
    assert sample["event_order"]["across_sources"] == "unspecified"
    for source in sample["sources"]:
        events = [
            r["event"]
            for r in sample["events"]
            if r["event"]["source_id"] == source["source_id"]
        ]
        assert [e["sequence_no"] for e in events] == list(range(len(events)))
        assert events[-1]["event_type"] == "source_finished"
        assert all("token_ids" not in e and "arrival_time" not in e for e in events)
    assert set(sample["finish"]["after_source_ids"]) == {
        source["source_id"] for source in sample["sources"]
    }


def test_overwrite_is_scoped_and_removes_old_generated_files(tmp_path: Path) -> None:
    """拒绝覆盖未授权目录，减量重建只能清理本场景的旧请求。"""
    tokenizer = CharacterTokenizer()
    original_config = make_config(tmp_path, append_tokens=(24,))
    [scenario] = run(original_config, tokenizer)
    with pytest.raises(FileExistsError):
        run(original_config, tokenizer)
    run(
        make_config(tmp_path, append_tokens=(24,), num_requests=1, overwrite=True),
        tokenizer,
    )
    assert len(list((scenario / "requests").glob("*.json"))) == 1
    summary = json.loads((scenario / "dataset_summary.json").read_text())
    assert summary["scenario"]["append_tokens"] == 24
    assert summary["num_samples"] == 1
    (scenario / "dataset_summary.json").write_text("{}")
    with pytest.raises(ValueError, match="foreign dataset"):
        run(make_config(tmp_path, append_tokens=(24,), overwrite=True), tokenizer)


@pytest.mark.parametrize(
    "overrides",
    [
        {"knowledge_tokens": 0},
        {"num_requests": 0},
        {"append_chunks": 0},
        {"update_lcp_ratio": 1},
        {"append_tokens": (24, 24)},
        {"append_tokens": (128,)},
        {"append_chunks": 25, "append_tokens": (24,)},
    ],
)
def test_invalid_configuration_is_rejected(
    tmp_path: Path, overrides: dict[str, Any]
) -> None:
    """用户输入不变量失败时不得开始写实验数据。"""
    with pytest.raises(ValueError):
        validate_config(make_config(tmp_path, **overrides))


def test_physical_length_budget_includes_padding_and_output(tmp_path: Path) -> None:
    """即使知识长度很短，也必须计入模板、问题、padding 与输出预算。"""
    config = make_config(tmp_path, max_model_len=128)
    with pytest.raises(ValueError, match="physical sequence tokens"):
        build_sample(CharacterTokenizer(), config, 0, 24)
    with pytest.raises(ValueError, match="physical sequence tokens"):
        run(config, CharacterTokenizer())
    assert not any(tmp_path.iterdir())
