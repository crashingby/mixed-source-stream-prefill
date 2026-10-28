# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""稳定 KV 回放器的 CPU-only SSE 与计量边界测试。"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from typing import Any

import httpx
import pytest

SCRIPT_DIR = (
    Path(__file__).resolve().parents[2] / "research/scripts/replay/stable_kv_push"
)
sys.path.insert(0, str(SCRIPT_DIR))
from replay_stable_kv_push import (  # noqa: E402
    StreamObservation,
    consume_stable_sse,
    validate_transfer_metrics,
)


def test_prefill_token_does_not_set_decode_first_token() -> None:
    """P 帧、纯 usage 帧不能提前触发 D 计时；空文本 token_ids 可以。"""
    observation = StreamObservation()
    observation.observe(
        {
            "token_source": "prefill",
            "choices": [{"text": "A", "token_ids": [10]}],
        },
        1.0,
    )
    observation.observe({"token_source": "decode", "choices": [], "usage": {}}, 1.5)
    assert observation.first_prefill_time == 1.0
    assert observation.first_decode_time is None
    observation.observe(
        {
            "token_source": "decode",
            "choices": [{"text": "", "token_ids": [11]}],
        },
        2.0,
    )
    observation.observe(
        {
            "choices": [],
            "stable_kv_metrics": {"final_transfer_ms": 3.0},
        },
        2.5,
    )
    assert observation.first_decode_time == 2.0
    assert observation.decode_tokens == [11]
    assert observation.stable_kv_metrics["final_transfer_ms"] == 3.0


def test_sse_multiline_and_final_metadata() -> None:
    """多行 data 拼接解析，最终统计必须在 DONE 前被读取。"""
    response = httpx.Response(
        200,
        content=(
            b'data: {"token_source":"decode",\n'
            b'data: "choices":[{"text":"B","token_ids":[2]}]}\n\n'
            b'data: {"choices":[],"stable_kv_metrics":{"final_transfer_ms":4}}\n\n'
            b"data: [DONE]\n\n"
        ),
    )
    observed = asyncio.run(consume_stable_sse(response))
    assert observed.decode_tokens == [2]
    assert observed.stable_kv_metrics == {"final_transfer_ms": 4}


def test_ready_event_does_not_contribute_answer_tokens() -> None:
    """新协议的 P 计时取完成通知，D 首 token 才是用户回答的起点。"""
    observation = StreamObservation()
    observation.observe({"event": "prefill_ready", "choices": []}, 1.0)
    assert observation.first_prefill_time == 1.0
    assert observation.first_decode_time is None
    assert observation.prefill_tokens == []
    assert observation.completion_text == ""
    observation.observe(
        {"token_source": "decode", "choices": [{"text": "A", "token_ids": [10]}]},
        2.0,
    )
    assert observation.first_decode_time == 2.0
    assert observation.decode_tokens == [10]
    assert observation.completion_text == "A"


def test_disconnect_after_token_is_not_success() -> None:
    """已经有首 token 仍需识别中途连接断开，不能污染完整 PD 指标。"""
    response = httpx.Response(
        200,
        content=(b'data: {"token_source":"decode","choices":[{"token_ids":[2]}]}\n\n'),
    )
    with pytest.raises(RuntimeError, match="before \\[DONE\\]"):
        asyncio.run(consume_stable_sse(response))


def test_proxy_error_cannot_be_hidden_by_done() -> None:
    """HTTP 200 之后的 SSE 错误属于失败响应。"""
    response = httpx.Response(
        200,
        content=(b'data: {"error":{"message":"transfer failed"}}\n\ndata: [DONE]\n\n'),
    )
    with pytest.raises(RuntimeError, match="transfer failed"):
        asyncio.run(consume_stable_sse(response))


def test_unknown_token_origin_is_rejected() -> None:
    """接错普通 proxy 时不能把 P token 当作 D 输出得到虚假的低延迟。"""
    with pytest.raises(ValueError, match="token_source"):
        StreamObservation().observe({"choices": [{"token_ids": [10]}]}, 1.0)


def test_missing_transfer_data_is_not_zero() -> None:
    """传输统计缺失与真正的零传输必须区分。"""
    errors = validate_transfer_metrics({"early_stable_kv_push": False}, False, True)
    assert any("final_transfer_ms" in error for error in errors)
    assert any("total_transferred_bytes" in error for error in errors)


def test_finish_only_control_rejects_early_bytes() -> None:
    """对照组被误设为提前发送时不能纳入正式对照。"""
    errors = validate_transfer_metrics(
        {
            "early_stable_kv_push": False,
            "p_finish_to_d_first_token_ms": 10.0,
            "d_request_to_first_token_ms": 2.0,
            "early_transferred_bytes": 16,
            "final_transferred_bytes": 32,
            "total_transferred_bytes": 48,
            "early_transferred_blocks": 1,
            "final_transferred_blocks": 2,
            "final_transfer_ms": 1.0,
        },
        False,
        True,
    )
    assert errors == ["Finish-only control unexpectedly transferred early bytes"]


def test_stable_summary_tail_quantiles_keep_failed_denominator(tmp_path: Path) -> None:
    """P90/P95 使用同一成功配对，而失败请求仍占成功率分母。"""
    import json

    from summarize_stable_kv_push import compare_runs

    scenario = {
        "knowledge_tokens": 2048,
        "append_tokens": 256,
        "update_tokens": 1792,
        "append_fraction": 0.125,
    }
    directories = {}
    for mode, enabled in (("early", True), ("finish", False)):
        directory = tmp_path / mode
        directory.mkdir()
        config = {
            "early_push": enabled,
            "sample_ids": ["a", "b", "failed"],
            "qps": 1.0,
            "controlled_dataset_metadata": {"scenario": scenario},
        }
        (directory / "config.json").write_text(json.dumps(config), encoding="utf-8")
        rows = [
            {
                "sample_id": sample_id,
                "transport_status": "ok",
                "metrics_status": "valid",
                "decode_observed": True,
                "p_finish_to_d_first_token_ms": duration,
                "completed_bytes_before_p_finish": 25 if enabled else 0,
                "total_transferred_bytes": 100,
            }
            for sample_id, duration in (("a", 10.0), ("b", 20.0))
        ]
        rows.append({"sample_id": "failed", "transport_status": "error"})
        (directory / "requests.jsonl").write_text(
            "\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8"
        )
        directories[mode] = directory
    comparison = compare_runs(directories["early"], directories["finish"])
    assert comparison["num_selected"] == 3
    assert comparison["num_formal_pairs"] == 2
    assert comparison["scenario"] == scenario
    early = comparison["modes"]["early_push"]
    assert early["success_rate"] == pytest.approx(2 / 3)
    assert early["paired_metrics"]["p_finish_to_d_first_token_ms"]["p90"] == 19.0
    assert early["paired_metrics"]["p_finish_to_d_first_token_ms"]["p95"] == 19.5
    assert (
        early["paired_metrics"]["completed_byte_fraction_before_p_finish"]["mean"]
        == 0.25
    )
    assert early["paired_metrics"]["received_blocks_at_finalize"]["n"] == 0


def test_stable_matrix_does_not_pool_repeat_percentiles() -> None:
    """重复轮次按各轮尾分位数汇总，不把小样本混合制造伪精确 P99。"""
    import copy

    from summarize_stable_kv_matrix import aggregate_comparisons

    first: dict[str, Any] = {
        "num_selected": 100,
        "num_formal_pairs": 90,
        "qps": 1.0,
        "controlled_settings": {
            "dataset_sha256": "same-dataset",
            "qps": 1.0,
            "random_seed": 2026,
        },
        "scenario": {
            "knowledge_tokens": 2048,
            "append_tokens": 256,
            "update_tokens": 1792,
            "append_fraction": 0.125,
        },
        "modes": {
            mode: {
                "num_transport_success": 95,
                "num_formal": 90,
                "success_rate": 0.95,
                "paired_metrics": {
                    "p_finish_to_d_first_token_ms": {
                        "n": 90,
                        "mean": 20.0,
                        "p90": 30.0,
                        "p95": 40.0,
                        "p99": 50.0,
                    }
                },
            }
            for mode in ("early_push", "finish_only")
        },
    }
    second = copy.deepcopy(first)
    for mode in second["modes"].values():
        mode["num_transport_success"] = 85
        mode["num_formal"] = 80
        mode["success_rate"] = 0.85
        mode["paired_metrics"]["p_finish_to_d_first_token_ms"]["p95"] = 60.0
        mode["paired_metrics"]["p_finish_to_d_first_token_ms"]["n"] = 80
    per_run, aggregate = aggregate_comparisons([first, second])
    assert len(per_run) == 4
    assert len(aggregate) == 2
    for row in aggregate:
        assert row["num_runs"] == 2
        assert row["p95_median"] == 50.0
        assert row["p95_min"] == 40.0
        assert row["p95_max"] == 60.0
        assert row["success_rate"] == 0.90
        assert row["n_min"] == 80
        assert row["n_max"] == 90


def test_stable_matrix_arrival_seed_changes_group_and_context() -> None:
    """不同泊松到达种子改变实际排队时序，不能悄悄并为同一重复组。"""
    from summarize_stable_kv_matrix import aggregate_comparisons

    comparisons = [
        {
            "num_selected": 1,
            "num_formal_pairs": 1,
            "qps": 1.0,
            "controlled_settings": {
                "dataset_sha256": "same-dataset",
                "qps": 1.0,
                "arrival_process": "poisson",
                "random_seed": seed,
            },
            "scenario": {
                "knowledge_tokens": 2048,
                "append_tokens": 256,
                "update_tokens": 1792,
                "append_fraction": 0.125,
            },
            "modes": {
                mode: {
                    "num_transport_success": 1,
                    "num_formal": 1,
                    "success_rate": 1.0,
                    "paired_metrics": {
                        "p_finish_to_d_first_token_ms": {
                            "n": 1,
                            "mean": 20.0,
                            "p90": 20.0,
                            "p95": 20.0,
                            "p99": 20.0,
                        }
                    },
                }
                for mode in ("early_push", "finish_only")
            },
        }
        for seed in (2026, 7)
    ]
    per_run, aggregate = aggregate_comparisons(comparisons)
    assert len({row["group_id"] for row in per_run}) == 2
    assert len({row["context_id"] for row in per_run}) == 2
    assert len(aggregate) == 4
    assert all(row["num_runs"] == 1 for row in aggregate)


def test_stable_matrix_plot_keeps_overall_client_intervals(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """阶段加速不能隐藏 P 等待；总图保留真实 D 续写与完整响应面板。"""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.figure import Figure
    from summarize_stable_kv_matrix import plot_matrix

    captured: list[Figure] = []
    original_subplots = plt.subplots

    def capture_subplots(*args: object, **kwargs: object) -> object:
        """保留实际 figure，保存 PNG 后仍能检查轴标签。"""
        figure, axes = original_subplots(*args, **kwargs)
        captured.append(figure)
        return figure, axes

    monkeypatch.setattr(plt, "subplots", capture_subplots)
    rows = [
        {
            "context_id": "test-context",
            "qps": 0.0,
            "mode": mode,
            "metric": metric,
            "append_fraction": 0.5,
            "success_rate": 1.0,
            "p95_median": 20.0,
            "p95_min": 18.0,
            "p95_max": 22.0,
        }
        for mode in ("early_push", "finish_only")
        for metric in (
            "p_finish_to_d_first_token_ms",
            "finish_to_d_first_client_ms",
            "finish_to_done_ms",
        )
    ]
    outputs = plot_matrix(rows, tmp_path, "p95")
    assert len(outputs) == 1
    assert Path(outputs[0]).stat().st_size > 0
    assert len(captured[0].axes) == 8
    labels = {axis.get_ylabel() for axis in captured[0].axes}
    assert "Client finish to D continuation token (ms)" in labels
    assert "Client finish to full response (ms)" in labels


def test_stable_paired_tail_changes_include_overall_client_intervals() -> None:
    """整体指标的配对降幅采用已测分位数，不能拼接两个阶段的分位数。"""
    from summarize_stable_kv_matrix import paired_tail_changes

    comparison = {
        "group_id": "test-group",
        "context_id": "test-context",
        "scenario": {"append_fraction": 0.5},
        "qps": 1.0,
        "num_selected": 128,
        "num_formal_pairs": 128,
        "modes": {
            mode: {
                "paired_metrics": {
                    "finish_to_d_first_client_ms": {"p95": duration},
                    "finish_to_done_ms": {"p95": duration * 2},
                }
            }
            for mode, duration in (("early_push", 120.0), ("finish_only", 100.0))
        },
    }
    rows = paired_tail_changes([comparison])
    for metric in ("finish_to_d_first_client_ms", "finish_to_done_ms"):
        row = next(
            row
            for row in rows
            if row["metric"] == metric and row["percentile"] == "p95"
        )
        assert row["reduction_percent"] == -20.0
