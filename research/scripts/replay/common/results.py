# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""可复现实验配置和逐请求结果持久化。"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
from collections.abc import Iterable, Sequence
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .load import read_load_summary
from .metrics import (
    EVENT_METRIC_FIELDS,
    REQUEST_METRIC_FIELDS,
    request_metric_row,
    summarize_metrics,
)
from .models import RenderedReplayPrompt, RequestMetric, SampleReplayInput
from .prompt import ChatTemplateRenderer

logger = logging.getLogger(__name__)


def utc_timestamp() -> str:
    """返回用于输出目录名称的 UTC 时间戳。"""
    return datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")


def json_default(value: Any) -> Any:
    """把 argparse Path 等少量对象转换为 JSON 可序列化值。"""
    if isinstance(value, Path):
        return value.as_posix()
    raise TypeError(f"Unsupported JSON value: {type(value)!r}.")


def write_json(path: Path, value: Any) -> None:
    """写入格式稳定的 UTF-8 JSON 文件。

    Args:
        path: 输出文件路径。
        value: 可 JSON 序列化的数据。
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        f"{json.dumps(value, ensure_ascii=False, indent=2, default=json_default)}\n",
        encoding="utf-8",
    )


def write_csv(
    path: Path, field_names: Sequence[str], rows: Iterable[dict[str, Any]]
) -> None:
    """以固定列顺序写入 CSV 指标文件。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=field_names, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_run_outputs(
    run_directory: Path,
    args: argparse.Namespace,
    renderer: ChatTemplateRenderer,
    samples: Sequence[SampleReplayInput],
    metrics: Sequence[RequestMetric],
) -> None:
    """写入可复现实验的配置、逐请求指标、逐事件指标和摘要。"""
    request_rows = [request_metric_row(metric) for metric in metrics]
    event_rows = [
        asdict(event_metric)
        for metric in metrics
        for event_metric in metric.event_metrics
    ]
    selected_sample_manifest = [
        {
            "sample_id": sample.sample_id,
            "query_id": sample.query_id,
            "status": sample.status,
            "content_sha256": hashlib.sha256(
                json.dumps(
                    sample.raw_sample,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode("utf-8")
            ).hexdigest(),
        }
        for sample in samples
    ]
    selected_sample_set_sha256 = hashlib.sha256(
        json.dumps(
            selected_sample_manifest,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    write_csv(
        run_directory / "request_metrics.csv", REQUEST_METRIC_FIELDS, request_rows
    )
    write_csv(run_directory / "event_metrics.csv", EVENT_METRIC_FIELDS, event_rows)
    write_json(
        run_directory / "config.json",
        {
            "args": vars(args),
            "prompt_format": {
                "name": "model_chat_template_system_user/v1",
                "tokenizer_model": renderer.tokenizer_model,
                "chat_template_sha256": renderer.chat_template_sha256,
                "transport": getattr(args, "input_transport", "text"),
            },
            "num_selected_samples": len(samples),
            "selected_sample_set_sha256": selected_sample_set_sha256,
            "selected_samples": selected_sample_manifest,
        },
    )
    write_json(
        run_directory / "summary.json",
        {
            "replay_mode": args.replay_mode,
            "ttft_definition": (
                "P-only: prefill_ready; full PD: D first answer token. "
                "Existing request/finish fields exclude client admission waiting; "
                "arrival_to_first_token_ms includes all client waiting."
            ),
            "load_summary": read_load_summary(run_directory),
            "num_selected_samples": len(samples),
            "num_completed_requests": len(metrics),
            "num_uncompleted_requests": len(samples) - len(metrics),
            "summary_by_mode": summarize_metrics(metrics),
        },
    )
    if args.save_responses:
        response_directory = run_directory / "responses"
        for metric in metrics:
            if not metric.response_payloads:
                continue
            write_json(
                response_directory / f"{metric.mode}-{metric.sample_id}.json",
                metric.response_payloads,
            )


def make_run_directory(args: argparse.Namespace) -> tuple[str, Path]:
    """创建本次回放的独立结果目录。"""
    run_id = args.run_name or f"{utc_timestamp()}-{args.replay_mode}"
    run_directory = args.output_dir / run_id
    if run_directory.exists():
        raise FileExistsError(
            f"Replay output directory already exists: {run_directory}. "
            "Choose a new --run-name."
        )
    run_directory.mkdir(parents=True)
    return run_id, run_directory


class RequestJournal:
    """立即保存已完成请求；CSV 和汇总仍沿用原实验输出格式。"""

    def __init__(self, directory: Path) -> None:
        self.metrics: list[RequestMetric] = []
        self._stream = (directory / "request_metrics.jsonl").open("w", encoding="utf-8")

    def record(self, metric: RequestMetric) -> None:
        """持久化一次请求结果后再允许 fail-fast 终止。"""
        self.metrics.append(metric)
        self._stream.write(json.dumps(asdict(metric), ensure_ascii=False) + "\n")
        self._stream.flush()

    def close(self) -> None:
        """关闭结果日志。"""
        self._stream.close()


def write_prompt_previews(
    directory: Path,
    samples: Sequence[SampleReplayInput],
    prompts: Sequence[RenderedReplayPrompt],
) -> None:
    """为 dry-run 保存最终输入文本和边界，便于核对模板与证据，不写评价答案。"""
    with (directory / "prompt_preview.jsonl").open("w", encoding="utf-8") as stream:
        for sample, prompt in zip(samples, prompts):
            record = {
                "sample_id": sample.sample_id,
                "input_transport": prompt.input_transport,
                "start_text": prompt.start_text,
                "finish_text": prompt.finish_text,
                "logical_prompt_text": prompt.logical_prompt_text,
                "append_chars": prompt.append_chars,
                "final_update_chars": prompt.final_update_chars,
            }
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
