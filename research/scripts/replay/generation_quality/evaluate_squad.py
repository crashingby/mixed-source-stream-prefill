# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""使用 SQuAD v2 官方 EM/F1 口径评估两种 generation quality 结果。"""

from __future__ import annotations

import argparse
import csv
import json
import math
import string
import sys
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import regex as re

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))
from research.scripts.replay.generation_quality.evaluators import score_answer

EVALUATION_SCHEMA_VERSION = "generation_quality_squad_eval/v1"
# 显式限定单词字符，避免 regex 将组合音标和连接符计入冠词边界。
ARTICLES_RE = re.compile(r"(?<![\p{L}\p{N}_])(a|an|the)(?![\p{L}\p{N}_])")

PER_REQUEST_FIELDS = (
    "sample_id",
    "query_id",
    "question",
    "gold_answers_json",
    "normal_status",
    "normal_prediction",
    "normal_exact",
    "normal_f1",
    "normal_error",
    "segmented_status",
    "segmented_prediction",
    "segmented_exact",
    "segmented_f1",
    "segmented_error",
    "paired_evaluable",
    "exact_outcome",
    "f1_delta_segmented_minus_normal",
)


@dataclass(frozen=True)
class ScoredPrediction:
    """一个成功生成结果及其 SQuAD 官方分数。"""

    prediction: str
    exact: float
    f1: float


def normalize_answer(text: str) -> str:
    """按 SQuAD 官方实现归一化一个答案字符串。

    归一化依次执行小写化、删除 ASCII 标点、删除英语冠词和合并空白。该函数
    故意不做语义改写或答案抽取，以保持指标与官方定义一致。
    """

    def remove_articles(value: str) -> str:
        return ARTICLES_RE.sub(" ", value)

    def remove_punctuation(value: str) -> str:
        return "".join(
            character for character in value if character not in string.punctuation
        )

    def white_space_fix(value: str) -> str:
        return " ".join(value.split())

    return white_space_fix(remove_articles(remove_punctuation(text.lower())))


def compute_exact(gold_answer: str, prediction: str) -> float:
    """计算一个预测与一个官方答案的归一化 Exact Match。"""
    return float(normalize_answer(gold_answer) == normalize_answer(prediction))


def compute_f1(gold_answer: str, prediction: str) -> float:
    """计算一个预测与一个官方答案的 token-level F1。"""
    gold_tokens = normalize_answer(gold_answer).split()
    prediction_tokens = normalize_answer(prediction).split()
    if not gold_tokens or not prediction_tokens:
        # SQuAD v2 以空字符串表示无法回答；两侧同时为空才记为 1。
        return float(gold_tokens == prediction_tokens)

    common_tokens = Counter(gold_tokens) & Counter(prediction_tokens)
    num_same_tokens = sum(common_tokens.values())
    if num_same_tokens == 0:
        return 0.0
    precision = num_same_tokens / len(prediction_tokens)
    recall = num_same_tokens / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def _score_prediction(
    prediction: str,
    gold_answers: Sequence[str],
) -> ScoredPrediction:
    """对多个官方标注取最高 EM 和最高 F1，与 SQuAD evaluator 一致。"""
    references = list(gold_answers) or [""]
    return ScoredPrediction(
        prediction=prediction,
        exact=max(compute_exact(reference, prediction) for reference in references),
        f1=max(compute_f1(reference, prediction) for reference in references),
    )


def _resolve_generations_path(path: Path) -> Path:
    """同时接受运行目录或具体 ``generations.jsonl`` 路径。"""
    resolved_path = path / "generations.jsonl" if path.is_dir() else path
    if not resolved_path.is_file():
        raise FileNotFoundError(f"Generation result file not found: {resolved_path}.")
    return resolved_path


def _load_generation_records(path: Path) -> dict[str, dict[str, Any]]:
    """读取一个模式的 JSONL 结果，并按 sample id 去重。"""
    generations_path = _resolve_generations_path(path)
    records: dict[str, dict[str, Any]] = {}
    with generations_path.open(encoding="utf-8") as stream:
        lines = list(stream)
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"Invalid JSON at {generations_path}:{line_number}: {exc}"
            ) from exc
        if not isinstance(record, dict) or not isinstance(record.get("sample_id"), str):
            raise ValueError(
                f"Invalid generation record at {generations_path}:{line_number}."
            )
        sample_id = record["sample_id"]
        if sample_id in records:
            raise ValueError(
                f"Duplicate sample_id={sample_id!r} in {generations_path}."
            )
        records[sample_id] = record
    if not records:
        raise ValueError(f"No generation records found in {generations_path}.")
    return records


def _validate_mode(records: dict[str, dict[str, Any]], expected_mode: str) -> None:
    """防止把 normal 与 segmented 结果路径传反。"""
    observed_modes = {str(record.get("mode")) for record in records.values()}
    if observed_modes != {expected_mode}:
        raise ValueError(
            f"Expected mode={expected_mode!r}, observed modes={observed_modes}."
        )


def _gold_answers(record: dict[str, Any]) -> list[str]:
    """读取结果内冻结的官方答案列表。"""
    raw_answers = record.get("gold_answers")
    if not isinstance(raw_answers, list) or not all(
        isinstance(answer, str) for answer in raw_answers
    ):
        raise ValueError(
            f"Record {record.get('sample_id')!r} has invalid gold_answers."
        )
    if bool(record.get("is_impossible", False)):
        return raw_answers or [""]
    if not raw_answers:
        raise ValueError(
            f"Answerable record {record.get('sample_id')!r} has no gold answer."
        )
    return raw_answers


def _score_record(record: dict[str, Any] | None) -> ScoredPrediction | None:
    """只对 REST 链路成功并完整返回的记录计算官方分数。"""
    if record is None or record.get("result_status") != "ok":
        return None
    prediction = record.get("response_text")
    if not isinstance(prediction, str):
        raise ValueError(
            f"Successful record {record.get('sample_id')!r} has no response_text."
        )
    exact, f1 = score_answer(
        prediction.strip(),
        _gold_answers(record),
        str(record.get("evaluation_protocol", "squad_v2")),
    )
    return ScoredPrediction(prediction.strip(), exact, f1)


def _mode_summary(
    records: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """汇总一个模式的成功率、EM 和平均 F1。"""
    scored_predictions = [
        score
        for record in records.values()
        if (score := _score_record(record)) is not None
    ]
    num_total = len(records)
    num_evaluated = len(scored_predictions)
    exact_sum = sum(score.exact for score in scored_predictions)
    f1_sum = sum(score.f1 for score in scored_predictions)
    return {
        "num_records": num_total,
        "num_evaluated": num_evaluated,
        "num_failed_or_incomplete": num_total - num_evaluated,
        "evaluation_protocols": sorted(
            {
                record.get("evaluation_protocol", "squad_v2")
                for record in records.values()
            }
        ),
        "exact_match_all_requests": 100 * exact_sum / num_total if num_total else None,
        "f1_all_requests": 100 * f1_sum / num_total if num_total else None,
        "success_rate": num_evaluated / num_total if num_total else None,
        # SQuAD leaderboard 使用 0--100 的百分数口径。
        "exact_match": 100 * exact_sum / num_evaluated if num_evaluated else None,
        "f1": 100 * f1_sum / num_evaluated if num_evaluated else None,
    }


def _mcnemar_exact_p_value(normal_only: int, segmented_only: int) -> float:
    """计算配对 EM 分歧样本上的双侧 exact McNemar p-value。"""
    num_disagreements = normal_only + segmented_only
    if num_disagreements == 0:
        return 1.0
    smaller_count = min(normal_only, segmented_only)
    lower_tail = sum(
        math.comb(num_disagreements, index) for index in range(smaller_count + 1)
    ) / (2**num_disagreements)
    return min(1.0, 2 * lower_tail)


def _record_value(record: dict[str, Any] | None, key: str) -> Any:
    """从可选结果记录读取字段，缺失模式返回 ``None``。"""
    return record.get(key) if record is not None else None


def _paired_rows_and_summary(
    normal_records: dict[str, dict[str, Any]],
    segmented_records: dict[str, dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """生成逐请求得分，并统计 normal→segmented 的配对质量变化。"""
    all_sample_ids = sorted(set(normal_records) | set(segmented_records))
    rows: list[dict[str, Any]] = []
    outcome_counts = {
        "both_exact": 0,
        "normal_only_exact": 0,
        "segmented_only_exact": 0,
        "neither_exact": 0,
    }
    paired_f1_deltas: list[float] = []

    for sample_id in all_sample_ids:
        normal_record = normal_records.get(sample_id)
        segmented_record = segmented_records.get(sample_id)
        base_record = normal_record or segmented_record
        assert base_record is not None

        if normal_record is not None and segmented_record is not None:
            if normal_record.get(
                "evaluation_protocol", "squad_v2"
            ) != segmented_record.get("evaluation_protocol", "squad_v2"):
                raise ValueError(f"Modes disagree on evaluation protocol: {sample_id}")
            if _gold_answers(normal_record) != _gold_answers(segmented_record):
                raise ValueError(
                    f"Modes disagree on gold answers for sample_id={sample_id!r}."
                )
            if normal_record.get("logical_prompt_sha256") != segmented_record.get(
                "logical_prompt_sha256"
            ):
                raise ValueError(
                    f"Modes used different logical prompts for sample_id={sample_id!r}."
                )
            if normal_record.get(
                "input_transport", "token_ids"
            ) != segmented_record.get("input_transport", "token_ids"):
                raise ValueError(f"Modes disagree on input transport: {sample_id}")
            normal_token_hash = normal_record.get("logical_prompt_token_sha256")
            segmented_token_hash = segmented_record.get("logical_prompt_token_sha256")
            if (
                normal_token_hash is not None or segmented_token_hash is not None
            ) and normal_token_hash != segmented_token_hash:
                raise ValueError(
                    "Modes used different logical prompt token sequences for "
                    f"sample_id={sample_id!r}."
                )

        normal_score = _score_record(normal_record)
        segmented_score = _score_record(segmented_record)
        paired_evaluable = normal_score is not None and segmented_score is not None
        exact_outcome: str | None = None
        f1_delta: float | None = None
        if paired_evaluable:
            assert normal_score is not None and segmented_score is not None
            normal_exact = bool(normal_score.exact)
            segmented_exact = bool(segmented_score.exact)
            if normal_exact and segmented_exact:
                exact_outcome = "both_exact"
            elif normal_exact:
                exact_outcome = "normal_only_exact"
            elif segmented_exact:
                exact_outcome = "segmented_only_exact"
            else:
                exact_outcome = "neither_exact"
            outcome_counts[exact_outcome] += 1
            f1_delta = segmented_score.f1 - normal_score.f1
            paired_f1_deltas.append(f1_delta)

        gold_answers = _gold_answers(base_record)
        rows.append(
            {
                "sample_id": sample_id,
                "query_id": base_record.get("query_id"),
                "question": base_record.get("question"),
                "gold_answers_json": json.dumps(
                    gold_answers, ensure_ascii=False, sort_keys=True
                ),
                "normal_status": _record_value(normal_record, "result_status"),
                "normal_prediction": (
                    normal_score.prediction if normal_score is not None else None
                ),
                "normal_exact": (
                    normal_score.exact if normal_score is not None else None
                ),
                "normal_f1": normal_score.f1 if normal_score is not None else None,
                "normal_error": _record_value(normal_record, "error"),
                "segmented_status": _record_value(segmented_record, "result_status"),
                "segmented_prediction": (
                    segmented_score.prediction if segmented_score is not None else None
                ),
                "segmented_exact": (
                    segmented_score.exact if segmented_score is not None else None
                ),
                "segmented_f1": (
                    segmented_score.f1 if segmented_score is not None else None
                ),
                "segmented_error": _record_value(segmented_record, "error"),
                "paired_evaluable": paired_evaluable,
                "exact_outcome": exact_outcome,
                "f1_delta_segmented_minus_normal": f1_delta,
            }
        )

    num_paired = sum(outcome_counts.values())
    num_normal_exact = (
        outcome_counts["both_exact"] + outcome_counts["normal_only_exact"]
    )
    f1_delta_mean = (
        sum(paired_f1_deltas) / len(paired_f1_deltas) if paired_f1_deltas else None
    )
    paired_summary = {
        "num_union_samples": len(all_sample_ids),
        "num_paired_evaluable": num_paired,
        "exact_outcome_counts": outcome_counts,
        "segmented_em_regression_rate_given_normal_exact": (
            outcome_counts["normal_only_exact"] / num_normal_exact
            if num_normal_exact
            else None
        ),
        "segmented_em_retention_rate_given_normal_exact": (
            outcome_counts["both_exact"] / num_normal_exact
            if num_normal_exact
            else None
        ),
        "mean_f1_delta_segmented_minus_normal": f1_delta_mean,
        "mean_f1_delta_percentage_points": (
            100 * f1_delta_mean if f1_delta_mean is not None else None
        ),
        "mcnemar_exact_p_value": _mcnemar_exact_p_value(
            outcome_counts["normal_only_exact"],
            outcome_counts["segmented_only_exact"],
        ),
    }
    return rows, paired_summary


def _write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    """写入固定字段顺序的逐请求评分 CSV。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=PER_REQUEST_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def _write_json(path: Path, value: Any) -> None:
    """以稳定格式写入汇总 JSON。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def parse_args() -> argparse.Namespace:
    """解析 SQuAD EM/F1 评测参数。"""
    parser = argparse.ArgumentParser(
        description="按结果记录的数据集协议计算 normal/segmented 答案 EM/F1。"
    )
    parser.add_argument("--normal-results", type=Path, required=True)
    parser.add_argument("--segmented-results", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    """加载两个模式的结果，计算官方指标并写出配对统计。"""
    args = parse_args()
    normal_records = _load_generation_records(args.normal_results)
    segmented_records = _load_generation_records(args.segmented_results)
    _validate_mode(normal_records, "normal")
    _validate_mode(segmented_records, "segmented")

    rows, paired_summary = _paired_rows_and_summary(
        normal_records,
        segmented_records,
    )
    summary = {
        "schema_version": EVALUATION_SCHEMA_VERSION,
        "metric_definition": {
            "normalization": (
                "Dataset-specific official normalization selected by "
                "evaluation_protocol."
            ),
            "multiple_gold_answers": "Maximum EM and F1 over official answers.",
            "score_scale": "Mode-level exact_match and f1 use 0--100.",
        },
        "normal": _mode_summary(normal_records),
        "segmented": _mode_summary(segmented_records),
        "paired": paired_summary,
    }
    _write_csv(args.output_dir / "per_request_scores.csv", rows)
    _write_json(args.output_dir / "evaluation_summary.json", summary)

    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    print(f"Evaluation written to {args.output_dir}")


if __name__ == "__main__":
    main()
