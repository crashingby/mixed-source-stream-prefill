# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""按官方答案协议评估完整生成，分开报告分段近似和块稀疏的质量变化。"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

from research.scripts.replay.generation_quality.evaluators import score_answer

MODES = ("normal", "segmented", "sparse")
EVALUATION_SCHEMA_VERSION = "generation_quality_answer_eval/v3"


@dataclass
class QualityRun:
    """保存一组完整生成及其预先选定的样本，发现中断后缺失的结果。"""

    mode: str
    path: Path
    records: dict[str, dict[str, Any]]
    selected: dict[str, dict[str, Any]]  # 来自 config，缺失结果仍属于评分分母。


def load_run(mode: str, path: Path) -> QualityRun:
    """读取一个模式的结果；拒绝重复、传反的模式和 dry-run 伪答案。"""
    generations = path / "generations.jsonl" if path.is_dir() else path
    records: dict[str, dict[str, Any]] = {}
    with generations.open(encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            record = json.loads(line)
            sample_id = record["sample_id"]
            if sample_id in records:
                raise ValueError(f"重复样本：{generations}: {sample_id}")
            if record.get("mode") != mode:
                raise ValueError(f"结果模式不是 {mode}：{generations}")
            if record.get("result_status") == "dry_run":
                raise ValueError("dry-run 没有模型答案，不能进行质量评分")
            if record.get("result_status") not in ("ok", "error", "pending"):
                raise ValueError(f"不是完整生成结果：{generations}: {sample_id}")
            records[sample_id] = record
    config = generations.parent / "config.json"
    selected_rows = (
        json.loads(config.read_text()).get("selected_samples", [])
        if config.exists()
        else []
    )
    selected = {row["sample_id"]: row for row in selected_rows}
    if len(selected) != len(selected_rows):
        raise ValueError(f"选中清单包含重复 sample_id：{config}")
    if not selected:
        selected = dict(records)  # 兼容只保留 JSONL 的旧结果。
    if not selected or records.keys() - selected.keys():
        raise ValueError(f"结果与选中清单不一致，或没有可评分样本：{generations}")
    return QualityRun(mode, generations, records, selected)


def score_record(record: dict[str, Any] | None) -> tuple[float, float] | None:
    """只对成功且完整消费 SSE 的预测评分；失败与缺失由汇总层记零。"""
    if record is None or record.get("result_status") != "ok":
        return None
    answers = record.get("gold_answers")
    if not isinstance(answers, list) or not all(
        isinstance(value, str) for value in answers
    ):
        raise ValueError(f"非法官方答案：{record.get('sample_id')}")
    if not answers and record.get("is_impossible"):
        answers = [""]
    prediction = record.get("response_text")
    if not isinstance(prediction, str):
        raise ValueError(f"成功结果缺少文本：{record.get('sample_id')}")
    return score_answer(
        prediction, answers, record.get("evaluation_protocol", "squad_v2")
    )


def mode_summary(run: QualityRun) -> dict[str, Any]:
    """按完整选中集合报告 EM/F1；另列成功样本分数，不隐藏请求失败。"""
    scores = [
        value
        for record in run.records.values()
        if (value := score_record(record)) is not None
    ]
    count, successful = len(run.selected), len(scores)
    exact_sum, f1_sum = (sum(value[index] for value in scores) for index in (0, 1))
    protocols = sorted(
        {
            record.get("evaluation_protocol", "squad_v2")
            for record in (
                run.records.get(sample_id) or metadata
                for sample_id, metadata in run.selected.items()
            )
        }
    )
    return {
        "num_selected": count,
        "num_records": len(run.records),
        "num_successful": successful,
        "num_failed": len(run.records) - successful,
        "num_missing": count - len(run.records),
        "success_rate": successful / count,
        "evaluation_protocols": protocols,
        "probe_keep_ratios": sorted(
            {
                record["probe_keep_ratio"]
                for record in run.records.values()
                if record.get("probe_keep_ratio") is not None
            }
        ),
        "exact_match": 100 * exact_sum / count,
        "f1": 100 * f1_sum / count,
        "exact_match_successful_only": 100 * exact_sum / successful
        if successful
        else None,
        "f1_successful_only": 100 * f1_sum / successful if successful else None,
        "num_length_limited": sum(
            row.get("finish_reason") == "length" for row in run.records.values()
        ),
    }


def validate_pair(reference: QualityRun, candidate: QualityRun) -> None:
    """拒绝不同样本、答案、Prompt、历史事件或生成配置的伪配对。"""
    if reference.selected.keys() != candidate.selected.keys():
        raise ValueError(f"两组选择的样本不同：{reference.mode}, {candidate.mode}")
    for sample_id in reference.selected:
        left, right = reference.records.get(sample_id), candidate.records.get(sample_id)
        # 缺失结果计零；先校验清单摘要，不能因某侧中断而掩盖不同输入。
        left_meta, right_meta = (
            reference.selected[sample_id],
            candidate.selected[sample_id],
        )
        if left_meta.get("request_sha256") != right_meta.get("request_sha256"):
            raise ValueError(f"选中样本来源摘要不一致：{sample_id}")
        if left is None or right is None:
            continue
        for key in (
            "question",
            "gold_answers",
            "is_impossible",
            "evaluation_protocol",
            "logical_prompt_sha256",
            "input_transport",
            "logical_prompt_token_sha256",
            "source_events_sha256",
            "generation_settings",
        ):
            if left.get(key) != right.get(key):
                raise ValueError(f"两组 {key} 不一致：{sample_id}")
        if not left.get("logical_prompt_sha256"):
            raise ValueError(f"缺少逻辑 Prompt 摘要：{sample_id}")


def compare_runs(
    reference: QualityRun, candidate: QualityRun
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """计算全样本净变化、参考正确时的退化，以及成功配对的辅助统计。"""
    validate_pair(reference, candidate)
    rows: list[dict[str, Any]] = []
    outcomes = dict.fromkeys(
        ("both_exact", "reference_only_exact", "candidate_only_exact", "neither_exact"),
        0,
    )
    exact_deltas, f1_deltas = [], []
    successful_exact_deltas, successful_f1_deltas = [], []
    for sample_id in reference.selected:
        left, right = reference.records.get(sample_id), candidate.records.get(sample_id)
        lscore, rscore = score_record(left), score_record(right)
        lexact, lf1 = lscore or (0.0, 0.0)
        rexact, rf1 = rscore or (0.0, 0.0)
        both_successful = lscore is not None and rscore is not None
        outcome = (
            "both_exact"
            if lexact and rexact
            else "reference_only_exact"
            if lexact
            else "candidate_only_exact"
            if rexact
            else "neither_exact"
        )
        outcomes[outcome] += 1
        exact_deltas.append(rexact - lexact)
        f1_deltas.append(rf1 - lf1)
        if both_successful:
            successful_exact_deltas.append(rexact - lexact)
            successful_f1_deltas.append(rf1 - lf1)
        metadata = left or right or reference.selected[sample_id]
        rows.append(
            {
                "reference_mode": reference.mode,
                "candidate_mode": candidate.mode,
                "sample_id": sample_id,
                "query_id": metadata.get("query_id"),
                "question": metadata.get("question"),
                "evaluation_protocol": metadata.get("evaluation_protocol", "squad_v2"),
                "gold_answers_json": json.dumps(
                    metadata.get("gold_answers"), ensure_ascii=False
                ),
                "reference_status": left.get("result_status") if left else "missing",
                "candidate_status": right.get("result_status") if right else "missing",
                "reference_prediction": left.get("response_text") if left else None,
                "candidate_prediction": right.get("response_text") if right else None,
                "reference_exact": lexact,
                "candidate_exact": rexact,
                "reference_f1": lf1,
                "candidate_f1": rf1,
                "exact_delta": rexact - lexact,
                "f1_delta": rf1 - lf1,
                "both_successful": both_successful,
                "exact_outcome": outcome,
                "reference_error": left.get("error") if left else None,
                "candidate_error": right.get("error") if right else None,
            }
        )
    count = len(rows)
    reference_correct = outcomes["both_exact"] + outcomes["reference_only_exact"]
    disagreements = outcomes["reference_only_exact"] + outcomes["candidate_only_exact"]
    # exact McNemar 双侧检验不代表质量等价性证明。
    smaller = min(outcomes["reference_only_exact"], outcomes["candidate_only_exact"])
    p_value = min(
        1.0,
        2
        * (
            sum(math.comb(disagreements, k) for k in range(smaller + 1))
            / 2**disagreements
        ),
    )
    return rows, {
        "reference_mode": reference.mode,
        "candidate_mode": candidate.mode,
        "num_selected": count,
        "num_both_successful": len(successful_f1_deltas),
        "exact_outcome_counts": outcomes,
        "em_delta_percentage_points": 100 * sum(exact_deltas) / count,
        "f1_delta_percentage_points": 100 * sum(f1_deltas) / count,
        "regression_rate_given_reference_exact": outcomes["reference_only_exact"]
        / reference_correct
        if reference_correct
        else None,
        "retention_rate_given_reference_exact": outcomes["both_exact"]
        / reference_correct
        if reference_correct
        else None,
        "em_delta_successful_pairs_percentage_points": 100
        * sum(successful_exact_deltas)
        / len(successful_exact_deltas)
        if successful_exact_deltas
        else None,
        "f1_delta_successful_pairs_percentage_points": 100
        * sum(successful_f1_deltas)
        / len(successful_f1_deltas)
        if successful_f1_deltas
        else None,
        "mcnemar_exact_p_value": p_value,
    }


def evaluate_runs(paths: dict[str, Path], output_dir: Path) -> dict[str, Any]:
    """评分各模式及全部可用配对，将 JSON 汇总和逐请求 CSV 写入结果目录。"""
    runs = {mode: load_run(mode, paths[mode]) for mode in MODES if mode in paths}
    rows: list[dict[str, Any]] = []
    comparisons = {}
    for reference, candidate in itertools.combinations(runs.values(), 2):
        paired_rows, paired = compare_runs(reference, candidate)
        rows.extend(paired_rows)
        comparisons[f"{candidate.mode}_vs_{reference.mode}"] = paired
    summary = {
        "schema_version": EVALUATION_SCHEMA_VERSION,
        "metric_definition": {
            "protocol": "Official answer normalization; maximum EM/F1 over references.",
            "score_scale": "Summary scores: 0--100; deltas: pp; CSV scores: 0--1.",
            "denominator": "Selected samples; failed or missing results score 0.",
            "hotpotqa_scope": "Answer EM/F1 only; no supporting-fact or joint scores.",
            "primary_sparse_comparison": "sparse_vs_segmented",
        },
        "modes": {mode: mode_summary(run) for mode, run in runs.items()},
        "comparisons": comparisons,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "evaluation_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    with (output_dir / "per_request_scores.csv").open(
        "w", encoding="utf-8", newline=""
    ) as stream:
        if rows:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    return summary


def main() -> None:
    """接受可重复的 mode=path，同时兼容历史 normal/segmented 选项。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="append", default=[], metavar="MODE=PATH")
    for mode in MODES:
        parser.add_argument(f"--{mode}-results", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    paths = {
        mode: path
        for mode in MODES
        if (path := getattr(args, f"{mode}_results")) is not None
    }
    for spec in args.run:
        mode, separator, path = spec.partition("=")
        if not separator or mode not in MODES or not path or mode in paths:
            parser.error(f"无效或重复的 --run：{spec}")
        paths[mode] = Path(path)
    if not paths:
        parser.error("至少提供一个 --run MODE=PATH 或 --MODE-results")
    print(
        json.dumps(evaluate_runs(paths, args.output_dir), ensure_ascii=False, indent=2)
    )


if __name__ == "__main__":
    main()
