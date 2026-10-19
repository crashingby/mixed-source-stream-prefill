# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""按数据集官方答案协议选择评分器；评价字段不参与 prompt 构造。"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from . import hotpotqa, squad, triviaqa

PROTOCOLS = {
    "squad_v2": squad,
    "squad_v2_question_level_em_f1": squad,
    "triviaqa_question_level_em_f1": triviaqa,
    "hotpotqa_answer_em_f1": hotpotqa,
}


def references(evaluation: dict[str, Any]) -> list[str]:
    """读取冻结的官方答案；TriviaQA 使用 normalized aliases 与 human answers。"""
    protocol = evaluation.get("protocol")
    if protocol not in PROTOCOLS:
        raise ValueError(f"不支持的评分协议：{protocol}")
    answer = evaluation.get("answer")
    if protocol == "triviaqa_question_level_em_f1":
        if not isinstance(answer, dict):
            raise ValueError("TriviaQA answer 必须是官方答案对象")
        aliases = answer.get("NormalizedAliases")
        if not isinstance(aliases, list):
            raise ValueError("TriviaQA 缺少 NormalizedAliases")
        values = aliases + [
            triviaqa.normalize_answer(value) for value in answer.get("HumanAnswers", [])
        ]
    elif protocol == "hotpotqa_answer_em_f1":
        values = [answer]
    else:
        values = evaluation.get("answers", [])
    if (
        not isinstance(values, list)
        or not values
        or not all(isinstance(value, str) for value in values)
    ):
        raise ValueError(f"官方答案缺失或类型不合法：{protocol}")
    return list(dict.fromkeys(values))


def score_answer(
    prediction: str, answers: Sequence[str], protocol: str
) -> tuple[float, float]:
    """对该数据集的多个官方答案分别取最高 EM 和 F1。"""
    if protocol not in PROTOCOLS:
        raise ValueError(f"不支持的评分协议：{protocol}")
    evaluator = PROTOCOLS[protocol]
    values = [evaluator.score(prediction, answer) for answer in answers]
    if not values:
        raise ValueError("评分需要至少一个参考答案")
    return max(value[0] for value in values), max(value[1] for value in values)
