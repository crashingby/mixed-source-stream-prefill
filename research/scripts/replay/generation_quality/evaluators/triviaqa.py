# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""TriviaQA 官方 question-level 答案评分口径。"""

from __future__ import annotations

import string
from collections import Counter

import regex as re

PUNCTUATION = frozenset(string.punctuation + "‘’´`")
# 与原评价器一致，单词边界按字母、数字、下划线划分。
ARTICLES_RE = re.compile(r"(?<![\p{L}\p{N}_])(a|an|the)(?![\p{L}\p{N}_])")


def normalize_answer(text: str) -> str:
    """标点和下划线替换为空格，再去冠词并合并空白。"""
    value = "".join(
        " " if char in PUNCTUATION else char for char in text.lower().replace("_", " ")
    )
    return " ".join(ARTICLES_RE.sub(" ", value).split())


def score(prediction: str, answer: str) -> tuple[float, float]:
    """归一化后计算 EM；无共同 token 时 F1 为零。"""
    predicted, expected = normalize_answer(prediction), normalize_answer(answer)
    left, right = predicted.split(), expected.split()
    common = sum((Counter(left) & Counter(right)).values())
    return float(predicted == expected), 2 * common / (
        len(left) + len(right)
    ) if common else 0.0
