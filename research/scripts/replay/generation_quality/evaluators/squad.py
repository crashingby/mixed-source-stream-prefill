# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""SQuAD v2 的答案归一化、EM 和 token F1。"""

from __future__ import annotations

import string
from collections import Counter

import regex as re

# 与原评价器一致，单词边界按字母、数字、下划线划分。
ARTICLES_RE = re.compile(r"(?<![\p{L}\p{N}_])(a|an|the)(?![\p{L}\p{N}_])")


def normalize_answer(text: str) -> str:
    """小写、删除 ASCII 标点与冠词，再合并空白。"""
    value = "".join(char for char in text.lower() if char not in string.punctuation)
    return " ".join(ARTICLES_RE.sub(" ", value).split())


def score(prediction: str, answer: str) -> tuple[float, float]:
    """空答案按 SQuAD v2 无答案规则计分。"""
    predicted, expected = normalize_answer(prediction), normalize_answer(answer)
    exact = float(predicted == expected)
    left, right = predicted.split(), expected.split()
    if not left or not right:
        return exact, float(left == right)
    common = sum((Counter(left) & Counter(right)).values())
    return exact, 2 * common / (len(left) + len(right))
