# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""HotpotQA 答案 EM/F1；不把它标记为 supporting-fact 或 joint 评分。"""

from __future__ import annotations

from collections import Counter

from .squad import normalize_answer

SPECIAL_ANSWERS = frozenset(("yes", "no", "noanswer"))


def score(prediction: str, answer: str) -> tuple[float, float]:
    """yes/no/noanswer 不匹配时，F1 必须为零。"""
    predicted, expected = normalize_answer(prediction), normalize_answer(answer)
    exact = float(predicted == expected)
    if (predicted in SPECIAL_ANSWERS or expected in SPECIAL_ANSWERS) and not exact:
        return exact, 0.0
    left, right = predicted.split(), expected.split()
    common = sum((Counter(left) & Counter(right)).values())
    return exact, 2 * common / (len(left) + len(right)) if common else 0.0
