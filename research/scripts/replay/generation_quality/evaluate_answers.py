# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""根据每条结果的协议评估 SQuAD、TriviaQA 或 HotpotQA 答案。"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))
from research.scripts.replay.generation_quality import evaluate_squad


def main() -> None:
    """复用配对统计和历史 CSV 布局，标记通用答案评测 schema。"""
    evaluate_squad.EVALUATION_SCHEMA_VERSION = "generation_quality_answer_eval/v2"
    evaluate_squad.main()


if __name__ == "__main__":
    main()
