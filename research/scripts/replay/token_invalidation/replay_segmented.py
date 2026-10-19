# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""通过 proxy TTFT-only 统计 segmented 布局的实际失效 token。"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

from research.scripts.replay.token_invalidation.replay_common import main

if __name__ == "__main__":
    main("segmented")
