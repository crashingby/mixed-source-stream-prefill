# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""稳定 KV 控制消息；具体内存约束由 EngineCore 再验证。"""

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

StableKVOperation = Literal[
    "describe", "reserve", "bind", "status", "finalize", "abort"
]


class StableKVControlRequest(BaseModel):
    """接收会话控制体，完全避免把未完成 prompt 伪装成生成请求。"""

    model_config = ConfigDict(extra="forbid")

    op: StableKVOperation = "describe"  # 路由别名可覆盖操作名称。
    session_id: str | None = Field(default=None, min_length=1, max_length=256)
    request_id: str | None = None  # P start 返回的实际内部 ID。
    peer: dict[str, Any] | None = None  # 对侧 connector 注册信息。
    early_push: bool = True  # False 为相同传输栈的 finish-only 对照。
    kv_transfer_params: dict[str, Any] | None = None  # 最终精确物理布局。
    physical_prompt_token_ids: list[int] | None = None  # 不包含 P 新采样 token。
