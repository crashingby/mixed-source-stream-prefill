# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""把内部 REST 控制交给 EngineCore，HTTP 层不持有 KV 所有权。"""

from typing import Any

from vllm.engine.protocol import EngineClient
from vllm.entrypoints.serve.stable_kv.protocol import StableKVControlRequest


class StableKVServing:
    """稳定 KV 的无输出收集器控制桥。"""

    def __init__(self, engine_client: EngineClient) -> None:
        self.engine_client = engine_client  # 复用当前 API server 的 engine。

    async def control(self, request: StableKVControlRequest) -> dict[str, Any]:
        """校验显式启用配置后执行一次主线程控制操作。

        reserve/bind/finalize/abort 会修改 EngineCore 的会话与块引用；
        status/describe 只读取状态。所有操作均不执行模型生成。
        """
        config = self.engine_client.vllm_config.kv_transfer_config
        if config is None or not config.kv_connector_extra_config.get(
            "stable_kv_push", False
        ):
            raise ValueError("stable_kv_push is not enabled in kv-transfer-config.")
        if request.op != "describe" and not request.session_id:
            raise ValueError(f"Stable KV {request.op} requires session_id.")
        # 只提交实际携带的字段，避免 None 覆盖 EngineCore 的默认值。
        return await self.engine_client.stable_kv_control(
            request.model_dump(exclude_none=True)
        )
