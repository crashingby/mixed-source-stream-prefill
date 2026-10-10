# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""稳定 KV 内部控制路由；最终生成继续使用普通 completions。"""

from fastapi import APIRouter, FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse

from vllm.entrypoints.serve.stable_kv.protocol import (
    StableKVControlRequest,
    StableKVOperation,
)
from vllm.entrypoints.serve.stable_kv.serving import StableKVServing

router = APIRouter()


async def _control(body: StableKVControlRequest, raw_request: Request) -> JSONResponse:
    """调用引擎桥，返回 ACK 而不占用任何生成输出队列。"""
    serving = StableKVServing(raw_request.app.state.engine_client)
    try:
        payload = await serving.control(body)
    except (ValueError, RuntimeError) as exc:
        # 配置、状态和 manifest 错误属于当前会话冲突，不伪装成成功。
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return JSONResponse(content=payload)


@router.post("/v1/stable_kv/control")
async def stable_kv_control(
    body: StableKVControlRequest, raw_request: Request
) -> JSONResponse:
    """接收包含 op 的控制消息。"""
    return await _control(body, raw_request)


@router.post("/v1/stable_kv/{operation}")
async def stable_kv_operation(
    operation: StableKVOperation,
    body: StableKVControlRequest,
    raw_request: Request,
) -> JSONResponse:
    """提供 reserve/bind/status/finalize/abort/describe 的明确路由。"""
    return await _control(body.model_copy(update={"op": operation}), raw_request)


def attach_router(app: FastAPI) -> None:
    """注册实验控制路由；实际调用仍检查 stable_kv_push 配置。"""
    app.include_router(router)
