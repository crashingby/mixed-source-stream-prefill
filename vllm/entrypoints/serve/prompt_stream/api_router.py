# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from http import HTTPStatus

from fastapi import APIRouter, FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse

from vllm.engine.protocol import EngineClient
from vllm.entrypoints.serve.prompt_stream.protocol import (
    PromptStreamEventEnvelope,
    PromptStreamStartRequest,
)
from vllm.entrypoints.serve.prompt_stream.serving import PromptStreamServing
from vllm.logger import init_logger

logger = init_logger(__name__)

router = APIRouter()


def engine_client(request: Request) -> EngineClient:
    """返回当前 OpenAI server 持有的 EngineClient。"""
    return request.app.state.engine_client


def prompt_stream_serving(request: Request) -> PromptStreamServing:
    """懒加载 P 节点本地 prompt stream serving 对象。"""
    serving = getattr(request.app.state, "prompt_stream_serving", None)
    if serving is None:
        serving = PromptStreamServing(engine_client(request))
        request.app.state.prompt_stream_serving = serving
    return serving


@router.post("/v1/prompt_stream/start")
async def start_prompt_stream(
    request: PromptStreamStartRequest,
    raw_request: Request,
):
    """创建 P 节点上的长期 prompt stream 请求。"""
    try:
        response = await prompt_stream_serving(raw_request).start(request)
    except ValueError as err:
        raise HTTPException(
            status_code=HTTPStatus.BAD_REQUEST,
            detail=str(err),
        ) from err
    logger.info("Started prompt stream request %s.", response.request_id)
    return JSONResponse(content=response.model_dump())


@router.post("/v1/prompt_stream/event")
async def add_prompt_stream_event(
    request: PromptStreamEventEnvelope,
    raw_request: Request,
):
    """添加 Append / Update / Source Finished 事件。"""
    try:
        response = await prompt_stream_serving(raw_request).add_event(request)
    except ValueError as err:
        raise HTTPException(
            status_code=HTTPStatus.BAD_REQUEST,
            detail=str(err),
        ) from err
    return JSONResponse(content=response.model_dump())


@router.post("/v1/prompt_stream/finish")
async def finish_prompt_stream(
    request: PromptStreamEventEnvelope,
    raw_request: Request,
):
    """提交 PROMPT_FINISHED，并返回 P 节点的 KV 参数与实验统计。"""
    try:
        response = await prompt_stream_serving(raw_request).finish(request)
    except ValueError as err:
        raise HTTPException(
            status_code=HTTPStatus.BAD_REQUEST,
            detail=str(err),
        ) from err
    logger.info("Finished prompt stream prefill for request %s.", request.request_id)
    return JSONResponse(content=response.model_dump())


@router.post("/v1/prompt_stream/abort")
async def abort_prompt_stream(raw_request: Request):
    """取消 P 节点上的 prompt stream 请求。"""
    body = await raw_request.json()
    request_id = body.get("request_id")
    if not isinstance(request_id, str):
        raise HTTPException(
            status_code=HTTPStatus.BAD_REQUEST,
            detail="prompt_stream abort requires string request_id.",
        )
    await prompt_stream_serving(raw_request).abort(request_id)
    return JSONResponse(content={"request_id": request_id, "status": "aborted"})


def attach_router(app: FastAPI) -> None:
    """把 prompt stream 实验性 API 注册到 OpenAI server。"""
    app.include_router(router)
