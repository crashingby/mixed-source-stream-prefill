# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""稳定 Prefix KV 提前 Push 的独立实验代理。

外部仍使用 start/event/finish；start 只建立一次 P/D 接收关系，event 只发 P。
finish 先返回 P 首 token，再等待 D 接收屏障并启动普通 decode。所有交接时间
均由本进程的 perf_counter 计量，避免跨主机时钟差污染实验。
"""

import argparse
import asyncio
import json
import logging
import time
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Any

import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse

try:
    from . import proxy_server as common
except ImportError:
    # 直接执行脚本时，兄弟模块位于 Python 的脚本搜索目录。
    import proxy_server as common

logger = logging.getLogger(__name__)


@dataclass
class StableProxySession:
    """绑定唯一 P/D、传输会话及原始生成参数。"""

    prompt: common.PromptStreamProxySession  # 复用已验证的物理 token 处理。
    session_id: str  # 独立 UUID，不能从请求 ID 截断或推测。
    early_push: bool  # A/B 只改变是否在 finish 前发布稳定区间。
    finishing: bool = False  # finish 一旦提交，拒绝重复或后续事件。
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)


def parse_args() -> argparse.Namespace:
    """解析单 P、单 D 的代理启动参数。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--prefiller-host", default="g2c9")
    parser.add_argument("--prefiller-port", type=int, default=8100)
    parser.add_argument("--decoder-host", default="g2c9")
    parser.add_argument("--decoder-port", type=int, default=8200)
    parser.add_argument("--http-timeout", type=float, default=600.0)
    parser.add_argument("--finalize-timeout", type=float, default=120.0)
    parser.add_argument("--status-poll-interval", type=float, default=0.005)
    parser.add_argument(
        "--early-stable-kv-push", action=argparse.BooleanOptionalAction, default=True
    )
    args = parser.parse_args()
    if min(args.http_timeout, args.finalize_timeout, args.status_poll_interval) <= 0:
        parser.error("Timeouts and status poll interval must be positive.")
    return args


def _client_info(host: str, port: int, timeout: float) -> dict[str, Any]:
    """创建固定节点 HTTP 客户端，不把本机 localhost 当作计算节点。"""
    return {
        "host": host,
        "port": port,
        "client": httpx.AsyncClient(
            base_url=f"http://{host}:{port}/v1",
            timeout=httpx.Timeout(timeout),
            trust_env=False,
            limits=httpx.Limits(max_connections=None, max_keepalive_connections=64),
        ),
    }


@asynccontextmanager
async def lifespan(application: FastAPI) -> AsyncIterator[None]:
    """建立连接，并在代理关闭时回收尚未完成的双侧会话。"""
    args = application.state.args
    application.state.prefiller = _client_info(
        args.prefiller_host, args.prefiller_port, args.http_timeout
    )
    application.state.decoder = _client_info(
        args.decoder_host, args.decoder_port, args.http_timeout
    )
    application.state.sessions = {}
    try:
        yield
    finally:
        for session in list(application.state.sessions.values()):
            await _cleanup(application, session)
        await application.state.prefiller["client"].aclose()
        await application.state.decoder["client"].aclose()


app = FastAPI(lifespan=lifespan)


async def _control(
    client: dict[str, Any], op: str, session_id: str | None = None, **fields: Any
) -> dict[str, Any]:
    """提交一个 ACK 控制请求，绝不通过生成请求伪造接收 work。"""
    body = dict(fields)
    if session_id is not None:
        body["session_id"] = session_id
    return await common._post_json(
        client,
        f"/stable_kv/{op}",
        body,
        session_id or "stable-kv-describe",
        service_name="stable-kv",
    )


def _find_session(request: Request, body: dict[str, Any]) -> StableProxySession:
    """查找由 start 建立的绑定，未知 ID 不允许重新创建目标。"""
    session = request.app.state.sessions.get(body.get("request_id"))
    if session is None:
        raise HTTPException(status_code=404, detail="Unknown stable KV request_id.")
    return session


async def _cleanup(application: FastAPI, session: StableProxySession) -> None:
    """通知两侧中止并释放，EngineCore 保证在途 WRITE 完成前不复用内存。"""
    application.state.sessions.pop(session.prompt.request_id, None)
    calls = [
        _control(application.state.prefiller, "abort", session.session_id),
        _control(application.state.decoder, "abort", session.session_id),
        common._post_json(
            application.state.prefiller,
            "/prompt_stream/abort",
            {"request_id": session.prompt.request_id},
            session.prompt.request_id,
        ),
    ]
    # 一侧失败不能阻止另一侧清理，清理请求本身也不能无限占用代理任务。
    try:
        async with asyncio.timeout(15.0):
            results = await asyncio.gather(*calls, return_exceptions=True)
        for result in results:
            if isinstance(result, BaseException):
                logger.warning("Stable KV cleanup %s: %s", session.session_id, result)
    except TimeoutError:
        logger.warning("Stable KV cleanup timed out: %s", session.session_id)


@app.get("/health")
async def health() -> dict[str, str]:
    """返回代理进程健康状态；P/D 状态由 start 握手单独验证。"""
    return {"status": "ok", "proxy": "stable-kv-push"}


@app.post("/v1/prompt_stream/start")
async def start(request: Request) -> JSONResponse:
    """创建 P 长期 Request，然后 reserve D 并绑定一次传输会话。"""
    body = await request.json()
    early_push = body.pop(
        "early_stable_kv_push", request.app.state.args.early_stable_kv_push
    )
    if type(early_push) is not bool:
        raise HTTPException(
            status_code=400, detail="early_stable_kv_push must be bool."
        )
    if common._get_api_endpoint(body) != "/completions":
        raise HTTPException(status_code=400, detail="Stable KV supports completions.")
    base_request = common._build_decode_base_request(body)
    if base_request.get("n", 1) != 1 or base_request.get("echo", False):
        raise HTTPException(
            status_code=400, detail="Stable KV requires n=1, echo=false."
        )
    producer = request.app.state.prefiller
    consumer = request.app.state.decoder
    peer = await _control(producer, "describe")
    prefill = await common._post_json(
        producer,
        "/prompt_stream/start",
        # bind 为稳定 Push 建立独立生命周期；旧 do_remote_decode 标志会额外
        # 注册整请求发送状态，而分批完成不能替它清理，故此处不传旧参数。
        common._build_prefill_start_request(body, enable_kv_transfer=False),
        body.get("request_id", "stable-kv-start"),
    )
    request_id = prefill["request_id"]
    session = StableProxySession(
        prompt=common.PromptStreamProxySession(
            request_id=request_id,
            prefill_client_info=producer,
            decode_client_info=consumer,
            api_endpoint="/completions",
            base_request=base_request,
            decode_prompt_text=common._get_initial_decode_prompt_text(body),
            decode_prompt_token_ids=common._get_initial_decode_prompt_token_ids(
                body, prefill
            ),
        ),
        session_id=uuid.uuid4().hex,
        early_push=early_push,
    )
    request.app.state.sessions[request_id] = session
    try:
        reserved = await _control(
            consumer,
            "reserve",
            session.session_id,
            request_id=request_id,
            peer=peer,
            early_push=early_push,
        )
        await _control(
            producer,
            "bind",
            session.session_id,
            request_id=request_id,
            peer=reserved["peer"],
            early_push=early_push,
        )
    except BaseException:
        await asyncio.shield(_cleanup(request.app, session))
        raise
    prefill["stable_kv_session_id"] = session.session_id
    prefill["early_stable_kv_push"] = early_push
    return JSONResponse(content=prefill)


@app.post("/v1/prompt_stream/event")
async def event(request: Request) -> JSONResponse:
    """只向 P 转发事件；稳定块实际完成后由 connector 通知 D。"""
    body = await request.json()
    session = _find_session(request, body)
    async with session.lock:
        if session.finishing:
            raise HTTPException(status_code=409, detail="Request is already finishing.")
        response = await common._post_json(
            session.prompt.prefill_client_info,
            "/prompt_stream/event",
            body,
            session.prompt.request_id,
        )
        common._append_decode_prompt_token_ids(session.prompt, response)
        common._append_decode_prompt_text(session.prompt, body.get("event", {}))
    return JSONResponse(content=response)


async def _finalize(
    application: FastAPI,
    session: StableProxySession,
    prefill: dict[str, Any],
) -> tuple[float, dict[str, Any]]:
    """等待 D 最终接收屏障，返回耗时和 finalize 到达时的接收快照。

    early 标记仅说明范围发布时机；received_blocks_at_finalize 才直接说明
    其中多少块在最终交接之前已经到达 D。
    """
    started = time.perf_counter()
    async with asyncio.timeout(application.state.args.finalize_timeout):
        status = await _control(
            application.state.decoder,
            "finalize",
            session.session_id,
            kv_transfer_params=prefill["kv_transfer_params"],
            physical_prompt_token_ids=prefill["prompt_token_ids"],
        )
        while not status.get("ready", False):
            if status.get("state") in ("failed", "aborted"):
                raise RuntimeError(f"Stable KV finalization failed: {status}")
            await asyncio.sleep(application.state.args.status_poll_interval)
            status = await _control(
                application.state.decoder, "status", session.session_id
            )
    received_metrics = status.get("metrics") or {}
    return (time.perf_counter() - started) * 1000, received_metrics


def _has_token(payload: dict[str, Any]) -> bool:
    """把空文本特殊 token 也作为真实输出，排除纯 usage / 结束帧。"""
    return any(
        choice.get("token_ids") or choice.get("text")
        for choice in payload.get("choices", [])
    )


async def _response_events(
    application: FastAPI,
    session: StableProxySession,
    prefill: dict[str, Any],
    decode_request: dict[str, Any],
    p_finish_time: float,
    finish_without_decode: bool,
) -> AsyncIterator[dict[str, Any]]:
    """保留 P 首 token，再标记 D 续写帧和一次性实验指标。

    无论客户端选择 JSON 还是 SSE，上游 D 均使用 SSE，确保首 D token
    计时不是整个生成完成时间。Generator 的 finally 回收双侧会话。
    """
    metrics: dict[str, Any] = {
        "early_stable_kv_push": session.early_push,
        "p_finish_to_d_first_token_ms": None,
        "d_request_to_first_token_ms": None,
        "finalize_wait_ms": None,
    }
    try:
        first = common._build_prefill_stream_chunk(
            session.prompt, prefill, is_final_chunk=finish_without_decode
        )
        first["token_source"] = "prefill"
        yield first
        if not finish_without_decode:
            finalize_wait_ms, received_metrics = await _finalize(
                application, session, prefill
            )
            metrics["finalize_wait_ms"] = finalize_wait_ms
            metrics["received_blocks_at_finalize"] = received_metrics.get(
                "received_blocks_at_finalize"
            )
            # 收养与生成仍走标准 completions，严禁再次发起普通全量 KV load。
            params = dict(prefill["kv_transfer_params"])
            params.update(
                stable_kv_session_id=session.session_id,
                do_remote_prefill=False,
                do_remote_decode=False,
            )
            decode_request["kv_transfer_params"] = params
            decode_request["stream"] = True
            decode_request["return_token_ids"] = True
            decode_request["stream_options"] = {"include_usage": True}
            d_request_time = time.perf_counter()
            async with application.state.decoder["client"].stream(
                "POST",
                "/completions",
                json=decode_request,
                headers=common._authorization_headers(session.prompt.request_id),
            ) as response:
                if response.is_error:
                    await response.aread()
                    raise RuntimeError(f"D completions failed: {response.text}")
                saw_done = False
                async for line in response.aiter_lines():
                    if not line.startswith("data:"):
                        continue
                    value = line[5:].strip()
                    if value == "[DONE]":
                        saw_done = True
                        break
                    payload = json.loads(value)
                    if "error" in payload:
                        raise RuntimeError(f"D completions error: {payload['error']}")
                    if (
                        _has_token(payload)
                        and metrics["d_request_to_first_token_ms"] is None
                    ):
                        now = time.perf_counter()
                        metrics["p_finish_to_d_first_token_ms"] = (
                            now - p_finish_time
                        ) * 1000
                        metrics["d_request_to_first_token_ms"] = (
                            now - d_request_time
                        ) * 1000
                    payload["id"] = f"cmpl-{session.prompt.request_id}"
                    payload["token_source"] = "decode"
                    usage = payload.get("usage")
                    if isinstance(usage, dict):
                        usage["prompt_tokens"] = max(0, usage["prompt_tokens"] - 1)
                        usage["completion_tokens"] += 1
                    yield payload
                if not saw_done:
                    raise RuntimeError("D stream closed without [DONE].")
        # WRITE handle 的实际计时来自 P，不把控制轮询耗时冒充 KV 传输时间。
        async with asyncio.timeout(application.state.args.finalize_timeout):
            while True:
                producer_status = await _control(
                    application.state.prefiller, "status", session.session_id
                )
                network_metrics = producer_status.get("metrics") or {}
                completed_blocks = network_metrics.get(
                    "early_transferred_blocks", 0
                ) + network_metrics.get("final_transferred_blocks", 0)
                expected_blocks = prefill["kv_transfer_params"][
                    "prompt_stream_remote_block_counts"
                ][0]
                # 很短的 decode 可能早于 P 主线程回收 WRITE 统计；在首 D token
                # 计时完成后补齐统计，不改变 handoff 指标的计量截止点。
                if finish_without_decode or completed_blocks == expected_blocks:
                    break
                if producer_status.get("state") in ("failed", "aborted"):
                    raise RuntimeError(
                        f"P transfer accounting failed: {producer_status}"
                    )
                await asyncio.sleep(application.state.args.status_poll_interval)
        metrics.update(producer_status.get("metrics") or {})
        metrics["early_stable_kv_push"] = session.early_push
        yield {
            "id": f"cmpl-{session.prompt.request_id}",
            "choices": [],
            "stable_kv_metrics": metrics,
            "prompt_stream_stats": prefill.get("prompt_stream_stats"),
        }
    finally:
        await asyncio.shield(_cleanup(application, session))


async def _sse(
    events: AsyncIterator[dict[str, Any]],
) -> AsyncIterator[bytes]:
    """把响应事件编码为 SSE，headers 发出后的错误必须显式传给客户端。"""
    try:
        async for payload in events:
            yield common._encode_sse_payload(payload)
    except (
        HTTPException,
        httpx.HTTPError,
        RuntimeError,
        TimeoutError,
        ValueError,
    ) as exc:
        logger.exception("Stable KV finish failed")
        yield common._encode_sse_payload({"error": {"message": str(exc)}})
    finally:
        await events.aclose()
    yield b"data: [DONE]\n\n"


async def _json_response(
    events: AsyncIterator[dict[str, Any]],
    session: StableProxySession,
    prefill: dict[str, Any],
) -> JSONResponse:
    """汇合 P/D 文本与 token IDs，同时保留从上游 SSE 测到的 D 首 token。"""
    result = common._build_prefill_only_response(session.prompt, prefill)
    choice = result["choices"][0]
    choice["text"] = ""
    choice["token_ids"] = []
    try:
        async for payload in events:
            if "stable_kv_metrics" in payload:
                result["stable_kv_metrics"] = payload["stable_kv_metrics"]
            if payload.get("usage"):
                result["usage"] = payload["usage"]
            for delta in payload.get("choices", []):
                choice["text"] += delta.get("text") or ""
                choice["token_ids"].extend(delta.get("token_ids") or [])
                if delta.get("finish_reason") is not None:
                    choice["finish_reason"] = delta["finish_reason"]
                    choice["stop_reason"] = delta.get("stop_reason")
    except (httpx.HTTPError, RuntimeError, TimeoutError, ValueError) as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    finally:
        await events.aclose()
    return JSONResponse(content=result)


@app.post("/v1/prompt_stream/finish", response_model=None)
async def finish(request: Request) -> JSONResponse | StreamingResponse:
    """提交 Question，并在最终 manifest 验证后让 D 收养目标 blocks。"""
    body = await request.json()
    session = _find_session(request, body)
    async with session.lock:
        if session.finishing:
            raise HTTPException(status_code=409, detail="finish is not retryable.")
        session.finishing = True
        try:
            prefill = await common._post_json(
                session.prompt.prefill_client_info,
                "/prompt_stream/finish",
                body,
                session.prompt.request_id,
            )
            p_finish_time = time.perf_counter()
            common._finish_decode_prompt_token_ids(session.prompt, prefill)
            count = common._append_prefill_output_token_ids(session.prompt, prefill)
            if count != 1:
                raise HTTPException(status_code=502, detail="P must return one token.")
            common._validate_prompt_stream_transfer(session.prompt, prefill, count)
            decode_request = dict(session.prompt.base_request)
            finish_without_decode = common._should_finish_without_decode(
                decode_request, prefill, count
            )
            common._shrink_decode_max_tokens(decode_request, count)
            common._apply_decode_prompt_to_request(session.prompt, decode_request)
        except BaseException:
            await asyncio.shield(_cleanup(request.app, session))
            raise
    events = _response_events(
        request.app,
        session,
        prefill,
        decode_request,
        p_finish_time,
        finish_without_decode,
    )
    if session.prompt.base_request.get("stream", False):
        return StreamingResponse(
            _sse(events),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )
    return await _json_response(events, session, prefill)


@app.post("/v1/prompt_stream/abort")
async def abort(request: Request) -> JSONResponse:
    """幂等清理双侧会话；D 已收养时由引擎中止对应普通 Request。"""
    body = await request.json()
    request_id = body.get("request_id")
    session = request.app.state.sessions.get(request_id)
    if session is not None:
        await _cleanup(request.app, session)
    return JSONResponse(content={"request_id": request_id, "status": "aborted"})


def main() -> None:
    """启动独立代理；请在 g2c14 运行，P/D 地址指向分配的 g2c9。"""
    import uvicorn

    args = parse_args()
    app.state.args = args
    logging.basicConfig(level=logging.INFO)
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
