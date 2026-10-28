# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""验证普通 PD 的块边界、生成预算与独立 P 完成通知。"""

import argparse
import asyncio
import json
import uuid
from pathlib import Path

import httpx

from research.scripts.replay.common.transport import iter_sse_payloads


async def run(args: argparse.Namespace) -> list[dict]:
    """比较 PD 与直接 D 的确定性输出，覆盖两种响应形式和单 token 预算。"""
    records = []
    run_id = uuid.uuid4().hex  # 隔离各次验证，避免本地 prefix hit 掩盖实际 KV 交接。
    async with httpx.AsyncClient(timeout=300, trust_env=False) as client:
        for length in (31, 32, 33):
            for budget in (1, 4):
                # 普通词元形成可复现的输入；长度直接控制在 KV 页边界两侧。
                payload = {
                    "model": args.model,
                    "prompt": [100] * length,
                    "add_special_tokens": False,
                    "max_tokens": budget,
                    "temperature": 0,
                    "seed": 2026,
                    "ignore_eos": True,
                    "return_token_ids": True,
                    "stream": False,
                    "cache_salt": f"{run_id}-baseline-{length}-{budget}",
                }
                response = await client.post(
                    f"{args.decode_url}/v1/completions", json=payload
                )
                response.raise_for_status()
                expected = response.json()["choices"][0]["token_ids"]
                for stream in (False, True):
                    payload["stream"] = stream
                    payload["cache_salt"] = f"{run_id}-pd-{length}-{budget}-{stream}"
                    if stream:
                        async with client.stream(
                            "POST", f"{args.proxy_url}/v1/completions", json=payload
                        ) as response:
                            response.raise_for_status()
                            frames = [
                                frame async for frame in iter_sse_payloads(response)
                            ]
                        assert frames[0]["event"] == "prefill_ready"
                        assert frames[0]["choices"] == []
                        choices = [
                            choice
                            for frame in frames
                            for choice in frame.get("choices", [])
                        ]
                        actual = [
                            token
                            for choice in choices
                            for token in choice.get("token_ids", [])
                        ]
                        assert choices[-1]["finish_reason"] == "length"
                    else:
                        response = await client.post(
                            f"{args.proxy_url}/v1/completions", json=payload
                        )
                        response.raise_for_status()
                        result = response.json()
                        actual = result["choices"][0]["token_ids"]
                        assert result["usage"]["prompt_tokens"] == length
                        assert result["usage"]["completion_tokens"] == budget
                    assert len(actual) == budget and actual == expected, (
                        length,
                        budget,
                        actual,
                        expected,
                    )
                    records.append(
                        {
                            "prompt_tokens": length,
                            "max_tokens": budget,
                            "stream": stream,
                            "status": "ok",
                        }
                    )
        # P-only 路由只返回控制事件，且不能把任何采样结果当作回答交付。
        response = await client.post(
            f"{args.proxy_url}/v1/completions_ttft", json=payload
        )
        response.raise_for_status()
        frames = [frame async for frame in iter_sse_payloads(response)]
        assert len(frames) == 1 and frames[0]["event"] == "prefill_ready"
        assert frames[0]["choices"] == []
        records.append({"mode": "p_only", "status": "ok"})
    return records


def main() -> None:
    """解析服务地址并把每个成功案例写入指定结果文件。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--proxy-url", default="http://127.0.0.1:18000")
    parser.add_argument("--decode-url", default="http://127.0.0.1:18200")
    parser.add_argument("--model", default="llama-3.1-8b-instruct")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    records = asyncio.run(run(args))
    args.output.write_text(json.dumps(records, ensure_ascii=False, indent=2) + "\n")
    print(f"PD handoff: {len(records)} checks passed; {args.output}")


if __name__ == "__main__":
    main()
