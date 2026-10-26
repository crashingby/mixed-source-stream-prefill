# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""隔离测量分页 attention 与 Query 打分；支持 nsys 的 CUDA capture range。"""

import argparse
import hashlib
import importlib.util
import json
import os
import socket
import sys
import time
from dataclasses import replace
from functools import partial
from pathlib import Path
from types import ModuleType

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

from vllm.v1.attention.ops import query_probe_attention as optimized
from vllm.v1.attention.ops.triton_unified_attention import unified_attention
from vllm.v1.query_probe import QueryProbeWork


def load_reference(path: Path) -> ModuleType:
    """从保存的源码加载原始内核，避免切换整个工作区版本。"""
    spec = importlib.util.spec_from_file_location("probe_reference", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@torch.inference_mode()
def main() -> None:
    """按相同形状测量 GPU 时间、CPU 提交时间，以及首次调用耗时。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--torch-profile-dir", type=Path)
    parser.add_argument("--query-tile", type=int, choices=[16, 32, 64])
    parser.add_argument("--key-tile", type=int, choices=[32, 64, 128])
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--contexts", type=int, nargs="+", default=[1024, 4096, 8192])
    parser.add_argument("--query-length", type=int, default=512)
    parser.add_argument("--repeats", type=int, default=100)
    parser.add_argument("--warmup-iterations", type=int, default=100)
    args = parser.parse_args()
    if args.query_tile:
        optimized.QUERY_TILE = args.query_tile
    if args.key_tile:
        optimized.KEY_TILE = args.key_tile
    reference = load_reference(args.reference) if args.reference else None
    device = torch.device("cuda")
    torch.manual_seed(7)
    records = []
    for context in args.contexts:
        for phase in ("update", "joint"):
            count, batch = args.query_length, args.batch
            end = context + count
            pages = (end + 15) // 16
            query = torch.randn(
                batch * count, 16, 128, device=device, dtype=torch.bfloat16
            )
            cache = torch.randn(
                batch * pages, 2, 16, 2, 128, device=device, dtype=query.dtype
            )
            key, value = cache.unbind(1)
            table = torch.randperm(
                batch * pages, device=device, dtype=torch.int32
            ).view(batch, pages)
            output = torch.empty_like(query)
            offsets = [i * count for i in range(batch + 1)]
            req_ids = [str(i) for i in range(batch)]
            probe_start = end - 32 if phase == "joint" else -1
            prefix = probe_start - 5 if phase == "joint" else context - 5
            padding = (prefix, probe_start if phase == "joint" else context)
            candidates = (
                list(range(context // 16, prefix // 16)) if phase == "joint" else []
            )
            dropped = list(range(2, context // 16 - 1, 2))
            works = {
                req_id: QueryProbeWork(
                    1, context, count, prefix, probe_start, candidates, dropped, padding
                )
                for req_id in req_ids
            }
            metadata = optimized.QueryProbeAttention.build(
                works, req_ids, offsets, "last", device
            )
            # 全保留消融使用同一内核；Update 不伪装成 probe，也不计算评分。
            dense_works = {
                req_id: replace(
                    work,
                    dropped_blocks=[],
                )
                for req_id, work in works.items()
            }
            dense_metadata = optimized.QueryProbeAttention.build(
                dense_works, req_ids, offsets, "last", device
            )
            starts = torch.tensor(offsets, device=device, dtype=torch.int32)
            lengths = torch.full((batch,), end, device=device, dtype=torch.int32)
            pad = torch.tensor([padding] * batch, device=device, dtype=torch.int32)
            keep = torch.ones((batch, pages), device=device, dtype=torch.bool)
            keep[:, dropped] = False
            if reference:
                original_metadata = reference.QueryProbeAttention(
                    works, req_ids, offsets, keep, "last"
                )

            dense = partial(
                unified_attention,
                q=query,
                k=key,
                v=value,
                out=output,
                cu_seqlens_q=starts,
                max_seqlen_q=count,
                seqused_k=lengths,
                max_seqlen_k=end,
                softmax_scale=128**-0.5,
                causal=True,
                window_size=(-1, -1),
                block_table=table,
                softcap=0,
                q_descale=None,
                k_descale=None,
                v_descale=None,
                prompt_stream_padding_ranges=pad,
            )
            sparse = partial(
                optimized.query_probe_attention,
                query,
                key,
                value,
                output,
                table,
                metadata,
                128**-0.5,
                "last",
            )
            compact_dense = partial(
                optimized.query_probe_attention,
                query,
                key,
                value,
                output,
                table,
                dense_metadata,
                128**-0.5,
                "middle" if phase == "update" else "last",
            )
            cases = [
                ("native_dense", dense),
                ("optimized_dense", compact_dense),
                ("optimized_sparse", sparse),
            ]
            if reference:
                cases.append(
                    (
                        "original_sparse",
                        partial(
                            reference.query_probe_attention,
                            query,
                            key,
                            value,
                            output,
                            table,
                            original_metadata,
                            128**-0.5,
                            "last",
                        ),
                    )
                )
            for name, call in cases:
                torch.accelerator.synchronize()
                before = time.perf_counter()
                call()
                torch.accelerator.synchronize()
                cold_ms = (time.perf_counter() - before) * 1000
                for _ in range(args.warmup_iterations):
                    call()
                torch.accelerator.synchronize()
                if args.profile:
                    torch.cuda.profiler.start()
                begin, finish = (
                    torch.cuda.Event(enable_timing=True),
                    torch.cuda.Event(enable_timing=True),
                )
                torch.cuda.nvtx.range_push(
                    f"{name}/{phase}/context{context}/batch{batch}"
                )
                begin.record()
                before = time.perf_counter()
                for _ in range(args.repeats):
                    call()
                cpu_ms = (time.perf_counter() - before) * 1000 / args.repeats
                finish.record()
                finish.synchronize()
                elapsed_ms = begin.elapsed_time(finish) / args.repeats
                torch.cuda.nvtx.range_pop()
                if args.profile:
                    torch.cuda.profiler.stop()
                # GPU 事件区间可能包含 CPU 启动空隙；nsys 可进一步分解内核时长。
                record = dict(
                    case=name,
                    phase=phase,
                    context=context,
                    batch=batch,
                    query_tile=optimized.QUERY_TILE,
                    key_tile=optimized.KEY_TILE,
                    query_length=count,
                    cold_ms=cold_ms,
                    warmup_iterations=args.warmup_iterations,
                    repeats=args.repeats,
                    stream_ms=elapsed_ms,
                    cpu_submit_ms=cpu_ms,
                )
                print(json.dumps(record), flush=True)
                records.append(record)
                if args.torch_profile_dir:
                    args.torch_profile_dir.mkdir(parents=True, exist_ok=True)
                    stem = args.torch_profile_dir / f"{name}-{phase}-{context}"
                    with torch.profiler.profile(
                        activities=[
                            torch.profiler.ProfilerActivity.CPU,
                            torch.profiler.ProfilerActivity.CUDA,
                        ],
                    ) as profiler:
                        with torch.profiler.record_function(name):
                            for _ in range(5):
                                call()
                        torch.accelerator.synchronize()
                    profiler.export_chrome_trace(str(stem.with_suffix(".json")))
                    stem.with_suffix(".txt").write_text(
                        profiler.key_averages().table(
                            sort_by="self_device_time_total", row_limit=35
                        )
                    )
            if phase == "joint":
                # 原版每层评分已同步，优化版在模型结束时同步；单独记录归约开销。
                before = time.perf_counter()
                for _ in range(args.repeats):
                    metadata.collect_results()
                records.append(
                    dict(
                        case="score_collect",
                        context=context,
                        wall_ms=(time.perf_counter() - before) * 1000 / args.repeats,
                    )
                )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(records, indent=2))
    sources = [Path(__file__), Path(optimized.__file__)]
    if args.reference:
        sources.append(args.reference)
    environment = {
        "host": socket.gethostname(),
        "job_id": os.environ.get("SLURM_JOB_ID"),
        "gpu": torch.cuda.get_device_name(device),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "args": vars(args),
        "source_sha256": {
            str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in sources
        },
    }
    args.output.with_suffix(".environment.json").write_text(
        json.dumps(environment, default=str, indent=2)
    )


if __name__ == "__main__":
    main()
