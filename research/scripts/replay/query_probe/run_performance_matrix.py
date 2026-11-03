# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""在一个 Slurm GPU 配额内启动 P/proxy，依次测量 dense/sparse 并发矩阵。"""

import argparse
import hashlib
import json
import math
import os
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

from research.scripts.replay.query_probe.summarize_performance import compare

REPO = Path(__file__).resolve().parents[4]


def wait_ready(process: subprocess.Popen, url: str) -> None:
    """等待服务启动；进程退出或启动超时则保留日志并停止实验。"""
    deadline = time.monotonic() + 900
    with httpx.Client(trust_env=False, timeout=5) as client:
        while time.monotonic() < deadline:
            if process.poll() is not None:
                raise RuntimeError(f"服务提前退出：{url}, exit={process.returncode}")
            try:
                if client.get(url).status_code == 200:
                    return
            except httpx.RequestError:
                pass
            time.sleep(2)
    raise RuntimeError(f"服务启动超过 900 秒：{url}")


def stop(process: subprocess.Popen) -> None:
    """结束本作业创建的进程组，包含 P 的模型 worker。"""
    if process.poll() is None:
        os.killpg(process.pid, signal.SIGTERM)
        try:
            process.wait(timeout=30)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait()


def snapshot(root: Path, args: argparse.Namespace) -> None:
    """记录硬件、版本和未提交源码，保证长时间运行后的结果可追溯。"""
    metadata = vars(args).copy()
    metadata.update(
        host=socket.gethostname(),
        python=sys.executable,
        attention_backend="CHUNK_SPARSE_ATTN",
    )
    metadata["slurm_job_id"] = os.environ.get("SLURM_JOB_ID")
    for key, command in (
        ("git_head", ["git", "rev-parse", "HEAD"]),
        (
            "gpu",
            [
                "nvidia-smi",
                "--query-gpu=name,uuid,memory.total,driver_version",
                "--format=csv,noheader",
            ],
        ),
    ):
        metadata[key] = subprocess.check_output(command, text=True).strip()
    paths = subprocess.check_output(
        ["git", "diff", "--name-only", "--", "vllm", "research/scripts"], text=True
    ).splitlines()
    paths += subprocess.check_output(
        [
            "git",
            "ls-files",
            "--others",
            "--exclude-standard",
            "vllm",
            "research/scripts/replay/query_probe",
            "research/scripts/replay/common",
        ],
        text=True,
    ).splitlines()
    fingerprints = {}
    for relative in sorted(set(paths)):
        source = REPO / relative
        if source.is_file():
            target = root / "source_snapshot" / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)
            fingerprints[relative] = hashlib.sha256(source.read_bytes()).hexdigest()
    metadata["source_sha256"] = fingerprints
    (root / "environment.json").write_text(json.dumps(metadata, default=str, indent=2))


def main() -> None:
    """依次运行配对实验，每对完成即更新对比图；结束后释放本作业资源。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--concurrency", type=int, nargs="+", default=[4, 8, 16])
    parser.add_argument(
        "--qps", type=float, nargs="+", help="开环 QPS 列表；省略时扫描闭环 concurrency"
    )
    parser.add_argument(
        "--max-num-seqs", type=int, default=32, help="P 节点调度上限，与客户端负载独立"
    )
    parser.add_argument(
        "--max-inflight", type=int, default=0, help="开环保护上限；0 不限"
    )
    parser.add_argument(
        "--arrival-process", choices=("fixed", "poisson"), default="fixed"
    )
    parser.add_argument("--max-requests", type=int, default=1000)
    parser.add_argument("--warmup-requests", type=int, default=16)
    parser.add_argument("--keep-ratio", type=float, default=0.5)
    parser.add_argument("--reverse-order", action="store_true")
    parser.add_argument(
        "--profile-iterations",
        type=int,
        default=0,
        help="单独诊断作业采集的 engine step 数；默认关闭",
    )
    parser.add_argument("--profile-delay", type=int, default=0)
    parser.add_argument("--p-port", type=int, default=0)
    parser.add_argument("--proxy-port", type=int, default=0)
    parser.add_argument("--model-path", default=str(Path.home() / "Models/Qwen2.5-3B"))
    parser.add_argument(
        "--config", type=Path, default=Path(__file__).with_name("performance.json")
    )
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    if (
        not args.concurrency
        or min(args.concurrency) < 1
        or args.max_requests < 1
        or args.max_num_seqs < 1
    ):
        parser.error("并发度和请求数必须为正数")
    if args.max_inflight < 0 or (args.qps is None and args.max_inflight):
        parser.error("max-inflight 仅用于 QPS 扫描，且必须非负")
    if args.qps is not None and any(
        not math.isfinite(qps) or qps <= 0 for qps in args.qps
    ):
        parser.error("QPS 扫描值必须有限且大于零")
    os.chdir(REPO)
    # 同节点可有多个 GPU 作业；先同时保留两个空闲端口，不能连接到别人的服务。
    with socket.socket() as p_socket, socket.socket() as proxy_socket:
        p_socket.bind(("127.0.0.1", args.p_port))
        proxy_socket.bind(("127.0.0.1", args.proxy_port))
        args.p_port = p_socket.getsockname()[1]
        args.proxy_port = proxy_socket.getsockname()[1]
    suffix = os.environ.get("SLURM_JOB_ID", time.strftime("%Y%m%d-%H%M%S"))
    root = (
        args.output_dir or Path(__file__).parent / "results" / f"performance-{suffix}"
    )
    root = root.resolve()
    root.mkdir(parents=True, exist_ok=False)
    snapshot(root, args)
    shutil.copyfile(args.config, root / "performance_config.json")
    # 缓存和子进程临时文件限定在仓库 tmp，全部模式共用这份缓存。
    temporary_root = REPO / "tmp"
    temporary_root.mkdir(exist_ok=True)
    cache = Path(tempfile.mkdtemp(prefix=f"query-probe-{suffix}-", dir=temporary_root))
    env = dict(os.environ)
    env.update(
        # Unix IPC 路径上限为 107 字节，不能把 UUID socket 放进过长的缓存子目录。
        TMPDIR=str(temporary_root),
        PYTHON_BIN=sys.executable,
        MODEL_PATH=args.model_path,
        PORT=str(args.p_port),
        ATTENTION_BACKEND="CHUNK_SPARSE_ATTN",
        MAX_NUM_SEQS=str(args.max_num_seqs),
        MAX_NUM_BATCHED_TOKENS="2048",
        MAX_MODEL_LEN="16384",
        GPU_MEMORY_UTILIZATION="0.8",
        VLLM_USE_FLASHINFER_SAMPLER="0",
        VLLM_USE_V2_MODEL_RUNNER="0",
        TRITON_CACHE_DIR=str(cache / "triton"),
        TORCHINDUCTOR_CACHE_DIR=str(cache / "inductor"),
        NO_PROXY="127.0.0.1,localhost",
        no_proxy="127.0.0.1,localhost",
    )
    env.pop("PROMPT_STREAM_UPDATE_COALESCING_CONFIG", None)
    env.pop("PROFILER_CONFIG", None)
    if args.profile_iterations:
        env["PROFILER_CONFIG"] = json.dumps(
            {
                "profiler": "torch",
                "torch_profiler_dir": str(root / "profiles"),
                "torch_profiler_with_stack": False,
                "ignore_frontend": True,
                "max_iterations": args.profile_iterations,
                "delay_iterations": args.profile_delay,
            }
        )
    processes: list[subprocess.Popen] = []
    statuses = []

    def interrupted(signum: int, frame: object) -> None:
        """让 Slurm TERM 也进入 finally，回收服务与临时缓存。"""
        raise SystemExit(128 + signum)

    signal.signal(signal.SIGTERM, interrupted)
    try:
        with (
            (root / "p.log").open("w") as p_log,
            (root / "proxy.log").open("w") as proxy_log,
        ):
            p = subprocess.Popen(
                ["bash", "research/scripts/replay/query_probe/start_p.sbatch"],
                env=env,
                stdout=p_log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            processes.append(p)
            wait_ready(p, f"http://127.0.0.1:{args.p_port}/health")
            proxy = subprocess.Popen(
                [
                    sys.executable,
                    "research/scripts/server/proxy_server.py",
                    "--host",
                    "127.0.0.1",
                    "--port",
                    str(args.proxy_port),
                    "--prefiller-hosts",
                    "127.0.0.1",
                    "--prefiller-ports",
                    str(args.p_port),
                    "--decoder-hosts",
                    "127.0.0.1",
                    "--decoder-ports",
                    "1",
                ],
                env=env,
                stdout=proxy_log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            processes.append(proxy)
            wait_ready(proxy, f"http://127.0.0.1:{args.proxy_port}/openapi.json")
            # 正式 1000 请求前先跑两种模式的短检验，避免长作业晚发现接口错误。
            cases = [
                (f"preflight-{mode}", mode, 2, 2, 0, 0.0)
                for mode in ("dense", "sparse")
            ]
            settings = (
                [(f"qps{qps:g}", 1, qps) for qps in args.qps]
                if args.qps is not None
                else [
                    (f"c{concurrency}", concurrency, 0.0)
                    for concurrency in args.concurrency
                ]
            )
            for index, (label, concurrency, qps) in enumerate(settings):
                modes = ("dense", "sparse") if index % 2 == 0 else ("sparse", "dense")
                if args.reverse_order:
                    modes = tuple(reversed(modes))
                cases.extend(
                    (
                        f"{mode}-{label}",
                        mode,
                        concurrency,
                        args.max_requests,
                        args.warmup_requests,
                        qps,
                    )
                    for mode in modes
                )
            for name, mode, concurrency, count, warmup, qps in cases:
                print(f"Starting {name}: {count} requests", flush=True)
                command = [
                    sys.executable,
                    "research/scripts/replay/query_probe/replay.py",
                    "--config",
                    str(args.config),
                    "--mode",
                    mode,
                    "--load-mode",
                    "open_loop" if qps else "closed_loop",
                    "--qps",
                    str(qps),
                    "--max-inflight",
                    str(args.max_inflight if qps else 0),
                    "--arrival-process",
                    args.arrival_process,
                    "--max-concurrency",
                    str(concurrency),
                    "--max-requests",
                    str(count),
                    "--warmup-requests",
                    str(warmup),
                    "--keep-ratio",
                    str(args.keep_ratio),
                    "--tokenizer-model",
                    args.model_path,
                    "--proxy-url",
                    f"http://127.0.0.1:{args.proxy_port}",
                    "--output-dir",
                    str(root),
                    "--run-name",
                    name,
                ]
                # 诊断与正式性能作业分别运行，避免把 profiler 开销计为真实吞吐。
                profile_case = bool(args.profile_iterations) and not name.startswith(
                    "preflight"
                )
                if profile_case:
                    profiles = root / "profiles"
                    previous_traces = set(profiles.rglob("*.pt.trace.json*"))
                    with httpx.Client(trust_env=False, timeout=180) as client:
                        client.post(
                            f"http://127.0.0.1:{args.p_port}/start_profile"
                        ).raise_for_status()
                with (root / f"{name}.log").open("w") as log:
                    replay = subprocess.Popen(
                        command,
                        env=env,
                        stdout=log,
                        stderr=subprocess.STDOUT,
                        start_new_session=True,
                    )
                    processes.append(replay)
                    code = replay.wait()
                if profile_case:
                    with httpx.Client(trust_env=False, timeout=180) as client:
                        client.post(
                            f"http://127.0.0.1:{args.p_port}/stop_profile"
                        ).raise_for_status()
                    # worker 文件名包含时间戳，用清单记录归属的测试组。
                    current_traces = set(profiles.rglob("*.pt.trace.json*"))
                    traces = [
                        str(path.relative_to(root))
                        for path in current_traces - previous_traces
                    ]
                    # worker 默认覆盖文本汇总，因此每组结束立即保存独立副本。
                    for timing in profiles.glob("profiler_out_*.txt"):
                        target = profiles / f"{name}-{timing.name}"
                        shutil.copyfile(timing, target)
                        traces.append(str(target.relative_to(root)))
                    traces.sort()
                    (root / f"{name}-profile-files.json").write_text(
                        json.dumps(traces, indent=2)
                    )
                statuses.append({"case": name, "exit_code": code})
                (root / "matrix_status.json").write_text(json.dumps(statuses, indent=2))
                compare(root)
                if code:
                    log_path = root / f"{name}.log"
                    raise RuntimeError(f"{name} 失败；已停止矩阵，见 {log_path}")
    finally:
        for process in reversed(processes):
            stop(process)
        variants = []
        for path in (cache / "triton").rglob("*.json"):
            try:
                entry = json.loads(path.read_text())
            except (OSError, ValueError):
                continue
            if entry.get("name") in (
                "_query_probe_attention_kernel",
                "_query_probe_score_kernel",
            ):
                variants.append(
                    {
                        "name": entry["name"],
                        "hash": entry.get("hash"),
                        "num_warps": entry.get("num_warps"),
                        "num_stages": entry.get("num_stages"),
                    }
                )
        (root / "jit_variants.json").write_text(json.dumps(variants, indent=2))
        shutil.rmtree(cache)
    print(f"Completed matrix: {root}", flush=True)


if __name__ == "__main__":
    main()
