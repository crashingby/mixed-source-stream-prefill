# 稳定 KV 提前传输的 nsys 排查

本目录保留 2026-10-04 的诊断方式与结论。**这次 trace 不支持把显式
`torch.cuda.synchronize()` 当作几百毫秒退化的主要原因**；更明显的现象是
大量细粒度 KV 拷贝与模型的 CPU 端 CUDA 提交相互重叠。

没有修改传输算法、删除同步、替换 backend 或改动正常生成逻辑。这里只增加
诊断包装器、短区间回放与只读 SQLite 分析。

## 采集设置

| 项目 | 实际设置 |
| --- | --- |
| P/D | g2c9，单个 Slurm allocation，两张 A30，TP=1 |
| Proxy/replay | g2c14 |
| 模型 | Qwen2.5-3B |
| 显存比例 | 0.90 |
| 历史实验注意力 | TRITON_ATTN，Masked Padding 开启，同步调度 |
| 数据 | 稳定知识占比 87.5%，Append=1792、Update=256 tokens |
| 来源间隔 | Append=0.1 s，Update=0.2 s，source finish delay=0 |
| 回放 | QPS=2，Poisson，seed=2026，并发上限 8，完整生成 64 tokens |
| 采集 | 仅 P 的进程树；nsys 2023.4.4；CUDA/NVTX，关闭 CPU sampling |
| 对照 | 每组 8 个请求，两轮；finish-only→early，再 early→finish-only |

预热分别用 8 条请求，不采集。四段正式诊断共 32 次请求，全部完整成功且
统计有效。第一轮还尝试过 64 请求、含 OS runtime 的采集，但报告生成过慢，
没有取得可用的完整报告；那一段不进入本节 CUDA 证据。

诊断目录为 `../results/nsys-light-20261004/`，四份可用报告对应如下：

| Capture | 模式 | 重复 | Nsight 报告 / SQLite |
| ---: | --- | ---: | --- |
| 1 | finish-only | 1 | `p-job-6390.1.nsys-rep` / `.sqlite` |
| 2 | early | 1 | `p-job-6390.2.nsys-rep` / `.sqlite` |
| 3 | early | 2 | `p-job-6390.3.nsys-rep` / `.sqlite` |
| 4 | finish-only | 2 | `p-job-6390.4.nsys-rep` / `.sqlite` |

每段的派生证据在 `trace-<N>-summary.json`。四段 raw `qdstrm` 也已保留，
可重新转换，不依赖仍然运行的 GPU 服务。采集结束后 P/D 与 proxy 已释放。

## 实测发现

### 显式 DeviceSynchronize 不是本轮的主要耗时

这是 CPU 端 API 调用耗时，不是 GPU kernel 时长。每段均包含 8 个请求：

| Capture | 模式 | 调用次数 | 累计耗时 | 最长一次 |
| ---: | --- | ---: | ---: | ---: |
| 1 | finish-only | 8 | 0.115 ms | 0.0209 ms |
| 2 | early | 17 | 0.179 ms | 0.0201 ms |
| 3 | early | 19 | 0.188 ms | 0.0173 ms |
| 4 | finish-only | 7 | 0.089 ms | 0.0243 ms |

提前发送增加了同步机会，但观察到的调用只需约 10–20 微秒，无法解释此前
客户端 P 首 token P95 增加几百毫秒。源码中的同步仍有正确性作用，不能
因为名字含 Synchronize 就认定它是瓶颈或直接删掉。

这不表示所有等待都很小：主执行线程的 `cudaEventSynchronize` 累计耗时在
finish-only 两段为 789/809 ms，early 两段为 598/587 ms。它包含模型执行的
完成等待，不能直接归因于传输；提前组在这项统计中也没有增加。本文排除的
只是显式 DeviceSynchronize 作为主要退化来源的猜测。

### KV 搬运呈现大量 8 KiB 的 GPU→CPU 拷贝

四段都观察到 **76,608 次 Device-to-Host 拷贝，每次 8192 bytes**：

- 合计 627,572,736 bytes，即 598.5 MiB，正好等于 8 条请求的全部 KV
  传输量；每请求 74.8125 MiB、9576 次这样的拷贝。
- 两种策略总量相同，提前 Push 没有重复传输全部 KV；改变的是搬运时机。
- 提前组在 P finish 前已完成约 85.8% 的传输，因此把大量搬运放进了仍有
  prefill 工作的阶段。
- P 日志明确显示 NIXL 使用 UCX，且 `kv_buffer_device=cuda`、
  `use_host_buffer=False`。缓存池仍位于 GPU；后者不等于底层传输一定没有
  host staging。trace 证明了当前链路实际发生 D2H，但没有证明底层为什么
  选择这个路径，也不能据此断言其它机器都采用相同路径。

这些小拷贝通过多个后台线程发起。GPU 上 D2H 的累计执行时间约为
140–143 ms，但多个 copy 线程的 CPU API 累计时间远大于这个值。
不能把不同线程的累计时长相加后当作请求的墙钟等待。

### 长时间模型提交与拷贝在时间线上重叠

筛选主执行线程上耗时至少 5 ms 的 `cudaLaunchKernel` / `cuLaunchKernelEx`：

| Capture | 模式 | 慢提交次数 | 其中与 copy API 重叠 | 最长一次 |
| ---: | --- | ---: | ---: | ---: |
| 1 | finish-only | 7 | 7 | 301.0 ms |
| 2 | early | 17 | 17 | 240.6 ms |
| 3 | early | 22 | 22 | 203.6 ms |
| 4 | finish-only | 8 | 8 | 269.0 ms |

两种模式都存在上述重叠：finish-only 也会在一个请求发送最终 KV 时遇到其它请求
仍在 prefill。early 使这些传输更早、更分散，两个重复中长提交的数量都更多。
例如 Capture 2 的最长 launch 调用持续 240.6 ms，其区间内同时存在
8640 个后台 copy API 调用。

**这里慢的是 CPU 提交调用，不意味着对应 kernel 在 GPU 上算了 240 ms。**
GPU kernel 累计执行时间分别约为 1685、1537、1537、1664 ms，提前组并没有
在这项统计里变得更长。不同策略改变了批处理时序与执行次数，因此这些累计值
也不是固定计算工作量下的算子性能对照。

## 当前结论与未确认部分

当前定位优先级应从“删除全设备同步”转向：**NIXL 搬运的细粒度 host staging、
CUDA API 提交路径的竞争，以及把这类搬运前移后对计算准入的影响**。

trace 确认了小拷贝、长提交和时间重叠，但还没有确认具体阻塞锁位于 CUDA
driver、UCX/NIXL、其它运行时层还是 CPU 调度。同步数据反驳了此前把显式
DeviceSynchronize 当成主要原因的猜测；copy/launch 重叠则是很强的相关证据，
还不是已经完成控制变量实验的因果证明。

profiling 本身会增加开销。这里的 8 请求窗口只用于定位，不作为正式 P90/P95
或原来 128 请求实验的替代，更不声称解释了正式实验里增加的每一毫秒。
后续可验证的方向是：

- 核对本机 UCX 的实际传输路径，确认是否能使用更合适的 CUDA IPC/RDMA
  路径；不预先假设切换一定可用或有收益。
- 在仍然保持 KV 地址与完成屏障正确的前提下，评估减少 descriptor/小拷贝
  数量，以及合并提前发布批次的可行性。
- 如果需要进一步定位内部锁，再做 CPU 调用栈采集或底层后端消融。
  本次没有为了定位而更改这些执行条件。

## 脚本与复查命令

| 文件 | 用途 |
| --- | --- |
| `profile_python.sh` | 作为启动器的 PYTHON_BIN；只包裹 P API 进程，D 和显存检查不进入 nsys。 |
| `run_capture.sh` | 预热，再用 HTTP start/stop profile 控制四段短 A/B。 |
| `summarize_trace.py` | 只读 SQLite，统计 API、GPU 活跃区间并集、copy 粒度和慢提交重叠。 |

从仓库根目录，在 g2c14 启动唯一一对服务。每次重跑换新的输出目录：

```shell
mkdir -p research/scripts/logs

NSYS_OUTPUT_DIR="${PWD}/research/scripts/replay/stable_kv_push/results/nsys-new" \
PYTHON_BIN="${PWD}/research/scripts/replay/stable_kv_push/nsys/profile_python.sh" \
REPO_ROOT="${PWD}" \
MODEL_PATH="${HOME}/Models/Qwen2.5-3B" \
SERVED_MODEL_NAME=qwen2.5-3b \
GPU_MEMORY_UTILIZATION=0.8 \
MAX_MODEL_LEN=4096 \
MAX_NUM_SEQS=64 \
PROMPT_STREAM_MASKED_PADDING=1 \
srun --job-name=stable-kv-pd --partition=gpu --nodelist=g2c9 \
  --nodes=1 --ntasks=2 --ntasks-per-node=2 \
  --gpus-per-task=1 --gpu-bind=single:1 \
  --cpus-per-task=8 --mem=24G --time=04:00:00 --kill-on-bad-exit=1 \
  --output="${PWD}/research/scripts/logs/stable-kv-pd-%j-rank-%t.out" \
  --error="${PWD}/research/scripts/logs/stable-kv-pd-%j-rank-%t.out" \
  bash -c '
    case "$SLURM_PROCID" in
      0) export ATTENTION_BACKEND=CHUNK_SPARSE_ATTN ;;
      1) export ATTENTION_BACKEND=TRITON_ATTN ;;
    esac
    exec bash "$REPO_ROOT/research/scripts/server/start_stable_kv_node.sh"
  '
```

与上层 README 相同，rank 0 的 P 使用自定义后端，rank 1 的 D 使用原 Triton
后端，只屏蔽 padding。按 rank 设置避免把 P 的后端配置同时应用于 D。

另一个 g2c14 终端启动专用 proxy，命令见上层 README。确认两端健康后：

```shell
OUTPUT_DIR=research/scripts/replay/stable_kv_push/results/nsys-new \
MAX_REQUESTS=8 \
QPS=2 \
bash research/scripts/replay/stable_kv_push/nsys/run_capture.sh
```

报告最终导出可能晚于 `/stop_profile` HTTP 返回；不要立刻把 stop 的导出开销
混进业务指标。保存完 raw/report 再结束自己的 P/D allocation。示例离线汇总：

```shell
.venv/bin/python \
  research/scripts/replay/stable_kv_push/nsys/summarize_trace.py \
  --sqlite research/scripts/replay/stable_kv_push/results/nsys-light-20261004/p-job-6390.2.sqlite \
  --output research/scripts/replay/stable_kv_push/results/nsys-light-20261004/trace-2-summary.json
```

图形界面可直接打开 `.nsys-rep`，查看主执行线程的长 launch 与 copy 线程。
无需重新启动服务或重新联网下载数据。

当前启动命令的 P 使用 `CHUNK_SPARSE_ATTN`，D 使用 `TRITON_ATTN`。上面的历史采样结果两端均使用 `TRITON_ATTN`，重测后需分别保存和比较。
