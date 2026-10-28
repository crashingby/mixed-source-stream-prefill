# Query probe 功能回放

当前 P/D 交接由 D 生成全部回答；P-only 使用 `prefill_ready` 控制事件测量 P 侧 TTFT。协议、兼容字段与验证命令见 [P/D 交接说明](../../server/PD-Handoff.md)。

本目录验证新 Append 与临时 Query 的联合 prefill。Append 行按已发布策略稀疏读取历史块，Query 行完整因果读取，并在最后一个 attention 层统计新 Append 的块分数。评分完成后保留正式 KV、释放临时 Query 页。

默认从 `research/data/append_chunks/triviaqa/web_unfiltered_dev_v1` 和 `research/data/pipeline_pools/triviaqa/wiki_dev_k5_ef8_v1` 按 sample ID 加载三条请求。问题清单使用公共 replay 的 TriviaQA 默认路径。输入均传原文，由 P 节点分词。

两种模式固定走同一个新版 32×128 attention kernel。dense 使用全部逻辑 KV 页，sparse 使用保留页；最终 Query / decode 使用全部页。`TRITON_ATTN` 保留原 `unified_attention`，且会拒绝 probe 请求。当前后端支持 CUDA SM80+、FP16/BF16、auto KV dtype、16-token KV block、普通因果注意力、TP=1、eager 和同步调度。

## 启动

在仓库根目录提交 P 节点任务，脚本固定使用 `g2c10`、Qwen2.5-3B、单 GPU 和 `CHUNK_SPARSE_ATTN` 后端：

```shell
mkdir -p research/scripts/replay/query_probe/results
sbatch research/scripts/replay/query_probe/start_p.sbatch
```

P 默认监听 `8110`，上下文上限为 16384，调度 token 预算为 1024，显存比例为 `0.8`。可通过环境变量调整 `MODEL_PATH`、`PYTHON_BIN`、`PORT`、`MAX_MODEL_LEN`、`MAX_NUM_BATCHED_TOKENS`、`MAX_NUM_SEQS` 和 `GPU_MEMORY_UTILIZATION`。模型 forward 次数取决于调度预算。`PYTHON_BIN` 必须在计算节点上可执行；若本地 `.venv/bin/python` 指向仅本机存在的 `/opt` 路径，请指定计算节点可用、已安装当前仓库依赖的环境。

在 `g2c14` 启动专用 proxy：

```shell
NO_PROXY=127.0.0.1,localhost,g2c10,g2c14 \
no_proxy=127.0.0.1,localhost,g2c10,g2c14 \
.venv/bin/python research/scripts/server/proxy_server.py \
  --host 127.0.0.1 --port 8001 \
  --prefiller-hosts g2c10 --prefiller-ports 8110 \
  --decoder-hosts g2c10 --decoder-ports 8200
```

proxy 的启动参数仍需声明 decoder 地址；本 replay 只调用 `start_ttft` 和 `finish_ttft`，不会连接 D 节点，也不启用 KV transfer。

```shell
.venv/bin/python research/scripts/replay/query_probe/replay.py \
  --config research/scripts/replay/query_probe/config.json
```

## 配置与结果

JSON 配置和 CLI 使用相同参数，CLI 优先。`keep_ratio=1` 验证全保留 probe，`keep_ratio=0.5` 验证固定比例选择。`append_chunk_interval`、`update_chunk_interval` 和各自的 initial delay 控制事件时间；`qps=0` 与 `max_concurrency=1` 默认逐请求回放。

可使用 `--questions-file`、`--append-dir`、`--pipeline-pools-dir` 指定其他版本，或以 `--sample-id` / `--query-id` 选择请求。`--dry-run` 只验证数据加载和事件计划。

结果默认保存在本目录 `results/run-*`，包含实际配置、逐请求 `requests.jsonl` 和汇总。功能验证检查以下内容：

- 每个非空 Append 都有已发布策略，候选仅包含完整独占 KV 块。
- 保留数量符合固定比例，保留集合符合原始 Query 分数的稳定 Top-K。
- 原始分数和长度校正分数均有限且非负。
- 临时 Query 页释放数量覆盖全部 Append 评分。

原始 `prompt_stream_stats.query_probe` 还保留每个 chunk 的逻辑范围、块分数、保留集合以及实际 forward 数。这些记录用于功能排错，当前不据此宣称性能或回答质量收益。

## API 和实现范围

在 start 请求中传 `probe_query` 和可选的 `probe_keep_ratio`。Query 必须与最终 finish 的完整问题后缀一致；两者分词结果不一致时，finish 会被拒绝。未提供 Query 的请求沿用原流程。

当前接入要求 eager、TP=1、masked padding 和普通 full-attention KV。最终 Query 与 decode 完整读取正式 KV。私有 probe 页不会进入正式 P/D 导出清单；本次功能验证不运行 D 节点。

CPU 状态和 GPU 算子检查分别运行：

```shell
.venv/bin/python -m pytest tests/v1/test_query_probe.py -q
srun --partition=gpu --nodelist=g2c10 --nodes=1 --ntasks=1 --gres=gpu:1 \
  --cpus-per-task=8 --mem=32G --time=00:20:00 \
  .venv/bin/python -m pytest tests/kernels/attention/test_query_probe_attention.py -q
```

## 本次功能验证

2026-10-22 使用 Qwen2.5-3B、A30 完成功能验证。P 和专用 proxy 在获准的 `g2c14` 运行；交付的 Slurm 脚本仍默认 `g2c10`，未启动 D。

| 检查 | 结果 |
| --- | --- |
| 请求、调度、masked padding 与 proxy 回归 | 108 项通过 |
| FP16/BF16、NHD/HND、GQA 与混合行 GPU 数值对照 | 32 项通过 |
| 单请求 TriviaQA，保留比例 0.5 | 10 个 Append 均联合执行并发布，20 个临时页释放 |
| 3 条 TriviaQA，最大并发 2 | 全部通过 |
| 128-token 预算，全保留，固定采样种子 | 每个 Append 拆为 5 批，全部完成 |
| 同一 TriviaQA 加长问题后缀的边界测试 | Query 为 243 token，跨 2–3 批归约，160 个临时页释放 |
| finish 不匹配、会话中止、普通生成共存 | 分别返回 400、200、200，符合预期 |
| 全部改动的 pre-commit | 通过 |

加长后缀只存在于测试请求中，没有修改数据集。以上结果验证执行和生命周期正确性，未评价回答质量或性能收益。`query_forward_count` 可用于检查 Query 是否实际跨批；`joint_forward_count` 记录同批包含真实 Append 与 Query 的次数。

## 并发性能矩阵

`run_performance.sbatch` 在一份 GPU 配额内启动 P、proxy 和回放客户端，依次比较两种配置。两边均使用分段 masked padding，并在每个响应中校验 Update 合并关闭。

| 模式 | 行为 |
| --- | --- |
| `dense` | 分段 prefill 完整读取历史 KV，不注册 probe Query |
| `sparse` | 注册 Query，每次 Append 执行 probe，默认保留 50% 可选择块 |

`performance.json` 默认选择相同的前 1000 条可回放 TriviaQA，Append 间隔 50 ms、Update 间隔 20 ms。矩阵默认并发度为 4、8、16，共 6 组正式测试、6000 次请求。每组先用前 16 条样本预热，预热不进入统计。正式矩阵前还有两种模式各 2 条请求的接口预检。

负载采用 **closed-loop（闭环）**：保持至多指定数量的在途请求，一个完成后立即补下一个，不固定 QPS。对比时应同时看 TTFT 和完成吞吐。不同并发度的两种模式交替先后顺序，全部串行使用同一张 GPU。

P 固定使用 Qwen2.5-3B、TP=1、eager、`CHUNK_SPARSE_ATTN`、16384 上下文、2048 单轮 token 预算、至少 32 个调度槽位及 0.8 显存利用率。整个矩阵不启动 D，不启用 KV transfer。仓库 `tmp/` 下的临时编译缓存由所有测试组共用，退出时自动清理；日志、源码快照和结果保存在本目录 `results/performance-<job_id>/`。

从仓库根目录提交，节点名按当前空闲资源选择：

```shell
mkdir -p research/scripts/replay/query_probe/results
sbatch --nodelist=g2c12 --time=7-00:00:00 \
  research/scripts/replay/query_probe/run_performance.sbatch
```

当前 `gpu` 分区默认时限为 24 小时，硬上限为 7 天，无法申请真正无限时长。脚本本身不设置额外时限；示例提交参数使用分区上限。计算节点需能够执行项目 `.venv/bin/python`，也可通过 `PYTHON_BIN` 指定兼容环境。

矩阵参数可以随提交命令传入，例如 `--concurrency 4 8 16 --max-requests 1000 --keep-ratio 0.5`。其他事件时间和 HTTP 超时放在 `performance.json`；也可用 `--config` 指定另一份配置。

### 时间口径与结果

- **整体 TTFT：** `request_to_first_token_ms`，从 start 请求提交前到 P 完成通知到达客户端，包含来源事件等待、HTTP 和服务端排队、probe、评分及正式 prefill。
- **finish TTFT：** `finish_to_first_token_ms`，从最后的 finish POST 提交到同一个完成通知，便于与已有 replay 对比。
- `source_dispatch_ms`、事件实际迟发量和 `start_ack_ms` 用于判断客户端发送是否受到服务端阻塞影响。
- 每组报告均值、P50、P95、P99、成功数、失败数及完成吞吐。正式结果即时写入 `requests.jsonl`，`requests_summary.json` 提供运行中的进度；完成后写 `summary.json`。
- `comparison/` 生成配对 CSV、JSON 和 `ttft_comparison.png`。两边样本集合、最终正文和完整事件计划的 SHA256 必须一致；只对双方都成功的请求计算配对指标，并保留失败计数。`mean_speedup` 为 dense 平均延迟除以 sparse 平均延迟，大于 1 表示 sparse 更快。

当前矩阵强制 `ATTENTION_BACKEND=CHUNK_SPARSE_ATTN`，并在 `environment.json` 记录后端。两边都使用新版算子；比较包含 probe、评分和稀疏元数据的全部成本。请求长度和边界已改为运行时元数据，不再逐长度编译稀疏 kernel；dtype、布局和内核配置仍可能产生编译变体，`jit_variants.json` 记录本作业的稀疏内核变体。正式测量期间的编译也计入延迟。结果反映系统运行表现，不能直接解读为纯 attention kernel 的加速比。

P 与 proxy 默认自动选择本节点空闲端口，记录到 `environment.json`，避免同节点不同 GPU 作业互相连错服务。可以用 `--p-port` 和 `--proxy-port` 显式指定；已被占用的端口会在启动前报错。`TMPDIR` 指向仓库 `tmp/`，编译缓存使用它下面的独立子目录，避免 Unix socket 路径超过长度上限。

查看任务与进度：

```shell
squeue -u "$USER"
tail -f research/scripts/replay/query_probe/results/performance-<job_id>.out
cat research/scripts/replay/query_probe/results/performance-<job_id>/matrix_status.json
```

每组细节在同名日志中，例如 `dense-c4.log`；P 和 proxy 日志分别为 `p.log`、`proxy.log`。任何一组失败会停止后续矩阵并释放本作业资源，已完成结果保留。也可重新汇总已有完整配对：

```shell
.venv/bin/python research/scripts/replay/query_probe/summarize_performance.py \
  research/scripts/replay/query_probe/results/performance-<job_id>
```

正式实验采用并发 4/8/16、每组 1000 条，具体方案与提交方式见 [正式性能实验](PERFORMANCE.md)。旧结果和结论已清理。

## 小规模复现与性能诊断

先用少量请求验证，再扩大样本。下面的第二次实验反转每个并发度的模式顺序：

```shell
sbatch --nodelist=g2c13 research/scripts/replay/query_probe/run_performance.sbatch \
  --concurrency 4 8 --max-requests 32 --warmup-requests 8
sbatch --nodelist=g2c13 research/scripts/replay/query_probe/run_performance.sbatch \
  --concurrency 4 8 --max-requests 32 --warmup-requests 8 --reverse-order
```

`benchmark_attention.py` 隔离测量原生 dense、同一新内核全保留、新内核跳块；可用 `--reference` 指向旧源码快照加入旧稀疏实现。全保留消融用于区分专用 prefill 内核改进与真正跳块的收益。微基准的两类调用是 Update attention 和联合 Append / Query 最后一层；后者包含评分 kernel，GPU 分数回传另记为 `score_collect`。

```shell
mkdir -p tmp/query-probe-profile
TMPDIR="$PWD/tmp" TRITON_CACHE_DIR="$PWD/tmp/query-probe-profile/triton" \
  .venv/bin/python research/scripts/replay/query_probe/benchmark_attention.py \
  --batch 4 --contexts 1024 4096 8192 --query-length 512 \
  --output research/scripts/replay/query_probe/results/kernels.json
```

微基准需在分配的 GPU 上执行。`--torch-profile-dir tmp/query-probe-profile/traces` 保存 CUPTI 时间线；`--profile` 则发出 CUDA profiler 启停标记，可配合 `nsys profile --capture-range=cudaProfilerApi --capture-range-end=repeat --kill=none`。必须检查报告实际包含 CUDA 事件，不能只根据生成了 `.nsys-rep` 判断采集成功。`--query-tile`、`--key-tile` 用于独立微基准调参，不改变服务端配置。

真实系统诊断使用单独的作业，避免把 profiler 开销混入正式吞吐结果：

```shell
sbatch --nodelist=g2c13 research/scripts/replay/query_probe/run_performance.sbatch \
  --concurrency 4 --max-requests 4 --warmup-requests 0 \
  --profile-iterations 40 --profile-delay 16
```

它在预检后通过 P 的 `/start_profile` 和 `/stop_profile` 控制内置 PyTorch profiler。trace 位于该作业的 `profiles/`；每组文件清单记录在 `*-profile-files.json`。`--profile-delay` 跳过开始的若干 engine step，`--profile-iterations` 限定采集范围。诊断作业的 TTFT 含采集和导出开销，不用于最终性能对比。正式实验方案和结果记录见 [PERFORMANCE.md](PERFORMANCE.md)。
