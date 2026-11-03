# 普通请求与流式请求的混合负载

本实验观察：流式请求采用分段管理和 Query probe 稀疏后，是否能降低同一个 P 节点上**普通请求的 TTFT**，以及整个混合负载的延迟和积压。

普通请求与流式请求在一次回放中共享 HTTP client、到达时间线、客户端上限和 proxy/P 节点。两种流式算法分别运行一轮；每轮内部都是混合负载。

## 请求类型与算法命名

| 字段 | 值 | 含义 |
| --- | --- | --- |
| `request_kind` | `regular` | 普通完整 prefill，一次提交最终 prompt |
| `request_kind` | `stream` | 分事件到达，最后提交问题后缀 |
| `stream_mode` | `normal` | 流式基线：每个原始 Append/Update 到达后，把当前全部证据发送为同一个 Update 来源的完整快照 |
| `stream_mode` | `segmented_sparse` | 分段 + Query probe 稀疏：保留独立 Append/Update 来源，每次 Append 联合问题 probe，后续知识计算使用已发布的块选择 |

这里的 `normal` **只表示流式请求采用纯 Update 建模**，普通请求始终叫 `regular`。两组的普通请求使用相同接口和正文，不注册 probe。

基线的完整快照固定采用 **Append 在前、当前 Update 在后**。分段模式采用同一最终正文顺序，且只有最终 Update 快照进入最终 prompt。两组不启用 Update 合并，同一时刻的多个数据事件也逐条发送，避免基线因客户端折叠事件而减少更新次数。

当前只测 P-only：普通请求调用 `/v1/completions_ttft`；流式请求调用 `start_ttft`、`event`、`finish_ttft`。计时终点统一为 `prefill_ready`，不启动 D，不生成答案。这些数值是 **P 侧 TTFT / prefill 完成延迟**，不包含 KV 传输或 D 的回答生成。

## 数据与混合比例

`configs/triviaqa.json` 和 `configs/hotpotqa.json` 通过公共读取器加载 `research/data/questions`、`research/data/append_chunks`、`research/data/pipeline_pools`。`--questions-file`、`--append-dir`、`--pipeline-pools-dir` 可覆盖版本路径；配置中的相对路径相对 JSON 所在目录解析。

- `max_requests` 是两类请求的**合计数量**。默认 1000 条、`stream_ratio=0.5`，即 500 条普通请求和 500 条流式请求。
- 流式数量取 `floor(N × stream_ratio + 0.5)`，通过 `mix_seed` 从所选样本位置中抽取。每条样本只发送一次，两类样本不重复。
- 样本先按公共读取器的顺序选择，再随机分配类型。切换 `stream_mode` 不改变样本、类型分配、顺序或到达种子。
- `stream_ratio=0` 为全普通，`stream_ratio=1` 为全流式。有限样本的实际比例写入摘要。
- 普通请求把全部 Append、最终 Update、问题和模型模板一次提交；这里模拟有完整检索上下文的普通 RAG 请求。它们的长度未被另外缩短。

原始事件计划复用 `token_invalidation/plan.py` 的纯数据转换函数。两个流式算法使用相同原始事件顺序和绝对时间；同一请求内逐事件等待 ACK，处理慢时保留计划时间并记录迟发。所有数据发送后封口来源，再发送全局 finish。`source_finish_delay` 从各原始来源最后一次 ACK 起算，空来源从 initial delay 起算。

## 负载配置

沿用 [公共负载与计时](../LOAD.md)，这里只使用一个全局 runner：

| 参数 | 默认值 | 含义 |
| --- | --- | --- |
| `load_mode` | `auto` | 正 QPS 使用开环，零 QPS 使用闭环 |
| `qps` | `1` | 两类请求合计的到达率；不是每类各 1 QPS，也不是事件 QPS |
| `arrival_process` | `poisson` | 可改为 `fixed` |
| `random_seed` | `2026` | 到达时间种子 |
| `mix_seed` | `2026` | 类型分配种子，与到达随机数独立 |
| `max_concurrency` | `8` | 闭环两类请求合计的在途上限 |
| `max_inflight` | `0` | 开环可选保护上限；0 不限，正 QPS 不使用 `max_concurrency` |
| `stream_ratio` | `0.5` | 逻辑请求数量中的流式比例 |
| `keep_ratio` | `0.5` | 稀疏模式保留的可选择 KV 块比例 |
| `append_chunk_interval` / `update_chunk_interval` | `0.05` / `0.02` 秒 | 单请求内的原始事件间隔 |
| `append_initial_delay` / `update_initial_delay` | `0` / `0` 秒 | start ACK 后的首次数据到达延迟 |
| `source_finish_delay` | `0` 秒 | 最后 ACK 后的来源封口延迟 |

类型在开始前固定，完成较快的普通请求不会额外抢占后续样本的类型配额。闭环实际到达时刻仍受请求完成速度影响，瞬时在途类型比例也不保证等于 `stream_ratio`。要比较相同外部输入压力下普通请求的变化，优先采用开环，并核对实际发送速率、成功率及积压。

## 运行方式

先启动同一个 P 和 proxy。P 使用 `CHUNK_SPARSE_ATTN`、masked padding、关闭 Update coalescing；两组均保持相同后端、模型、token 预算和调度上限。现有启动方式见 [Query probe 启动说明](../query_probe/README.md#启动)，模型默认 `~/Models/Qwen2.5-3B`。proxy 的 decoder 地址可保留占位配置，本实验不会连接 D。

两轮会重放相同普通请求，正式对比应在 P 的启动命令中加 `--no-enable-prefix-caching`，或在每轮前按相同规则清空 prefix cache，避免后跑的一组复用前一轮的 KV。客户端脚本不会替你重置服务端缓存。

下面命令在仓库根目录执行。先跑混合负载中的纯 Update 基线：

```shell
.venv/bin/python research/scripts/replay/mixed_workload/replay.py \
  --config research/scripts/replay/mixed_workload/configs/triviaqa.json \
  --proxy-url http://<PROXY_HOST>:8000 \
  --stream-mode normal --run-name triviaqa-normal
```

再用相同负载替换流式算法：

```shell
.venv/bin/python research/scripts/replay/mixed_workload/replay.py \
  --config research/scripts/replay/mixed_workload/configs/triviaqa.json \
  --proxy-url http://<PROXY_HOST>:8000 \
  --stream-mode segmented_sparse --run-name triviaqa-segmented-sparse
```

默认开环合计 1 QPS。需要闭环时，两条命令都加 `--qps 0 --max-concurrency 8`；改变比例时两边都加例如 `--stream-ratio 0.75`。独立数据配置可替换为 `configs/hotpotqa.json`。

只检查数据与类型清单可加 `--dry-run`：写入配置、输入指纹和分配清单，不发送 HTTP，也不等待到达时间。正式回放不自动预热；初期请求仍计入统计。重复实验可交换运行顺序，预热也应另存结果目录。

### 后台依次运行两组

`run_mixed_workload.sbatch` 只申请客户端资源，复用已经启动的 P/proxy。它依次跑两组混合负载并自动生成配对报告：

```shell
mkdir -p research/scripts/replay/mixed_workload/results
PROXY_URL=http://<PROXY_HOST>:8000 \
DATASET=triviaqa MAX_REQUESTS=1000 STREAM_RATIO=0.5 QPS=1 \
sbatch research/scripts/replay/mixed_workload/run_mixed_workload.sbatch
```

JSON 为共享参数来源，只有显式提供的环境变量才覆盖它。可设置 `REPLAY_CONFIG`、`LOAD_MODE`、`MAX_CONCURRENCY`、`MAX_INFLIGHT`、`ARRIVAL_PROCESS`、`RANDOM_SEED`、`MIX_SEED`、`KEEP_RATIO` 和来源时间参数。`REVERSE_ORDER=1` 反转两组先后顺序。某组出现失败会返回非零并停止脚本，已完成记录保留。

这是独立 Slurm 作业，提交后不依赖 SSH 会话。它没有额外指定运行时限，仍受集群分区限制；P 和 proxy 也须保持可用。

## 结果与对比

默认输出到 `mixed_workload/results/<run-name>/`：

- `config.json`：实际参数、模板指纹与 `workload_sha256`。同一 P 配置可用 `experiment_profile` 标记，它是人工标签，不能替代服务端配置核对。
- `plan.jsonl`：样本位置、类型、最终正文指纹、原始事件计划指纹、正文长度；开环还包含计划到达偏移。
- `requests.jsonl` / `requests.csv`：逐请求结果、类型、TTFT、客户端等待。JSON 还保留事件 ACK/迟发、服务端流式统计及 probe 信息。
- `load_requests.jsonl` / `load_summary.json`：公共负载统计；可按 `sample_index` 与类型清单关联，包括取消和未发请求。
- `summary.json`：`groups.regular`、`groups.stream` 和 `groups.all`，分别报告成功/失败、发送速率、完成吞吐和延迟的 mean/P50/P95/P99/max。

优先查看普通请求的两个指标：

| 指标 | 含义 |
| --- | --- |
| `arrival_to_first_token_ms` | 计划到达至 P 完成，包含客户端等待与准备 |
| `sent_to_first_token_ms` | 首次 HTTP 提交至 P 完成，包含网络与服务端排队/计算 |
| `dispatch_lag_ms` | 计划到达到 HTTP 提交的全部客户端迟发 |
| `finish_to_first_token_ms` | 流式请求从最后 finish 提交至 P 完成；普通请求从完整 prompt 提交前计时 |

流式请求的前两个指标还包含来源事件等待，因此普通、流式延迟分别汇总。每类完成吞吐都除以**同一轮总耗时**，可以相加；每类实际发送 QPS 则从该类发送间隔计算，不要求两者相加等于全局观测速率。`stream_ratio` 是请求数量比例，不代表计算量比例。

手动配对：

```shell
.venv/bin/python research/scripts/replay/mixed_workload/compare.py \
  --baseline research/scripts/replay/mixed_workload/results/triviaqa-normal \
  --optimized research/scripts/replay/mixed_workload/results/triviaqa-segmented-sparse \
  --output-dir research/scripts/replay/mixed_workload/results/triviaqa-comparison
```

报告校验相同输入与负载指纹，输出 `comparison.csv` / `comparison.json`。延迟只配对两组都成功的相同样本，报告同时保留两组各类成功率、完成吞吐和公共负载摘要。`mean_speedup` 大于 1、`mean_reduction_fraction` 大于 0 表示优化组延迟更低。

比较时先看 `request_kind=regular`，再看流式和整体；同时检查成功率及实际发送速率，避免把少发请求或大量失败误判为缓解压力。这个两组对比体现**分段管理与稀疏的组合收益**，不能单独归因于块稀疏；分离贡献仍需已有分段 dense 对照。

本次仅实现代码和文档，未运行回放、单元测试或 GPU 实验。
