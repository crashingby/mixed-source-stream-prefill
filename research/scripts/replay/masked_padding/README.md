# Masked Padding 性能对比

本目录评估 `PROMPT_STREAM_MASKED_PADDING=0/1` 对分段流式 Prefill 的性能影响。
两组实验都使用同一份 mixed-source 数据、相同事件到达计划、相同 Triton
attention 后端和相同 P/D 资源。唯一主动变量是 P、D 是否启用中间 padding
的 attention mask 与逻辑位置修正。

## 共享数据与文本传输

本入口使用 [公共回放实现](../README.md)，支持独立 questions、append_chunks 和
pipeline_pools，按 `sample_id` 对齐。默认 `--input-transport text`，客户端渲染
模型模板，P 分词；start 是模板前缀，事件是原始文本，finish 带问题及 assistant
header。`--input-transport token_ids` 用于历史预分词实验复现。

`run_masked_padding_sweep.sbatch` 可设置
`REPLAY_CONFIG=research/scripts/replay/configs/triviaqa.json`。
其 Python 入口仍是 stream 回放器；`--return-token-ids` 读取输出计量，与输入
传输模式独立。保留 padding manifest 校验和完整 P→D 生成。

批量脚本支持 `INPUT_TRANSPORT`，默认 text；脚本显式参数优先于配置。

## 为什么同时测 TTFT 和 TPS

Masked Padding 同时作用于 P、D 两侧：

- P 节点在 Append、Update 和最终问题的 prefill 中使用 range mask，因此可能影响
  finish-to-first-token TTFT。
- D 节点读取 P 的物理 KV 后继续 decode，同样需要使用 range mask 和逻辑
  position，因此只跑 `/finish_ttft` 无法观察完整代价。
- CPU 路径只多传每请求两个边界整数，但 Triton kernel 每个 attention tile
  需要额外应用 range mask。实际开销必须通过同配置 A/B 实验确认。

实验分为两个阶段：

| 阶段 | REST 终点 | 是否启动 D | 主要指标 | 回答的问题 |
| --- | --- | --- | --- | --- |
| `ttft` | `/v1/prompt_stream/finish_ttft` | 否 | TTFT p50/p95 | P 侧 masked attention 是否拖慢最终 prefill |
| `full` | `/v1/prompt_stream/finish` | 是 | TTFT、TPOT、单请求 TPS、aggregate TPS、成功率 | P→D 完整链路和 D decode 的总体代价 |

`full` 阶段固定传递 `return_token_ids=true`。vLLM streaming completion 返回的是每个
chunk 的增量 `token_ids`，因此脚本按真实 token 数计算吞吐，不使用字符数或 SSE
chunk 数进行近似。

## 指标定义

对包含 $N$ 个输出 token 的成功请求，定义：

$$
\operatorname{TTFT}=t_1-t_{finish}
$$

$$
\operatorname{TPOT}=\frac{t_N-t_1}{N-1}
$$

$$
\operatorname{OutputTPS}=\frac{N-1}{t_N-t_1}
$$

其中，$t_{finish}$ 是客户端提交 `finish` 的时刻，$t_1$ 和 $t_N$ 分别是首个与
最后一个输出 token 到达客户端的时刻。这个 TPOT 包含 P→D handoff、D 节点调度和
后续 decode，是客户端可观察到的完整 post-first-token 代价。

`aggregate_output_tps` 使用所有有效请求的真实输出 token 总数除以本轮观测窗口。
它反映带负载时系统整体吞吐；不能用单请求 `OutputTPS` 的均值替代。

## 目录内容

- `run_masked_padding_sweep.sbatch`：对当前已经启动的一个 P/D 配置执行 warmup、
  P-only TTFT 和完整 P→D sweep。
- `summarize_masked_padding.py`：校验 enabled/disabled 样本配对，生成逐轮汇总和
  百分比差异。
- `plot_masked_padding_comparison.py`：绘制 P-only TTFT 图，以及完整链路的
  TTFT、TPOT、aggregate TPS 三联图。
- `results/`：默认实验输出目录；脚本首次运行时自动创建。

底层请求回放复用
`research/scripts/replay/segmented_prefill/replay_prompt_stream.py`，避免维护另一套
事件时序实现。

## 公平对比约束

两组实验必须满足以下条件：

- **P 和 D 同时切换开关**。不允许 P 开启、D 关闭，或反过来。
- **两组都使用 `TRITON_ATTN`**。关闭组不能退回自动选择的 FlashAttention，
  否则测到的是后端差异。
- **两组都使用 `GENERATION_CONFIG=vllm`**。这会统一 Qwen 的 sampling 默认值，
  避免模型目录中的 `repetition_penalty=1.05` 只让开启组失败。
- **关闭 Update coalescing**。本实验只隔离 padding mask，不混入另一项优化。
- 模型、数据目录、P/D 节点、GPU 数、`GPU_MEMORY_UTILIZATION`、
  `MAX_MODEL_LEN`、`MAX_NUM_SEQS` 和 TP 配置保持一致。
- `START_INDEX`、`MAX_REQUESTS`、QPS、并发、事件 interval、sampling seed 和
  `MAX_TOKENS` 保持一致。
- 每次重启后先运行完整 P→D warmup，排除 Triton JIT 和首次 NIXL 建链。
- 正式结果至少重复 3 次。两个 profile 的同一 repeat 使用相同
  `random_seed`，从而得到相同的泊松到达时间表。

汇总器会根据 `selected_sample_set_sha256` 检查每个
`phase × QPS × repeat × seed` 是否存在一对相同样本。错配时默认拒绝生成结论。

## 启动关闭组

下面用通用占位符展示启动方式。P、D 可以在不同节点，也可以由 Slurm 在同节点
分配不同 GPU；proxy 必须能访问二者的 OpenAI 端口。

### P 节点

```shell
MODEL_PATH=<MODEL_PATH> \
SERVED_MODEL_NAME=<SERVED_MODEL_NAME> \
GPU_MEMORY_UTILIZATION=0.6 \
MAX_MODEL_LEN=4096 \
MAX_NUM_SEQS=64 \
ATTENTION_BACKEND=TRITON_ATTN \
GENERATION_CONFIG=vllm \
PROMPT_STREAM_MASKED_PADDING=0 \
PROMPT_STREAM_UPDATE_COALESCING_CONFIG='' \
sbatch --nodelist=<P_NODE> research/scripts/server/start_p_node.sbatch
```

### D 节点

```shell
MODEL_PATH=<MODEL_PATH> \
SERVED_MODEL_NAME=<SERVED_MODEL_NAME> \
GPU_MEMORY_UTILIZATION=0.6 \
MAX_MODEL_LEN=4096 \
MAX_NUM_SEQS=64 \
ATTENTION_BACKEND=TRITON_ATTN \
GENERATION_CONFIG=vllm \
PROMPT_STREAM_MASKED_PADDING=0 \
sbatch --nodelist=<D_NODE> research/scripts/server/start_d_node.sbatch
```

等待两个日志都出现服务启动完成后，启动或重启 proxy：

```shell
.venv/bin/python research/scripts/server/proxy_server.py \
  --host 0.0.0.0 \
  --port 8000 \
  --prefiller-hosts <P_NODE> \
  --prefiller-ports 8100 \
  --decoder-hosts <D_NODE> \
  --decoder-ports 8200
```

## 回放关闭组

```shell
MASKED_PADDING=0 \
PROXY_URL=http://<PROXY_HOST>:8000 \
DATA_DIR=research/data/mix_source_perf_qwen \
OUTPUT_DIR=research/scripts/replay/masked_padding/results \
MODEL=qwen2.5-3b \
TOKENIZER_MODEL=/workplace/home/huangxinying/Models/Qwen2.5-3B \
START_INDEX=0 \
MAX_REQUESTS=512 \
QPS_LIST="0.5 1 2 4" \
MAX_CONCURRENCY=32 \
NUM_REPEATS=3 \
APPEND_CHUNK_INTERVAL=0.5 \
UPDATE_CHUNK_INTERVAL=0.2 \
MAX_TOKENS=128 \
RUN_PREFIX=qwen-triton \
sbatch --nodelist=<REPLAY_NODE> \
  research/scripts/replay/masked_padding/run_masked_padding_sweep.sbatch
```

`MASKED_PADDING` 只是实验标签和响应校验条件，**不能远程修改 P/D 配置**。如果它与
首个 SSE payload 中的 manifest 不一致，请求会被标记为失败，从而阻止错误标注。

## 启动并回放开启组

先结束旧的 P/D 作业，然后用完全相同的资源参数重启；只把两条启动命令中的：

```shell
PROMPT_STREAM_MASKED_PADDING=0
```

改为：

```shell
PROMPT_STREAM_MASKED_PADDING=1
```

P、D 和 proxy 都就绪后，复用上一节的 replay 命令，只修改：

```shell
MASKED_PADDING=1
```

不要修改 `RUN_PREFIX`、数据范围、QPS、interval、并发或 repeat 数。结果目录名已经
包含 `masked-padding-enabled/disabled`、seed、repeat 和 Slurm job id，不会互相
覆盖。

## 常用参数

| 参数 | 默认值 | 含义 |
| --- | ---: | --- |
| `MASKED_PADDING` | 必填 | 当前 P/D 的真实模式；`0` 为 padding 参与 attention，`1` 为 mask |
| `QPS_LIST` | `0.5 1 2 4` | offered load；可额外加入 `0` 做闭环串行测试 |
| `MAX_REQUESTS` | `512` | 每个 QPS、phase、repeat 的逻辑请求数 |
| `MAX_CONCURRENCY` | `32` | replay 客户端允许的最大在途逻辑请求数；QPS 为 `0` 时脚本强制为 `1` |
| `NUM_REPEATS` | `3` | 每个配置的正式重复次数 |
| `MAX_TOKENS` | `128` | `full` 阶段总输出上限；P 首 token 计入其中 |
| `RUN_TTFT_SUITE` | `1` | 是否运行 P-only TTFT 阶段 |
| `RUN_FULL_SUITE` | `1` | 是否运行完整 P→D 阶段 |
| `RUN_WARMUP` | `1` | 是否先运行不进入正式汇总的完整链路 warmup |
| `WARMUP_REQUESTS` | `8` | warmup 请求数量 |
| `APPEND_CHUNK_INTERVAL` | `0.5` | 同一 Append 来源的 chunk 间隔，单位秒 |
| `UPDATE_CHUNK_INTERVAL` | `0.2` | 同一 Update 来源的快照间隔，单位秒 |
| `SSE_TIMEOUT` | `300` | 完整消费一个生成流的最长秒数 |
| `FAIL_FAST` | `0` | 是否在首个请求失败时终止整轮 |

只测串行 decode 开销时，可以缩小请求数并关闭 TTFT-only 阶段：

```shell
MASKED_PADDING=<0_OR_1> \
QPS_LIST="0" \
MAX_REQUESTS=64 \
MAX_CONCURRENCY=1 \
NUM_REPEATS=3 \
RUN_TTFT_SUITE=0 \
RUN_FULL_SUITE=1 \
PROXY_URL=http://<PROXY_HOST>:8000 \
MODEL=<SERVED_MODEL_NAME> \
TOKENIZER_MODEL=<TOKENIZER_MODEL> \
sbatch research/scripts/replay/masked_padding/run_masked_padding_sweep.sbatch
```

## 汇总结果

两种 profile 都完成后运行：

```shell
.venv/bin/python \
  research/scripts/replay/masked_padding/summarize_masked_padding.py \
  --results-dir research/scripts/replay/masked_padding/results
```

输出位于 `results/analysis/`：

- `run_summary.csv`：每个 profile、phase、QPS、repeat 的独立结果。
- `comparison.csv`：相同 phase/QPS 下开启组相对关闭组的平均变化。
- `paired_request_deltas.csv`：按 `sample_id` 对齐两组后的请求级 TTFT、TPOT、
  TPS 差异；padding 长度取开启组返回的语义 manifest。
- `padding_bucket_summary.csv`：按 `0`、`1-4`、`5-8`、`9-12`、`13+` 个
  padding token 分桶，观察开销是否随 gap 长度变化。
- `summary.json`：便于后续脚本读取的结构化比较结果。

`*_change_percent` 使用：

$$
\frac{\text{enabled}-\text{disabled}}{\text{disabled}}\times 100\%
$$

因此 TTFT、TPOT 的负值表示开启 mask 后更快；TPS 的正值表示开启后吞吐更高。
必须同时检查成功率。如果一组靠请求失败减少了负载，其延迟和吞吐不能作为有效
性能结论。

## 绘图

```shell
.venv/bin/python \
  research/scripts/replay/masked_padding/plot_masked_padding_comparison.py \
  --run-summary-csv \
    research/scripts/replay/masked_padding/results/analysis/run_summary.csv \
  --output-dir research/scripts/replay/masked_padding/results/analysis
```

生成：

- `masked_padding_ttft_only.png`：P-only TTFT p95 随 QPS 的变化。
- `masked_padding_full_generation.png`：完整 P→D 的 TTFT p95、TPOT p50 和
  aggregate output TPS。

## 结果解释边界

- 关闭组与开启组的输出语义本来就不同：关闭组让物理 padding 参与 attention，
  开启组恢复逻辑序列语义。因此本实验用于量化“消除该近似需要多少性能代价”，
  不是要求两组逐 token 输出相同。
- `ttft` 阶段不会启动 D，不能据此判断 decode TPS。
- `full` 阶段的 TPOT 包含 P→D handoff 和 D 首次调度。这是客户端视角的真实代价；
  如果需要纯 kernel decode 吞吐，应额外运行 `QPS_LIST="0"` 的串行组，并结合
  profiler 分离 NIXL 时间。
- 首轮 Triton JIT、首次 NIXL 建链、不同节点的后台 GPU 负载和抢占都会显著影响
  尾延迟。warmup 与重复实验不能省略。
- 当前 Masked Padding 仅支持项目文档声明的 Triton self-attention 范围；不要把
  FlashAttention、MLA、滑窗、多模态等不支持组合纳入这组数据。
