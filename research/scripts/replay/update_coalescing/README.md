# Update 合并回放实验

本目录用于评估 Prompt Stream 的 **Update coalescing（Update 合并）** 功能。实验通过
RESTFUL API 回放同一批 mixed-source 请求，同时测量：

- 最终问题到达后，P 节点生成首 token 的 TTFT。
- Update 合并减少的逻辑 Prefill token。
- 已物化、后来又因快照变化而失效的 Update KV token。
- 不同策略、窗口、Update 到达频率和 QPS 下的成功率与尾延迟。

合并策略是 **P 节点 Engine 启动级配置**。P 节点存活期间不能由 replay 请求切换
策略。每次修改 profile、Append gate、window 或 Adaptive 参数，都必须重启 P。

## 目录与脚本

```text
research/scripts/replay/update_coalescing/
├── README.md
├── coalescing_profiles.sh
├── start_p_node_coalescing.sbatch
├── replay_update_coalescing.py
├── run_policy_sweep.sbatch
├── summarize_coalescing.py
├── plot_coalescing_tradeoff.py
├── plot_strategy_matrix.py
└── results/
```

| 文件 | 用途 | 如何使用 |
| --- | --- | --- |
| `coalescing_profiles.sh` | 根据 profile 和环境变量生成 P 节点配置 JSON。 | 通常不单独运行，两个 sbatch 会自动 `source`。 |
| `start_p_node_coalescing.sbatch` | 申请 GPU，生成合并配置并启动一个 P 节点。 | 每个 Engine 配置单独提交一次。 |
| `replay_update_coalescing.py` | 执行一个 mixed-source REST 回放实验点。 | 调试单个点时前台运行。 |
| `run_policy_sweep.sbatch` | 在一个已启动的 P 配置下扫描 QPS、Update interval 和重复次数。 | 正式采集时提交。 |
| `summarize_coalescing.py` | 合并多个 run，生成 run 级、配置级和请求级汇总。 | 一组实验结束后运行。 |
| `plot_coalescing_tradeoff.py` | 绘制 TTFT、saved token、invalidated token 和 Pareto 图。 | 汇总结束后运行。 |
| `plot_strategy_matrix.py` | 按 Update interval 和 QPS 绘制策略 TTFT / token-saving 矩阵。 | 汇总结束后运行。 |

### `coalescing_profiles.sh`

这是**配置生成器**，不是需要提前启动的服务：

- P 启动 sbatch 用它生成真正传给 vLLM 的 JSON。
- replay sbatch 用它生成审计 JSON，并与 P 返回的实际配置核对。

正常实验不需要手动执行。只想查看某个 profile 的最终 JSON 时可以运行：

```shell
source research/scripts/replay/update_coalescing/coalescing_profiles.sh

COALESCING_PROFILE=fixed-window \
SOFT_WINDOW_MS=20 \
HARD_DEADLINE_MS=80 \
build_coalescing_config
```

### `start_p_node_coalescing.sbatch`

这个脚本只启动**一个指定配置的 P 节点**。它会：

- 读取 `COALESCING_PROFILE` 和相关参数。
- 生成并打印最终配置 JSON。
- 在当前 Slurm allocation 内复用
  `research/scripts/server/start_p_node.sbatch` 启动 vLLM。

它不会运行 replay，也不会自动轮换策略。

#### 提交方式

环境变量必须写在 `sbatch` 前面；`--nodelist` 等 Slurm 参数写在 sbatch 命令后面：

```shell
COALESCING_PROFILE=fixed-window \
SOFT_WINDOW_MS=20 \
HARD_DEADLINE_MS=80 \
MODEL_PATH="${MODEL_PATH}" \
SERVED_MODEL_NAME="${SERVED_MODEL_NAME}" \
GPU_MEMORY_UTILIZATION=0.6 \
MAX_MODEL_LEN=4096 \
MAX_NUM_SEQS=16 \
sbatch --nodelist="${P_NODE}" \
  research/scripts/replay/update_coalescing/start_p_node_coalescing.sbatch
```

这条命令分为三部分：

- `COALESCING_PROFILE` 及其配套变量决定 **Update 合并策略**。
- `MODEL_PATH`、`MAX_MODEL_LEN` 等变量决定 **vLLM Engine 配置**。
- `--nodelist`、`--partition` 等选项决定 **Slurm 资源分配**。

#### Slurm 资源参数

脚本内置的默认 allocation 是：一个节点、一张 GPU、16 个 CPU、64 GiB 内存、最长
12 小时。常用提交选项如下：

| `sbatch` 选项 | 脚本默认值 | 何时调整 |
| --- | ---: | --- |
| `--nodelist` | 未固定 | 指定 P 实际运行节点，例如 `--nodelist=g2c13`。 |
| `--partition` | `gpu` | 集群存在多个 GPU partition 时显式选择。 |
| `--gres` | `gpu:1` | 当前实现按单 GPU P 节点测试，不建议修改。 |
| `--cpus-per-task` | `16` | tokenization 或 HTTP 前端 CPU 不足时调整。 |
| `--mem` | `64G` | 模型加载或数据处理需要更多主存时调整。 |
| `--time` | `12:00:00` | P 必须覆盖该 profile 下全部 replay 的运行时间。 |

Slurm 会设置当前 job 可见的 GPU。通常不要再手写 `CUDA_VISIBLE_DEVICES`，否则可能
覆盖 Slurm 的 GPU 隔离结果。

#### P 节点通用参数

`start_p_node_coalescing.sbatch` 最终会调用通用 P 启动脚本，因此还支持以下环境变量：

| 环境变量 | 默认值 | 是否建议显式设置 | 含义 |
| --- | ---: | --- | --- |
| `REPO_ROOT` | `SLURM_SUBMIT_DIR` | 从仓库根目录提交时不需要 | vLLM 仓库根目录。 |
| `PYTHON_BIN` | `${REPO_ROOT}/.venv/bin/python` | 通常不需要 | 启动 vLLM 的 Python。 |
| `MODEL_PATH` | `${HOME}/Models/Qwen2.5-3B/` | **建议设置** | 模型目录。正式实验不要依赖个人环境的隐式默认值。 |
| `SERVED_MODEL_NAME` | `qwen2.5-3b` | **建议设置** | REST 请求中的模型名称，必须与 replay 的 `MODEL` 一致。 |
| `HOST` | `0.0.0.0` | 通常不需要 | P 节点 OpenAI API 监听地址。 |
| `PORT` | `8100` | 端口冲突时设置 | P 节点 REST 端口，proxy 的 prefiller port 必须与它一致。 |
| `GPU_MEMORY_UTILIZATION` | `0.65` | **建议设置** | vLLM 可使用的 GPU 显存比例。过高可能因其它进程占用而 OOM。 |
| `MAX_MODEL_LEN` | `4096` | **建议设置** | 最大序列长度；超过它的实验请求会失败。 |
| `MAX_NUM_SEQS` | `16` | **建议固定并记录** | Scheduler 同时容纳的序列上限，不是数据集请求总数。增大后并发能力和 KV 显存占用都会增加。 |
| `NIXL_SIDE_CHANNEL_PORT` | `5600` | 同节点多实例时设置 | NIXL P 侧 side-channel 端口。 |
| `VLLM_NIXL_SIDE_CHANNEL_HOST` | 实际 Slurm hostname | 特殊网络环境时设置 | 发布给 D 节点的 NIXL 可达地址，不能错误写成跨节点不可达的 localhost。 |
| `UCX_NET_DEVICES` | `all` | 特殊网卡绑定时设置 | NIXL/UCX 使用的网络设备。 |

`MAX_NUM_SEQS` 控制 Engine 同时调度多少个**逻辑请求**，不控制一个请求内部有多少
Append/Update 事件。提高它可能缓解并发请求 admission 限制，但也会提高 KV Cache
压力；对比实验中必须固定，不能只给某个策略单独调大。

#### 合并配置参数

| 环境变量 | 适用 profile | 用途 |
| --- | --- | --- |
| `COALESCING_PROFILE` | 全部 | 必须显式指定，避免意外使用默认 `fixed-window`。 |
| `APPEND_GATE` | 除 `disabled`、`immediate-controller` 外 | 选择 Append gate；主实验默认 `hold_until_append_finished`。 |
| `SOFT_WINDOW_MS` | fixed、debounce、adaptive | soft window，默认 `20` ms。 |
| `HARD_DEADLINE_MS` | fixed、debounce、adaptive | hard deadline，默认 `80` ms。 |
| `TTFT_DEADLINE_MS` | fixed、debounce、adaptive | 可选请求级 deadline；不启用时不传或设为 `null`。 |
| EWMA、prior、price 变量 | adaptive | 控制在线估计与等待成本。 |
| `PROMPT_STREAM_UPDATE_COALESCING_CONFIG` | custom | 完整 JSON；其它 profile 会由 helper 自动生成，不要手动同时设置。 |

**同一个 P job 的 Engine 配置不会随 replay 改变。** 修改上述任意合并参数后，都要
停止旧 P 并重新提交这个 sbatch。

### `replay_update_coalescing.py`

这是单次回放入口。它通过 proxy 依次发送 start、Append/Update、source finished 和
finish_ttft 请求。客户端只复现事件时间线；Update admission、window、deadline 和
Append gate 全部由 P 节点执行。

### `run_policy_sweep.sbatch`

这个脚本只服务于**当前存活的一个 P 配置**，扫描：

```text
QPS_LIST × UPDATE_INTERVAL_LIST × NUM_REPEATS
```

例如：

```text
QPS_LIST="0.5 1 2 4"
UPDATE_INTERVAL_LIST="0.2 0.01"
NUM_REPEATS=3
```

会产生 $4 \times 2 \times 3=24$ 个正式 run。replay 中的 profile 参数只是配置
审计，不会改变 P 的 Engine 配置。

该 sbatch 固定使用 `--ttft-only`。P 生成首 token 后，proxy 直接返回，不再向 D
节点发送 decode 请求。因此 TTFT 主实验不会混入 D 节点 KV load 和 decode 噪声。

#### 提交方式

批量回放脚本本身不启动 P，也不需要 GPU。它只需要 CPU 和网络连接，并要求 proxy、
P 已经就绪。典型提交方式是：

```shell
PROXY_URL="${PROXY_URL}" \
COALESCING_PROFILE=fixed-window \
APPEND_GATE=hold_until_append_finished \
SOFT_WINDOW_MS=20 \
HARD_DEADLINE_MS=80 \
MAX_REQUESTS=512 \
QPS_LIST="0.5 1 2 4" \
UPDATE_INTERVAL_LIST="0.5 0.2 0.01" \
NUM_REPEATS=3 \
sbatch research/scripts/replay/update_coalescing/run_policy_sweep.sbatch
```

这里的合并参数不会传给 P。它们只重建预期 JSON，用来检查当前 P 是否真的以相同
配置启动。只改 replay 侧参数不会改变 Engine 行为，反而会触发配置不一致错误。

#### 必须与 P 保持一致的参数

| replay 环境变量 | 对应 P 配置 | 约束 |
| --- | --- | --- |
| `COALESCING_PROFILE` | P 的 profile | 必须相同。 |
| `APPEND_GATE` | P 的 `append_gate` | 使用该变量的 profile 必须相同。 |
| `SOFT_WINDOW_MS` | P 的 `soft_window_ms` | fixed、debounce、adaptive 必须相同。 |
| `HARD_DEADLINE_MS` | P 的 `hard_deadline_ms` | fixed、debounce、adaptive 必须相同。 |
| `TTFT_DEADLINE_MS` | P 的 `ttft_deadline_ms` | P 设置时 replay 也必须设置。 |
| Adaptive EWMA、prior、price 变量 | P 的 Adaptive 字段 | 每个字段都必须相同。 |
| `PROMPT_STREAM_UPDATE_COALESCING_CONFIG` | custom JSON | `custom` profile 下必须完全相同。 |

`MODEL` 还必须等于 P 的 `SERVED_MODEL_NAME`，但它不是 coalescing JSON 的一部分。

#### 回放矩阵参数

| 环境变量 | 默认值 | 实际含义 |
| --- | ---: | --- |
| `MAX_REQUESTS` | `256` | **每个 run** 选择的请求数，不是整个 sbatch 的总请求数。 |
| `QPS_LIST` | `1` | 空格分隔的逻辑请求到达率；不是 HTTP event QPS。 |
| `UPDATE_INTERVAL_LIST` | `0.2 0.01` | 单个逻辑请求内部相邻 Update 事件的计划间隔，单位秒。 |
| `NUM_REPEATS` | `3` | 每个 `(QPS, Update interval)` 点独立重复多少次。 |
| `BASE_RANDOM_SEED` | `2026` | 第一次重复的到达过程 seed；第 $i$ 次重复使用 `base+i-1`。 |
| `ARRIVAL_PROCESS` | `poisson` | 请求到达过程，当前常用 `poisson`；也支持 `fixed`。 |
| `MAX_CONCURRENCY` | `8` | 客户端最多同时保留多少个未完成逻辑请求。 |

正式 run 数为：

$$
N_{\mathrm{runs}}
= |\mathrm{QPS\_LIST}| \times |\mathrm{UPDATE\_INTERVAL\_LIST}|
\times \mathrm{NUM\_REPEATS}.
$$

正式请求总数为：

$$
N_{\mathrm{requests}}
= N_{\mathrm{runs}} \times \mathrm{MAX\_REQUESTS}.
$$

例如 4 个 QPS、3 个 Update interval、3 次重复、每次 512 个请求，会生成 36 个 run，
计划回放 $36 \times 512=18432$ 个正式请求。

`QPS_LIST` 控制逻辑请求的计划启动时间。请求启动后，它自己的 Append/Update 事件仍按
source interval 继续发送。因此流式 workload 的 HTTP event 到达率会高于逻辑 QPS。
当 `MAX_CONCURRENCY` 已满时，新请求会在客户端等待空位，实际 achieved QPS 可能低于
offered QPS；这正是高负载结果必须同时报告成功率和完成吞吐的原因。

#### Source 时序参数

| 环境变量 | 默认值 | 含义 |
| --- | ---: | --- |
| `APPEND_INITIAL_DELAY` | `0.0` | 请求 start ACK 后到第一条 Append 的计划延迟。 |
| `UPDATE_INITIAL_DELAY` | `0.0` | 请求 start ACK 后到第一条 Update 的计划延迟。 |
| `APPEND_CHUNK_INTERVAL` | `0.5` | 同一请求 Append source 内相邻 chunk 的间隔。 |
| `UPDATE_INTERVAL_LIST` | `0.2 0.01` | 同一请求 Update source 内相邻快照的间隔，也是 sweep 维度。 |
| `SOURCE_FINISH_DELAY` | `0.0` | 最后数据事件 ACK 后，再发送 source finished 的延迟。 |

这些参数改变 source 的相对结束顺序，可能显著影响 Append gate。正式对比时必须固定
除目标消融维度以外的所有 source timing。

#### 连接、输出与失败处理参数

| 环境变量 | 默认值 | 含义 |
| --- | ---: | --- |
| `PROXY_URL` | 无 | **必填**；必须从 replay 所在计算节点可达。 |
| `DATA_DIR` | `research/data/mix_source` | mixed-source 输入目录。 |
| `OUTPUT_DIR` | `.../update_coalescing/results` | 正式结果根目录。 |
| `MODEL` | `qwen2.5-3b` | REST 模型名，必须等于 P 的 served model name。 |
| `HTTP_TIMEOUT` | `600` | 普通 HTTP 操作超时，单位秒。 |
| `SSE_TIMEOUT` | `120` | 等待 P 首 token 的 SSE 超时，单位秒。 |
| `MAX_TOKENS` | `64` | 生成参数；本 sbatch 固定 TTFT-only，P 首 token 后即结束。 |
| `TEMPERATURE` | `0` | 生成温度；延迟对比建议固定为 0。 |
| `RUN_PREFIX` | `coalescing` | 自动生成 run name 的前缀。 |
| `EVENT_PROGRESS_LOG_INTERVAL` | `0` | 每 N 个 event 打印一次进度；`0` 关闭。 |
| `FAIL_FAST` | `0` | 设为 `1` 后首个 HTTP/SSE 或首 token 失败即停止；stats invalid 不触发。 |
| `FORMAL_FAIL_FAST` | `0` | 设为 `1` 后首个 `formal_result_status=invalid` 即停止。 |
| `EXTRA_REPLAY_ARGS` | 空 | 追加给 Python replay 的简单参数；带空格的复杂值不适合通过它传递。 |

`PROXY_URL=http://127.0.0.1:8000` 只在 replay job 与 proxy 位于同一节点时有效。
Slurm replay 通常运行在其它节点，应使用 proxy 所在节点 hostname 或 IP。

#### Warmup 行为

`WARMUP_REQUESTS` 默认为 `0`。设为正数后，脚本在正式矩阵前只执行**一次** warmup：

- 使用 `UPDATE_INTERVAL_LIST` 的第一个值。
- 使用 `BASE_RANDOM_SEED`。
- 使用 QPS `0` 和当前 `MAX_CONCURRENCY`，因此是突发式 warmup。
- 写入 `${OUTPUT_DIR}/warmup`，汇总脚本默认跳过。

它不是每个 QPS 点或每次重复前都重新 warmup。若需要严格顺序 warmup，可以同时把
`MAX_CONCURRENCY=1`，但正式采集时仍恢复统一的并发配置。

### 汇总与绘图

`summarize_coalescing.py` 生成：

- `run_summary.csv`：每次独立 run 的指标。
- `configuration_summary.csv`：相同 workload 和配置的重复实验汇总。
- `combined_request_metrics.csv`：全部请求级记录与实验元数据。
- `summary.json`：结构化汇总。

- `plot_coalescing_tradeoff.py` 使用 `configuration_summary.csv` 绘制四联
  trade-off 图。
- `plot_strategy_matrix.py` 使用同一份汇总结果，按 Update interval 和 QPS
  绘制 TTFT 与 Update Prefill token saving 策略矩阵。

## P 节点配置

真实 CLI 参数为：

```shell
--prompt-stream-update-coalescing-config '<JSON>'
```

实验脚本用 `COALESCING_PROFILE` 统一生成 JSON。

| Profile | `append_gate` | `update_policy` | 实验含义 |
| --- | --- | --- | --- |
| `disabled` | 不适用 | 不启用 controller | 原始逐 Update FIFO，是主要无合并基线。 |
| `immediate-controller` | `none` | `immediate` | 每条 Update 立即 admission，用于测 controller 开销。 |
| `strict-immediate` | 默认 `hold_until_append_finished` | `immediate` | policy 立即 ready，但受 Append gate 阻挡。 |
| `fixed-window` | 默认 `hold_until_append_finished` | `fixed_window` | 从窗口第一条 Update 开始等待固定 soft window。 |
| `quiet-debounce` | 默认 `hold_until_append_finished` | `quiet_debounce` | 每条有效 Update 刷新 soft deadline。 |
| `adaptive` | 默认 `hold_until_append_finished` | `adaptive` | 使用 EWMA 在线决定等待时间。 |
| `full-merge` | 默认 `hold_until_append_finished` | `full_merge` | 等待 source finished 或最终 Prompt barrier。 |
| `custom` | 由 JSON 决定 | 由 JSON 决定 | 使用完整手写 JSON。 |

除 `immediate-controller` 固定使用 `none` 外，其它 profile 可通过 `APPEND_GATE` 选择：

```text
none
hold_until_append_finished
hold_until_append_or_update_finished
```

`fixed-window`、`quiet-debounce`、`adaptive` 和 `full-merge` 都能与这些 gate 组合。
主实验固定 `hold_until_append_finished`，只对胜出策略做 gate 消融。

### 窗口参数

`fixed-window`、`quiet-debounce` 和 `adaptive` 使用：

- `SOFT_WINDOW_MS`：soft window 上限，默认 `20` ms。
- `HARD_DEADLINE_MS`：从窗口首事件计算的硬截止时间，默认 `80` ms。
- `TTFT_DEADLINE_MS`：可选请求级 TTFT deadline，默认 `null`。

实验脚本要求 `SOFT_WINDOW_MS <= HARD_DEADLINE_MS`。`full-merge` 不使用这些
window deadline。

### Adaptive 参数

| 环境变量 | 默认值 | 含义 |
| --- | ---: | --- |
| `INTERVAL_EWMA_ALPHA` | `0.25` | Update 到达间隔 EWMA 的平滑系数。 |
| `GAIN_EWMA_ALPHA` | `0.25` | 合并收益 EWMA 的平滑系数。 |
| `COLD_START_PRIOR_WEIGHT` | `8` | 冷启动先验的等效观测权重。 |
| `ARRIVAL_RATE_PRIOR_PER_S` | `null` | 可选的 Update 到达率先验。 |
| `MERGE_GAIN_PRIOR_TOKENS` | `null` | 可选的 token 收益先验。 |
| `DELAY_PRICE` | `1` | 等待成本权重。 |
| `LOAD_PRICE_SCALE` | `0` | 系统负载对等待成本的缩放。 |

## 实验设计

Update 合并不是简单地让所有 Update 等待更久，而是在有限等待内减少重复 Prefill，
同时控制最终用户问题的 TTFT。实验需要回答：

- **计算是否减少**：相同 Update 序列下，admitted work 和 Update Prefill token
  是否下降。
- **节省是否转化为性能收益**：不同 QPS 下，TTFT 和尾延迟是否改善。
- **等待代价是否可控**：window、deadline 和 Append gate 是否造成额外等待。
- **Adaptive 是否值得**：是否比调优后的 fixed/debounce 更稳健。

当前实验边界是一个真实 Append source 和一个 Update source。

### 控制组与主策略

- `disabled` 是原始逐 Update FIFO 行为。
- `immediate-controller` 启用 controller，但不等待、不使用 gate。它应与
  `disabled` 接近，用于排除 controller 热路径开销。
- `full-merge` 只计算最终目标快照，给出最少 Update work 的参考，但不保证 TTFT
  最优。

主策略比较固定 `append_gate=hold_until_append_finished`，再比较 `immediate`、
`fixed_window`、`quiet_debounce`、`adaptive` 和 `full_merge`。最后单独做 gate
消融，避免把 policy 和 gate 的影响混在一起。

### Workload 维度

#### Update 到达频率

| 类型 | `UPDATE_INTERVAL_LIST` 代表值 | 预期 |
| --- | --- | --- |
| 低频 | `0.5` | 窗口通常只有一条 Update，不应无故增加 TTFT。 |
| 中频 | `0.2` 或 `0.1` | 观察 fixed/debounce 的常规取舍。 |
| 高频 | `0.05` 或 `0.01` | Update burst 明显，最容易体现合并收益。 |

#### 系统负载

建议从 `QPS_LIST="0.5 1 2 4"` 开始，覆盖低负载、中负载和接近饱和但成功率仍
可接受的高负载点。若某个 QPS 出现大量 HTTP/SSE 或首 token 失败，应标记为
overload，不能删除失败请求后只报告较低的 TTFT。已经收到首 token、但后置 stats
校验失败的请求仍进入 TTFT 分布；其 token/work 指标不进入正式聚合，并由
`token_stats_coverage` 单独报告覆盖率。

#### Update 快照波动

客户端记录相邻 Update 快照的 source-local LCP：

$$
\rho_i = \frac{\operatorname{LCP}(U_{i-1}, U_i)}{\max(1, |U_i|)}.
$$

高 LCP 表示内容较稳定，低 LCP 表示快照变化剧烈。正式分析可以按 LCP ratio 分桶，
观察策略对 Update 波动程度的敏感性。

#### Append 与 Update 的结束顺序

后续 gate 消融至少覆盖 Append 先结束、两者接近同时结束、Update 先结束三种时序。
通过 initial delay、chunk interval 和事件数量控制，不与所有策略参数做笛卡尔积。

### 每次回放自动采集的指标

`run_policy_sweep.sbatch` 会同时采集 TTFT、saved token 和 invalidated token，不需要
为每个指标重新运行一套实验。

#### 延迟与可靠性

| 指标 | 含义 |
| --- | --- |
| `finish_to_first_token_ms` | `/finish_ttft` 发出到 P 返回首 token 的时间。 |
| TTFT P50 / P90 / P95 / P99 | 中位数和尾延迟。 |
| `transport_success_rate` | HTTP/SSE 链路成功率。 |
| `first_token_success_rate` | 实际收到首 token 的请求比例，也是 TTFT 的样本口径。 |
| `stats_valid_rate` | 通过 schema v2 recovery-aware 守恒校验的请求比例。 |
| `formal_valid_rate` | 首 token、配置和 stats 全部满足正式实验条件的比例。 |
| `token_stats_coverage` | stats-valid 请求数占首 token 成功请求数的比例。 |
| `completed_requests_per_s` | 当前 offered load 下的完成吞吐。 |

#### Update work 与 token

正式实验使用 `metrics_schema_version=2`。下表中的 `update_works_*`、
`admitted_update_prefill_tokens`、`saved_update_prefill_tokens`、
`extra_update_prefill_tokens`、`scheduled_update_prefill_tokens` 和
`update_tokens_invalidated` 是为了兼容旧 CSV 保留的别名；schema v2 的正式字段和
恢复语义见后表。

| 指标 | 含义 |
| --- | --- |
| `update_events_accepted` | P 接受且真实改变快照的 Update 事件数。 |
| `update_works_admitted` | 策略真正创建的 Update work 数。 |
| `update_works_completed` | 请求结束前完成的 admission work 数。 |
| `eager_update_prefill_tokens` | 每条 Update 立即 admission 的反事实 token 数。 |
| `admitted_update_prefill_tokens` | 当前策略 admission 的 Update token 数。 |
| `scheduled_update_prefill_tokens` | Scheduler 推进的 Update token；重放会重复累计。 |
| `saved_update_prefill_tokens` | `max(0, eager - admitted)`。 |
| `extra_update_prefill_tokens` | `max(0, admitted - eager)`。 |
| `update_tokens_invalidated` | 已物化、后来因新快照而失效的 Update KV token。 |

| schema v2 指标 | 含义 |
| --- | --- |
| `update_policy_works_admitted` | 策略放行的逻辑 Update work 数。 |
| `update_policy_works_completed` | 已完成的逻辑 Update work 数。 |
| `update_policy_works_cancelled_by_recovery` | 被抢占恢复路径明确取消的逻辑 work 数。 |
| `update_policy_works_outstanding` | finish 时仍未完成或取消的逻辑 work 数；正式结果必须为 0。 |
| `update_execution_attempts_started` | 物理执行尝试次数；同一逻辑 work 可因抢占多次执行。 |
| `update_execution_attempts_completed` | 正常完成的物理执行尝试次数。 |
| `update_execution_attempts_preempted` | 被抢占中断的物理执行尝试次数。 |
| `update_recovery_works_created` | 创建的恢复 work 数；可能尚未启动就被后续恢复替换。 |
| `update_tokens_scheduled_total` | 原始执行与恢复重放累计调度的 Update token。 |
| `update_tokens_scheduled_for_recovery` | 其中由恢复 work 重放的 Update token。 |
| `prefix_append_tokens_scheduled_for_recovery` | 恢复时重算的 PREFIX_APPEND token。 |
| `update_tokens_discarded_by_preemption` | 抢占时丢弃的 Update 已计算 token。 |
| `prefix_append_tokens_discarded_by_preemption` | 抢占时丢弃的 PREFIX_APPEND 已计算 token。 |
| `question_tokens_discarded_by_preemption` | 抢占时丢弃的 QUESTION 已计算 token。 |
| `update_tokens_invalidated_by_snapshot` | 新 Update 快照造成的语义失效 token，不含抢占丢弃。 |
| `prompt_stream_*preemptions/evictions/resets/failures*` | 请求压力、驱逐、prefix cache reset 和分配失败计数。 |

跨请求汇总使用：

$$
\mathrm{coalescing\_ratio}
= 1 - \frac{\sum N_{\mathrm{admitted}}}{\sum N_{\mathrm{accepted}}},
$$

$$
\mathrm{token\_saving\_ratio}
= \frac{\sum C_{\mathrm{saved}}}{\sum C_{\mathrm{eager}}}.
$$

跨请求使用总量之比，不平均请求级 ratio。分母为 0 或字段不适用时保留空值。
`scheduled_update_prefill_tokens` 是调度口径；算法节省主要比较 eager 与 admitted。

当前尚未导出 `flush_reason`、`window_wait_ms` 和 `append_gate_block_ms`，第一版实验
不从客户端时序推测这些内部指标。

### 分阶段实验计划

不测试全部参数的笛卡尔积，而是逐层筛选。

#### 阶段 A：正确性与 controller 开销

- 使用单请求或低 QPS 比较 `disabled` 和 `immediate-controller`。
- 检查成功率、最终 Update 快照和 QUESTION barrier。
- 结合 `research/scripts/simple_test` 覆盖零 token Update、截断和 source finished
  顺序。

#### 阶段 B：策略初筛

固定 QPS `1`、Update interval `0.2 0.01`、严格 Append gate 和 `20/80 ms` 窗口，
依次比较：

```text
disabled
immediate-controller
strict-immediate
fixed-window
quiet-debounce
adaptive
full-merge
```

每个 profile 单独重启 P。

#### 阶段 C：窗口筛选

对 `fixed-window` 和 `quiet-debounce` 分别测试：

| `SOFT_WINDOW_MS` | `HARD_DEADLINE_MS` |
| ---: | ---: |
| 5 | 20 |
| 10 | 40 |
| 20 | 80 |
| 40 | 160 |
| 80 | 320 |

每一行都是新的 Engine 配置，都要重启 P。选出合理 soft window 后，再单独改变 hard
或 `TTFT_DEADLINE_MS`。

#### 阶段 D：Adaptive 敏感性

- EWMA alpha 同步测试 `0.1 / 0.25 / 0.5`。
- `COLD_START_PRIOR_WEIGHT` 测试 `0 / 4 / 8`。
- `DELAY_PRICE` 测试 `0.5 / 1 / 2`。
- `LOAD_PRICE_SCALE` 先比较 `0` 与一个非零值。
- arrival/gain prior 只比较 `null` 与从调优集得到的一组先验。

一次只改变一组参数，P 启动与 replay 审计必须使用相同值。

#### 阶段 E：主负载实验

选出 `disabled`、最佳 fixed/debounce、最佳 Adaptive 和 `full-merge`，再运行完整
QPS 和 Update interval sweep，用于论文主表与主图。

#### 阶段 F：Append gate 消融

固定胜出 policy 及参数，只改变 `APPEND_GATE`。三个 gate 对应三个独立 P 启动。

## 实验执行流程

下面给出从启动 P 到生成图表的完整操作。命令均从仓库根目录运行。

### 准备公共变量

```shell
REPO_ROOT=/path/to/vllm
cd "${REPO_ROOT}"
mkdir -p research/scripts/logs

export P_NODE=g2c9
export MODEL_PATH=/workplace/home/huangxinying/Models/Qwen2.5-3B
export SERVED_MODEL_NAME=qwen2.5-3b

# 必须是所有 Slurm replay 节点都能访问的 proxy 地址。
export PROXY_HOST=g2c14
export PROXY_URL="http://${PROXY_HOST}:8000"
```

不要在 replay sbatch 中使用 `http://127.0.0.1:8000`，除非 replay job 与 proxy
确定在同一节点。Slurm 通常把 replay 分配到其它节点，此时 localhost 指向错误机器。

P 节点启用合并时要求 DP=1、单 in-flight batch、关闭 async scheduling，并使用
NIXL pull producer。通用 P 启动脚本已传入 `--no-async-scheduling`。D 节点不要
设置合并配置。

### 按策略启动 P 节点

以下命令使用上面的公共变量，每次只启动一个 profile。

每个小节先给出 P 启动命令，再给出 P ready 后参数匹配的正式 sweep 命令。
下面统一使用 512 个请求、4 个 QPS、3 个 Update interval 和 3 次重复；
这些 workload 参数可以调整，但不同策略做公平对比时应保持一致。

#### `disabled`

```shell
COALESCING_PROFILE=disabled \
MODEL_PATH="${MODEL_PATH}" SERVED_MODEL_NAME="${SERVED_MODEL_NAME}" \
GPU_MEMORY_UTILIZATION=0.6 MAX_MODEL_LEN=4096 MAX_NUM_SEQS=16 \
sbatch --nodelist="${P_NODE}" \
  research/scripts/replay/update_coalescing/start_p_node_coalescing.sbatch
```

P ready 后，对应的 sweep 命令为：

```shell
PROXY_URL="${PROXY_URL}" \
COALESCING_PROFILE=disabled \
MODEL="${SERVED_MODEL_NAME}" \
MAX_REQUESTS=512 \
QPS_LIST="0.5 1 2 4" \
UPDATE_INTERVAL_LIST="0.5 0.2 0.01" \
NUM_REPEATS=3 BASE_RANDOM_SEED=2026 \
MAX_CONCURRENCY=8 APPEND_CHUNK_INTERVAL=0.5 \
SOURCE_FINISH_DELAY=0 WARMUP_REQUESTS=8 \
sbatch research/scripts/replay/update_coalescing/run_policy_sweep.sbatch
```

#### `immediate-controller`

```shell
COALESCING_PROFILE=immediate-controller \
MODEL_PATH="${MODEL_PATH}" SERVED_MODEL_NAME="${SERVED_MODEL_NAME}" \
GPU_MEMORY_UTILIZATION=0.6 MAX_MODEL_LEN=4096 MAX_NUM_SEQS=16 \
sbatch --nodelist="${P_NODE}" \
  research/scripts/replay/update_coalescing/start_p_node_coalescing.sbatch
```

P ready 后，对应的 sweep 命令为：

```shell
PROXY_URL="${PROXY_URL}" \
COALESCING_PROFILE=immediate-controller \
MODEL="${SERVED_MODEL_NAME}" \
MAX_REQUESTS=512 \
QPS_LIST="0.5 1 2 4" \
UPDATE_INTERVAL_LIST="0.5 0.2 0.01" \
NUM_REPEATS=3 BASE_RANDOM_SEED=2026 \
MAX_CONCURRENCY=8 APPEND_CHUNK_INTERVAL=0.5 \
SOURCE_FINISH_DELAY=0 WARMUP_REQUESTS=8 \
sbatch research/scripts/replay/update_coalescing/run_policy_sweep.sbatch
```

#### `strict-immediate`

```shell
COALESCING_PROFILE=strict-immediate \
APPEND_GATE=hold_until_append_finished \
MODEL_PATH="${MODEL_PATH}" SERVED_MODEL_NAME="${SERVED_MODEL_NAME}" \
GPU_MEMORY_UTILIZATION=0.6 MAX_MODEL_LEN=4096 MAX_NUM_SEQS=16 \
sbatch --nodelist="${P_NODE}" \
  research/scripts/replay/update_coalescing/start_p_node_coalescing.sbatch
```

P ready 后，对应的 sweep 命令为：

```shell
PROXY_URL="${PROXY_URL}" \
COALESCING_PROFILE=strict-immediate \
APPEND_GATE=hold_until_append_finished \
MODEL="${SERVED_MODEL_NAME}" \
MAX_REQUESTS=512 \
QPS_LIST="0.5 1 2 4" \
UPDATE_INTERVAL_LIST="0.5 0.2 0.01" \
NUM_REPEATS=3 BASE_RANDOM_SEED=2026 \
MAX_CONCURRENCY=8 APPEND_CHUNK_INTERVAL=0.5 \
SOURCE_FINISH_DELAY=0 WARMUP_REQUESTS=8 \
sbatch research/scripts/replay/update_coalescing/run_policy_sweep.sbatch
```

#### `fixed-window`

```shell
COALESCING_PROFILE=fixed-window \
APPEND_GATE=hold_until_append_finished \
SOFT_WINDOW_MS=20 HARD_DEADLINE_MS=80 \
MODEL_PATH="${MODEL_PATH}" SERVED_MODEL_NAME="${SERVED_MODEL_NAME}" \
GPU_MEMORY_UTILIZATION=0.6 MAX_MODEL_LEN=4096 MAX_NUM_SEQS=16 \
sbatch --nodelist="${P_NODE}" \
  research/scripts/replay/update_coalescing/start_p_node_coalescing.sbatch
```

P ready 后，对应的 sweep 命令为：

```shell
PROXY_URL="${PROXY_URL}" \
COALESCING_PROFILE=fixed-window \
APPEND_GATE=hold_until_append_finished \
SOFT_WINDOW_MS=20 HARD_DEADLINE_MS=80 \
MODEL="${SERVED_MODEL_NAME}" \
MAX_REQUESTS=512 \
QPS_LIST="0.5 1 2 4" \
UPDATE_INTERVAL_LIST="0.5 0.2 0.01" \
NUM_REPEATS=3 BASE_RANDOM_SEED=2026 \
MAX_CONCURRENCY=8 APPEND_CHUNK_INTERVAL=0.5 \
SOURCE_FINISH_DELAY=0 WARMUP_REQUESTS=8 \
sbatch research/scripts/replay/update_coalescing/run_policy_sweep.sbatch
```

#### `quiet-debounce`

```shell
COALESCING_PROFILE=quiet-debounce \
APPEND_GATE=hold_until_append_finished \
SOFT_WINDOW_MS=20 HARD_DEADLINE_MS=80 \
MODEL_PATH="${MODEL_PATH}" SERVED_MODEL_NAME="${SERVED_MODEL_NAME}" \
GPU_MEMORY_UTILIZATION=0.6 MAX_MODEL_LEN=4096 MAX_NUM_SEQS=16 \
sbatch --nodelist="${P_NODE}" \
  research/scripts/replay/update_coalescing/start_p_node_coalescing.sbatch
```

P ready 后，对应的 sweep 命令为：

```shell
PROXY_URL="${PROXY_URL}" \
COALESCING_PROFILE=quiet-debounce \
APPEND_GATE=hold_until_append_finished \
SOFT_WINDOW_MS=20 HARD_DEADLINE_MS=80 \
MODEL="${SERVED_MODEL_NAME}" \
MAX_REQUESTS=512 \
QPS_LIST="0.5 1 2 4" \
UPDATE_INTERVAL_LIST="0.5 0.2 0.01" \
NUM_REPEATS=3 BASE_RANDOM_SEED=2026 \
MAX_CONCURRENCY=8 APPEND_CHUNK_INTERVAL=0.5 \
SOURCE_FINISH_DELAY=0 WARMUP_REQUESTS=8 \
sbatch research/scripts/replay/update_coalescing/run_policy_sweep.sbatch
```

#### `adaptive`

```shell
COALESCING_PROFILE=adaptive \
APPEND_GATE=hold_until_append_finished \
SOFT_WINDOW_MS=20 HARD_DEADLINE_MS=80 \
INTERVAL_EWMA_ALPHA=0.25 GAIN_EWMA_ALPHA=0.25 \
COLD_START_PRIOR_WEIGHT=8 DELAY_PRICE=1 LOAD_PRICE_SCALE=0 \
MODEL_PATH="${MODEL_PATH}" SERVED_MODEL_NAME="${SERVED_MODEL_NAME}" \
GPU_MEMORY_UTILIZATION=0.6 MAX_MODEL_LEN=4096 MAX_NUM_SEQS=16 \
sbatch --nodelist="${P_NODE}" \
  research/scripts/replay/update_coalescing/start_p_node_coalescing.sbatch
```

如需冷启动先验，在 P 启动和 replay 两边同时增加：

```shell
ARRIVAL_RATE_PRIOR_PER_S=20 \
MERGE_GAIN_PRIOR_TOKENS=256
```

P ready 后，对应的 sweep 命令如下。EWMA、prior 和 price 参数必须与上面的 P 启动
命令一致：

```shell
PROXY_URL="${PROXY_URL}" \
COALESCING_PROFILE=adaptive \
APPEND_GATE=hold_until_append_finished \
SOFT_WINDOW_MS=20 HARD_DEADLINE_MS=80 \
INTERVAL_EWMA_ALPHA=0.25 GAIN_EWMA_ALPHA=0.25 \
COLD_START_PRIOR_WEIGHT=8 DELAY_PRICE=1 LOAD_PRICE_SCALE=0 \
MODEL="${SERVED_MODEL_NAME}" \
MAX_REQUESTS=512 \
QPS_LIST="0.5 1 2 4" \
UPDATE_INTERVAL_LIST="0.5 0.2 0.01" \
NUM_REPEATS=3 BASE_RANDOM_SEED=2026 \
MAX_CONCURRENCY=8 APPEND_CHUNK_INTERVAL=0.5 \
SOURCE_FINISH_DELAY=0 WARMUP_REQUESTS=8 \
sbatch research/scripts/replay/update_coalescing/run_policy_sweep.sbatch
```

#### `full-merge`

```shell
COALESCING_PROFILE=full-merge \
APPEND_GATE=hold_until_append_finished \
MODEL_PATH="${MODEL_PATH}" SERVED_MODEL_NAME="${SERVED_MODEL_NAME}" \
GPU_MEMORY_UTILIZATION=0.6 MAX_MODEL_LEN=4096 MAX_NUM_SEQS=16 \
sbatch --nodelist="${P_NODE}" \
  research/scripts/replay/update_coalescing/start_p_node_coalescing.sbatch
```

P ready 后，对应的 sweep 命令为：

```shell
PROXY_URL="${PROXY_URL}" \
COALESCING_PROFILE=full-merge \
APPEND_GATE=hold_until_append_finished \
MODEL="${SERVED_MODEL_NAME}" \
MAX_REQUESTS=512 \
QPS_LIST="0.5 1 2 4" \
UPDATE_INTERVAL_LIST="0.5 0.2 0.01" \
NUM_REPEATS=3 BASE_RANDOM_SEED=2026 \
MAX_CONCURRENCY=8 APPEND_CHUNK_INTERVAL=0.5 \
SOURCE_FINISH_DELAY=0 WARMUP_REQUESTS=8 \
sbatch research/scripts/replay/update_coalescing/run_policy_sweep.sbatch
```

#### `custom`

```shell
COALESCING_PROFILE=custom \
PROMPT_STREAM_UPDATE_COALESCING_CONFIG='{"append_gate":"none","update_policy":"fixed_window","soft_window_ms":10,"hard_deadline_ms":40}' \
MODEL_PATH="${MODEL_PATH}" SERVED_MODEL_NAME="${SERVED_MODEL_NAME}" \
GPU_MEMORY_UTILIZATION=0.6 MAX_MODEL_LEN=4096 MAX_NUM_SEQS=16 \
sbatch --nodelist="${P_NODE}" \
  research/scripts/replay/update_coalescing/start_p_node_coalescing.sbatch
```

P ready 后，对应的 sweep 命令如下。`PROMPT_STREAM_UPDATE_COALESCING_CONFIG` 必须与
上面的 P 启动命令完全相同：

```shell
PROXY_URL="${PROXY_URL}" \
COALESCING_PROFILE=custom \
PROMPT_STREAM_UPDATE_COALESCING_CONFIG='{"append_gate":"none","update_policy":"fixed_window","soft_window_ms":10,"hard_deadline_ms":40}' \
MODEL="${SERVED_MODEL_NAME}" \
MAX_REQUESTS=512 \
QPS_LIST="0.5 1 2 4" \
UPDATE_INTERVAL_LIST="0.5 0.2 0.01" \
NUM_REPEATS=3 BASE_RANDOM_SEED=2026 \
MAX_CONCURRENCY=8 APPEND_CHUNK_INTERVAL=0.5 \
SOURCE_FINISH_DELAY=0 WARMUP_REQUESTS=8 \
sbatch research/scripts/replay/update_coalescing/run_policy_sweep.sbatch
```
### 确认 P 已就绪

```shell
P_JOB_ID=1234
squeue -j "${P_JOB_ID}"
tail -f "research/scripts/logs/vllm-p-coalescing-${P_JOB_ID}.out"
```

日志开头应显示 `COALESCING_PROFILE` 和最终 JSON。等待 server ready 后再提交 replay。
如果 P hostname 改变，要让 proxy 的 `--prefiller-hosts` 指向新节点；P 始终使用相同
hostname 和端口时，proxy 不需要因 profile 改变而重启。

### 先跑小规模实验

以下命令假定当前 P 是 `fixed-window 20/80 ms`：

```shell
PROXY_URL="${PROXY_URL}" \
COALESCING_PROFILE=fixed-window \
APPEND_GATE=hold_until_append_finished \
SOFT_WINDOW_MS=20 HARD_DEADLINE_MS=80 \
QPS_LIST="1" UPDATE_INTERVAL_LIST="0.01" \
MAX_REQUESTS=64 NUM_REPEATS=1 WARMUP_REQUESTS=0 \
sbatch research/scripts/replay/update_coalescing/run_policy_sweep.sbatch
```

profile、gate 和 window 必须与当前 P 完全一致。查看 replay 日志：

```shell
REPLAY_JOB_ID=1235
squeue -j "${REPLAY_JOB_ID}"
tail -f "research/scripts/logs/replay-coalescing-${REPLAY_JOB_ID}.out"
```

成功结果应满足：

- `num_first_token_success == num_requests`，或至少达到预设首 token 成功率阈值；
  TTFT 仅使用这些请求。
- `token_stats_coverage` 达到实验要求；token/work 结论只使用 schema v2
  `stats_validation_status=valid` 的请求。
- 每个正式请求满足 `admitted = completed + cancelled_by_recovery + outstanding`，
  且 finish 时 `outstanding == 0`。
- 每个正式请求满足 `attempts_started = attempts_completed + attempts_preempted`。
- `update_tokens_scheduled_for_recovery <= update_tokens_scheduled_total`。恢复 work
  可能尚未启动就被替换，因此不要求 `recovery_works_created <= attempts_started`。
- P 实际配置与 replay 审计配置一致。
- run 目录存在 `config.json`、`request_metrics.csv`、`event_metrics.csv` 和
  `summary.json`。

### 为当前 P 跑正式矩阵

仍以 `fixed-window 20/80 ms` 为例：

```shell
PROXY_URL="${PROXY_URL}" \
COALESCING_PROFILE=fixed-window \
APPEND_GATE=hold_until_append_finished \
SOFT_WINDOW_MS=20 HARD_DEADLINE_MS=80 \
MODEL=qwen2.5-3b MAX_REQUESTS=512 \
QPS_LIST="0.5 1 2 4" \
UPDATE_INTERVAL_LIST="0.5 0.2 0.01" \
NUM_REPEATS=3 BASE_RANDOM_SEED=2026 \
MAX_CONCURRENCY=8 APPEND_CHUNK_INTERVAL=0.5 \
SOURCE_FINISH_DELAY=0 WARMUP_REQUESTS=8 \
sbatch research/scripts/replay/update_coalescing/run_policy_sweep.sbatch
```

这条命令运行 $4 \times 3 \times 3=36$ 个正式 run。每个 run 同时记录 TTFT、
成功率、Update work、saved token 和 invalidated token。

### 停止 P 并切换策略

当前 profile 的 replay 全部完成后：

```shell
scancel "${P_JOB_ID}"
```

然后重新执行“启动 P → 等待 ready → 小规模检查 → 正式矩阵”。P 启动和 replay
必须使用同一个 profile 及参数，不能只修改输出目录标签。

### 批量回放参数

| 环境变量 | 默认值 | 含义 |
| --- | ---: | --- |
| `PROXY_URL` | 无 | 必填；所有 replay 节点都能访问的 proxy URL。 |
| `DATA_DIR` | `research/data/mix_source` | mixed-source 数据目录。 |
| `OUTPUT_DIR` | `.../update_coalescing/results` | 正式结果根目录。 |
| `MODEL` | `qwen2.5-3b` | REST 请求中的 served model。 |
| `MAX_REQUESTS` | `256` | 每个 run 的样本数。 |
| `QPS_LIST` | `1` | 空格分隔的逻辑请求到达率。 |
| `UPDATE_INTERVAL_LIST` | `0.2 0.01` | Update source 内事件间隔，单位秒。 |
| `NUM_REPEATS` | `3` | 每个 QPS/interval 点的重复次数。 |
| `BASE_RANDOM_SEED` | `2026` | 第一次 Poisson seed；后续依次加 1。 |
| `MAX_CONCURRENCY` | `8` | 客户端同时在途的请求上限。 |
| `APPEND_INITIAL_DELAY` | `0` | Append 首事件延迟。 |
| `UPDATE_INITIAL_DELAY` | `0` | Update 首事件延迟。 |
| `APPEND_CHUNK_INTERVAL` | `0.5` | Append source 内 chunk 间隔。 |
| `SOURCE_FINISH_DELAY` | `0` | 最后数据事件 ACK 后发送 source finished 的延迟。 |
| `MAX_TOKENS` | `64` | 普通生成上限；TTFT-only 在 P 首 token 后结束。 |
| `WARMUP_REQUESTS` | `0` | 正式矩阵前的 warmup 数；正式采集建议 `8`。 |
| `EVENT_PROGRESS_LOG_INTERVAL` | `0` | 每 N 个 event 打印进度；`0` 关闭。 |
| `FAIL_FAST` | `0` | 设为 `1` 后首个 HTTP/SSE 或首 token 失败即结束；stats invalid 不触发。 |
| `FORMAL_FAIL_FAST` | `0` | 设为 `1` 后首个 formal-invalid 结果即结束；默认关闭。 |
| `EXTRA_REPLAY_ARGS` | 空 | 追加给 Python replay 的简单参数。 |

warmup 使用矩阵中的第一个 Update interval，写入 `results/warmup/`，汇总默认跳过。

## 结果文件与检查

每个 run 目录包含：

- `config.json`：replay 参数、P 配置审计、样本列表和输入内容指纹。
- `request_metrics.csv`：请求级 TTFT、成功状态、work/token 和派生比率。
- `event_metrics.csv`：事件计划时间、发送时间、ACK 延迟、token 数和 Update LCP。
- `summary.json`：当前 run 的 TTFT 分位数和全局 token/work 总量。
- `responses/`：仅在显式使用 `--save-responses` 时生成。

HTTP/SSE 或首 token 失败才会把网络结果标记为错误并触发必要的 abort。已经收到首
token 后，服务端 stats 缺失、类型错误、recovery-aware 守恒不一致、coalescing
状态与预期不一致，或 P 实际配置与审计 JSON 不一致，只会把 stats/formal 状态标为
无效：该请求仍进入 TTFT 分布，不再 abort；其 token/work 数据不会填成 0，也不会
进入正式 token 聚合。schema v1 只作为 legacy 读取，formal 状态为 `not_evaluated`。

## 汇总与绘图

```shell
.venv/bin/python \
  research/scripts/replay/update_coalescing/summarize_coalescing.py \
  --input-root research/scripts/replay/update_coalescing/results \
  --output-dir research/scripts/replay/update_coalescing/results/analysis
```

汇总只池化 `workload_id`、P 配置、QPS 和 Update interval 相同的重复实验；不同
样本、Append 时序、并发度或模型不会被静默混合。

```shell
.venv/bin/python \
  research/scripts/replay/update_coalescing/plot_coalescing_tradeoff.py \
  --summary-csv \
    research/scripts/replay/update_coalescing/results/analysis/configuration_summary.csv \
  --output \
    research/scripts/replay/update_coalescing/results/analysis/coalescing_tradeoff.png \
  --ttft-percentile p95
```

四联图适合观察一个实验切片中的 TTFT、节省 token、失效 token 和 Pareto
trade-off。若要同时比较不同 Update interval 和 QPS 下的策略表现，运行：

```shell
.venv/bin/python \
  research/scripts/replay/update_coalescing/plot_strategy_matrix.py \
  --summary-csv \
    research/scripts/replay/update_coalescing/results/analysis/configuration_summary.csv \
  --output \
    research/scripts/replay/update_coalescing/results/analysis/coalescing_strategy_matrix.png \
  --ttft-percentile p95 \
  --baseline-profile disabled
```

策略矩阵的上半部分显示所选 TTFT 分位数，并以 `disabled` 计算每个单元格中的
相对变化；下半部分显示 Update Prefill token saving。橙色边框表示同一
`(Update interval, QPS)` 列中表现最好的可靠策略，灰色表示缺失、不适用或未达到
可靠性阈值。策略矩阵要求输入中只包含一个 `workload_id`；若汇总结果含有多个
workload，必须通过 `--workload-id <WORKLOAD_ID>` 选择一个。

两个绘图脚本均可用 `--qps`、`--append-chunk-interval`、
`--update-chunk-interval`、`--profile` 和 `--workload-id` 过滤数据。需要保留多个值
时重复传入同一个参数，例如 `--qps 1 --qps 2 --qps 4`。绘图默认同时要求首 token
成功率和 recovery-safe stats 覆盖率至少为 `0.95`。
只有分析 overload 或旧结果时，才分别使用 `--min-success-rate 0` 或
`--min-stats-coverage 0` 放宽阈值；这类图不能作为正式 token trade-off 结论。

`plot_coalescing_tradeoff.py` 只有在过滤后剩下一个
`(workload_id, QPS, Update interval)` 切片时才计算 Pareto frontier。不同 offered
load 不会互相计算支配关系。`disabled` 的 eager/saved 字段为空，因此只进入 TTFT
和 invalidated 图。

## 推荐执行顺序

1. 用 `MAX_REQUESTS=64`、QPS `1`、Update interval `0.01` 验证每个 profile。
2. 用 QPS `1`、Update interval `0.2 0.01`、每点 3 次进行策略初筛。
3. 对 `fixed-window` 和 `quiet-debounce` 筛选窗口，每个窗口重启一次 P。
4. 对 Adaptive 做少量 EWMA 和 price 参数敏感性实验。
5. 选出 `disabled`、最佳 fixed/debounce、最佳 Adaptive 和 `full-merge`，运行
   `QPS_LIST="0.5 1 2 4"` 的正式负载实验。
6. 固定胜出参数，只改变 `APPEND_GATE` 做消融。
7. 汇总结果，检查成功率后绘制四联图。

每一步都遵循：

```text
启动一个 P 配置
→ 等待 P ready
→ 确认 proxy 指向该 P
→ 提交匹配配置的 replay sbatch
→ 检查结果完整性
→ 停止 P
→ 启动下一个配置
```

不要在一个存活的 P 上只修改 replay 的 profile 标签。这样目录名虽然不同，Engine
实际执行的仍是旧配置，结果不能用于策略对比。
