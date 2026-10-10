# 稳定 KV 提前 Push 实验

本目录比较两种完整 P/D 生成流程：**提前 Push 已完成的稳定 Prefix+Append 整块**，以及 **P finish 后统一 Push 全部 KV**。两组使用相同 `NixlPushConnector`、相同接收会话协议、相同模型和 Masked Padding，只改变请求里的 `early_stable_kv_push`。

本实验不能使用 `--ttft-only`。P 首 token 已经能够由 proxy 提前返回，本功能主要缩短它之后 D 开始续写的交接间隔。设计依据见 [Early-Stable-KV-Push.md](../../../Early-Stable-KV-Push.md)。

本轮部署、固定变量和正式结果记录见 [EXPERIMENTS.md](EXPERIMENTS.md)。

## 脚本和职责

| 文件 | 用途 |
| --- | --- |
| `replay_stable_kv_push.py` | 读取混合数据，按两个来源的 interval 发送事件，消费完整 SSE，分别记录 P 和 D 输出。 |
| `summarize_stable_kv_push.py` | 校验两组配置和数据一致，按相同样本汇总均值、P90/P95/P99 与配对差值；同时保留 P50 供诊断。 |
| `plot_stable_kv_push.py` | 画 P→D 交接时延生存曲线、提前/最终传输量和最终批次传输耗时；默认展示 P95。 |
| `summarize_stable_kv_matrix.py` | 读取各场景逐轮 `comparison.json`，汇总稳定占比/QPS 曲线；重复轮次不池化请求延迟。 |
| `run_stable_kv_sweep.sbatch` | 在 g2c14 执行多 QPS、多重复 A/B 回放，随后汇总和绘图。 |
| `run_stable_kv_matrix.sbatch` | 顺序回放多个受控稳定占比数据目录，复用 QPS sweep，并生成整个矩阵的汇总和图。 |
| [`build_stable_kv_push_dataset.py`](../../data_processing/build_stable_kv_push_dataset.py) | 按 tokenizer 构造知识总长固定、Append 占比不同的合成性能集，不覆盖原来的质量集。 |
| [`start_stable_kv_pair.sh`](../../server/start_stable_kv_pair.sh) | 一个 `srun` allocation 在 g2c9 申请两张 GPU，启动恰好一个 P 和一个 D。 |
| [`start_stable_kv_node.sh`](../../server/start_stable_kv_node.sh) | 每个 Slurm task 的实际启动命令；rank 0 是 P，rank 1 是 D。 |
| [`stable_kv_proxy.py`](../../server/stable_kv_proxy.py) | 独立实验网关；在 g2c14 建立两端会话、转发事件、完成 finalize 和完整输出。 |

数据解析、`apply_chat_template` 以及 Append/Update 的事件发送复用 `segmented_prefill`。完整输出消费与传输计量采用本目录的实现，避免把 P 首 token 当成 D 首 token。

## 本轮部署约束

| 组件 | 所在主机 | 资源和地址 |
| --- | --- | --- |
| P | `g2c9` | 一张 Slurm 分配的 GPU；OpenAI `8100`；NIXL `5600`。 |
| D | `g2c9` | 另一张 Slurm 分配的 GPU；OpenAI `8200`；NIXL `5601`。 |
| Proxy | `g2c14` | HTTP `8000`，连接 `g2c9:8100/8200`。 |
| Replay | `g2c14` | 访问 `http://g2c14:8000`；不申请 GPU。 |

P/D 使用 Qwen2.5-3B、TP=1、PP=1、DP=1、`TRITON_ATTN`、`--prompt-stream-masked-padding`、`--generation-config vllm` 和同步调度。本轮命令将显存比例显式设为 `0.90`；启动脚本未传参的默认值仍为 `0.85`。实际设备已被其它进程占用时会在启动前检查失败，不会自行换到未分配的 GPU。

两组都使用下列 connector 配置，只按角色改变 `kv_role`：

```json
{
  "kv_connector": "NixlPushConnector",
  "kv_role": "kv_producer",
  "kv_load_failure_policy": "fail",
  "kv_connector_extra_config": {
    "stable_kv_push": true
  }
}
```

D 的角色是 `kv_consumer`。`stable_kv_push=true` 打开接收、增量传输和最终收养能力；实验对照通过每个 start 请求的布尔值控制，不需要为 A/B 重启 P/D。**不要把对照组换成原来的 Pull connector 或禁用整个 KV transfer。**

本轮 P/D 在同一主机的两张卡上，NIXL 可使用本机传输机制。它能够验证实现与本机传输收益，但不能直接代表跨主机网络上的传输耗时或收益。

## 启动一次 P/D 和 Proxy

以下为当前环境可直接复制的例子，发布时可以删除本地节点和模型路径。
所有命令从 vLLM 仓库根目录执行。

先在 **g2c14 的一个终端或 tmux 窗口**启动一对 P/D：

```shell
mkdir -p research/scripts/logs

SERVER_NODE=g2c9 \
MODEL_PATH="${HOME}/Models/Qwen2.5-3B" \
SERVED_MODEL_NAME=qwen2.5-3b \
GPU_MEMORY_UTILIZATION=0.90 \
MAX_MODEL_LEN=4096 \
MAX_NUM_SEQS=64 \
PROMPT_STREAM_MASKED_PADDING=1 \
ATTENTION_BACKEND=TRITON_ATTN \
bash research/scripts/server/start_stable_kv_pair.sh
```

这个命令会保持前台运行。只启动这一对服务，已有同端口实验实例时先结束它们。
Slurm 负责 GPU 隔离，不需要也不应该手动覆盖 `CUDA_VISIBLE_DEVICES`。

日志为 `research/scripts/logs/stable-kv-pd-<job_id>-rank-0.out` 和
`stable-kv-pd-<job_id>-rank-1.out`。rank 0/1 分别为 P/D，可以通过
`squeue -u "$USER"` 找到 job id。结束时在 srun 终端按 Ctrl-C，或执行
`scancel <job_id>`，同一 allocation 内的两个服务一起释放。

确认两端都已健康，再在 **g2c14 的另一个终端**启动专用网关：

```shell
curl --fail http://g2c9:8100/health
curl --fail http://g2c9:8200/health

.venv/bin/python research/scripts/server/stable_kv_proxy.py \
  --host 0.0.0.0 \
  --port 8000 \
  --prefiller-host g2c9 \
  --prefiller-port 8100 \
  --decoder-host g2c9 \
  --decoder-port 8200
```

网关的参数是单数 `--prefiller-host` / `--decoder-host`，与旧 proxy 的复数参数不同。
它提供 `/v1/prompt_stream/start`、`/event`、`/finish`、`/abort` 和 `/health`。

启动脚本可调整的环境变量：

| 变量 | 默认值 | 含义 |
| --- | --- | --- |
| `SERVER_NODE` | `g2c9` | P/D 两张卡必须来自同一个节点。 |
| `SERVER_MEMORY` | `24G` | 两个服务共用的 Slurm 主机内存申请量，不是 GPU 显存。 |
| `SERVER_CPUS_PER_TASK` | `8` | 每个服务的 CPU 数。 |
| `SERVER_TIME` | `04:00:00` | P/D allocation 时间上限。 |
| `MODEL_PATH` | `${HOME}/Models/Qwen2.5-3B` | P/D 的模型和 tokenizer 本地目录。 |
| `SERVED_MODEL_NAME` | `qwen2.5-3b` | 客户端提交的模型名。 |
| `GPU_MEMORY_UTILIZATION` | `0.85` | 每张卡上 vLLM 的显存比例。 |
| `MAX_MODEL_LEN` | `4096` | 模板、知识、Question、物理 padding 和生成的序列长度约束。 |
| `MAX_NUM_SEQS` | `64` | 模型调度的并发序列上限，不是接收会话占用的块数上限。 |
| `P_PORT` / `D_PORT` | `8100` / `8200` | 两端 OpenAI HTTP 端口。 |
| `P_SIDE_PORT` / `D_SIDE_PORT` | `5600` / `5601` | 两端 NIXL 端口。 |

`VLLM_NIXL_SIDE_CHANNEL_HOST` 默认使用 Slurm 实际运行节点的 hostname，不需写成提交命令所在的 g2c14。

## 先验证小规模完整生成

先分别用一条请求预热两种路径；预热结果不要混入正式样本。使用新的
`--run-name` 可避免覆盖已有结果：

```shell
.venv/bin/python research/scripts/replay/stable_kv_push/replay_stable_kv_push.py \
  --proxy-url http://g2c14:8000 \
  --max-requests 1 --qps 0 --max-concurrency 1 \
  --no-early-push --run-name warmup-finish

.venv/bin/python research/scripts/replay/stable_kv_push/replay_stable_kv_push.py \
  --proxy-url http://g2c14:8000 \
  --max-requests 1 --qps 0 --max-concurrency 1 \
  --early-push --run-name warmup-early
```

随后用相同数据的四条请求验证 A/B：

```shell
for mode in finish early; do
  flag=--no-early-push
  if [ "$mode" = early ]; then flag=--early-push; fi
  .venv/bin/python research/scripts/replay/stable_kv_push/replay_stable_kv_push.py \
    --data-dir research/data/mix_source_perf_qwen \
    --proxy-url http://g2c14:8000 \
    --model qwen2.5-3b \
    --tokenizer-model "${HOME}/Models/Qwen2.5-3B" \
    --max-requests 4 \
    --qps 0 --max-concurrency 1 \
    --append-chunk-interval 0.5 --update-chunk-interval 0.2 \
    --source-finish-delay 0 \
    --max-tokens 64 --temperature 0 --seed 2026 \
    --save-responses --run-name "smoke-${mode}" "$flag"
done
```

`--max-tokens` 至少为 2，因为 P 消耗第一个输出 token，D 才有机会生成续写。
即使设为 64，模型也可能在 P 首 token 就输出 EOS；这种请求记录为完整响应，
但没有 D 首 token，不能用于交接时延比较。

## 指标如何解释

| 指标 | 时间起点和终点 / 口径 | 作用 |
| --- | --- | --- |
| `p_finish_to_d_first_token_ms` | Proxy 读完 P finish 响应，到 proxy 收到第一个真实 D 输出 token。 | **主要指标**；包含 finalize、剩余传输屏障、D 请求构造/提交、排队和首次本地执行。 |
| `d_request_to_first_token_ms` | Proxy 提交 D completions，到收到 D 首 token。 | 当前路径在 finalize 后才提交 D，因此主要反映 D 的准入、排队和计算，不应代替上一个指标。 |
| `finalize_wait_ms` | Proxy 内部 finalize 调用阶段的等待。 | 包含剩余传输、控制轮询和最终就绪检查，不等同于纯网络传输。 |
| `finish_to_p_first_client_ms` | Replay 发送 finish 到收到 P token。 | 检查提前发送是否影响 P 的 prefill / 首 token。 |
| `finish_to_d_first_client_ms` | Replay 发送 finish 到收到 D token。 | 用户侧可观察的完整续写启动时间。 |
| `final_transfer_ms` | P worker 报告的最终传输批次 NIXL duration 累计值。 | 不包含 proxy 构造和 D 排队；多批重叠时，累计值也不等于端到端等待。 |
| `early_transferred_bytes` | 归类为提前批次的实际传输字节。 | 表示工作被移到提前通道；不单独证明这些字节全部在 P finish 之前完成。 |
| `final_transferred_bytes` | 最终批次实际传输字节。 | 直接观察 finish 阶段剩余传输量是否减少。 |
| `early_byte_fraction` | `early_transferred_bytes / total_transferred_bytes`。 | 已提前分类的传输比例；不能称为已隐藏耗时比例。 |
| `received_blocks_at_finalize` | 若服务端返回，表示 D 在 finalize 时已经收齐的块数。 | 区分“已提交提前发送”与“已在最终阶段前接收”。 |
| `completed_bytes_before_p_finish` / `completed_blocks_before_p_finish` | P finish 时主线程已经观测到 WRITE 完成的字节/块数。 | 保守的提前完成水位；worker 已完成但尚未回报的不计入。 |
| `decode_tpot_ms` | D 首至末 token 的间隔，除以 D token 数减一。 | D 开始续写后的平均 token 间隔，首个 token 前的迁移等待不混入。 |
| `decode_tps` | 上述区间的 token 数减一，除以秒数。 | 单请求 D 稳态输出速率；不把 SSE chunk 数当作 token 数。 |
| `finish_to_done_ms` | Replay 发送 finish 到完整 SSE DONE。 | 最终响应耗时。 |
| `aggregate_output_tokens_per_second` | 整轮成功请求的输出 token 数除以回放总墙钟时间。 | 包含检索事件等待和到达空隙的整体吞吐，不是单请求 D TPS。 |

Proxy 时间差均在同一 proxy 进程用单调时钟计算。NIXL 时长来自 worker，
客户端时长来自 replay；**不直接相减 P、D、proxy 不同主机的时间戳**。
回放和专用 proxy 的集群内 HTTP 连接均忽略外部 `HTTP_PROXY` / `HTTPS_PROXY`，
避免模型下载代理转发这些内部地址。

SSE 会标记 `token_source=prefill/decode`。最终 metadata 帧携带
`stable_kv_metrics`，replay 一直消费到 `[DONE]`，不在 P 首 token 后关闭连接。
空文本但有 `token_ids` 的帧仍然算输出 token。

`transport_status` 表示是否完整成功，`metrics_status` 表示统计字段和实验开关
是否有效。统计缺失不会被填成 0，也不会抹掉网络已经成功的事实。正式对照需要
两组相同样本都成功、有 D 输出、统计有效；汇总同时报告被排除数量和全部请求成功率。

## 批量回放

### 先测现有混合性能集

确认小样本正确后，在 g2c14 提交：

```shell
PROXY_URL=http://g2c14:8000 \
DATA_DIR=research/data/mix_source_perf_qwen \
MODEL=qwen2.5-3b \
TOKENIZER_MODEL="${HOME}/Models/Qwen2.5-3B" \
MAX_REQUESTS=128 \
QPS_LIST="0 0.5 1 2" \
MAX_CONCURRENCY=8 \
APPEND_CHUNK_INTERVAL=0.5 \
UPDATE_CHUNK_INTERVAL=0.2 \
SOURCE_FINISH_DELAY=0 \
MAX_TOKENS=64 \
NUM_REPEATS=2 \
sbatch --nodelist=g2c14 \
  research/scripts/replay/stable_kv_push/run_stable_kv_sweep.sbatch
```

这个 sbatch 不启动服务、不申请 GPU。它在每个 QPS 上依次跑两个模式，
每轮完整结束再跑下一组；偶数重复轮次交换先后顺序。默认两轮，因此可以发现
warmup 和背景负载造成的顺序效应。P/D 的四小时默认时限不足时，应在启动它们时
调大 `SERVER_TIME`，而不是让 replay 运行到服务被 Slurm 清理。

主要回放参数：

| 参数 | 默认值 | 含义 |
| --- | --- | --- |
| `--early-push` / `--no-early-push` | 开启 | 修改 start 请求中的提前传输开关。 |
| `--data-dir` | `research/data/mix_source_perf_qwen` | 读取其 `requests/*.json`，不改原始数据。 |
| `--max-requests` | `16` | 选择多少条完整数据；0 表示全部。指定 `--query-id` 时回放指定样本。 |
| `--start-index` | `0` | 从完整样本序列的哪个位置开始。 |
| `--qps` | `0` | 逻辑请求的目标到达率；0 为一个完成再发下一个。 |
| `--arrival-process` | `poisson` | 正 QPS 时采用泊松或 `fixed` 固定到达。 |
| `--max-concurrency` | `8` | 从 start 到完整输出的最大在途逻辑请求数，排队延迟单独统计。 |
| `--append-chunk-interval` / `--update-chunk-interval` | `0.5` / `0.2` 秒 | 两个来源内部目标事件间隔。来源并发，同一来源逐次等待 ACK。 |
| `--append-initial-delay` / `--update-initial-delay` | `0` 秒 | start ACK 后首事件的延迟。 |
| `--source-finish-delay` | `0` 秒 | 该来源最后一次数据 ACK 后发 source_finished 的延迟。 |
| `--max-tokens` | `64` | P 与 D 合计输出上限，D 获得扣除 P 首 token 后的余额。 |
| `--http-timeout` | `120` 秒 | HTTP 各网络阶段的超时。 |
| `--finish-timeout` | `180` 秒 | 覆盖 finish 提交、等待 headers、完整 SSE 的总期限。 |
| `--seed` / `--random-seed` | `2026` / `2026` | 分别控制采样和泊松到达序列。 |
| `--save-responses` | 关闭 | 在 JSONL 保存原始 SSE，便于核对 P/D token 和服务端统计。 |
| `--dry-run` | 关闭 | 只选择和渲染样本，不发网络请求。 |

结果目录包含 `config.json`、逐条 flush 的 `requests.jsonl`、
`request_metrics.csv` 和 `summary.json`。进程中断时 JSONL 保存已完成部分；
正式汇总会拒绝缺失选中请求的未完成实验，避免只统计快请求。

### 构造稳定占比受控的性能集

为了区分“稳定内容少”和“传输机制没有起效”，另建合成性能集，不覆盖
`mix_source_perf_qwen` 或 SQuAD 质量集。默认总知识长度为 2048 tokens：

| 场景目录 | Append tokens | 最终 Update tokens | 稳定知识占比 |
| --- | ---: | ---: | ---: |
| `append-0256-update-1792` | 256 | 1792 | 12.5% |
| `append-1024-update-1024` | 1024 | 1024 | 50.0% |
| `append-1792-update-0256` | 1792 | 256 | 87.5% |

三个场景都固定 Append 为 2 个 chunk、Update 为 5 个完整快照，相邻 Update
的 source-local LCP 比例为 0.75。生成器用真实 tokenizer 验证事件文本的编码
长度，避免“字符数近似等于 token 数”。这里的内容用于性能控制，不用于标准答案
正确率测试。各个占比的 P 计算代价仍不同，因此只在每个场景内部对比相同输入的 A/B。

从仓库根目录生成；同名目录已有数据时脚本拒绝覆盖，确需重建才加 `--overwrite`：

```shell
.venv/bin/python \
  research/scripts/data_processing/build_stable_kv_push_dataset.py \
  --tokenizer-model "${HOME}/Models/Qwen2.5-3B" \
  --output-dir research/data/stable_kv_push_controlled_qwen \
  --num-requests 128 \
  --knowledge-tokens 2048 \
  --append-tokens 256 1024 1792 \
  --append-chunks 2 \
  --update-snapshots 5 \
  --update-lcp-ratio 0.75 \
  --seed 2026
```

### 为三个稳定占比跑同一组 QPS

继续使用上面在 g2c9 启动的同一对 P/D 和 g2c14 的 proxy，不同时启动第二对服务。
矩阵脚本只申请 g2c14 的 CPU，不申请 GPU；逐场景、逐 QPS、逐 A/B 完整执行，
不会让不同实验条件同时给 P/D 施加负载：

```shell
PROXY_URL=http://g2c14:8000 \
DATA_ROOT=research/data/stable_kv_push_controlled_qwen \
SCENARIO_DIRS="append-0256-update-1792 append-1024-update-1024 append-1792-update-0256" \
MODEL=qwen2.5-3b \
TOKENIZER_MODEL="${HOME}/Models/Qwen2.5-3B" \
MAX_REQUESTS=128 \
QPS_LIST="0 1 2" \
NUM_REPEATS=2 \
WARMUP_REQUESTS=8 \
MAX_CONCURRENCY=8 \
APPEND_INITIAL_DELAY=0 \
UPDATE_INITIAL_DELAY=0 \
APPEND_CHUNK_INTERVAL=0.1 \
UPDATE_CHUNK_INTERVAL=0.2 \
SOURCE_FINISH_DELAY=0 \
MAX_TOKENS=64 \
RUN_PREFIX=stable-ratio-qwen \
sbatch --nodelist=g2c14 \
  research/scripts/replay/stable_kv_push/run_stable_kv_matrix.sbatch
```

Append 两个 chunk 在约 0.1 秒内到达，五个 Update 快照覆盖约 0.8 秒；这给
稳定 KV 留出较明确的提前传输窗口。但实际完成比例仍以水位指标为准，而不是
用设置的 interval 推断“肯定已经传完”。改两个 `*_INITIAL_DELAY` 可进一步控制
哪个来源先开始；想只测一个比例，就只保留 `SCENARIO_DIRS` 中一个目录名。

脚本按每轮顺序交换 A/B，预热不纳入正式样本。最后自动写入本轮独立分析根
`results/analysis/<RUN_PREFIX>/` 并调用矩阵汇总；重跑时更换 `RUN_PREFIX`，
避免混入旧实验或覆盖现有结果。P/D 的时限应覆盖整个矩阵所需时间。

## 汇总与画图

正式分析以 **P90/P95** 为主。每轮保留样本量和 P99，但 128 条样本的 P99
主要由最慢的少数请求决定，只用于探索性观察，不据此宣称稳定的极端尾部收益。
失败率必须和尾部分位数一起解释；成功配对的低延迟不代表失败请求的体验更好。

小规模对照的命令：

```shell
.venv/bin/python research/scripts/replay/stable_kv_push/summarize_stable_kv_push.py \
  --early-run research/scripts/replay/stable_kv_push/results/smoke-early \
  --finish-only-run research/scripts/replay/stable_kv_push/results/smoke-finish \
  --output-dir research/scripts/replay/stable_kv_push/results/analysis/smoke

.venv/bin/python research/scripts/replay/stable_kv_push/plot_stable_kv_push.py \
  --early-run research/scripts/replay/stable_kv_push/results/smoke-early \
  --finish-only-run research/scripts/replay/stable_kv_push/results/smoke-finish \
  --percentile p95 \
  --output research/scripts/replay/stable_kv_push/results/analysis/smoke/comparison.png
```

批量 sbatch 会自动对每个 QPS/重复轮次执行以上两个脚本。汇总使用原始样本
SHA256、模型模板哈希、采样和事件参数校验两组是否可比。

三张图分别展示交接时延生存曲线、平均传输字节拆分、最终批次传输 P95 耗时。
生存曲线图例同时标出 P90/P95；`--percentile p90` 或 `p99` 可切换第三张图。
`comparison.json` 和 `metrics.csv` 还提供 D TPS/TPOT 等指标。
配对改变量按 `early - finish_only` 计算，所以耗时为负表示提前传输较快。
逐请求差值的 P95 与“两组延迟 P95 之差”是不同统计量，不能混用。

### 多占比、多 QPS、重复轮次的总图

受控数据每个目录保存 `dataset_summary.json`，回放会把其中的场景参数保存进
`config.json`。占比是 `Append / (Append + 最终 Update)`，不含通用模板和
Question；实际提前传输量则包含满足整块要求的模板与 Append，所以两种占比
不应混称。旧数据没有受控场景元数据时，仍可画单对 A/B，但矩阵图会明确记录
跳过理由，而不是从文件夹名称猜测实际长度。

每个 QPS/重复轮次先生成上述 `comparison.json`，然后执行：

```shell
.venv/bin/python \
  research/scripts/replay/stable_kv_push/summarize_stable_kv_matrix.py \
  --analysis-dir research/scripts/replay/stable_kv_push/results/analysis/stable-ratio-qwen \
  --output-dir research/scripts/replay/stable_kv_push/results/analysis/stable-ratio-qwen/matrix \
  --percentile p95
```

要另画 P90，只需改输出目录和 `--percentile`，不会重跑服务或回放：

```shell
.venv/bin/python \
  research/scripts/replay/stable_kv_push/summarize_stable_kv_matrix.py \
  --analysis-dir research/scripts/replay/stable_kv_push/results/analysis/stable-ratio-qwen \
  --output-dir research/scripts/replay/stable_kv_push/results/analysis/stable-ratio-qwen/matrix-p90 \
  --percentile p90
```

输出包含：

- `per_run.csv`：每轮各模式的 P90/P95/P99、样本量、成功/正式统计数量和比例。
- `matrix_summary.csv`：相同占比/QPS/时序下，各轮分位数的中位数及最小/最大值。
- `paired_tail_changes.csv`：每轮交接、用户 finish→D 续写 token、完整响应、
  最终传输、P 首 token 的两组 P90/P95/P99
  之差及相对降幅；`early_minus_finish_ms < 0` 和 `reduction_percent > 0`
  表示提前传输更快。它与逐请求延迟差值的 P95 不同。
- `matrix_summary.json`：比较文件、受控配置、排除理由与图路径的完整清单。
- `matrix-<配置标识>-p95.png`：横坐标为稳定知识占比；颜色表示 QPS，实线/虚线
  表示提前 Push/finish-only。八个面板依次为交接 P95、用户 finish→D 续写 token
  P95、最终传输 P95、最终字节数、P finish 前实际已完成字节比例、P 首 token
  P95、全量请求成功率、用户 finish→完整响应 P95。

线表示**逐轮估计的中位数**，阴影仅为重复轮次最小/最大范围，不是置信区间。
不把两轮 128 条请求拼成 256 条再计算 P99。均值和分位数缺失时保留空值，不补零；
不同模型、时序、并发上限或总知识长度会生成不同配置图，避免曲线混入其它变量。
跨占比原始时延还受到 Update 长度与重算量变化影响；判断传输优化应优先看
同占比、同 QPS 的配对降幅，以及实际完成水位和最终传输量，不把不同占比间的
原始时延差全部归因于提前 Push。
**客户端真正收到的首 token 来自 P**，不是 D。图中的用户 finish→D 指标是
客户端从提交 finish 到收到第一个 D 续写 token 的实际间隔，并不等同于常规 TTFT。
它同时反映 P finish 之前的等待和之后的交接：提前发送可能减少交接却拖慢 P，
因此必须读取这个完整区间的实际分位数，不能把 P 首 token P95 和交接 P95 相加。

输出 token 一致性也会统计，但不能只凭两次整模型输出相同就证明传输正确：
两条来源仍存在竞争，传输开销可能改变其相对调度时刻。传输正确性还应核对
物理块对应、有效范围、最终 manifest、代次和掩码，以及接收未齐时禁止 decode。

## 实验结论的边界

先在 QPS=0 和低并发验证稳定字节确实提前接收，且最终传输量减少；再提高 QPS
检查交接 P95、成功率及整体完成时间。提前接收会更早占用 D 显存，也会增加控制
消息和 NIXL 提交次数，因此不预设所有负载都更快。

若 `final_transferred_bytes` 减少而 `p_finish_to_d_first_token_ms` 没改善，应继续
分解 finalize 等待、控制轮询、D 排队和计算；不能把字节减少直接当作同等比例
的 TTFT 改善。抢占、连接错误或不同数据造成的失败也必须保留在成功率分母中。
