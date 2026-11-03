# Replay 请求负载与计时

所有 replay 使用 `common/runner.py` 的同一个请求调度器。负载单位是**逻辑请求**：普通请求的一次完整 prompt，或流式请求从 start、多个 Append/Update 到 finish 的整个生命周期。HTTP keep-alive 连接、来源事件数和服务端 running 数都不等于这里的并发数。

## 两种负载模式

| 参数 | 闭环 `closed_loop` | 开环 `open_loop` |
| --- | --- | --- |
| `qps` | 必须为 0 | 必须大于 0，表示逻辑请求的计划到达速率 |
| `max_concurrency` | 维持的在途逻辑请求上限；完成后补入下一条 | 不参与限流 |
| `max_inflight` | 必须为 0 | 可选客户端保护上限；默认 0，不限 |
| `arrival_process` | 不生效 | `fixed` 固定间隔，`poisson` 指数分布间隔 |
| `inter_request_delay` | 可选的每槽位完成后空闲时间 | 不生效 |

`load_mode=auto` 是兼容默认值：正 QPS 自动选择开环，零 QPS 自动选择闭环。解析后的有效模式会写入配置；显式模式与 QPS 矛盾时直接报错。

闭环最多同时执行 C 个逻辑请求，最后一批不足 C 条时自然下降。开环预先根据 QPS 和 `random_seed` 生成每个请求的绝对到达时间，发起下一条不等待前一条完成。每请求内部的 Append/Update 节奏仍由原有事件间隔控制。

开环默认没有客户端槽位和 HTTP 连接池并发限制。显式设置 `max_inflight` 后，到期请求可以在客户端排队，**计划时间保持不变**。这时后端实际收到的请求速率可能低于计划 QPS，结果会保存这一差异。到期后因事件循环繁忙而迟发的请求同样保留原计划时间。

P 节点的 `max_num_seqs` 控制服务端调度，与客户端参数分别设置；本次改动只统一客户端负载。

[混合负载入口](mixed_workload/README.md) 让普通请求与流式请求共用上述 runner。QPS 和客户端上限均作用于两类请求的合计量，类型比例及普通请求 TTFT 另行保存。

## 使用示例

服务已经启动后，维持 8 个流式请求：

```shell
.venv/bin/python research/scripts/replay/query_probe/replay.py \
  --config research/scripts/replay/query_probe/performance.json \
  --load-mode closed_loop --qps 0 --max-concurrency 8
```

平均每秒到达 0.5 个请求，按 Poisson 加压：

```shell
.venv/bin/python research/scripts/replay/query_probe/replay.py \
  --config research/scripts/replay/query_probe/performance.json \
  --load-mode open_loop --qps 0.5 --arrival-process poisson
```

需要保护客户端资源时，在第二条命令后增加 `--max-inflight 64`。这些数值用于说明参数语义，实际负载范围应根据实验设置选择。normal、segmented、generation quality、token invalidation、stable KV push 使用相同参数。

Query probe 的 Slurm 矩阵支持两个独立入口，默认只启动 P 和 proxy：

```shell
# 闭环并发矩阵。
sbatch --nodelist=g2c9 research/scripts/replay/query_probe/run_performance.sbatch \
  --concurrency 4 8 16 --max-requests 1000

# 开环 QPS 矩阵；每个 QPS 分别跑 dense 和 sparse。
sbatch --nodelist=g2c9 research/scripts/replay/query_probe/run_performance.sbatch \
  --qps 0.25 0.5 1 --arrival-process poisson --max-requests 1000
```

节点按提交时的资源情况选择。QPS 列表存在时矩阵只扫描 QPS，不与默认 concurrency 列表做笛卡尔积。`--max-num-seqs` 单独控制该 launcher 的 P 调度上限，默认 32。质量全链路 launcher 用环境变量 `MAX_NUM_SEQS` 控制，默认 4。

既有 sbatch 入口可传 `LOAD_MODE` 和 `MAX_INFLIGHT`；不传时分别使用 `auto` 和 0。Query probe 的专用矩阵直接使用上述 CLI 参数。

## 结果与时钟

每个实际回放目录新增两份文件：

- `load_requests.jsonl`：逐逻辑请求的计划、准入、首个 HTTP 发送、首响应和完成时间。每条完成或取消后立即保存，按 `sample_index` 对齐输入清单。
- `load_summary.json`：负载模式、配置速率、实际发送速率、成功/失败/取消计数、最大在途数、积压和延迟分布。中途异常退出也保存已经观测到的结果。

各实验的主摘要嵌入 `load_summary`；逐请求 JSON 的 `load` 字段保存同一记录。公共指标 CSV 展开主要延迟列。Query probe 预热的负载记录位于 `warmup/`，正式记录位于 run 根目录，两个时钟互相独立。

**发送时刻**取 HTTP 客户端开始处理该逻辑请求的首个 start/completions 请求的 request hook。后续来源事件、finish 和 abort 不重复计数。这是客户端的 HTTP 提交时刻，包含之后的连接建立、网络和服务端处理，不宣称测到了服务器收到字节的时刻。

| 字段 | 定义 |
| --- | --- |
| `planned_qps` | 开环计划到达速率；闭环为 null |
| `actual_send_qps` | 实际首个 HTTP 发送时刻的平均间隔倒数，即 `(N-1)/(最后发送-首次发送)`；不足两条为 null |
| `actual_interarrival_ms` | 实际发送间隔分布，观察突发和追赶发送 |
| `successful_requests_per_second` | 成功且实际发送的请求数除以整轮耗时，包含末尾排空阶段 |
| `max_inflight` | 已准入但未完成的逻辑请求峰值，包含客户端准备与服务端等待 |
| `max_client_backlog` | 已到计划时间、尚未提交首个 HTTP 的请求峰值，包含准入等待和客户端准备 |
| `client_backlog_at_last_arrival` | 最后一条计划到达时的客户端积压；未到该时刻就终止则为 null |
| `arrival_lag_ms` | 到达协程唤醒比计划时间晚多少 |
| `client_queue_ms` | 唤醒后等待客户端准入的时间 |
| `queue_delay_ms` | 上述两项之和，保留既有字段口径 |
| `client_preparation_ms` | 准入后至 HTTP 提交，包括 prompt 构造 |
| `dispatch_lag_ms` | 计划到达至 HTTP 提交，包含全部客户端迟发 |
| `arrival_to_first_token_ms` | 计划到达至本实验首响应，包含客户端等待 |
| `sent_to_first_token_ms` | 实际 HTTP 提交至本实验首响应 |
| `arrival_to_done_ms` / `sent_to_done_ms` | 相同起点至逻辑请求执行结束 |

完整生成的首响应是 D 的首个回答 token；P-only 的首响应是 `prefill_ready`，由 `ttft_origin` 标识。不存在的响应时间保留 null；失败和取消仍保留在计数中。延迟分布使用实际观测到的相应时刻，不能仅凭成功请求延迟判断过载时的服务能力。

已有 `request_to_first_token_ms`、`finish_to_first_token_ms` 保留原计时起点。开环主对比使用新增的 `arrival_to_first_token_ms`，并同时查看实际 QPS、客户端积压和成功率。Query probe 的 QPS 图分别展示从计划到达、从 HTTP 发送到 P 完成的延迟；闭环旧图继续使用原指标。

## 与历史运行的区别

- 以前部分入口在 `qps=0` 时串行，现在所有入口都遵守 `max_concurrency`。复现串行实验时显式设为 1。
- 以前正 QPS 还受 `max_concurrency` 限制，现在默认独立发送。复现旧客户端上限时，把原并发数传给 `max_inflight`。
- HTTP 连接数不再作为开环的隐含限流器；闭环与有保护上限的开环会为并行来源事件预留连接容量。
- 历史结果缺少上述新计时字段，保留原解释，不把旧结果补成零等待或与新开环延迟混算。
