# Mixed-source REST 回放脚本

本目录提供两个相互独立的实验入口。它们读取同一批
`research/data/mix_source/requests/*.json`，但使用不同的 REST 请求链路：

- `replay_prompt_stream.py`：使用 mixed-source 流式 prefill API。
- `replay_normal.py`：把最终完整 prompt 一次性发送给普通 completions API。
- `replay_common.py`：两个入口共用的内部实现，不直接运行。

两个脚本都只配置 `--proxy-url`。普通模式不再使用 `--baseline-url`，因为
`research/scripts/proxy_server.py` 现在同时支持 prompt-stream 和普通 OpenAI
RESTFUL API。

## proxy 支持的路径

| 类型 | proxy 路径 | 后端 |
| --- | --- | --- |
| mixed-source 流式 prefill | `/v1/prompt_stream/start`、`/event`、`/finish`、`/abort` | P 节点完成流式 prefill，再把 KV 交给 D 节点 |
| 普通 completions | `/v1/completions` | 直接转发到一个 D 节点 |
| 普通 chat completions | `/v1/chat/completions` | 直接转发到一个 D 节点 |
| 普通 Responses API | `/v1/responses` | 直接转发到一个 D 节点 |

普通接口同时支持 `stream=false` JSON 响应和 `stream=true` SSE 响应。多个 D
节点由 proxy 按 round-robin 选择，因此实验客户端不需要知道具体 D 节点地址。

修改 proxy 后需要重启：

```shell
.venv/bin/python research/scripts/proxy_server.py \
  --host 0.0.0.0 \
  --port 8000 \
  --prefiller-hosts <P_HOST> \
  --prefiller-ports 8100 \
  --decoder-hosts <D_HOST> \
  --decoder-ports 8200
```

P/D 节点的完整启动方式见上一级 [scripts README](../README.md)。

## 两种模式的数据语义

### 流式 prefill

`replay_prompt_stream.py` 的请求顺序是：

1. 向 `/v1/prompt_stream/start` 发送通用模板。
2. 为每个来源创建独立异步任务。
3. Append 来源按 `sequence_no` 发送单调追加 chunk。
4. Update 来源按 `sequence_no` 发送完整快照。
5. 同一来源等待上一条 event 的 ack；不同来源之间并发竞争。
6. 所有来源的 `source_finished` 都收到 ack 后，才发送全局
   `prompt_finished`。
7. 从 `/finish` 请求发出到收到第一个 SSE token 的时间记为 TTFT。

正常启用 KV transfer 时，第一个 SSE token 来自 P 节点完成最终 prefill 后
生成的那个 token，结果中的 `ttft_origin` 通常为 `prefill`。

### 普通整段 prompt

`replay_normal.py` 不发送中间 event。它从数据文件构造：

```text
通用模板 + 所有 Append chunk + 每个 Update 来源的最终快照 + 最终用户问题
```

然后以 `stream=true` 请求 proxy 的 `/v1/completions`。proxy 将请求直接转发
给 D 节点。从普通 completions 请求发出到收到第一个 D 节点 SSE token 的时间
就是普通模式 TTFT，`ttft_origin` 为 `normal_decode`。

## 样本长度控制

回放脚本不再在请求开始前请求服务端做完整 prompt 预检。这样更贴近在线 RAG
场景：真正处理流式请求时，系统只知道已经到达的 chunk，不提前知道完整请求。

样本长度应在数据构建阶段控制，例如使用
`build_mixed_source_dataset.py` 的 `--max-web-chunk-tokens`、
`--max-anns-document-tokens` 和 `--max-update-snapshot-tokens`。回放阶段如果仍然
遇到超出 P/D `max_model_len` 的样本，该请求会沿正常 start/event/finish 链路
失败，并按实验错误记录处理。

## 离线检查

离线检查只解析数据并构造最终 prompt，不访问 proxy、P 或 D。两个入口都能
执行，输出目录不能重名。

```shell
.venv/bin/python research/scripts/replay/replay_prompt_stream.py \
  --data-dir research/data/mix_source \
  --query-id 0 \
  --dry-run \
  --run-name dry-stream-query-0
```

```shell
.venv/bin/python research/scripts/replay/replay_normal.py \
  --data-dir research/data/mix_source \
  --query-id 0 \
  --dry-run \
  --run-name dry-normal-query-0
```

## 流式 prefill 回放

### 单请求闭环

```shell
.venv/bin/python research/scripts/replay/replay_prompt_stream.py \
  --data-dir research/data/mix_source \
  --query-id 0 \
  --proxy-url http://<PROXY_HOST>:8000 \
  --model <SERVED_MODEL_NAME> \
  --qps 0 \
  --max-concurrency 1 \
  --append-chunk-interval 0.05 \
  --update-chunk-interval 0.20 \
  --max-tokens 64 \
  --temperature 0 \
  --run-name stream-query-0
```

### 多请求泊松到达

```shell
.venv/bin/python research/scripts/replay/replay_prompt_stream.py \
  --data-dir research/data/mix_source \
  --max-requests 100 \
  --proxy-url http://<PROXY_HOST>:8000 \
  --model <SERVED_MODEL_NAME> \
  --qps 1.0 \
  --arrival-process poisson \
  --random-seed 2026 \
  --max-concurrency 8 \
  --append-initial-delay 0.0 \
  --update-initial-delay 0.0 \
  --append-chunk-interval 0.05 \
  --update-chunk-interval 0.20 \
  --source-finish-delay 0.0 \
  --run-name stream-poisson-qps-1
```

## 普通模式回放

```shell
.venv/bin/python research/scripts/replay/replay_normal.py \
  --data-dir research/data/mix_source \
  --max-requests 100 \
  --proxy-url http://<PROXY_HOST>:8000 \
  --model <SERVED_MODEL_NAME> \
  --qps 1.0 \
  --arrival-process poisson \
  --random-seed 2026 \
  --max-concurrency 8 \
  --max-tokens 64 \
  --temperature 0 \
  --run-name normal-poisson-qps-1
```

为了公平比较，两个脚本应使用相同的：

- 样本范围：`--query-id`、`--start-index` 和 `--max-requests`。
- 逻辑请求负载：`--qps`、`--arrival-process`、`--random-seed` 和
  `--max-concurrency`。
- 生成参数：`--model`、`--max-tokens`、`--temperature` 和 `--seed`。

建议先完成一轮流式实验，再完成一轮普通实验，避免两种负载同时干扰同一组
GPU。

## 公共参数

| 参数 | 默认值 | 含义 |
| --- | --- | --- |
| `--data-dir` | `research/data/mix_source` | 包含 `requests/` 的数据目录。 |
| `--query-id` | 无 | 指定 query id，可重复使用；显式 ID 不受 `--max-requests` 截断。 |
| `--start-index` | `0` | 未指定 query id 时的排序后起始位置。 |
| `--max-requests` | `1` | 批量模式保留的最大样本数；`0` 表示全部。 |
| `--include-partial` | 关闭 | 允许使用 `status != complete` 的样本。 |
| `--proxy-url` | `http://127.0.0.1:8000` | 两种模式共同使用的 proxy 地址。 |
| `--model` | `qwen2.5-3b` | 必须匹配服务端的 `--served-model-name`。 |
| `--qps` | `1.0` | 逻辑请求启动速率；`0` 表示闭环串行。 |
| `--arrival-process` | `fixed` | 使用固定间隔或 `poisson` 泊松到达。 |
| `--max-concurrency` | `1` | 同时在飞的逻辑请求上限。 |
| `--max-tokens` | 不覆盖 | 覆盖数据中的最大生成 token 数。 |
| `--temperature` | 不覆盖 | 覆盖数据中的采样温度。 |
| `--seed` | 无 | 发送给 vLLM 的采样 seed。 |
| `--random-seed` | `0` | 只控制 Poisson 到达间隔。 |
| `--http-timeout` | `600` | HTTP 超时，单位秒。 |
| `--output-dir` | `research/scripts/replay/results` | 实验结果根目录。 |
| `--run-name` | 时间戳与模式 | 指定本次结果子目录名称。 |
| `--save-responses` | 关闭 | 保存解析后的 SSE payload。 |
| `--dry-run` | 关闭 | 仅校验数据，不发送 HTTP。 |
| `--fail-fast` | 关闭 | 第一个请求失败后立即停止。 |

正 QPS 下，请求先按计划时间到达，再等待并发 semaphore。并发限制引入的等待
时间记录在 `queue_delay_ms`。`--qps 0` 时，前一个请求完整结束后才开始下一个。

## 流式模式专用时序参数

这些参数只存在于 `replay_prompt_stream.py`，单位都是秒。它们属于实验负载，
不会写回 `mix_source` 数据。

| 参数 | 默认值 | 含义 |
| --- | --- | --- |
| `--append-initial-delay` | `0` | start ack 到首个 Append chunk 的延迟。 |
| `--update-initial-delay` | `0` | start ack 到首个 Update 快照的延迟。 |
| `--append-chunk-interval` | `0` | Append 来源内部相邻 chunk 的目标间隔。 |
| `--update-chunk-interval` | `0` | Update 来源内部相邻快照的目标间隔。 |
| `--source-finish-delay` | `0` | 最后数据事件到 `source_finished` 的目标延迟。 |

同一来源必须等待 event ack，所以实际间隔不会短于上游处理时间。不同来源有
独立任务，因此 Append 与 Update 谁先抵达 scheduler 由实际时序决定。

## 输出文件

每次运行创建 `<OUTPUT_DIR>/<RUN_ID>/`：

| 文件 | 内容 |
| --- | --- |
| `config.json` | CLI 参数、模式和选中样本。 |
| `request_metrics.csv` | 每个请求的 TTFT、完成时延、队列时延及内容规模。 |
| `event_metrics.csv` | 流式模式的逐 event 发送与 ack 指标；普通模式为空。 |
| `summary.json` | 当前模式的成功数、错误数及 mean/p50/p90/p95 等 TTFT。 |
| `responses/*.json` | 设置 `--save-responses` 后保存 SSE payload。 |

主指标列仍统一叫 `finish_to_first_token_ms`：

- 流式模式表示 `/prompt_stream/finish` 到首 token。
- 普通模式没有独立 finish 阶段，因此等于 `/v1/completions` 到首 token，
  同时也等于 `request_to_first_token_ms`。

## 常见问题

- **proxy 返回 404**：重启 proxy，确认日志或 OpenAPI 中存在
  `/v1/completions` 和 `/v1/prompt_stream/finish`。
- **Expected text/event-stream**：确认请求中的 `stream=true` 没有被中间层
  改写，并避免反向代理缓冲 SSE。
- **Unknown request id**：确认一个流式请求的 start/event/finish 始终进入同一
  proxy 实例；当前会话状态保存在 proxy 进程内。
- **D 或 NIXL 报错**：先用已有的 append/update/mixed frontend 检查 P/D
  链路，再运行批量回放。
