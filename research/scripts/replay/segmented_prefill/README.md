# 分段流式 Prefill 回放实验

本目录提供三个相互独立的实验入口。它们读取同一批
独立的 questions、append_chunks、pipeline_pools 数据，也兼容
`research/data/mix_source/requests/*.json`，使用不同的 REST 请求链路：

- `replay_prompt_stream.py`：使用 mixed-source 流式 prefill API，是我们的优化方案。
- `replay_merged_update.py`：把已经到达的 Append 与 Update 内容合并成单个 Update 快照，是朴素流式 baseline。
- `replay_normal.py`：把最终完整 prompt 一次性发送给普通 completions API。
- `replay_common.py`：历史公共接口兼容入口，实际实现位于 `../common/`。

## 模型模板与文本输入

数据、配置与时序的统一说明见 [回放总说明](../README.md)。`replay_common.py`
保留历史导入路径；公共实现位于 `../common/`。

默认 `--input-transport text`：normal 提交完整 prompt 字符串；stream/merged 的
start 提交模板前缀，数据事件提交 `event.text`，finish 提交问题和模板后缀。
客户端渲染模型模板，P 对原始文本分词。已渲染的 normal/start 使用
`add_special_tokens=false`，避免重复 BOS；finish 保留 assistant header。

`--tokenizer-model` 默认 `~/Models/Qwen2.5-3B`，必须对应服务模型模板。
normal 整段分词与 stream 逐块分词的边界差异保留。历史预分词实验可显式传
`--input-transport token_ids`。质量专用短答案指令仅由质量入口注入。

检查 TriviaQA 输入，不发送 HTTP：

```shell
.venv/bin/python research/scripts/replay/segmented_prefill/replay_prompt_stream.py \
  --config research/scripts/replay/configs/triviaqa.json \
  --dry-run --output-dir /tmp/replay-check --run-name triviaqa-stream-check
```

配置同样适用于 `replay_normal.py` 和 `replay_merged_update.py`；正式实验去掉
`--dry-run`，设置 `--proxy-url`，并按需要加 `--ttft-only`。

## proxy 支持的路径

| 类型 | proxy 路径 | 后端 |
| --- | --- | --- |
| mixed-source 流式 prefill | `/v1/prompt_stream/start`、`/event`、`/finish`、`/finish_ttft`、`/abort` | P 节点完成流式 prefill；完整生成走 D，TTFT-only 不启动 D |
| 普通 PD completions | `/v1/completions` | P 对完整 prompt prefill 并生成首 token；D 加载 P 的 KV 后继续 decode |
| 普通 PD completions TTFT-only | `/v1/completions_ttft` | 只请求 P，P 返回首 token 后立即结束，不创建 D 请求 |
| 普通 chat completions | `/v1/chat/completions` | 当前仍直接转发到一个 D 节点 |
| 普通 Responses API | `/v1/responses` | 当前仍直接转发到一个 D 节点 |

普通 completions 同时支持 `stream=false` JSON 和 `stream=true` SSE。完整请求走标准 P→D 分离；
TTFT-only 使用专用 P-only 路由。多个 P、D 节点均由 proxy 按 round-robin 选择，因此实验客户端不需要知道具体节点地址。

修改 proxy 后需要重启：

```shell
.venv/bin/python research/scripts/server/proxy_server.py \
  --host 0.0.0.0 \
  --port 8000 \
  --prefiller-hosts <P_HOST> \
  --prefiller-ports 8100 \
  --decoder-hosts <D_HOST> \
  --decoder-ports 8200
```

P/D 节点的完整启动方式见 [scripts README](../../README.md)。

## 三种模式的数据语义

### 流式 prefill

`replay_prompt_stream.py` 的请求顺序是：

1. 向 `/v1/prompt_stream/start` 发送通用模板。
2. 为每个来源创建独立异步任务。
3. Append 来源按 `sequence_no` 发送单调追加 chunk。
4. Update 来源按 `sequence_no` 发送完整快照。
5. 同一来源等待上一条 event 的 ack；不同来源之间并发竞争。
6. 所有来源的 `source_finished` 都收到 ack 后，才发送全局
   `prompt_finished`。
7. 从 finish 请求发出到收到第一个 SSE token 的时间记为 TTFT；开启
   `--ttft-only` 时使用 `/finish_ttft`，否则使用 `/finish`。

正常启用 KV transfer 时，第一个 SSE token 来自 P 节点完成最终 prefill 后
生成的那个 token，结果中的 `ttft_origin` 通常为 `prefill`。

### 合并 Update baseline

`replay_merged_update.py` 也走 prompt-stream API，但它不把 Append 和
Update 分开暴露给服务端。客户端本地维护两个状态：

- Append store：每个 Append chunk 到达时追加到末尾。
- Update store：每个 Update snapshot 到达时替换为最新完整快照。

只要任意一个来源的 interval 到达，脚本就先更新本地 store，然后按：

```text
当前所有 Append chunk + 当前最新 Update snapshot
```

拼成一个完整文本，并作为同一个 `update` 来源发送给
`/v1/prompt_stream/event`。如果 Append 和 Update 在同一计划时刻到达，脚本
会在本地先处理 Append、再处理 Update，并只发送一次合并快照。

这个模式用于衡量“朴素整段 Update”相对于我们分段 KV 管理方案的 TTFT
差距。开启 `--ttft-only` 时，它从 `/prompt_stream/finish_ttft` 到首个
SSE token 计 TTFT，不会启动 D 节点。

### 普通整段 prompt

`replay_normal.py` 不发送中间 event。它从数据文件构造：

```text
通用模板 + 所有 Append chunk + 每个 Update 来源的最终快照 + 最终用户问题
```

然后以 `stream=true` 请求 proxy 的 `/v1/completions`。这是标准 P→D 分离：

1. proxy 强制 P 使用 `stream=false`、`max_tokens=1` 和 `return_token_ids=true`。
2. P 对完整 prompt 做 prefill，返回一个首 token 和 `kv_transfer_params`。
3. 完整生成时，proxy 将 P 实际使用的 `prompt_token_ids` 加上 P 首 token 作为 D 的输入，扣除一个生成 budget，并把 KV 参数交给 D。
4. P 的首 token 先作为 SSE 发给客户端，D 再继续输出剩余 token。

开启 `--ttft-only` 时，`replay_normal.py` 改请求 `/v1/completions_ttft`。该路由仍让 P 对同一个完整 prompt 做 prefill 并固定生成一个 token，但在 P 返回后立即发送 `[DONE]`，不会向 D 发任何请求。

因此普通模式的 TTFT 从 normal 请求提交到收到 P 首 token 计量，`ttft_origin` 为 `normal_prefill`。它与另外两种模式一样测量 P 侧产生的首 token；区别只是普通模式必须在该请求开始时一次性 prefill 整个最终 prompt。

## 样本长度控制

回放脚本不再在请求开始前请求服务端做完整 prompt 预检。这样更贴近在线 RAG
场景：真正处理流式请求时，系统只知道已经到达的 chunk，不提前知道完整请求。

样本长度应在数据构建阶段控制。当前
`build_mixed_source_dataset.py` 固定保留完整 SQuAD context 作为两个 Append
chunk，并通过 `--max-anns-document-tokens` 和
`--max-update-snapshot-tokens` 控制 ANNS Update 内容。回放阶段如果仍然遇到
超出 P/D `max_model_len` 的样本，该请求会沿正常 start/event/finish 链路失败，
并按实验错误记录处理。

## 离线检查

离线检查只解析数据并构造最终 prompt，不访问 proxy、P 或 D。三个入口都能
执行，输出目录不能重名。

```shell
.venv/bin/python research/scripts/replay/segmented_prefill/replay_prompt_stream.py \
  --data-dir research/data/mix_source \
  --query-id 0 \
  --tokenizer-model <TOKENIZER_MODEL> \
  --dry-run \
  --run-name dry-stream-query-0
```

```shell
.venv/bin/python research/scripts/replay/segmented_prefill/replay_normal.py \
  --data-dir research/data/mix_source \
  --query-id 0 \
  --tokenizer-model <TOKENIZER_MODEL> \
  --dry-run \
  --run-name dry-normal-query-0
```

```shell
.venv/bin/python research/scripts/replay/segmented_prefill/replay_merged_update.py \
  --data-dir research/data/mix_source \
  --query-id 0 \
  --tokenizer-model <TOKENIZER_MODEL> \
  --dry-run \
  --run-name dry-merged-update-query-0
```

## 流式 prefill 回放

### 单请求闭环

```shell
.venv/bin/python research/scripts/replay/segmented_prefill/replay_prompt_stream.py \
  --data-dir research/data/mix_source \
  --query-id 0 \
  --proxy-url http://<PROXY_HOST>:8000 \
  --model <SERVED_MODEL_NAME> \
  --tokenizer-model <TOKENIZER_MODEL> \
  --qps 0 \
  --max-concurrency 1 \
  --append-chunk-interval 0.05 \
  --update-chunk-interval 0.20 \
  --max-tokens 64 \
  --temperature 0 \
  --ttft-only \
  --sse-timeout 120 \
  --run-name stream-query-0
```

### 多请求泊松到达

```shell
.venv/bin/python research/scripts/replay/segmented_prefill/replay_prompt_stream.py \
  --data-dir research/data/mix_source \
  --max-requests 100 \
  --proxy-url http://<PROXY_HOST>:8000 \
  --model <SERVED_MODEL_NAME> \
  --tokenizer-model <TOKENIZER_MODEL> \
  --qps 1.0 \
  --arrival-process poisson \
  --random-seed 2026 \
  --max-concurrency 32 \
  --append-initial-delay 0.0 \
  --update-initial-delay 0.0 \
  --append-chunk-interval 0.05 \
  --update-chunk-interval 0.20 \
  --source-finish-delay 0.0 \
  --ttft-only \
  --sse-timeout 120 \
  --run-name stream-poisson-qps-1
```

## 合并 Update baseline 回放

这个入口使用与流式 prefill 相同的到达参数，但服务端只看到一个 Update
来源。

```shell
.venv/bin/python research/scripts/replay/segmented_prefill/replay_merged_update.py \
  --data-dir research/data/mix_source \
  --max-requests 100 \
  --proxy-url http://<PROXY_HOST>:8000 \
  --model <SERVED_MODEL_NAME> \
  --tokenizer-model <TOKENIZER_MODEL> \
  --qps 1.0 \
  --arrival-process poisson \
  --random-seed 2026 \
  --max-concurrency 32 \
  --append-initial-delay 0.0 \
  --update-initial-delay 0.0 \
  --append-chunk-interval 0.05 \
  --update-chunk-interval 0.20 \
  --source-finish-delay 0.0 \
  --max-tokens 64 \
  --temperature 0 \
  --ttft-only \
  --sse-timeout 120 \
  --run-name merged-update-poisson-qps-1
```

## 普通模式回放

```shell
.venv/bin/python research/scripts/replay/segmented_prefill/replay_normal.py \
  --data-dir research/data/mix_source \
  --max-requests 100 \
  --proxy-url http://<PROXY_HOST>:8000 \
  --model <SERVED_MODEL_NAME> \
  --tokenizer-model <TOKENIZER_MODEL> \
  --qps 1.0 \
  --arrival-process poisson \
  --random-seed 2026 \
  --max-concurrency 32 \
  --max-tokens 64 \
  --temperature 0 \
  --ttft-only \
  --sse-timeout 120 \
  --run-name normal-poisson-qps-1
```

### 仅跑 normal 的 TTFT QPS sweep

普通 baseline 已改为 P-only TTFT 路径。以下命令只重跑 normal 的
QPS = 0.5、1、2、4，结果目录与完整 sweep 的命名一致：

```shell
for qps in 0.5 1 2 4; do
  qps_suffix="${qps/./p}"
  .venv/bin/python research/scripts/replay/segmented_prefill/replay_normal.py \
    --data-dir research/data/mix_source \
    --max-requests 512 \
    --proxy-url http://<PROXY_HOST>:8000 \
    --model <SERVED_MODEL_NAME> \
    --tokenizer-model <TOKENIZER_MODEL> \
    --qps "${qps}" \
    --arrival-process poisson \
    --random-seed 2026 \
    --max-concurrency 32 \
    --max-tokens 64 \
    --temperature 0 \
    --ttft-only \
    --sse-timeout 120 \
    --run-name "ttft-normal-qps-${qps_suffix}"
done
```

为了公平比较，多个脚本应使用相同的：

- 样本范围：`--query-id`、`--start-index` 和 `--max-requests`。
- 逻辑请求负载：`--qps`、`--arrival-process`、`--random-seed` 和
  `--max-concurrency`。
- 生成参数：`--model`、`--max-tokens`、`--temperature` 和 `--seed`。
- SSE 消费策略：若只比较 TTFT，建议所有对比模式都开启 `--ttft-only`，并
  使用相同的 `--sse-timeout`。

建议逐个完成不同模式的实验，避免多种负载同时干扰同一组 GPU。

## TTFT QPS sweep

`run_ttft_qps_sweep.sbatch` 用于提交一组固定配置的 TTFT sweep。它本身不申请 GPU，只作为 replay client 向已经启动好的 proxy 发请求。默认会串行运行：

```text
QPS = 0.5, 1, 2, 4
每个 QPS 下依次运行 stream、merged-update、normal
每个模式最多 512 个请求
```

三种模式都会开启 `--ttft-only`。

- 流式 prefill 与合并 Update baseline 走 `/v1/prompt_stream/finish_ttft`：P 在最终问题 prefill 后返回首 token，不启动 D。
- 普通 baseline 走 `/v1/completions_ttft`：P 对完整最终 prompt 做一次普通 prefill 并返回首 token，同样不启动 D。

这使三条线都排除了 D 侧 KV load、decode 以及完整输出长度带来的干扰。

默认提交命令：

```shell
mkdir -p research/scripts/logs
```

```shell
PROXY_URL=http://<PROXY_HOST>:8000 \
MODEL=<SERVED_MODEL_NAME> \
TOKENIZER_MODEL=<TOKENIZER_MODEL> \
sbatch research/scripts/replay/segmented_prefill/run_ttft_qps_sweep.sbatch
```

当前实验配置的完整写法：

```shell
DATA_DIR=research/data/mix_source \
OUTPUT_DIR=research/scripts/replay/segmented_prefill/results \
PROXY_URL=http://<PROXY_HOST>:8000 \
MODEL=<SERVED_MODEL_NAME> \
TOKENIZER_MODEL=<TOKENIZER_MODEL> \
MAX_REQUESTS=512 \
QPS_LIST="0.5 1 2 4" \
ARRIVAL_PROCESS=poisson \
RANDOM_SEED=2026 \
MAX_CONCURRENCY=32 \
APPEND_INITIAL_DELAY=0.0 \
UPDATE_INITIAL_DELAY=0.0 \
APPEND_CHUNK_INTERVAL=0.5 \
UPDATE_CHUNK_INTERVAL=0.01 \
SOURCE_FINISH_DELAY=0.0 \
MAX_TOKENS=64 \
TEMPERATURE=0 \
SSE_TIMEOUT=120 \
RUN_PREFIX=ttft \
sbatch research/scripts/replay/segmented_prefill/run_ttft_qps_sweep.sbatch
```

### 使用自定义后端重跑

attention backend 由各 P/D server 在启动时分别决定；replay client 不能按请求切换。两个启动脚本现已接受可选的 `ATTENTION_BACKEND` 环境变量。保留既有 FA 结果目录，并使用独立的 `RUN_PREFIX=chunk-ttft`：

```shell
ATTENTION_BACKEND=CHUNK_SPARSE_ATTN \
MODEL_PATH=<MODEL_PATH> \
SERVED_MODEL_NAME=<SERVED_MODEL_NAME> \
GPU_MEMORY_UTILIZATION=<GPU_UTIL> \
MAX_MODEL_LEN=4096 \
MAX_NUM_SEQS=64 \
sbatch --nodelist=<P_HOST> research/scripts/server/start_p_node.sbatch
```

若之后测量完整 P/D transfer 与 decode，D 节点显式使用 `ATTENTION_BACKEND=TRITON_ATTN`，通过 `unified_attention` 完整读取历史 KV，不做块稀疏。启用中间 padding mask 时，P/D 均设置 `PROMPT_STREAM_MASKED_PADDING=1`；完整启动命令见 [服务端 README](../../server/README.md)。当前 sweep 全部使用 `--ttft-only`，首 token 只由 P 节点生成，D 不会实际进行 KV load；但三种模式都必须通过同一个 `CHUNK_SPARSE_ATTN` P 节点运行。

```shell
DATA_DIR=research/data/mix_source \
OUTPUT_DIR=research/scripts/replay/segmented_prefill/results \
PROXY_URL=http://<PROXY_HOST>:8000 \
MODEL=<SERVED_MODEL_NAME> \
TOKENIZER_MODEL=<TOKENIZER_MODEL> \
MAX_REQUESTS=512 \
QPS_LIST="0.5 1 2 4" \
ARRIVAL_PROCESS=poisson \
RANDOM_SEED=2026 \
MAX_CONCURRENCY=32 \
APPEND_INITIAL_DELAY=0.0 \
UPDATE_INITIAL_DELAY=0.0 \
APPEND_CHUNK_INTERVAL=0.5 \
UPDATE_CHUNK_INTERVAL=0.2 \
SOURCE_FINISH_DELAY=0.0 \
MAX_TOKENS=64 \
TEMPERATURE=0 \
SSE_TIMEOUT=120 \
RUN_PREFIX=chunk-ttft \
sbatch research/scripts/replay/segmented_prefill/run_ttft_qps_sweep.sbatch
```

该命令依次写入 `chunk-ttft-stream-*`、`chunk-ttft-merged-update-*` 和 `chunk-ttft-normal-*`。它们应与旧的 `ttft-*` FA/auto 和 `triton-ttft-*` unified 结果分开画图和汇总。

输出目录名形如：

```text
research/scripts/replay/segmented_prefill/results/ttft-stream-qps-0p5
research/scripts/replay/segmented_prefill/results/ttft-merged-update-qps-0p5
research/scripts/replay/segmented_prefill/results/ttft-normal-qps-0p5
```

### 当前本地常用命令

这一节是当前实验环境的可复制命令。对外发布 README 时可以删除。

```shell
mkdir -p research/scripts/logs
```

```shell
DATA_DIR=research/data/mix_source \
OUTPUT_DIR=research/scripts/replay/segmented_prefill/results \
PROXY_URL=http://g2c14:8000 \
MODEL=llama-3.1-8b-instruct \
TOKENIZER_MODEL=/workplace/home/huangxinying/Models/Llama-3.1-8B-Instruct \
MAX_REQUESTS=512 \
QPS_LIST="0.5 1 2 4" \
ARRIVAL_PROCESS=poisson \
RANDOM_SEED=2026 \
MAX_CONCURRENCY=32 \
APPEND_INITIAL_DELAY=0.0 \
UPDATE_INITIAL_DELAY=0.0 \
APPEND_CHUNK_INTERVAL=1 \
UPDATE_CHUNK_INTERVAL=0.5 \
SOURCE_FINISH_DELAY=0.0 \
MAX_TOKENS=64 \
TEMPERATURE=0 \
SSE_TIMEOUT=120 \
RUN_PREFIX=ttft \
sbatch research/scripts/replay/segmented_prefill/run_ttft_qps_sweep.sbatch
```

## TTFT survival curve 画图

`plot_ttft_survival.py` 会读取一个或多个结果目录里的 `request_metrics.csv`，把
TTFT survival curve 画在同一张图中。横坐标是 TTFT，单位 ms；纵坐标是
`percent(TTFT > x)`，即 TTFT 大于当前横坐标的请求占比。

脚本统一使用重复 `--series` 指定曲线。格式是
`label=result_dir[:ttft_column]`。如果省略 `ttft_column`，默认使用
`finish_to_first_token_ms`。脚本只统计 `result_status=ok` 且 TTFT 字段有效的请求。

```shell
.venv/bin/python research/scripts/replay/segmented_prefill/plot_ttft_survival.py \
  --series "Segmented stream prefill=research/scripts/replay/segmented_prefill/results/ttft-stream-qps-1:finish_to_first_token_ms" \
  --series "Merged update baseline=research/scripts/replay/segmented_prefill/results/ttft-merged-update-qps-1:finish_to_first_token_ms" \
  --series "Normal full prompt=research/scripts/replay/segmented_prefill/results/ttft-normal-qps-1:request_to_first_token_ms" \
  --output research/scripts/replay/segmented_prefill/results/ttft-survival-three-modes-qps-1.svg \
  --title "TTFT survival, QPS=1"
```

批量画出 `0.5/1/2/4` 四个 QPS 的三模式图：

```shell
for qps in 0p5 1 2 4; do
  .venv/bin/python research/scripts/replay/segmented_prefill/plot_ttft_survival.py \
    --series "Segmented stream prefill=research/scripts/replay/segmented_prefill/results/ttft-stream-qps-${qps}:finish_to_first_token_ms" \
    --series "Merged update baseline=research/scripts/replay/segmented_prefill/results/ttft-merged-update-qps-${qps}:finish_to_first_token_ms" \
    --series "Normal full prompt=research/scripts/replay/segmented_prefill/results/ttft-normal-qps-${qps}:request_to_first_token_ms" \
    --output "research/scripts/replay/segmented_prefill/results/ttft-survival-three-modes-qps-${qps}.svg" \
    --title "TTFT survival, QPS=${qps}"
done
```

脚本会同时写出一个同名 `.summary.json`，记录每条曲线的样本数、均值和分位数。
当前实现使用 matplotlib，输出格式由 `--output` 的扩展名决定，常用 `.png`、
`.svg` 或 `.pdf`。

## 公共参数

| 参数 | 默认值 | 含义 |
| --- | --- | --- |
| `--data-dir` | `research/data/mix_source` | 包含 `requests/` 的数据目录。 |
| `--query-id` | 无 | 指定 query id，可重复使用；显式 ID 不受 `--max-requests` 截断。 |
| `--start-index` | `0` | 未指定 query id 时的排序后起始位置。 |
| `--max-requests` | `1` | 批量模式保留的最大样本数；`0` 表示全部。 |
| `--include-partial` | 关闭 | 允许使用 `status != complete` 的样本。 |
| `--proxy-url` | `http://127.0.0.1:8000` | 三种模式共同使用的 proxy 地址。 |
| `--model` | `qwen2.5-3b` | 必须匹配服务端的 `--served-model-name`。 |
| `--tokenizer-model` | `~/Models/Qwen2.5-3B` | 本地模型/tokenizer 目录；必须与 `--model` 对应的 P 模型使用同一 chat template。 |
| `--qps` | `1.0` | 逻辑请求启动速率；`0` 表示闭环串行。 |
| `--arrival-process` | `fixed` | 使用固定间隔或 `poisson` 泊松到达。 |
| `--max-concurrency` | `32` | 同时在飞的逻辑请求上限。 |
| `--max-tokens` | 不覆盖 | 覆盖数据中的最大生成 token 数。 |
| `--temperature` | 不覆盖 | 覆盖数据中的采样温度。 |
| `--seed` | 无 | 发送给 vLLM 的采样 seed。 |
| `--random-seed` | `0` | 只控制 Poisson 到达间隔。 |
| `--http-timeout` | `600` | HTTP 超时，单位秒。 |
| `--ttft-only` | 关闭 | 只记录 TTFT。prompt-stream 类模式调用 `/v1/prompt_stream/finish_ttft`；普通模式调用 `/v1/completions_ttft`。两者都是 P 返回一个首 token 后立即 `[DONE]`，不启动 D。 |
| `--sse-timeout` | 不限制 | 单个成功 SSE 响应的消费超时，单位秒；用于避免 HTTP 200 后一直等不到 `[DONE]`。 |
| `--output-dir` | `research/scripts/replay/segmented_prefill/results` | 实验结果根目录。 |
| `--run-name` | 时间戳与模式 | 指定本次结果子目录名称。 |
| `--save-responses` | 关闭 | 保存解析后的 SSE payload。 |
| `--dry-run` | 关闭 | 仅校验数据，不发送 HTTP。 |
| `--fail-fast` | 关闭 | 第一个请求失败后立即停止。 |

正 QPS 下，请求先按计划时间到达，再等待并发 semaphore。并发限制引入的等待
时间记录在 `queue_delay_ms`。`--qps 0` 时，前一个请求完整结束后才开始下一个。

## prompt-stream 类模式专用时序参数

这些参数存在于 `replay_prompt_stream.py` 和
`replay_merged_update.py`，单位都是秒。它们属于实验负载，不会写回
`mix_source` 数据。

| 参数 | 默认值 | 含义 |
| --- | --- | --- |
| `--append-initial-delay` | `0` | start ack 到首个 Append chunk 的延迟。 |
| `--update-initial-delay` | `0` | start ack 到首个 Update 快照的延迟。 |
| `--append-chunk-interval` | `0` | Append 来源内部相邻 chunk 的目标间隔。 |
| `--update-chunk-interval` | `0` | Update 来源内部相邻快照的目标间隔。 |
| `--source-finish-delay` | `0` | 最后数据事件到 `source_finished` 的目标延迟。 |

`replay_prompt_stream.py` 中同一来源必须等待 event ack，所以实际间隔不会短于
上游处理时间；不同来源有独立任务，因此 Append 与 Update 谁先抵达
scheduler 由实际时序决定。`replay_merged_update.py` 则在客户端侧合并
同一计划时刻的来源事件，再串行发送单个 Update 来源。

## 输出文件

每次运行创建 `<OUTPUT_DIR>/<RUN_ID>/`：

| 文件 | 内容 |
| --- | --- |
| `config.json` | CLI 参数、模式和选中样本。 |
| `request_metrics.csv` | 每个请求的 TTFT、完成时延、队列时延及内容规模。 |
| `event_metrics.csv` | prompt-stream 类模式的逐 event 发送与 ack 指标；普通模式为空。 |
| `summary.json` | 当前模式的成功数、错误数及 mean/p50/p90/p95 等 TTFT。 |
| `responses/*.json` | 设置 `--save-responses` 后保存 SSE payload。 |

主指标列仍统一叫 `finish_to_first_token_ms`：

- 流式 prefill 和合并 Update baseline 在 `--ttft-only` 下表示
  `/prompt_stream/finish_ttft` 到首 token；非 TTFT-only 调试时表示
  `/prompt_stream/finish` 到首 token。
- 普通模式没有独立 finish 阶段。在 `--ttft-only` 下，它表示
  `/v1/completions_ttft` 到 P 首 token；非 TTFT-only 调试时表示
  `/v1/completions` 到 P 首 token，同时也等于 `request_to_first_token_ms`。

## 常见问题

- **proxy 返回 404**：重启 proxy，确认日志或 OpenAPI 中存在
  `/v1/completions`、`/v1/completions_ttft`、
  `/v1/prompt_stream/finish` 和 `/v1/prompt_stream/finish_ttft`。
- **Expected text/event-stream**：确认请求中的 `stream=true` 没有被中间层
  改写，并避免反向代理缓冲 SSE。
- **Unknown request id**：确认一个流式请求的 start/event/finish 始终进入同一
  proxy 实例；当前会话状态保存在 proxy 进程内。
- **D 或 NIXL 报错**：先用已有的 append/update/mixed frontend 检查 P/D
  链路，再运行批量回放。
