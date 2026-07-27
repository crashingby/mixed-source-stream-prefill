# 混合来源数据生成

`build_mixed_source_dataset.py` 以 Stream2LLM 的 ANNS 问题和检索 trace
为基础，构造适用于当前 prompt-stream RESTFUL API 的混合来源数据：

- **ANNS 来源**使用 Update 语义。每一行 `PipelinePool` 都会物化成一份完整
  快照，并保持原始文档顺序。
- **网页来源**使用 Append 语义。脚本用 DDGS 搜索同一个问题，再用
  Trafilatura 抽取网页正文，每个成功网页形成一个独立 chunk。
- **最终问题**使用 `prompt_finished` 保存；每个来源最后都有
  `source_finished`。

## 时间与事件顺序

生成数据**不保存真实检索耗时、网页抓取耗时或事件到达间隔**。

- ANNS 来源内部按照 trace 的 CSV 行序和 `sequence_no` 排列。
- 网页来源内部按照搜索结果的 `search_rank` 和 `sequence_no` 排列。
- `events` 在文件中按来源分组，只是为了稳定存储，不代表实验到达顺序。
- Append 与 Update 如何交错、事件之间等待多久、何时设置运行时
  `arrival_time`，都由后续实验回放脚本控制。
- `--query-delay` 只负责全量构建数据时降低网页搜索请求频率，不会写入
  数据，也不是 prompt-stream 实验的事件间隔。

当前 schema 为 `prompt_stream_mixed/v2`。样本中使用下面的声明明确区分
来源内顺序和跨来源调度：

```json
{
  "event_order": {
    "storage_layout": "grouped_by_source",
    "within_source": "sequence_no",
    "across_sources": "unspecified"
  }
}
```

## 目录结构

默认输出到 `research/data/mix_source`：

```text
mix_source/
├── dataset.json
├── manifest.jsonl
├── requests/
│   └── anns-query-000000.json
└── .cache/
    └── web/
        └── query_0.json
```

- `requests/*.json`：可供实验回放器读取的完整请求数据。
- `manifest.jsonl`：样本路径、问题、状态和事件数量索引。
- `dataset.json`：schema、样本数量和生成配置摘要。
- `.cache/web`：网页搜索与正文抽取缓存，用于中断恢复和离线重建。

`research/data` 是独立的数据子模块。生成脚本属于 vLLM 父仓库，生成结果
属于数据子模块，后续需要分别提交。

## 安装依赖

在 vLLM 项目根目录执行：

```shell
uv pip install -r research/scripts/data_processing/requirements.txt
```

当前验证版本为 `ddgs==9.14.4` 和 `trafilatura==2.1.0`。

## 快速开始

### 生成一个默认样本

不指定 query 范围时，默认只生成数值排序后的第一个问题，避免意外启动
全量网页检索：

```shell
.venv/bin/python \
  research/scripts/data_processing/build_mixed_source_dataset.py
```

### 生成指定问题

`--query-id` 可以重复使用。指定它以后，`--start-index` 和
`--max-queries` 不再参与选择：

```shell
.venv/bin/python \
  research/scripts/data_processing/build_mixed_source_dataset.py \
  --query-id 0 \
  --query-id 17
```

### 生成一个连续范围

下面的命令从数值排序后的第 100 个 query 开始，生成 20 个问题：

```shell
.venv/bin/python \
  research/scripts/data_processing/build_mixed_source_dataset.py \
  --start-index 100 \
  --max-queries 20
```

`--start-index` 是排序后的**位置**，不是 query id。

### 生成全部 ANNS 问题

`--max-queries 0` 表示处理 `--start-index` 之后的全部问题：

```shell
.venv/bin/python \
  research/scripts/data_processing/build_mixed_source_dataset.py \
  --anns-dir research/data/anns \
  --output-dir research/data/mix_source \
  --start-index 0 \
  --max-queries 0 \
  --max-web-results 5 \
  --fetch-workers 4 \
  --query-delay 2
```

已有 v2 请求文件会自动跳过；已有且配置匹配的网页缓存会直接复用。因此命令
被中断后，可以直接重新执行同一条命令续跑。

### 生成适合 4096 窗口的短样本

正式跑当前 P/D 节点的 `--max-model-len 4096` 时，建议在数据生成阶段就启用
token 预算。token 预算既可以用于**联网重新检索**，也可以用于**复用已有网页
缓存重建请求 JSON**。

#### 复用已有网页缓存

如果前面已经爬完了一部分网页，推荐先用下面的命令把已有缓存重建成短样本：

```shell
.venv/bin/python \
  research/scripts/data_processing/build_mixed_source_dataset.py \
  --anns-dir research/data/anns \
  --output-dir research/data/mix_source \
  --start-index 0 \
  --max-queries 0 \
  --max-web-results 5 \
  --tokenizer-model <TOKENIZER_OR_MODEL_PATH> \
  --max-web-chunk-tokens 256 \
  --max-anns-document-tokens 256 \
  --max-update-snapshot-tokens 1536 \
  --only-cached-web \
  --offline \
  --overwrite
```

这条命令只处理已有 `.cache/web/query_<id>.json` 的问题，不会重新联网检索，
也不会为尚未爬完的问题刷失败日志。它会重写对应的 `requests/*.json`，把每个
网页 Append chunk 控制在约 256 tokens，并把每个 Update 快照控制在约 1536
tokens。回放阶段不再提前请求完整 prompt 预检，因此这里的 token 预算是控制
样本长度的主要手段。

当前本机 Qwen 复用缓存示例：

```shell
.venv/bin/python \
  research/scripts/data_processing/build_mixed_source_dataset.py \
  --anns-dir research/data/anns \
  --output-dir research/data/mix_source \
  --start-index 0 \
  --max-queries 0 \
  --max-web-results 5 \
  --tokenizer-model /workplace/home/huangxinying/Models/Qwen2.5-3B/ \
  --max-web-chunk-tokens 256 \
  --max-anns-document-tokens 256 \
  --max-update-snapshot-tokens 1536 \
  --only-cached-web \
  --offline \
  --overwrite
```

#### 从头联网生成短样本

如果希望重新搜索、重新抓取网页，同时生成短样本，就去掉 `--offline` 和
`--only-cached-web`，并加上 `--refresh-web --overwrite`：

```shell
.venv/bin/python \
  research/scripts/data_processing/build_mixed_source_dataset.py \
  --anns-dir research/data/anns \
  --output-dir research/data/mix_source \
  --start-index 0 \
  --max-queries 0 \
  --max-web-results 5 \
  --fetch-workers 4 \
  --query-delay 2 \
  --tokenizer-model <TOKENIZER_OR_MODEL_PATH> \
  --max-web-chunk-tokens 256 \
  --max-anns-document-tokens 256 \
  --max-update-snapshot-tokens 1536 \
  --refresh-web \
  --overwrite
```

当前本机 Qwen 从头联网示例：

```shell
.venv/bin/python \
  research/scripts/data_processing/build_mixed_source_dataset.py \
  --anns-dir research/data/anns \
  --output-dir research/data/mix_source \
  --start-index 0 \
  --max-queries 0 \
  --max-web-results 5 \
  --fetch-workers 4 \
  --query-delay 2 \
  --tokenizer-model /workplace/home/huangxinying/Models/Qwen2.5-3B/ \
  --max-web-chunk-tokens 256 \
  --max-anns-document-tokens 256 \
  --max-update-snapshot-tokens 1536 \
  --refresh-web \
  --overwrite
```

这里的长度限制作用在最终写入 `requests/*.json` 的事件文本上；网页缓存仍保存
抽取后的正文，便于后续用不同 token 预算反复重建。

## 参数说明

### 输入、输出与问题选择

| 参数 | 默认值 | 作用 |
| --- | --- | --- |
| `--anns-dir` | `research/data/anns` | ANNS 数据目录，必须包含 `query_trace_map_5k.json`、`res/` 和 corpus shards。 |
| `--output-dir` | `research/data/mix_source` | 请求 JSON、manifest 和网页缓存的输出目录。 |
| `--query-id` | 未设置 | 生成指定 query id；可以重复传入。使用后忽略 `--start-index` 和 `--max-queries`。 |
| `--start-index` | `0` | 数值排序后的 query 列表起始位置，不是 query id。 |
| `--max-queries` | `1` | 最多处理多少个问题；`0` 表示处理剩余全部问题。 |

### 网页搜索与正文抓取

| 参数 | 默认值 | 作用 |
| --- | --- | --- |
| `--max-web-results` | `5` | 每个问题最多取得并尝试抓取多少条搜索结果。 |
| `--min-successful-web-pages` | `1` | 达到多少个有效网页时样本状态为 `complete`；低于该值时为 `partial`。 |
| `--fetch-workers` | `4` | 单个问题内部并发抓取网页的最大线程数。它不决定 Append 的实验到达顺序。 |
| `--search-timeout` | `20` 秒 | 一次 DDGS 搜索的超时时间。 |
| `--fetch-timeout` | `20` 秒 | 单个网页 HTTP 请求的超时时间。 |
| `--max-search-retries` | `2` | 搜索首次失败后允许的额外重试次数。 |
| `--max-fetch-retries` | `1` | 网页首次抓取失败后允许的额外重试次数。 |
| `--retry-backoff` | `1.0` 秒 | 重试的初始指数退避时间。仅影响数据采集过程。 |
| `--query-delay` | `1.0` 秒 | 相邻两个在线问题检索之间的节流等待。**不写入数据，不控制实验事件间隔。** |
| `--region` | `wt-wt` | DDGS 检索地区。改变后需要重新生成网页缓存。 |
| `--safesearch` | `moderate` | DDGS SafeSearch，可选 `on`、`moderate`、`off`。 |
| `--search-backend` | `auto` | DDGS 文本检索后端。 |
| `--max-page-bytes` | `5242880` | 单个网页允许下载的最大字节数，默认 5 MiB。 |
| `--min-web-chars` | `200` | 抽取正文少于该字符数时，将该网页记为失败。 |
| `--max-web-chars` | `12000` | 每个网页正文的最大字符数；`0` 表示不截断。 |
| `--tokenizer-model` | 未设置 | 本地 tokenizer 或模型目录。启用任意 token 级长度预算时必须设置。 |
| `--max-web-chunk-tokens` | `0` | 每个格式化后网页 Append chunk 的最大 tokenizer token 数；`0` 表示不启用。 |
| `--user-agent` | 脚本默认值 | 网页 HTTP 请求使用的 `User-Agent`。正式发布数据前建议改成有效联系方式。 |

### ANNS 内容和生成请求

| 参数 | 默认值 | 作用 |
| --- | --- | --- |
| `--max-anns-document-chars` | `0` | 每个 ANNS 文档的最大字符数；`0` 表示完整保留。截断模板保持稳定，避免无谓破坏 Update LCP。 |
| `--max-anns-document-tokens` | `0` | 每个格式化后 ANNS 文档的最大 tokenizer token 数；`0` 表示不启用。 |
| `--max-update-snapshot-tokens` | `0` | 每个完整 ANNS Update 快照的最大 tokenizer token 数；`0` 表示不启用。 |
| `--max-tokens` | `64` | 写入每个样本 `start.max_tokens` 的默认 decode 上限。 |
| `--temperature` | `0.0` | 写入每个样本 `start.temperature` 的默认采样温度。 |
| `--prefix` | 内置通用模板 | 写入 `start.prompt` 的固定 prefix/template。 |

原始 ANNS 和网页正文可能超过实际模型的 `max_model_len`。默认行为仍然保留
原始内容；要让更多样本适配 4096 窗口，应当在生成阶段显式启用
`--tokenizer-model`、`--max-web-chunk-tokens`、
`--max-anns-document-tokens` 和 `--max-update-snapshot-tokens`。正式回放时若
仍遇到超长样本，会由 P/D 的正常请求路径 fail closed。

### 缓存、覆盖和失败处理

| 参数 | 默认值 | 作用 |
| --- | --- | --- |
| `--offline` | 关闭 | 禁止网络访问，只允许使用已有 `.cache/web/query_<id>.json`。不能与 `--refresh-web` 同时使用。 |
| `--overwrite` | 关闭 | 允许重写已有 `requests/*.json`。如果网页缓存存在且配置相同，仍然复用缓存。 |
| `--refresh-web` | 关闭 | 忽略已有网页缓存并重新搜索、抓取。脚本要求同时使用 `--overwrite`。 |
| `--only-cached-web` | 关闭 | 只处理已经存在 `.cache/web/query_<id>.json` 的问题，适合部分爬取后离线重建短样本。 |
| `--require-web` | 关闭 | 有效网页数低于 `--min-successful-web-pages` 时让该 query 失败，而不是生成 `partial` 样本。 |
| `--fail-fast` | 关闭 | 任意 query 失败后立即停止；默认记录失败并继续处理其它 query。 |
| `--random-seed` | `0` | 只控制 retry jitter，不进入生成数据，也不决定事件顺序。 |
| `--log-level` | `INFO` | 日志等级，可选 `DEBUG`、`INFO`、`WARNING`、`ERROR`。 |

## 缓存与覆盖模式

### 断点续跑

重新执行原来的生成命令即可：

- 已存在的 v2 请求直接跳过。
- 尚未生成请求但已有网页缓存时，复用缓存。
- 尚无请求和缓存时，重新联网检索。

### 使用缓存重新生成请求 JSON

只重写请求文件，不重新联网搜索：

```shell
.venv/bin/python \
  research/scripts/data_processing/build_mixed_source_dataset.py \
  --query-id 0 \
  --max-web-results 5 \
  --overwrite
```

缓存配置必须与当前内容相关参数一致；不一致时脚本会要求显式刷新。

如果只是想把已经抓好的网页缓存重建成短样本，可以在这条命令上加入 token
预算参数，例如：

```shell
.venv/bin/python \
  research/scripts/data_processing/build_mixed_source_dataset.py \
  --query-id 0 \
  --max-web-results 5 \
  --tokenizer-model <TOKENIZER_OR_MODEL_PATH> \
  --max-web-chunk-tokens 256 \
  --max-anns-document-tokens 256 \
  --max-update-snapshot-tokens 1536 \
  --only-cached-web \
  --offline \
  --overwrite
```

### 完全离线重建请求 JSON

```shell
.venv/bin/python \
  research/scripts/data_processing/build_mixed_source_dataset.py \
  --query-id 0 \
  --max-web-results 5 \
  --offline \
  --overwrite
```

### 完全重新检索并覆盖全部结果

下面这条命令会针对当前 `research/data/anns` 中的全部问题，忽略旧网页
缓存、重新搜索和抓取，并覆盖已有请求 JSON：

```shell
.venv/bin/python \
  research/scripts/data_processing/build_mixed_source_dataset.py \
  --anns-dir research/data/anns \
  --output-dir research/data/mix_source \
  --start-index 0 \
  --max-queries 0 \
  --max-web-results 5 \
  --fetch-workers 4 \
  --query-delay 2 \
  --tokenizer-model <TOKENIZER_OR_MODEL_PATH> \
  --max-web-chunk-tokens 256 \
  --max-anns-document-tokens 256 \
  --max-update-snapshot-tokens 1536 \
  --refresh-web \
  --overwrite
```

这里的 `--query-delay 2` 只是在连续搜索之间等待 2 秒，用于降低限流风险；
它不会出现在 `dataset.json`、网页缓存或请求 JSON 中。

## 回放数据时的约束

每个 `requests/*.json` 的 `events[].event` 与当前
`PromptStreamEventRequest` 基本同构。正式回放器应当：

1. 复制 `start`，注入实际 `model`，发送 `/v1/prompt_stream/start`。
2. 按 `source_id` 分组事件，并在每个来源内部按 `sequence_no` 排序。
3. 根据实验策略选择下一条 Append 或 Update 事件，并自行控制 sleep、并发和
   运行时 `arrival_time`。
4. 某个来源的数据事件发完后，发送该来源的 `source_finished`。
5. 所有 `finish.after_source_ids` 对应的 barrier 都收到 ack 后，再发送
   `finish.event` 到 `/v1/prompt_stream/finish`。

Update 事件携带完整 `text` 快照，不是相对上一版的增量。
`PipelinePool=()` 保存为 `text: ""`，表示清空 Update 来源；不能过滤该事件。
