# 混合来源数据处理

**build_mixed_source_dataset.py** 将 **SQuAD v2 官方 context** 与
**Stream2LLM ANNS PipelinePool trace** 组合为当前 prompt-stream RESTFUL API
可以直接回放的混合来源数据。

当前版本已经完全移除在线网页搜索、网页抓取和网页缓存依赖。生成过程只读取
本地数据，结果具有确定性。

## 脚本用途

### build_mixed_source_dataset.py

构造 **research/data/mix_source/requests/*.json**：

- **Append 来源**：当前 ANNS 问题在 SQuAD v2 中对应的官方 context；默认完整
  保留，也可按每个 chunk 的 token 上限截断。
- **Update 来源**：Stream2LLM ANNS trace 中不断变化的 PipelinePool。
- **最终问题**：以 prompt_finished 事件保存原始 ANNS 问题。
- **事件时间**：不写入数据文件，由 replay 的 interval 参数控制。

### align_squad_gold_answers.py

重新扫描生成后的请求，将问题与 SQuAD v2 官方答案对齐，输出质量评测使用的
gold-answer manifest。该脚本不会修改请求 JSON。

### build_stable_kv_push_dataset.py

离线构造 **稳定 KV 提前传输的受控性能集**。它不读取或覆盖已有 SQuAD/ANNS
数据，不需要联网，也没有标准答案；只用于比较提前传输与 finish-only 的性能。
各场景使用相同的请求编号、问题、模板和随机种子，只调整知识中稳定 Append
与最终完整 Update 的 token 配比。每个请求的模板开头包含独有标识，减少长
公共前缀缓存命中对对照实验的影响。

## 稳定 KV 提前传输的受控数据

固定知识总长 2048 tokens，默认生成三种配比：

| 场景目录 | Append 总长 | 最终 Update 长度 | Append 占知识比例 |
| --- | ---: | ---: | ---: |
| `append-0256-update-1792` | 256 | 1792 | 12.5% |
| `append-1024-update-1024` | 1024 | 1024 | 50% |
| `append-1792-update-0256` | 1792 | 256 | 87.5% |

每个请求默认有两个 Append chunk、五代等长 Update 完整快照。相邻 Update
快照的实际 source-local LCP 为 `floor(Update长度 × update_lcp_ratio)`，默认
比例为 0.75。生成器用自然文档语句组成内容并严格验证重新编码的 token 长度；
如果 tokenizer 无法满足精确切块或 LCP，不会静默改成近似长度。

通用生成命令：

```shell
.venv/bin/python research/scripts/data_processing/build_stable_kv_push_dataset.py \
  --tokenizer-model "<本地模型或tokenizer目录>" \
  --output-dir research/data/stable_kv_push_controlled_qwen \
  --num-requests 128 \
  --knowledge-tokens 2048 \
  --append-tokens 256 1024 1792 \
  --append-chunks 2 \
  --update-snapshots 5 \
  --update-lcp-ratio 0.75 \
  --seed 2026 \
  --block-size 16 \
  --max-model-len 4096 \
  --max-tokens 64
```

本地可直接复制的 tokenizer 写法：

```shell
.venv/bin/python research/scripts/data_processing/build_stable_kv_push_dataset.py \
  --tokenizer-model /workplace/home/huangxinying/Models/Qwen2.5-3B \
  --output-dir research/data/stable_kv_push_controlled_qwen \
  --num-requests 128 \
  --knowledge-tokens 2048 \
  --append-tokens 256 1024 1792 \
  --append-chunks 2 \
  --update-snapshots 5 \
  --update-lcp-ratio 0.75 \
  --seed 2026
```

| 参数 | 默认值 | 含义 |
| --- | --- | --- |
| `--tokenizer-model` | 必填 | 本地 tokenizer，必须与回放一致且具有 chat template；只读本地文件 |
| `--output-dir` | 必填 | 三种场景的公共父目录，不要指向质量集 |
| `--num-requests` | 128 | 每种场景的请求数 |
| `--knowledge-tokens` | 2048 | Append 加最终完整 Update 的 token 总数，不包含模板与问题 |
| `--append-tokens` | `256 1024 1792` | 多个场景的稳定 Append 总 token 长度，必须不同且小于知识总长 |
| `--append-chunks` | 2 | 每请求 Append 数据事件数，默认尽量等分，实际切点受 token 边界约束 |
| `--update-snapshots` | 5 | 每请求 Update 完整快照数；每代长度相同 |
| `--update-lcp-ratio` | 0.75 | 相邻完整快照的精确 source-local LCP 比例，取值 `[0,1)` |
| `--seed` | 2026 | 数据内容随机种子，与 replay 的到达过程随机种子独立 |
| `--block-size` | 16 | P 节点 Prefix+Append 的块对齐粒度，用于物理 padding 预算 |
| `--max-model-len` | 4096 | 物理 prompt 加完整输出预算的最大长度 |
| `--max-tokens` | 64 | 生成时预留的完整输出 token 数，不在 P 首 token 之外重复计数 |
| `--overwrite` | 关闭 | 重建本生成器已有场景，移除旧生成请求；拒绝覆盖其他数据集 |

**长度检查按实际分段传输计算**：模型 chat-template prefix、Append 两个事件
的 token 总长、最终完整 Update、问题和 assistant header、Prefix+Append
对齐 padding，以及完整输出预算，都计入上限。知识总长不等于模型输入总长。
回放改用不同模型、输出长度或 block size 时，应按新配置重新生成。

每个场景都有 `requests/*.json`、`manifest.jsonl` 和 `dataset_summary.json`。
请求格式继续使用 `prompt_stream_mixed/v2`，事件只保存 `text`；现有回放器
根据目标 tokenizer 再编码为 `token_ids`。汇总中记录精确长度与 padding
上下界；单样本的 `controlled_workload` 记录每块长度和实际 LCP，便于审计。
数据中不保存时间间隔，到达节奏仍通过 replay 的 interval/initial-delay 控制。

## 数据语义

每个请求包含一个 Append source 和一个 Update source。

### SQuAD Append source

SQuAD v2 的 context 本身就是一个完整 paragraph。生成器会：

1. 先按问题原文精确匹配 SQuAD train/dev。
2. 精确匹配失败时，才使用消除大小写、重音、标点和重复空白的归一化键。
3. 如果同一问题指向多个不同 context，则拒绝静默选择并报告失败。
4. 找到官方 context 后，在靠近中点的句子边界切成前后两个连续部分。
5. 如果没有句子边界，则退回普通空白边界。
6. 切分点不得位于任何官方答案 span 内部。

因此，每个请求固定产生两个 Append 数据事件：

~~~text
SQuAD 官方 context
├── Append chunk 0：context 前半部分
└── Append chunk 1：context 后半部分
~~~

默认情况下，两个 chunk 合起来完整覆盖官方 context，不会把整篇 article 的
全局前两个 paragraph 当作请求证据，也不会对 SQuAD context 做 token 截断。
这套无损模式用于质量评测。

`--max-append-chunk-tokens` 可为每个格式化后的 Append 事件设置独立 token
上限，包装标签也计入预算。它适合 4K 上下文窗口的性能实验：超过预算的正文会
被直接截断并写入 `[truncated]` 标记。启用后官方答案可能落在被删除的后缀，
因此不能将该数据集用于 SQuAD EM/F1 质量结论。

Append 文本格式为：

~~~text
<squad_context
  split="train"
  title="University_of_Notre_Dame"
  squad_id="..."
  segment="1/2">
...
</squad_context>
~~~

每条事件的 provenance 还会保存：

- split
- title
- paragraph_index
- squad_id
- match_method
- context_sha256
- segment_index
- char_start 和 char_end
- contains_gold_answer

Append source 最后固定发送一条 source_finished。

### ANNS Update source

Update 的处理方式保留原有设计：

- 每一行 trace CSV 表示一版完整 PipelinePool。
- 每个 PipelinePool 按原始文档顺序物化为完整 Update snapshot。
- 文档标签只包含稳定 document id，避免人为破坏 source-local LCP。
- 空 PipelinePool 会生成空字符串 Update，语义是清空当前快照。
- Update source 最后发送一条 source_finished。

ANNS 文档格式为：

~~~text
<retrieved_document id="...">
...
</retrieved_document>
~~~

Update 可以设置单文档和整版快照 token 预算；Append 的上限由独立的
`--max-append-chunk-tokens` 控制。

### 最终逻辑 Prompt

普通模式最终构造：

~~~text
固定模板
+ 两个 SQuAD Append chunk
+ 最终 ANNS Update snapshot
+ 用户问题
~~~

Segmented 模式通过两个 source 的 RESTFUL 事件得到相同的逻辑内容。跨来源
事件的实际交错顺序仍由 replay 控制。

## RESTFUL 事件兼容性

请求继续使用：

~~~text
schema_version = prompt_stream_mixed/v2
data_revision = squad_append_v1
~~~

保留 v2 是为了兼容现有：

- replay_normal.py
- replay_prompt_stream.py
- replay_merged_update.py
- update-coalescing replay
- generation-quality replay

每条 events[].event 继续包含：

~~~json
{
  "source_id": "squad:0",
  "event_type": "append",
  "source_type": "append",
  "chunk_id": "squad:0-context-0",
  "version": 0,
  "sequence_no": 0,
  "text": "..."
}
~~~

Update 事件仍使用 event_type="update" 和 source_type="update"。
两个来源内部的 sequence_no 都从 0 连续递增，最后一条均为
source_finished。

数据文件只声明：

~~~json
{
  "event_order": {
    "storage_layout": "grouped_by_source",
    "within_source": "sequence_no",
    "across_sources": "unspecified"
  }
}
~~~

因此，文件中的来源分组顺序不代表运行时到达顺序。

## 输入数据

默认读取：

~~~text
research/data/
├── anns/
│   ├── query_trace_map_5k.json
│   ├── res/
│   │   └── *_pipeline_trace.csv
│   └── retrieved_corpus_content.part.*.json
└── squad_v2/
    ├── train-v2.0.json
    └── dev-v2.0.json
~~~

query_trace_map_5k.json 提供：

- query：最终问题，同时用于查找 SQuAD context。
- trace_file：该问题对应的 ANNS PipelinePool trace。

## 输出目录

默认写入：

~~~text
research/data/mix_source/
├── dataset.json
├── manifest.jsonl
├── requests/
│   └── anns-query-000000.json
└── quality_eval/
    ├── squad_gold_answers.jsonl
    └── squad_alignment_summary.json
~~~

- requests/*.json：可直接交给现有 replay 的请求数据。
- manifest.jsonl：只索引当前 squad_append_v1 请求。
- dataset.json：记录数据版本、来源语义、切分策略和 ANNS token 预算。
- quality_eval/*：运行 align_squad_gold_answers.py 后重新生成。

旧版本可能遗留：

~~~text
research/data/mix_source/.cache/web/
~~~

当前生成器不会读取、更新或删除该目录。它不再属于当前数据链路。

## 环境依赖

不启用 token 预算时，生成器只依赖 Python 标准库。

启用 **--max-anns-document-tokens** 或
**--max-update-snapshot-tokens** 时，会从当前 .venv 使用
transformers.AutoTokenizer。生成器强制 local_files_only，只接受本地路径或
已经存在于本机 Hugging Face cache 的 tokenizer，不会在生成过程中下载模型。
vLLM 开发环境已经包含 transformers，不再需要安装 ddgs、httpx 或
trafilatura。

所有 Python 命令都从项目根目录使用：

~~~shell
.venv/bin/python
~~~

## 快速检查一个请求

不指定范围时默认只处理数值排序后的第一个 query：

~~~shell
.venv/bin/python \
  research/scripts/data_processing/build_mixed_source_dataset.py \
  --overwrite
~~~

这会真实写入 research/data/mix_source/requests/anns-query-000000.json，
但不会联网。

也可以显式选择多个 query：

~~~shell
.venv/bin/python \
  research/scripts/data_processing/build_mixed_source_dataset.py \
  --query-id 0 \
  --query-id 17 \
  --query-id 95 \
  --overwrite
~~~

## 按范围生成

从数值排序后的第 100 个 query 开始生成 20 个：

~~~shell
.venv/bin/python \
  research/scripts/data_processing/build_mixed_source_dataset.py \
  --start-index 100 \
  --max-queries 20 \
  --overwrite
~~~

--start-index 是排序后的位置，不是 query id。

只要传入任意 --query-id，生成器就只处理显式 id，并忽略 --start-index 与
--max-queries。

## 完全重新生成全部请求

--max-queries 0 表示处理 --start-index 之后的全部 ANNS query。

下面的完整重建命令带 --overwrite，因此旧网页版本和已经是当前版本的请求都会
按新规则重写：

~~~shell
.venv/bin/python \
  research/scripts/data_processing/build_mixed_source_dataset.py \
  --anns-dir research/data/anns \
  --squad-dir research/data/squad_v2 \
  --output-dir research/data/mix_source \
  --start-index 0 \
  --max-queries 0 \
  --overwrite
~~~

这条命令不启用任何 ANNS token 截断。它最接近原始 PipelinePool 内容，但生成
结果可能超过实际 P/D 服务的 --max-model-len。

## 适配 16K Llama 的推荐命令

如果 P、D 使用：

~~~text
MAX_MODEL_LEN=16384
~~~

建议保留完整 SQuAD context，同时只限制 ANNS 内容：

~~~shell
.venv/bin/python \
  research/scripts/data_processing/build_mixed_source_dataset.py \
  --anns-dir research/data/anns \
  --squad-dir research/data/squad_v2 \
  --output-dir research/data/mix_source \
  --start-index 0 \
  --max-queries 0 \
  --tokenizer-model <TOKENIZER_OR_MODEL_PATH> \
  --max-anns-document-tokens 1024 \
  --max-update-snapshot-tokens 12288 \
  --overwrite
~~~

参数语义：

- 1024 限制每个格式化后的 ANNS document。
- 12288 限制每一版完整 Update snapshot。
- 两个 SQuAD Append chunk 不会被这两个参数裁剪。
- 16K 窗口剩余空间用于固定模板、SQuAD context、问题和生成 token。

当前本机 Llama 示例：

~~~shell
.venv/bin/python \
  research/scripts/data_processing/build_mixed_source_dataset.py \
  --anns-dir research/data/anns \
  --squad-dir research/data/squad_v2 \
  --output-dir research/data/mix_source \
  --start-index 0 \
  --max-queries 0 \
  --tokenizer-model \
    /workplace/home/huangxinying/Models/Llama-3.1-8B-Instruct \
  --max-anns-document-tokens 1024 \
  --max-update-snapshot-tokens 12288 \
  --overwrite
~~~

如果实际服务只能使用 8K，可以改成更保守的：

~~~shell
--max-anns-document-tokens 512
--max-update-snapshot-tokens 6144
~~~

## Qwen 4K 性能数据集

质量评测使用的 `research/data/mix_source` 应保持完整 SQuAD context。若使用
Qwen2.5-3B 的 4K 窗口做纯性能实验，请写入独立目录，并同时限制两个来源：

~~~shell
.venv/bin/python \
  research/scripts/data_processing/build_mixed_source_dataset.py \
  --anns-dir research/data/anns \
  --squad-dir research/data/squad_v2 \
  --output-dir research/data/mix_source_perf_qwen \
  --start-index 0 \
  --max-queries 0 \
  --tokenizer-model ~/Models/Qwen2.5-3B \
  --max-append-chunk-tokens 256 \
  --max-anns-document-tokens 256 \
  --max-update-snapshot-tokens 1536 \
  --overwrite
~~~

该配置最多保留两个各 256-token 的 Append chunk，以及 1536-token 的 Update
snapshot；它不覆盖质量集，也不保证 SQuAD 答案仍被保留。

token 预算使用数据生成时指定的 tokenizer。跨模型实验应尽量复用同一批请求，
避免因 tokenizer 不同而改变 ANNS 文本内容；回放前再按目标模型检查实际长度。

## 断点续跑

当前版本请求满足以下标记时，默认会跳过：

~~~json
{
  "schema_version": "prompt_stream_mixed/v2",
  "data_revision": "squad_append_v1",
  "provenance": {
    "append_dataset": "squad_v2"
  }
}
~~~

因此全量命令中途停止后，可以重新执行同一条命令继续。

- 已经是当前版本且没有 --overwrite：跳过。
- 旧网页版本：即使没有 --overwrite 也会自动重建。
- 带 --overwrite：所有选中请求都重新构建。
- 关闭 --fail-fast 时，单个 trace、SQuAD 匹配或 corpus 文档错误只会让对应
  query 失败，其余 query 继续生成；进程最终仍以非零状态报告失败总数。

断点续跑应沿用相同 tokenizer、token 预算、prefix 和生成默认值。dataset.json
记录的是最后一次运行参数；若要改变这些参数，应使用同一套新参数全量
--overwrite，避免一个目录混入不同构建配置。

manifest.jsonl 只索引当前 SQuAD 版本请求。如果日志提示仍存在 legacy request，
不要立刻开始 replay；应执行覆盖全部 query id 的全量命令。生成器不会删除
不属于当前 ANNS query map 的孤立 JSON；这类文件也应在 replay 前从 requests/
移走，因为现有 replay 会直接扫描该目录，而不依赖 manifest。

## 生成后重新对齐官方答案

请求全部重建完成后，重新生成质量评测 manifest：

~~~shell
.venv/bin/python \
  research/scripts/data_processing/align_squad_gold_answers.py \
  --squad-dir research/data/squad_v2 \
  --anns-query-map research/data/anns/query_trace_map_5k.json \
  --requests-dir research/data/mix_source/requests \
  --output \
    research/data/mix_source/quality_eval/squad_gold_answers.jsonl \
  --summary-output \
    research/data/mix_source/quality_eval/squad_alignment_summary.json \
  --strict
~~~

对齐脚本会：

- 校验请求中的 query_id 和 question 是否仍与 ANNS query map 一致。
- 恢复 gold_answers 和 is_impossible。
- 拒绝有歧义的答案匹配。
- 输出请求级 JSONL 和聚合覆盖率摘要。

虽然 builder 已经在每个请求的 squad_context 中写入匹配信息，质量回放仍使用
独立 gold manifest，以避免生成输入和评分读取逻辑相互耦合。

## 主要参数

| 参数 | 默认值 | 作用 |
|---|---:|---|
| --anns-dir | research/data/anns | ANNS query、trace 和 corpus 根目录 |
| --squad-dir | research/data/squad_v2 | SQuAD v2 train/dev 根目录 |
| --output-dir | research/data/mix_source | 请求和数据集元信息输出目录 |
| --query-id | 未设置 | 显式选择 query id，可重复；设置后忽略位置范围 |
| --start-index | 0 | 数值排序后的起始位置 |
| --max-queries | 1 | 最大请求数，0 表示全部 |
| --tokenizer-model | 未设置 | 所有 token 预算使用的 tokenizer |
| --max-append-chunk-tokens | 0 | 单个格式化 SQuAD Append 事件上限；0 保留完整 context |
| --max-anns-document-tokens | 0 | 单个 ANNS 文档上限，0 表示关闭 |
| --max-update-snapshot-tokens | 0 | 单版 Update 上限，0 表示关闭 |
| --max-anns-document-chars | 0 | 单个 ANNS 文档字符上限 |
| --max-tokens | 64 | 请求保存的默认生成 token 上限 |
| --temperature | 0.0 | 请求保存的默认温度 |
| --prefix | 内置模板 | 固定 Prompt 前缀 |
| --overwrite | 关闭 | 重建选中的当前版本请求 |
| --fail-fast | 关闭 | 第一个问题失败时立即退出 |
| --log-level | INFO | 日志级别 |

Append chunk 数量固定为 2，不提供命令行参数。这样不同实验生成的数据具有相同
Append 事件数量，来源 interval 的含义也保持一致。

## 数据正确性约束

生成器会在写入每个请求前检查：

- 必须恰好声明一个 Append source 和一个 Update source。
- Append 数据事件必须恰好为 2 个。
- 每个来源的 sequence_no 必须从 0 连续递增。
- 数据事件的 version 必须从 0 连续递增。
- 每个来源最后一条事件必须是 source_finished。
- Append 文本不能为空。
- Update 文本不能为 null，但允许空字符串清空快照。
- finish.after_source_ids 必须同时包含两个来源。
- event_order.across_sources 必须是 unspecified。
- 数据中不得保存抓取耗时、trace 原始时间或 replay interval。

## 注意点

- **SQuAD context 不做截断**：这是保证官方证据存在的核心约束。
- **ANNS 仍可能不含答案**：Update 保留原始近似检索 workload 的噪声。
- **未匹配问题会失败**：生成器不会为没有唯一 SQuAD context 的问题伪造
  Append。
- **旧网页缓存不会自动删除**：它已脱离当前链路，但仍可能占用数据子模块空间。
- **父仓库与数据子模块分开提交**：脚本属于 vLLM 父仓库，
  research/data/mix_source 的生成结果属于数据子模块。
