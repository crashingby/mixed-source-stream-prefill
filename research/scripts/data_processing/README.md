# 科研数据处理

新数据的 Append 组织方式见 [Append 证据与文本块设计](APPEND_CHUNKS_DESIGN.md)。
该文档讨论 TriviaQA、HotpotQA 的证据来源、prompt 文本与独立存储格式，
已实现两个独立 Append 生成器及 Wikipedia Update 采集器；
旧混合数据生成器的说明保留在后文。

## 新数据 Append 生成

安装读取依赖后，每个脚本不带参数即可生成对应数据集的全部 dev 请求：

```shell
uv pip install --python .venv/bin/python \
  -r research/scripts/data_processing/requirements.txt

.venv/bin/python research/scripts/data_processing/build_triviaqa_append_chunks.py
.venv/bin/python research/scripts/data_processing/build_hotpotqa_append_chunks.py
```

默认输入和输出路径由脚本所在位置推导，运行时的工作目录不会改变默认位置。
上述相对命令从项目根目录执行；在其他工作目录使用脚本和解释器的绝对路径。

| 脚本 | 默认输入 | 默认 Append 输出 |
| --- | --- | --- |
| `build_triviaqa_append_chunks.py` | `research/data/raw/triviaqa/` 中的 unfiltered Web dev QA 和 `rc/evidence/web/` 正文 | `research/data/append_chunks/triviaqa/web_unfiltered_dev_v1/` |
| `build_hotpotqa_append_chunks.py` | `research/data/raw/hotpotqa/qa/fullwiki/validation-00000-of-00001.parquet` | `research/data/append_chunks/hotpotqa/fullwiki_dev_context_v1/` |

默认独立问题清单分别写入：

- `research/data/questions/triviaqa/web_unfiltered_dev_v1.jsonl`
- `research/data/questions/hotpotqa/fullwiki_dev_context_v1.jsonl`

每个版本包含 `metadata.json`、`manifest.jsonl`、`samples/*.json`。
manifest 的 `sample_id` 与问题清单对应，`path` 指向该版本内的样本相对路径。
文件名采用 sample ID 的 SHA-256，避免原始 ID 中的特殊字符影响路径。
每个样本的 `chunks[].text` 是实际进入 prompt 的文本，包含证据编号、标题、
已有 URL、正文与分隔符；来源信息存于 `documents`。TriviaQA 一份选定摘录一块，
HotpotQA 一份 context 文档一块。问题、答案和 supporting facts 只写入独立问题清单。

TriviaQA 只采用具有本地正文的候选，用问题对正文段落进行 BM25 排序；历史排名
用于读取顺序和同分时的决策。没有 `Filename`、缺少文件或正文为空时记录原因。
HotpotQA 使用原始 context 顺序，保留句子 ID
与正文区间映射。Fullwiki dev context 是固定的官方检索结果；训练 context
单独标记。两个脚本均离线运行，证据选择不使用答案或 supporting facts。

### TriviaQA 原文摘录与预算

默认 tokenizer 是项目所属用户的 `Models/Qwen2.5-3B/`，当前机器对应
`/workplace/home/huangxinying/Models/Qwen2.5-3B/`。只读取本地分词文件，不加载
模型权重，不下载模型，不调用 GPU。使用其他生成模型时，指定它的本地
`--tokenizer` 目录，预算按实际使用的分词器计算。目录必须包含 `tokenizer.json`，
脚本使用 `tokenizers` 直接读取它，不依赖 `transformers` 的模型加载入口。

1. 把整份网页正文切成最多 512 tokens 的连续窗口，优先在句末或换行处结束，
   相邻窗口目标重叠 80 tokens。短行和标题与邻近原文一起进入窗口，避免每行独立
   成为短摘录；不足半窗的末尾向前补上下文。原网页本身较短时保留其实际长度。
2. 在当前问题的全部候选正文中计算 BM25（k1=1.5、b=0.75），按大小写无关的
   单词匹配，去除固定英语停用词，不使用标题代替正文匹配。问题中的年份、数字
   全部匹配时，排序分数最多再乘以 1.5；缺少数字的证据仍可入选。
3. 默认至少匹配 2 个不同问题词，且覆盖 30% 的有效问题词；问题词更少时
   自动降低词数要求。保留加权分数至少为最高合格候选 20% 的段落。
4. 过滤同源重叠窗口和跨网页近重复文本。已采用来源的排序分数乘以 0.7，软性
   优先不同网页；同一页仍可贡献多个不同摘录。不为凑满数量保留低匹配评论。
5. 每问题最多 10 块，每块最多 640 tokens，拼接后的完整 Append 最多
   6,144 tokens。标题、URL 和分隔符都计入预算；超限候选跳过，继续检查其他窗口。

`documents[].provenance` 保存完整正文哈希、原始文档 ID、`source_body_span`
字符区间、原始 BM25 分数、加权分数和匹配词数。摘录始终是统一换行后的原文连续子串，没有模型
摘要或改写。`append_tokens` 与 `chunks[].text_tokens` 保存实际渲染文本的计数；
总长重新编码最终拼接字符串，不能简单相加各块计数。

| TriviaQA 参数 | 默认值 | 含义 |
| --- | --- | --- |
| `--tokenizer` | 上述本地 Qwen 目录 | 必须提供支持字符偏移的 fast tokenizer |
| `--append-tokens` | `6144` | 每问题完整 Append 上限 |
| `--chunk-tokens` | `640` | 单块上限，包含来源头 |
| `--max-chunks` | `10` | 摘录数量上限，不要求填满 |
| `--passage-tokens` / `--overlap-tokens` | `512` / `80` | 候选正文窗口与重叠目标；末尾补上下文可能增加重叠，设为 0 则关闭重叠 |
| `--min-matched-terms` | `2` | 不同匹配词的数量下限 |
| `--min-query-coverage` | `0.3` | 不同问题词的覆盖比例下限 |
| `--min-score-ratio` | `0.2` | 相对于最高合格分数的下限 |

BM25 按字词匹配，不理解同义改写，不能保证选中的摘录包含答案。有本地正文但
没有满足匹配和预算规则的段落时，也生成 `empty` 样本，原因见
`diagnostics.selection_empty_reason`；问题与评价仍保存。

默认沿用 `web_unfiltered_dev_v1/` 和原 v1 问题清单路径。已有数据时加
`--overwrite`，用新规则重建 Append；脚本完成后才替换结果。
如果 Append 目录已删除而问题清单仍存在，可以不带 `--overwrite` 重新生成。
脚本逐行校验问题清单与本次输入完全一致后复用它，不改写已有 QA；内容或请求数量
不同则停止发布，需要指定新的 `--questions-file`。
内容生成规则在 metadata 中记录为版本 3，算法为
`query_bm25_context_windows/v3`，目录名不代表筛选算法版本。
相同 split 的问题和 `sample_id` 不变，已有 PipelinePool 可按 ID 复用，
不需要重新上传或检索 Wikipedia。自定义筛选参数或 tokenizer 自动追加
`_cfg<指纹>`，防止不同配置混淆。

覆盖已有默认数据：

```shell
.venv/bin/python research/scripts/data_processing/build_triviaqa_append_chunks.py --overwrite
```

### 常用选项

| 参数 | 默认值 | 含义 |
| --- | --- | --- |
| `--split` | `dev` | 生成 train/dev；TriviaQA 另支持 test；`all` 处理所有已约定 split |
| `--variant` | `web-unfiltered` | 仅 TriviaQA；可选 `web-rc`、`wikipedia-rc` |
| `--max-requests` | `0` | 全部问题；正数用于小规模试跑 |
| `--max-documents` | `0` | TriviaQA 不额外限制不同来源数；HotpotQA 正数保留前若干 context 文档 |
| `--chunk-chars` | `0` | 仅 HotpotQA 可设正数，按 Unicode 字符连续分段；TriviaQA 使用 token 预算 |
| `--data-dir` | 对应 `raw/<dataset>/` | 改用其他原始数据根目录 |
| `--output-dir` | 上表路径 | 指定 Append 数据版本目录 |
| `--questions-file` | 上述问题清单 | 指定独立问题清单 |
| `--overwrite` | 关闭 | 重建同一生成器已有输出；拒绝覆盖其他目录或额外文件 |
| `--log-every` | `500` | 每处理指定数量问题输出进度；0 关闭周期日志 |

非零数量限制和分段选项会自动追加版本后缀，例如 `--max-requests 16` 生成
`web_unfiltered_dev_v1_first16/`，问题清单也带相同后缀。这些试跑输出独立于
默认完整版本；显式提供输出路径时使用指定路径。

HotpotQA 的 `--chunk-chars` 是字符级分段参数。标题较长时首块可能超过目标字符数。
所有分段按顺序直接拼接即可恢复同一文档的完整 prompt 文本；改变分段不会
重复标题或正文。HotpotQA 仍保留 context 全文，没有 token 总预算。
TriviaQA 的预算只覆盖 Append；回放端还需要给 Update 最大候选池、问题、模板和
输出留出空间，并检查服务端上下文上限。默认 replay 传原文，由服务端分词，分段边界可导致
normal 与 stream 的 token 序列不同。

本地 Qwen2.5-3B 的 `config.json` 记录模型窗口为 32,768 tokens，服务实际窗口
仍由 P 节点的 `MAX_MODEL_LEN` 限制。使用新的 TriviaQA 预算时建议设置
`MAX_MODEL_LEN=16384`；旧的 8,192 不能仅凭模型支持 32K 就认为足够。
Update 会替换已有区域，容量按**最大的中间快照**计算，而不是累加所有更新版本，
也不能只看最终快照。开启 masked padding 时还要计入额外占位 token；扩大窗口
不代表 GPU 能同时容纳原先同样数量的长请求。

没有可用证据的问题仍生成 `empty` 样本。逐请求处理失败生成 `error` 样本，
汇总到 metadata，命令返回非零状态；正常缺失候选不等同于处理错误。
完整结果先在临时目录构建，再发布 Append 与问题清单；失败重建保留旧输出。
新格式可通过 `replay --dataset triviaqa/hotpotqa` 或 `--config` 回放；入口与
使用说明见 [replay README](../replay/README.md)。

运行生成器测试：

```shell
.venv/bin/python -m unittest discover \
  -s research/scripts/data_processing/tests -v
```

## Wikipedia Update 采集

两个入口先将整个 Wikipedia 集合送入检索服务，再读取已有独立问题清单，
逐请求采集 `/search/stream` 的候选池。默认 **k=5、ef=8**；每次请求显式传入
ef，因此会覆盖服务启动时的默认值。候选池最多 8 块，最终结果最多 5 块。
较小 ef 是否足够满足检索质量，需要通过答案评价和检索指标确认。

### 启动服务并生成

在两个终端分别启动独立的服务数据库。现有接口没有数据集过滤，两个数据集
不能混用同一数据库。服务首次启动所需依赖见 hnswlib 的
`http_service/README.md`，采集脚本只使用 Python 标准库，不增加安装依赖。

TriviaQA 服务：

```shell
cd /workplace/home/huangxinying/Projects/hnswlib
.venv/bin/python -m http_service.server \
  --model-path /workplace/home/huangxinying/Models/bge-m3 \
  --db-path var/http_service/triviaqa_wiki_v1.sqlite3 \
  --device cuda:0 --host 127.0.0.1 --port 8091 \
  --chunk-size 800 --chunk-overlap 100 --ef 8 \
  --embedding-batch-size 64 --index-threads 8
```

HotpotQA 服务：

```shell
cd /workplace/home/huangxinying/Projects/hnswlib
.venv/bin/python -m http_service.server \
  --model-path /workplace/home/huangxinying/Models/bge-m3 \
  --db-path var/http_service/hotpotqa_wiki_v1.sqlite3 \
  --device cuda:1 --host 127.0.0.1 --port 8092 \
  --chunk-size 800 --chunk-overlap 100 --ef 8 \
  --embedding-batch-size 64 --index-threads 8
```

服务就绪后，从 vLLM 项目根目录运行：

```shell
.venv/bin/python research/scripts/data_processing/build_triviaqa_pipeline_pools.py
.venv/bin/python research/scripts/data_processing/build_hotpotqa_pipeline_pools.py
```

首次重启新版服务会添加名称索引和幂等标识表，再从已保存向量恢复 HNSW。
恢复进度打印在服务日志中；看到 `Application startup complete` 后运行采集器。
原有服务数据库和 `_state` 文件直接复用，已确认文章跳过，未确认文章先核对
旧名称和正文哈希。无需删除原数据库或客户端进度。

默认路径由脚本位置推导，不受启动目录影响。默认输入和输出如下：

| 数据集 | Wikipedia 输入（相对 research/data） | 问题输入（相对 research/data） | PipelinePool 输出（相对 research/data） |
| --- | --- | --- | --- |
| TriviaQA | `raw/triviaqa/rc/evidence/wikipedia/` 下的全部 `.txt` | `questions/triviaqa/web_unfiltered_dev_v1.jsonl` | `pipeline_pools/triviaqa/wiki_dev_k5_ef8_v1/` |
| HotpotQA | `raw/hotpotqa/corpus/enwiki-20171001-pages-meta-current-withlinks-abstracts/` 下的全部 `.bz2` | `questions/hotpotqa/fullwiki_dev_context_v1.jsonl` | `pipeline_pools/hotpotqa/wiki_dev_k5_ef8_v1/` |

TriviaQA 文章标题由文件名推导，Wikipedia URL 由该标题构造，并非实时验证。
HotpotQA 使用语料中的文章 ID、标题和 URL；读取无 HTML 的 `text` 字段。
HotpotQA 的语料是 Wikipedia 摘要，不是完整网页正文。两个入口保留可用文章
正文，不用答案、supporting facts 或逐问题 gold 文档关联筛选入库集合。
空文章跳过并输出日志，损坏文件或非法正文结构会终止入库，不能静默当成完整语料。

当前 TriviaQA Append 来自 Web 证据，本采集器 Update 来自 Wikipedia，
不能保证覆盖每一个 Web 问题的答案。即便某问题没有 Append chunks，仍照常检索。
问题清单必须已存在；采集器不修改 QA、评价字段或 Append 数据。

### 参数与续跑

| 参数 | 默认值 | 含义 |
| --- | --- | --- |
| `--server-url` | TriviaQA 为 `http://127.0.0.1:8091`，HotpotQA 为 `http://127.0.0.1:8092` | 对应数据集的检索服务 |
| `--k` / `--ef` | `5` / `8` | 最终块数与候选池容量；要求 `ef >= k` |
| `--data-dir` | `research/data/raw/<dataset>/` | 原始数据根目录 |
| `--questions-file` | 上表问题路径 | 改用其他问题清单 |
| `--output-dir` | 上表输出路径 | 覆盖版本目录 |
| `--state-db` | `pipeline_pools/<dataset>/_state/wiki_v1_<服务地址哈希>.sqlite3` | 客户端导入进度及文章来源映射，没有向量索引 |
| `--import-only` | 关闭 | 只入库，不要求问题清单存在 |
| `--skip-import` | 关闭 | 使用该状态文件已记录的文章，不遍历原始语料 |
| `--max-documents` | `0` | 全库入库上限，0 表示全部；不是每问题块数 |
| `--max-requests` | `0` | 问题清单前 N 个请求，0 表示全部 |
| `--timeout` / `--retries` | `300` 秒 / `3` 次 | 网络等待与失败重试次数 |
| `--import-batch-size` | `256` | 一次 HTTP 请求的文章上限，范围 1～512 |
| `--import-batch-bytes` | `4194304` | 每批正文 UTF-8 字节上限，最多 8 MiB；单篇超限会停止并提示增大上限 |
| `--log-every` | `100` | 入库及检索进度输出间隔；0 关闭周期日志 |

第一次运行导入整个语料，再检索。完整入库后，后续运行自动跳过原始语料遍历；
已经成功的请求不重复检索，失败请求重新尝试。也可先 `--import-only`，
稍后执行默认命令完成采集。修改 k/ef 会生成独立版本并复用同一客户端入库状态。
修改问题清单或显式复用输出目录时，脚本会检查配置和语料指纹，拒绝混合版本。

采集器使用新版服务的 `/documents/batch` 和 `/documents/lookup`。
每篇文章使用由数据集和原始来源 ID 派生的稳定 `ingest_key`；同一标识和正文
重复提交只返回原文档 ID，标识或旧名称对应不同正文时服务返回 HTTP 409。
脚本在 POST 前用一笔事务保存整批 pending，确认响应的 ID、名称和正文哈希后
用另一笔事务保存整批 ready。响应丢失可以安全重试，耗尽重试后保留 pending；
重新运行会通过名称索引批量核对，包括旧版单篇接口上传的记录。
同一服务数据库应始终保留并复用同一 `--state-db`，不要手工导入其他文章。

默认一次发送最多 256 篇，服务合并文章的文本块，按每组 64 块进行 GPU 编码，
用 8 个线程更新 HNSW，并整批提交服务数据库。两组参数分别由客户端
`--import-batch-size` 和服务 `--embedding-batch-size`、`--index-threads` 控制。
批量日志包含编码、索引和数据库毫秒数，进度日志包含本轮平均确认速度。
批量参数不改变客户端状态、默认输出路径、800/100 字符分块或 k/ef。
服务在内存中保留 HNSW 和 chunk 文本。
小规模试跑可用 `--max-documents 100 --max-requests 16`，自动生成
`wiki_dev_k5_ef8_v1_docs100_first16/`。试跑请使用独立的服务数据库和端口。
限量入库记录为 `subset`；沿用同一状态文件去掉限制可以继续补全语料，
但之前已采集的子集结果保持独立，不属于全库检索结果。

### PipelinePool 格式

每个版本包含 `metadata.json`、`manifest.jsonl`、`samples/<sample_id哈希>.json`。
metadata 记录输入文件指纹、语料实例指纹、服务模型及文档/块计数、k/ef 和采集状态。
manifest 的 `sample_id` 与 questions、Append 对齐，`path` 是样本相对路径。

- `chunks` 是去重后的块目录，保存服务 ID、原始来源、字符偏移、`body` 和
  可直接拼进 prompt 的 `text`。`text` 包含稳定引用编号、标题、URL、正文和分隔符。
- `snapshots` 按收到事件的顺序保存。每个快照的 `chunk_ids` 是完整有序候选池，
  按顺序拼接块目录中对应的 `text` 即为该次 Update 全文；快照不是增量。
- 每个快照另外保存距离、原始序号、原始耗时和渲染文本指纹。这些变化的数值
  不放进 prompt，避免干扰前缀复用。最后一个快照来自 `final`，仅保留最终 Top-k。
- `final` 保留服务的实际 k/ef、更新次数和耗时。`source_elapsed_ms` 是候选事件
  的原始检索计时，final 快照该字段为 null；replay 配置决定实际发送频率。

快照流必须有匹配的 `final` 和 `done` 才写为 `ready`。错误或断流记录为 `error`，
汇总到 metadata 并返回非零退出码；下次运行会重试。候选流接收期间不逐快照写盘，
收完整个请求后再构造样本并原子保存。采集期间应保持服务数据库固定。
新格式可通过 `replay --dataset triviaqa/hotpotqa` 或 `--config` 回放；入口与
使用说明见 [replay README](../replay/README.md)。

测试使用临时语料和本机模拟 HTTP 服务，不向真实向量服务上传数据：

```shell
.venv/bin/python -m unittest discover \
  -s research/scripts/data_processing/tests -p test_pipeline_pools.py -v
```

## 旧混合数据生成

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

```text
SQuAD 官方 context
├── Append chunk 0：context 前半部分
└── Append chunk 1：context 后半部分
```

默认情况下，两个 chunk 合起来完整覆盖官方 context，不会把整篇 article 的
全局前两个 paragraph 当作请求证据，也不会对 SQuAD context 做 token 截断。
这套无损模式用于质量评测。

`--max-append-chunk-tokens` 可为每个格式化后的 Append 事件设置独立 token
上限，包装标签也计入预算。它适合 4K 上下文窗口的性能实验：超过预算的正文会
被直接截断并写入 `[truncated]` 标记。启用后官方答案可能落在被删除的后缀，
因此不能将该数据集用于 SQuAD EM/F1 质量结论。

Append 文本格式为：

```text
<squad_context
  split="train"
  title="University_of_Notre_Dame"
  squad_id="..."
  segment="1/2">
...
</squad_context>
```

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

```text
<retrieved_document id="...">
...
</retrieved_document>
```

Update 可以设置单文档和整版快照 token 预算；Append 的上限由独立的
`--max-append-chunk-tokens` 控制。

### 最终逻辑 Prompt

普通模式最终构造：

```text
固定模板
+ 两个 SQuAD Append chunk
+ 最终 ANNS Update snapshot
+ 用户问题
```

Segmented 模式通过两个 source 的 RESTFUL 事件得到相同的逻辑内容。跨来源
事件的实际交错顺序仍由 replay 控制。

## RESTFUL 事件兼容性

请求继续使用：

```text
schema_version = prompt_stream_mixed/v2
data_revision = squad_append_v1
```

保留 v2 是为了兼容现有：

- replay_normal.py
- replay_prompt_stream.py
- replay_merged_update.py
- update-coalescing replay
- generation-quality replay

每条 events[].event 继续包含：

```json
{
  "source_id": "squad:0",
  "event_type": "append",
  "source_type": "append",
  "chunk_id": "squad:0-context-0",
  "version": 0,
  "sequence_no": 0,
  "text": "..."
}
```

Update 事件仍使用 event_type="update" 和 source_type="update"。
两个来源内部的 sequence_no 都从 0 连续递增，最后一条均为
source_finished。

数据文件只声明：

```json
{
  "event_order": {
    "storage_layout": "grouped_by_source",
    "within_source": "sequence_no",
    "across_sources": "unspecified"
  }
}
```

因此，文件中的来源分组顺序不代表运行时到达顺序。

## 输入数据

默认读取：

```text
research/data/
├── anns/
│   ├── query_trace_map_5k.json
│   ├── res/
│   │   └── *_pipeline_trace.csv
│   └── retrieved_corpus_content.part.*.json
└── squad_v2/
    ├── train-v2.0.json
    └── dev-v2.0.json
```

query_trace_map_5k.json 提供：

- query：最终问题，同时用于查找 SQuAD context。
- trace_file：该问题对应的 ANNS PipelinePool trace。

## 输出目录

默认写入：

```text
research/data/mix_source/
├── dataset.json
├── manifest.jsonl
├── requests/
│   └── anns-query-000000.json
└── quality_eval/
    ├── squad_gold_answers.jsonl
    └── squad_alignment_summary.json
```

- requests/*.json：可直接交给现有 replay 的请求数据。
- manifest.jsonl：只索引当前 squad_append_v1 请求。
- dataset.json：记录数据版本、来源语义、切分策略和 ANNS token 预算。
- quality_eval/*：运行 align_squad_gold_answers.py 后重新生成。

旧版本可能遗留：

```text
research/data/mix_source/.cache/web/
```

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

```shell
.venv/bin/python
```

## 快速检查一个请求

不指定范围时默认只处理数值排序后的第一个 query：

```shell
.venv/bin/python \
  research/scripts/data_processing/build_mixed_source_dataset.py \
  --overwrite
```

这会真实写入 research/data/mix_source/requests/anns-query-000000.json，
但不会联网。

也可以显式选择多个 query：

```shell
.venv/bin/python \
  research/scripts/data_processing/build_mixed_source_dataset.py \
  --query-id 0 \
  --query-id 17 \
  --query-id 95 \
  --overwrite
```

## 按范围生成

从数值排序后的第 100 个 query 开始生成 20 个：

```shell
.venv/bin/python \
  research/scripts/data_processing/build_mixed_source_dataset.py \
  --start-index 100 \
  --max-queries 20 \
  --overwrite
```

--start-index 是排序后的位置，不是 query id。

只要传入任意 --query-id，生成器就只处理显式 id，并忽略 --start-index 与
--max-queries。

## 完全重新生成全部请求

--max-queries 0 表示处理 --start-index 之后的全部 ANNS query。

下面的完整重建命令带 --overwrite，因此旧网页版本和已经是当前版本的请求都会
按新规则重写：

```shell
.venv/bin/python \
  research/scripts/data_processing/build_mixed_source_dataset.py \
  --anns-dir research/data/anns \
  --squad-dir research/data/squad_v2 \
  --output-dir research/data/mix_source \
  --start-index 0 \
  --max-queries 0 \
  --overwrite
```

这条命令不启用任何 ANNS token 截断。它最接近原始 PipelinePool 内容，但生成
结果可能超过实际 P/D 服务的 --max-model-len。

## 适配 16K Llama 的推荐命令

如果 P、D 使用：

```text
MAX_MODEL_LEN=16384
```

建议保留完整 SQuAD context，同时只限制 ANNS 内容：

```shell
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
```

参数语义：

- 1024 限制每个格式化后的 ANNS document。
- 12288 限制每一版完整 Update snapshot。
- 两个 SQuAD Append chunk 不会被这两个参数裁剪。
- 16K 窗口剩余空间用于固定模板、SQuAD context、问题和生成 token。

当前本机 Llama 示例：

```shell
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
```

如果实际服务只能使用 8K，可以改成更保守的：

```shell
--max-anns-document-tokens 512
--max-update-snapshot-tokens 6144
```

## Qwen 4K 性能数据集

质量评测使用的 `research/data/mix_source` 应保持完整 SQuAD context。若使用
Qwen2.5-3B 的 4K 窗口做纯性能实验，请写入独立目录，并同时限制两个来源：

```shell
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
```

该配置最多保留两个各 256-token 的 Append chunk，以及 1536-token 的 Update
snapshot；它不覆盖质量集，也不保证 SQuAD 答案仍被保留。

token 预算使用数据生成时指定的 tokenizer。跨模型实验应尽量复用同一批请求，
避免因 tokenizer 不同而改变 ANNS 文本内容；回放前再按目标模型检查实际长度。

## 断点续跑

当前版本请求满足以下标记时，默认会跳过：

```json
{
  "schema_version": "prompt_stream_mixed/v2",
  "data_revision": "squad_append_v1",
  "provenance": {
    "append_dataset": "squad_v2"
  }
}
```

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

```shell
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
```

对齐脚本会：

- 校验请求中的 query_id 和 question 是否仍与 ANNS query map 一致。
- 恢复 gold_answers 和 is_impossible。
- 拒绝有歧义的答案匹配。
- 输出请求级 JSONL 和聚合覆盖率摘要。

虽然 builder 已经在每个请求的 squad_context 中写入匹配信息，质量回放仍使用
独立 gold manifest，以避免生成输入和评分读取逻辑相互耦合。

## 主要参数

| 参数 | 默认值 | 作用 |
| --- | ---: | --- |
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
