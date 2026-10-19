# Append 证据与文本块设计

本文约定新数据格式，两个离线生成器已实现，运行命令见 [README](README.md)。
目标是把外部证据整理成可以
直接进入 prompt、可以追溯来源、可以重复回放的 Append 文本。

**建议保存“证据文档的来源信息 + 已经渲染好的有序文本块”。** 原始数据留在
`research/data/raw/`，问题和答案留在 `questions/`，Append 样本只保存选定证据。
时间频率由实验配置控制。

## Agent 与 RAG 中的证据长什么样

检索得到的文档通常包含两部分：正文，以及标题、URL、文档 ID 等元数据。
例如 LangChain 的 `Document` 使用 `page_content` 与 `metadata` 表达这两部分。
它是检索数据对象；进入模型前，还要由应用决定消息角色和文本格式。
见 [LangChain 文档对象源码](https://github.com/langchain-ai/langchain/blob/master/libs/core/langchain_core/documents/base.py)。

模型可见的内容可以是网页正文、带来源的摘录、搜索摘要，也可以是工具返回的
JSON。Agent 经常把它们放在工具结果消息中，RAG 也会把多个证据块整理进上下文。
**没有一个所有 Agent 都使用的统一字符串模板。** 对本项目，建议先使用简单的
“来源编号、标题、URL、正文”格式，便于模型识别边界和引用来源。

下面是自造示例，其正文并非来自数据集：

```markdown
### [A1] Example note
Source: https://example.org/note

Alpha started in 1990.
Beta ended in 2000.

---
```

`[A1]` 是当前问题的 Append 证据编号，标题和 URL 提供来源信息，正文提供事实。
缺少 URL 时省略 `Source:` 行；正文使用原文语言。
抓取状态、文件路径、检索分数、哈希和标准答案保存在数据字段中。

这是一种外部证据文本的表达方式。完整 Agent 的工具调用消息、推理轮次、用户
问题及模型 chat template 由请求层组织，Append 数据只负责其中的证据内容。

## 文档与发送块分别表示什么

| 对象 | 含义 | 示例 |
| --- | --- | --- |
| 证据文档 | 一份可识别来源的正文或摘录 | 一份网页抓取文本、一个 Wikipedia 段落 |
| 索引 chunk | 向量服务建立索引和返回候选时使用的单元 | 带重叠的检索窗口 |
| Append chunk | 本次请求中一次追加的不可修改文本 | 一份证据全文，或其连续片段 |

三者的大小可以不同。向量服务自行设计索引 chunk；Append 生成器负责把选定
证据按文档顺序渲染，再按发送粒度划分。

建议先定义每份文档的完整渲染文本 `R(document)`，再定义最终 Append：

```python
append_text = "".join(render_document(doc) for doc in selected_documents)
assert "".join(chunk["text"] for chunk in chunks) == append_text
```

同一份文档分一次或多次发送，都应得到相同的最终字符串。标题和 URL 在文档
开头出现一次，分隔符在末尾出现一次。发送块按连续区间划分，完整保留空白，
各块之间无需 replay 再加换行。

第一版默认采用 **一份证据文档一个 Append chunk**，接近网页抓取或检索结果
整份返回的场景。需要研究发送粒度时，再建立连续分段版本，优先在段落、句子
边界切分。可先试验约 512 tokens 的目标块长；这属于实验参数，不是行业标准。
长句的边界处理、实际最大块长和 tokenizer 版本都需要记录。

HotpotQA 提供 `--chunk-chars` 字符级连续分段，默认关闭。TriviaQA 的 v3 生成器
改为查询相关的 BM25 摘录，采用真实 tokenizer 限制正文窗口和渲染文本预算；
字符目标与 token 目标需要明确区分。

连续分段时保持文档头完整，块之间采用零重叠。索引中的重叠窗口若直接成为
证据，可作为真实检索结果保留；额外的 Append 分段不再人为重复正文。
多个文档之间也不通过重叠切块制造重复内容。

文本拼接等价还需要在请求层验证 token 等价。分别编码每个字符串块，得到的
token 拼接结果可能与编码完整文本不同。回放实现需要明确处理该边界问题，
比较发送粒度时应检查最终 token 序列；块级 token 数也不直接相加充当总长。

## 样本如何存储

每个数据版本使用 `metadata.json`、`manifest.jsonl` 和 `samples/*.json`。
例如 `research/data/append_chunks/triviaqa/web_unfiltered_dev_v1/`。
同一问题在不同证据版本中保持相同 `sample_id`，版本由所在目录和元数据标识。
文件名采用安全编码，manifest 保存 `sample_id` 到样本相对路径的映射。

每个样本的最小结构如下。这个自造示例把一份文档分成两个连续发送块；长度和
哈希对应 JSON 解码后的文本，开头的两个换行属于实际 prompt 内容。

```json
{
  "schema": "append_chunks/v1",
  "sample_id": "example:dev:q1",
  "status": "ready",
  "documents": [
    {
      "document_id": "example:note:v1",
      "citation_id": "A1",
      "source_type": "web",
      "title": "Example note",
      "url": "https://example.org/note",
      "provenance": {"kind": "synthetic_example"},
      "rendered_chars": 106,
      "rendered_sha256": "342e04717a86ee27c9309ad7811d575c40c682c9f05ce6b9ca1e181e48edad6a"
    }
  ],
  "chunks": [
    {
      "sequence_no": 0,
      "chunk_id": "a0000",
      "document_id": "example:note:v1",
      "rendered_span": [0, 81],
      "text": "\n\n### [A1] Example note\nSource: https://example.org/note\n\nAlpha started in 1990.\n"
    },
    {
      "sequence_no": 1,
      "chunk_id": "a0001",
      "document_id": "example:note:v1",
      "rendered_span": [81, 106],
      "text": "Beta ended in 2000.\n\n---\n"
    }
  ]
}
```

### 字段约定

`documents` 按实际采用的证据顺序排列，每个 `document_id` 唯一。这个 ID
标识来源及内容版本，不能只使用不带版本的 URL。`citation_id` 是模型可见的
简短编号，确定文档顺序后固定；改变发送频率或分段大小时保持不变。

`chunks` 按 `sequence_no` 从零连续排列。一块对应一份文档的一个连续片段，
其 `text` 是生成时已经确定的最终文本。replay 直接发送这些字符串。
JSON 是文件容器，模型实际看到的是解码后的 `text`。

`rendered_span` 使用 Unicode 字符的左闭右开区间，参照该文档完整的渲染文本，
包括标题、来源行与分隔符。它与原始文件的字节位置不同。需要句子级追溯时，
另存原始句子 ID 到正文或渲染文本区间的映射。

`provenance` 保存原始文件相对路径、QA 记录 ID、原始检索排名，以及内容版本。
HotpotQA 可以保存 parquet 文件与记录 ID、context 下标；语料文件可以保存
bz2 相对路径和页面 ID。路径均相对于 `research/data/`。

样本中保留实际发送的完整文本，来源路径只用于追溯，回放无需再次读取原文件。
第一版无需另存一份重复的完整正文；将来更换清洗或渲染规则时，从 `raw/` 重新
生成一个数据版本。选择规则、清洗规则和渲染模板是三个独立的版本参数。

### 版本元数据与清单

`metadata.json` 记录数据集及 split、原始输入指纹、证据选择规则、清洗规则、
渲染版本、分段策略、生成器版本，以及使用的 tokenizer 和内容预算。
这些参数决定证据内容；发送间隔、到达率和随机到达种子属于实验配置。

`manifest.jsonl` 每行记录 `sample_id`、相对路径、状态、文档数、块数、最终
Append 文本哈希及缺失证据统计。第一版建议区分：

- `ready`：按该版本约定完成证据生成。
- `empty`：该问题没有可用证据，仍保留问题及空块列表。
- `error`：解析或生成失败，记录原因。

`ready` 不承诺证据包含答案。选择本地可用证据的版本需要统计不可用候选；
要求完整恢复所有候选的版本遇到缺失文件时应报告失败。

## TriviaQA 如何转成 Append

本地抽样确认，Web 问答记录的 `SearchResults` 带有 `Title`、`Url`、`Rank`，
部分带 `Filename`，对应 `rc/evidence/web/` 下的文本。Wikipedia 问答记录的
`EntityPages` 带有标题和文件名，对应 `rc/evidence/wikipedia/`。

这些文本可以直接作为正文来源。Web 文本中仍可能有导航、广告和加载错误等
噪声；第一版建议只统一换行等必要格式，保留段落。进一步去噪需要单独的清洗
版本，按与答案无关的固定规则处理。

建议先构建 **unfiltered Web 的本地可用证据版本**：

1. 从 `unfiltered-web-dev.json` 读取问题 ID 与候选网页记录。
2. 按数值 `Rank` 排序，排名相同时沿用原始记录顺序。
3. 用 `Filename` 关联本地证据文件，保留标题、URL 和历史排名。
4. 对重复证据按固定规则去重，保留首次出现位置及其余来源记录。
5. 原生成规则渲染完整正文；当前规则按问题用 BM25 选择正文摘录，每份摘录一块。

v3 默认把相邻短行合成不超过 512 tokens 的原文连续窗口，优先沿句末或换行
切分，相邻窗口目标重叠 80 tokens。末尾不足半窗时向前补上下文，避免单独选出
标题碎片；短网页保留实际长度。筛选结合匹配词覆盖率、相对分数阈值、来源多样性
和重复过滤，最多选择 10 个摘录。单块渲染文本不超过 640 tokens，最终拼接
Append 不超过 6,144 tokens，均包含标题、URL 和分隔符。具体参数和运行方法
见 README。

每个摘录具有带字符区间的独立 `document_id`；`provenance.source_document_id`
关联完整来源，`source_body_sha256` 与 `source_body_span` 用于追溯统一换行后的
原文。BM25 分数只保存在来源字段中，不进入 prompt。空证据与无合格摘录分别
记录原因，不用标准答案补齐候选或凑满块数。默认仍使用原 v1 输出路径，
加 `--overwrite` 可重建 Append；metadata 中的生成规则版本为 3。
原始数据和已有 PipelinePool 保留，按相同 `sample_id` 关联新的 Append。

没有 `Filename` 或文件不存在时，记录不可用原因。`Description` 是搜索摘要，
可以用于另一个明确标记的摘要证据版本。正文版本只使用实际文件内容。
若后续进行实时搜索，保存真实搜索和抓取产物，再使用同一套渲染与存储结构。

官方说明 RC 数据经过答案字符串相关的筛选，unfiltered 更适合检索式 QA。
本地可用文件仍只覆盖历史候选的一部分，因此这个版本应标记为离线存档证据
回放。见 [TriviaQA 数据说明](https://nlp.cs.washington.edu/triviaqa/)。

可以另建 RC Web、RC Wikipedia 版本，用于给定证据条件下的性能和答案质量
实验。每个版本明确来源与筛选规则，结果中报告最终证据数量和问题覆盖率。

## HotpotQA 如何转成 Append

QA 的 `context` 表示标题和句子列表。可以保留标题及句子顺序，将一个 context
段落视作一份证据文档，再渲染为 Append 文本。句子连接时保留原有空白，缺少
词间分隔时按固定规则补空格，并保存原始句子编号映射。

官方 **Fullwiki dev** 的 context 是基线检索器返回的十个候选段落，可能缺少
正确证据。可以直接用它建立固定检索结果的 Append 版本；这批内容的来源应
标记为 `dataset_context`。训练 context 的构造条件需要单独标记，不能沿用
Fullwiki dev 的检索解释。见 [HotpotQA 数据格式说明](https://github.com/hotpotqa/hotpot/blob/master/README.md)。

语料包名称中的 `abstracts` 以及本地抽样的 `text` 表明，这里使用的是处理后
的 Wikipedia 摘要/段落正文，不能统一称为完整网页。把这份语料上传向量服务后，
服务返回的候选属于实际检索结果；是否保存到 Append 或 Update，由实验定义。

如果 Append 要模拟外部网页检索，HotpotQA 还需要额外的搜索与抓取步骤。
用当前 context 可以先研究固定证据的流式传入，并明确这种实验设置。

`answer` 和 `supporting_facts` 保存在问题/评价文件中。普通检索实验的证据
选择依据是 context 顺序或真实检索结果，不能依据标准答案挑选文档、只保留
supporting sentences，或把它们排到最前面。已知正确证据的受控实验可以另建
版本，并明确标记其使用了标注信息。

URL 优先使用数据源已有字段。依据标题构造的 Wikipedia URL 标记为派生地址，
它只方便展示来源，不代表抓取过当前页面。原始句子 ID 与新抓取网页之间的
对应关系也需要另行验证。

## 内容预算与评价

完整证据版本保存选定文档的全部正文。面向有限上下文窗口时，另建有明确预算
的版本，记录每文档上限、总 Append 预算、文档选择和截断规则。预算应计入
标题与分隔符，并为请求模板、Update、问题和输出预留空间。

改变发送粒度只调整分段边界；改变文档数量或截断正文会改变模型输入，属于
不同的证据版本。普通版本按排名、固定长度或真实检索结果选择内容，不使用
标准答案决定截断位置。答案覆盖统计可以在生成后离线计算。

TriviaQA 评价保留完整答案对象，采用官方的 alias 与归一化逻辑。若把多个网页
聚合成每问题一次预测，应明确使用问题级评价；官方 Web RC 的原始评测键是
问题与文档组合，不能直接声称两者分母一致。
见 [TriviaQA 评价代码](https://github.com/mandarjoshi90/triviaqa/blob/master/evaluation/triviaqa_evaluation.py)
及 [评测键构造](https://github.com/mandarjoshi90/triviaqa/blob/master/utils/dataset_utils.py)。

HotpotQA 可先报告答案 EM/F1。需要 supporting-fact 或 joint 指标时，再定义
包含证据句子编号的模型输出协议，并验证输出编号能映射回原始句子。
见 [HotpotQA 评价代码](https://github.com/hotpotqa/hotpot/blob/master/hotpot_evaluate_v1.py)。
缺失证据和空样本仍纳入问题覆盖率统计；有证据子集的指标单独标明。

## 第一版实现范围

统一的文档渲染与样本格式，以及 TriviaQA unfiltered Web 本地证据、HotpotQA
Fullwiki dev context 两个离线适配器已实现。TriviaQA 默认采用 BM25 原文摘录
与 token 预算，HotpotQA 保留 context 文档及可选的字符级连续分段。
真实网页抓取和向量服务结果采集可复用同一结构。

生成时检查文档 ID 唯一、顺序连续、分段区间无缝衔接，以及每份文档重组后的
字符数和 SHA-256。比较分段版本时检查最终 Append 哈希相同；回放端再验证
模型实际输入 token 等价。

两个数据集均保留原文语言；TriviaQA 选择原文连续摘录以控制冗长网页，HotpotQA
保留短 context 文档。无需把已有问答 JSON 或 parquet 整条序列化进 prompt。
