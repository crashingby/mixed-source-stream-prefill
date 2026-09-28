# 混合来源流式 Prefill：当前设计、实现与实验总结

本文是混合来源流式 Prefill 原型的阶段性总览，记录当前代码已经完成的设计、实现与实验，也明确区分尚未实现的下一阶段工作。目标是让第一次接触项目的读者能够理解：为什么需要分段 KV、事件如何进入长期请求、一个事件何时成为可调度 work、不同模式的 KV block 如何独立管理、P/D 节点如何交接，以及目前的性能和输出质量结论能够支持到什么程度。

文中的状态标签含义如下：

| 状态 | 含义 |
| --- | --- |
| **已实现并验证** | 已进入源码，并有单元测试、最小链路测试或正式实验结果支撑。 |
| **已实现，仍需扩大验证** | 主路径已经接入，但仍需更多配置、模型或压力场景验证。 |
| **设计完成，尚未实现** | 已完成源码调研和接口设计，当前运行结果仍使用旧行为。 |

## 当前结论摘要

- **Segmented KV 已实现并形成端到端闭环。** 一个逻辑请求在 `Request` 中长期存在，模板、Append、Update 和最终问题通过事件逐步到达；物理 KV 只分成 `PREFIX_APPEND` 与 `UPDATE` 两个独立域。
- **请求内工作采用显式 FIFO。** 模板 bootstrap 必须先完成；随后已经准入的 Append、Update、Question 都表示为不可变 `PromptStreamWorkItem`，依次经过 `pending_work_items -> active_work_item -> committed KV`。Question 不能越过尚未完成的知识事件。
- **Update 复用采用 source-local LCP。** LCP 只比较相邻 Update 完整快照，不把模板或 Append token 混入 LCP；对应 KV 能否复用还取决于该 Update 所绑定的 Append epoch。
- **P/D 精确交接已经显式描述物理布局。** P 节点导出 `PREFIX_APPEND blocks + UPDATE blocks`、物理 prompt token、block 数量和长度边界；D 节点按 manifest 拉取已经 forward 的 prompt KV，并在本地 forward P 产生的首 token。
- **历史实验包含两项有意近似。** 第一项是 `PREFIX_APPEND` 尾部 block 对齐 padding 会真正参与 attention；第二项是早先计算完成的 Update 不会因后来 Append 增长而回滚重算。新加入的可选 Masked Padding 路径针对第一项，历史结果不能视为新路径实验。
- **性能结果支持分段计算的主张，但存在高压拐点。** 在 Qwen2.5-3B + Triton、每组 512 请求的实验中，QPS 为 0.5、1、2 时 Segmented 的 TTFT 分位数总体最好；QPS=4 时事件拆分造成的调度排队使它落后于一次性 Normal 请求。
- **输出质量没有观察到总体退化，但不等价。** 4980 条 SQuAD v2 对齐请求中，Normal 为 51.81 EM / 66.36 F1，Segmented 为 52.89 EM / 67.12 F1；不过有 121 条 Normal 正确样本在 Segmented 中变错，因此不能宣称逐请求严格等价。
- **流式请求统一停放。** 所有 PromptStream 在模板完成、active/pending work 耗尽且无 GPU 在途步骤时进入 dormant，保留 KV 并释放 running 名额；新 work 到达后经 waiting 重新准入，与 Update 合并开关无关。完整规则见 [Scheduler 笔记](Scheduler.md)。
- **Update coalescing 已接入源码。** 它在 Update work 准入前维护唯一可覆盖窗口，支持六种 Update policy、时间与执行反馈唤醒、dormant 请求和抢占恢复指标；policy 满足即放行，不等待 Append source 结束。实验表明收益集中在 Update 高频、P 节点有排队压力的区域，不是无条件优化。
- **Masked Padding 已实现，等待完整模型/P-D 验证。** `--prompt-stream-masked-padding` 启用 Triton range mask 与 RoPE 位置修正，保持物理 block 拓扑；已通过无服务状态机、proxy 和单 kernel 测试，尚未重新运行完整性能及质量实验。

## 问题模型与核心思想

一个混合来源请求的最终逻辑 Prompt 写为：

$$
X = T \mathbin\Vert A \mathbin\Vert U \mathbin\Vert Q.
$$

其中：

- $T$ 是启动请求携带的固定模板，例如 system prompt 和 chat template 前缀。
- $A$ 是 Append 源单调累积的内容；新事件只在尾部增加 token。
- $U$ 是 Update 源的当前完整快照；新版本可以改写旧版本的任意后缀。
- $Q$ 是 `PROMPT_FINISHED` 携带的最终用户问题与答案起始模板。

普通推理只有在 $T+A+U+Q$ 全部准备好后才一次性 prefill。流式方案希望在知识来源生成过程中提前计算可复用 KV：

- Append 的历史不会被新 Append 推翻，因此 $T+A$ 适合维护为单调增长的稳定域。
- Update 会反复修改，只有相邻版本的最长公共前缀可以继续复用，因此必须拥有独立失效边界。
- 最终 Question 必须等所有已经准入的知识工作完成，但不应把已经完成的 Append 与 Update 全部重新 prefill。

由此形成当前最重要的设计决定：**逻辑上有四段，物理上只有两个 KV 域。**

~~~text
逻辑序列：T | A | U | Q

物理 KV：
  PREFIX_APPEND = T | A | block 对齐 padding
  UPDATE        = U | Q | 后续 decode token
~~~

Question 与 decode 没有第三个 KV key，而是接在 `UPDATE` 域后面。这样既保留了 Update 的独立失效能力，也避免再增加一个物理拼接边界。

## 端到端控制流

### 从 REST 请求到长期 Request

P 节点在 OpenAI server 中注册独立的 Prompt Stream router：

| 路由 | 作用 |
| --- | --- |
| `POST /v1/prompt_stream/start` | 创建长期 Prompt Stream 请求，提交固定模板 $T$。 |
| `POST /v1/prompt_stream/event` | 提交 Append、Update 或单来源结束事件并返回 ACK。 |
| `POST /v1/prompt_stream/finish` | 提交最终问题 $Q$，等待 P prefill 完成并返回首 token、KV manifest 和统计。 |
| `POST /v1/prompt_stream/abort` | 取消长期请求并清理状态。 |

REST 层支持两种输入形式：

- **文本模式**：事件携带 `text`，P 节点 serving 层使用模型 tokenizer 转成 token ids。
- **token 模式**：事件直接携带 `token_ids`，适合严格控制分词一致性的实验客户端。

`start` 后，EngineCore 中保存的是一个长期 `Request`。后续每个 REST 事件会被包装成只携带 `prompt_stream_event` 的临时 `EngineCoreRequest`；`Scheduler._apply_prompt_stream_request()` 找到相同 `request_id` 的长期 `Request`，把事件应用到已有状态，而不是创建新的逻辑请求。

~~~text
REST text/token
  -> PromptStreamServing 统一为 token_ids
  -> EngineCoreRequest(prompt_stream_event)
  -> Scheduler 找到长期 Request
  -> Request 更新 source / work / logical prompt 状态
  -> Scheduler 选择本轮 execution view
  -> KV manager 分配对应 segment blocks
  -> worker 执行 prefill
~~~

### Proxy 与 P/D 两种完成路径

实验 proxy 同时支持完整生成和只测 TTFT：

- `/v1/prompt_stream/finish`：P 先完成最后一次 prefill 并产生一个首 token，proxy 再携带物理 prompt、首 token 和 KV transfer 参数请求 D 节点继续 decode。
- `/v1/prompt_stream/finish_ttft`：只让 P 产生首 token，直接返回客户端，不创建 D 请求，也不要求 `kv_transfer_params`。
- `/v1/completions`：普通完整 Prompt 仍走标准 P/D proxy 路径。
- `/v1/completions_ttft`：普通 TTFT 对照也只请求 P 的一个首 token，使三种 TTFT 实验使用一致的 P-only 测量终点。

TTFT-only 请求从一开始就不携带 KV transfer 参数，因此 NIXL connector 不会参与该请求；这不是先生成 transfer manifest 再丢弃，而是完全跳过 D 侧准备与 KV 传输。

## 事件协议与来源状态

### 事件定义

`PromptStreamEventType` 定义四种事件：

| 事件 | token 语义 | 主要效果 |
| --- | --- | --- |
| `APPEND` | 本次新增的 delta token | 更新 Append 日志，并创建 `PREFIX_APPEND` work。 |
| `UPDATE` | 当前版本的完整 Update 快照 | 计算 source-local LCP；立即 admission 或进入合并窗口。 |
| `SOURCE_FINISHED` | 无内容 | 封闭指定来源；Update source 结束会触发最终窗口 flush。 |
| `PROMPT_FINISHED` | 非空最终问题 | 标记“用户问题已到达”，处理最终 Update，并尝试创建显式 Question work。 |

每个事件还携带 `request_id`、`source_id`、`source_type`、`chunk_id`、`version`、`sequence_no` 和 `arrival_time`。其中 `chunk_id` 是来源内幂等键；`sequence_no` 用于检测缺口和乱序；`arrival_time` 用于排队与合并统计。

### PromptSourceState 的职责

每个来源保存当前完整 token、版本、下一个期望序号、已见 `chunk_id`、结束状态和结束事件的幂等信息。

当前协议选择 **fail closed**：

- 重复重试的旧 `chunk_id` 可以作为幂等 no-op。
- `sequence_no` 有缺口、回退或跳号时直接拒绝。
- `pending_events` 字段是为将来乱序重排预留的结构，当前不会把缺口事件留在里面等待补齐。
- `SOURCE_FINISHED` 必须紧跟最后一个已接受事件；结束后再到达的数据事件会被拒绝。

当前 Update 合并第一版只支持一个真实 Append 源和一个 Update 源。启动时用于固定模板的合成来源不计入该限制。

## 请求内 work、准入与最终 barrier

### 三层状态不能混为一谈

一个长期请求同时维护三类状态：

| 层次 | 典型字段 | 含义 |
| --- | --- | --- |
| 逻辑输入层 | `PromptSourceState.token_ids`、`append_token_ids`、`suffix_token_ids` | 已经被协议接受的最新内容。 |
| 请求内执行层 | `pending_work_items`、`active_work_item` | 已经承诺执行的不可变工作及当前执行工作。 |
| 物理 KV 层 | committed 长度、快照、Append epoch、segment blocks | GPU 中已经真实 forward 并可复用的状态。 |

“事件已 ACK”只表示它进入了逻辑状态或准入控制器；“work 已 admission”表示它不可再被后续事件覆盖；“committed”才表示目标 KV 已经真实写入。

### pending_work_items 与 active_work_item

固定模板 bootstrap 是唯一必须最先完成的工作。bootstrap 完成后：

- 非空 Append 事件产生一个 `PREFIX_APPEND` work。
- 未启用合并器时，每个有效 Update 事件立即产生一个 `UPDATE` work。
- 启用合并器时，Update 先进入唯一合并窗口，满足 policy 后即产生 `UPDATE` work；物理执行仍等待冻结的 Append epoch。
- 最终问题满足 barrier 后产生唯一 `QUESTION` work。

所有已经 admission 的 work 按进入请求的顺序保存在同一个 `deque`：

~~~text
pending_work_items: [work_1, work_2, ..., QUESTION]
                              |
                              v
                      active_work_item
                              |
                              v
                         committed KV
~~~

关键不变量是：

- 一个逻辑请求同一时刻只有一个 `active_work_item`。
- 已进入 `pending_work_items` 的 work 内容被冻结，后续 Update 不能删除、覆盖或取消它。
- Append 与 Update 没有硬编码的“Append 永远优先”。它们在模板完成后按照实际 admission 顺序进入 FIFO；谁先进入队列，谁先成为该请求的 active work。
- 请求在全局 Scheduler 中仍然只出现一次。请求内两个 work 不会被拆成两个全局请求并行做两次 token-budget 分析。
- 已经 active 的 work 可以被 chunked prefill 分多轮执行，但中途不会切换到同请求的下一个 work。

### Update work 冻结什么

每个 Update work 至少冻结：

- 目标完整 Update 快照。
- 它相对前一执行基准的 source-local LCP。
- 需要读取的 Append 逻辑长度和 Append epoch。
- 用于执行链一致性检查的前一完整 Update 快照。
- policy admission ID 与物理 execution attempt ID。

完整快照必须保留。仅保存 LCP 长度或 LCP token 无法生成本次待算后缀，也无法与下一版本正确比较。

### PROMPT_FINISHED 到达时不会直接 decode

`PROMPT_FINISHED` 中的 `finished=True` 只表示最终问题已经到达，不表示 Question 已可调度。

未启用合并器时，Question 被追加到已有 FIFO 尾部，因此自然等待先前所有 Append/Update work。启用合并器时还必须满足：

- Append 与 Update 来源已经封闭；`PROMPT_FINISHED` 可以在确认无序号缺口后原子地封闭尚未显式结束的可选来源。
- 唯一 Update 合并窗口已被 flush，不存在 `latest_pending_token_ids`。
- 最新逻辑 Update 已经 admission。
- `question_work_enqueued` 仍为 false，以保证只创建一次。

随后 `QUESTION` work 才进入 FIFO 尾部。真正允许 decode 的有效条件包括：

- 模板 bootstrap 已完成。
- 请求已收到 `PROMPT_FINISHED`。
- 没有 `active_work_item` 和 `pending_work_items`。
- 显式 Question work 已完成。
- 合并路径的最终 Append epoch 与最终 Update admission 已在 Question 入队前通过依赖检查。

P 节点在中间 prefill 视图意外采样出的临时 token 会被丢弃；只有最终 Question 完成后产生的首 token 才进入 P/D 完成路径。

## Update 的 source-local LCP

设相邻两次 Update 完整快照为 $U_{i-1}$ 与 $U_i$：

$$
L_i = \operatorname{LCP}(U_{i-1}, U_i).
$$

这里的 LCP **只在 Update 源内部计算**。模板 $T$ 和 Append $A$ 不属于该数值；它们通过独立 `PREFIX_APPEND` KV 域作为本轮 attention 前缀。

当 work 真正轮到执行时，可复用 Update KV 数为：

$$
C_i = \min(C_{i-1}^{\mathrm{materialized}}, L_i),
$$

其中 $C_{i-1}^{\mathrm{materialized}}$ 是上一代 Update 已经真实写入 KV 的 token 数。Scheduler 只在旧 work 完成、下一代 work 被选中后，才把 `UPDATE` blocks 延后截断到 $C_i$；事件刚到达时不会立刻释放或改写 active work 的 blocks。

如果 $U_i$ 是上一版本的严格前缀，可能出现无需 forward 的零 token work。这个 work 仍必须完成一次状态转换：截断旧 Update 尾部、更新 committed snapshot，并解除 Question barrier，不能直接丢弃。

当 Update work 绑定的 Append epoch 与其执行基准所绑定的 epoch 不同时，即使 Update token 的 LCP 很长，也不能复用旧 Update KV，实际 LCP 必须视为 0。原因是相同 Update token 在不同前置 Append 上下文下产生的 KV 并不等价。

## Segmented KV 的物理管理

### 两个 key 仍归属于同一个请求

KV manager 为一个逻辑请求使用两个内部 key：

| 物理域 | KV manager key | 生命周期与失效规则 |
| --- | --- | --- |
| `PREFIX_APPEND` | 原始 `request_id` | 保存 $T+A+P_r$；只随 Append 增长，不因 Update 变化而截断。 |
| `UPDATE` | `request_id::prompt_stream::update` | 保存 $U+Q+decode$；Update 变化时只截断该域。 |

“两个 key”不是两个用户请求。它只是 KV coordinator 中的 block 所有权隔离；请求完成、abort、抢占或整体恢复时，Scheduler 仍以原始请求为单位同时处理两个域。

### 三个执行阶段的读取与写入

`PromptStreamExecutionView` 同时说明物理 token 视图、已计算位置、需要读取的 segments 和唯一写入 segment：

| 阶段 | worker token view | 读取的 block table | 新 KV 写入 |
| --- | --- | --- | --- |
| `PREFIX_APPEND` | $T+A+P_r$ | `PREFIX_APPEND` | `PREFIX_APPEND` |
| `UPDATE` | $T+A+P_r+U_i$ | `PREFIX_APPEND + UPDATE` | `UPDATE` |
| `QUESTION` | $T+A+P_r+U_{final}+Q$ | `PREFIX_APPEND + UPDATE` | `UPDATE` |

因此：

- Append 不会读取或写入 Update blocks。
- Update 和 Question 可以注意到 PREFIX_APPEND KV，但新 slots 只从 UPDATE 域申请。
- Question 直接复用最终 Update blocks，并继续占用同一个 UPDATE block table。

### Segment-local block 分配

普通 vLLM `allocate_slots()` 假设整个序列属于一张连续 block table。Prompt Stream 的 `_allocate_prompt_stream_slots()` 改为：

1. 从 execution view 取得本轮 `write_segment`。
2. 使用该 segment 内部的 `num_computed_tokens` 和目标 token 数计算所需 blocks。
3. 只为对应 segment key 申请新 blocks。
4. Scheduler 另行通过 `get_prompt_stream_execution_blocks()` 把本轮可读 segments 按顺序合并给 worker。

分配坐标和执行坐标必须分开：Update 阶段的全局已计算位置包含整个物理 PREFIX_APPEND 长度，但为 UPDATE key 申请 blocks 时只使用 Update 域内部的局部长度。

当前 segment blocks 有意保持 **unhashed、unshared**。Prompt Stream P 请求和从 Prompt Stream manifest 恢复的 D 请求都不读写普通 automatic prefix cache，避免近似物理序列污染正常 cache namespace。D 日志中的 External Prefix Cache 命中表示 NIXL 外部 KV 成功加载，不等价于本地 APC 命中。

### worker refresh

阶段切换会同时改变 token 序列和完整 block table。仅发送普通 cached-request 增量可能让 worker 保留上一阶段视图，因此当前实现设置 `prompt_stream_needs_worker_refresh`：

- 同一执行视图内的 chunked prefill 只推进计算位置，可以继续走 cached 更新。
- Append、Update、Question 相互切换，或 block 截断改变表结构时，Scheduler 通过 `NewRequestData` 对相同 `request_id` 做完整 remove/re-add。
- worker 接收的新 token view、`num_computed_tokens` 和 block ids 必须来自同一个 `PromptStreamExecutionView`。

### 截断、释放与恢复

- Update 新版本只调用 `truncate_prompt_stream_segment_blocks(..., UPDATE, keep_tokens)`，不会释放 PREFIX_APPEND blocks。
- 正常完成与 abort 使用 `free_prompt_stream_segments()` 同时释放两个 key，顺序上先尾部 UPDATE，再长期 PREFIX_APPEND。
- running preemption、dormant eviction、retained-waiting eviction 或 prefix-cache reset 会释放物理 KV，但保留来源的权威逻辑内容。
- 恢复采用 latest-only：先重建最新 `PREFIX_APPEND`，再重建仍需保留的最新 Update，最后重新创建 Question work；已经被后续快照覆盖且尚未完成的历史物理 work 不必逐代重放。
- recovery 与 policy admission 分开统计。抢占导致的重复 forward 会增加 execution attempt 和 recovery token，不能被误计为“算法又 admission 了一次 Update”或新的 token saving。

## Padding：当前机制与两个近似

### 为什么必须有中间 padding

设 KV block size 为 $B$，当前 PREFIX_APPEND 逻辑长度为：

$$
L_{PA}=|T+A|.
$$

为使 UPDATE 从独立 block table 的第一个 slot 开始，当前实现补齐：

$$
r=(-L_{PA})\bmod B, \qquad 0\le r<B.
$$

物理序列因此变成：

$$
X_{\mathrm{physical}}
=T\mathbin\Vert A\mathbin\Vert P_r\mathbin\Vert U\mathbin\Vert Q.
$$

若不补齐，直接拼接两张 block table 后，D 节点会认为 Update 的第一个 token 应位于 PREFIX_APPEND 最后一个半满 block 的空 slot，而实际 Update KV 却位于新 block 的开头，slot mapping 与 position 会错位。

当前 padding token id 使用 PREFIX_APPEND 最后一个真实 token；当前缀为空时使用 0。选择这个值不是因为它具有“空白”语义，而只是为了生成合法 token id。tokenizer 的 `pad_token_id`、空白字符 token 或 token id 0 都不会自动从 causal attention 中消失。

Append 继续增长时：

1. 新 Append 从旧的逻辑末尾开始，覆盖原先的尾部 padding slots。
2. 若跨越现有容量，则继续向 PREFIX_APPEND key 申请 blocks。
3. 在新的逻辑尾部重新补到 block 边界。
4. 一个 Append work 必须执行到新的物理边界后才提交，不能在半个 block 内切给 Update。

### 近似一：padding 会参与模型语义

当前 $P_r$ 会真正经过 embedding、forward 并写入 KV；后续 Update 与 Question 可以注意到这些 padding KV，而且后续真实 token 的 RoPE 位置也整体右移 $r$。

所以 padding 当前只解决 **block 拓扑正确性**，没有实现语义透明。专用 pad token、重复最后一个 token 或 token id 0 都不能消除这个影响。

### 近似二：后到 Append 不重算已完成 Update

若 Update 在 Append 长度 $A_t$ 时完成，后来 Append 又增长到 $A_{t+1}$，当前系统保留：

$$
\operatorname{KV}(U\mid T,A_t).
$$

最终 Question 却安装更新后的 PREFIX_APPEND blocks：

$$
T\mathbin\Vert A_{t+1}\mathbin\Vert U\mathbin\Vert Q.
$$

系统不会仅因新 Append 到达就使已经完成的 Update KV 失效。这保留了 Append/Update overlap 与少算 token 的收益，但意味着 Update 可能没有关注到后来 Append 的内容。

这两项近似互相独立。后续 Masked Padding 只能消除第一项；第二项需要在最终 Append 上下文确定后重算相关 Update；当前仍保留这项时序近似。

## P/D KV 交接与首 token 边界

### P 节点导出的 block 顺序

最终可传输 block ids 的顺序固定为：

~~~text
remote_block_ids = PREFIX_APPEND block ids + UPDATE block ids
~~~

此时 UPDATE 域已经包含最终 Update 与 Question。P Scheduler 在调用 connector 前按每个 KV cache group 校验：

$$
N_{\mathrm{blocks}}
=\left\lceil\frac{L_{\mathrm{physical}}}{B}\right\rceil.
$$

任何实际 block 数与物理 prompt 长度不一致的请求都会 fail closed，不把错误描述交给 D。

P 侧 `kv_transfer_params` 还附加权威 manifest：

- `prompt_stream_exact_transfer=true`
- `prompt_stream_prompt_token_ids`
- `prompt_stream_physical_prompt_len`
- `prompt_stream_remote_block_counts`
- `prompt_stream_remote_block_sizes`
- `prompt_stream_prefill_output_tokens`
- `remote_num_tokens=physical_prompt_len`

proxy、D scheduler、NIXL metadata 和 worker 使用同一组边界校验 local/remote block table。

### 为什么 P 的唯一首 token 也要交给 D

Question prefill 完成后，P 节点采样首个输出 token $y_1$。它是本次 forward 的输出，但尚未作为输入经过下一轮 forward，所以 P 侧没有 $y_1$ 的 KV。

~~~text
P 已有 KV：physical prompt
P 已采样但无 KV：y1

D 输入：physical prompt + y1
D 外部加载：仅 physical prompt 的 KV
D 本地 forward：y1
D 从 y1 的 logits 继续生成 y2, y3, ...
~~~

因此：

$$
L_{\mathrm{D,prompt}}
=L_{\mathrm{physical}}+N_{\mathrm{P,output}},
\qquad N_{\mathrm{P,output}}\in\{0,1\}.
$$

proxy 会把 P 首 token 合并进最终输出，并相应缩减 D 的 `max_tokens`，保证用户请求的总生成上限不因 P/D 拆分而增加。

## Update 合并：第二阶段设计与实现

### 为什么还要在 Segmented KV 上合并 Update

Segmented KV 解决了“Update 变化不应让 Append KV 一起失效”，但默认路径仍会让每个 Update 版本都创建 work。若 Update 高频到达，中间版本可能刚进入队列就被更新版本取代；逐代计算会增加 GPU token、KV churn 和队列压力。

合并的边界被严格放在 **work admission 之前**：

~~~text
Update source 最新逻辑快照
  -> UpdateCoalescingState 中唯一可覆盖窗口
  -> policy 判定 ready
  -> 冻结不可变 UPDATE work
  -> pending_work_items
~~~

一旦 work 进入 `pending_work_items`，后续 Update 不得再合并、删除或取消它。这样合并逻辑不会侵入已经开始的 KV 计算。

### 四个 Update 版本

| 记号 | 源码状态 | 含义 |
| --- | --- | --- |
| $U_o$ | `PromptSourceState.token_ids` | 最近接受的最新逻辑快照。 |
| $U_p$ | `latest_pending_token_ids` | 当前窗口内尚未 admission、可被覆盖的目标。 |
| $U_e$ | `last_admitted_token_ids` | 已承诺执行链的尾部，可能仍在 pending/active。 |
| $U_c$ | `update_committed_token_ids` | GPU 中当前真实物化的 Update 快照。 |

正常时间线是 $U_o -> U_p -> U_e -> U_c$。$U_e$ 可以领先于 $U_c$，因为 admission 不等于立即获得 GPU。flush 时，新目标必须相对 $U_e$ 计算逻辑执行 LCP，而不是一律相对较旧的 $U_c$；真正开始执行时再校验物理 committed KV 与冻结执行基准一致。

### Update 合并策略

Update policy 决定“窗口是否已经应该 flush”，达到条件便创建冻结 work，不再将 Append source
是否结束作为额外的准入约束。

| policy | flush 条件 |
| --- | --- |
| `immediate` | 每条有效 Update 都立即 flush。 |
| `fixed_window` | soft deadline 固定锚定窗口第一条 Update，不随新事件刷新。 |
| `quiet_debounce` | 每条有效新 Update 将 soft deadline 刷新为“最近事件时间 + 窗口”。 |
| `adaptive` | 无固定 deadline；本请求没有未完成 Update 且全局 token / slot 压力低于阈值时放行。 |
| `full_merge` | 仅在 Update source 结束或 `PROMPT_FINISHED` 时 flush。 |

`disabled` 表示启动配置为 `None`：每个 Update 直接走原来的即时 admission 路径。
`immediate-controller` 表示合并器开启且 `update_policy=immediate`，不等待合并窗口，但仍经过
controller 和统计路径。两者现在共享 dormant 停放 / waiting 唤醒机制；当前 controller 的
epoch 检查同样保留，因此两条路径的实际计算成本仍应分别观察。

### Flush 与 admission

- policy 尚未满足：保留 $U_p$，继续吸收新快照。
- policy 已满足：以 $U_e$ 为执行基准，将最新 $U_p$ 冻结成 work，推进 $U_e$ 并清空窗口。
- Append source 结束：记录来源封闭状态和最终 epoch；不为 Update 增加额外等待，也不重置窗口。
- Update source 结束或 `PROMPT_FINISHED`：flush 尚未放行的最新 Update。`full_merge` 不要求
  两个源同时结束，Update 先结束时也会放行。

对 fixed_window 和 quiet_debounce，soft deadline 控制合并等待，hard deadline 限制
最长等待，可选 TTFT
deadline 从请求 start 起限制放行时间。**这些 deadline 不保证 work 立即执行**：冻结的 Append
epoch 必须已经物理提交，FIFO、token budget 和 KV 资源仍可能产生排队。

来源状态位于 `PromptStreamState.coalescing_sources`，类型为 `PromptStreamSourceState`；
source id、封闭状态和 epoch 继续服务协议与执行依赖。固定模板先算、Question 最后执行的
barrier 都保留。启用 controller 后，跨 epoch 的下一代 Update 仍按零 LCP 重算；已完成 Update
不会只因 Append 后续增长而自动重放。

### Adaptive 的执行背压决策

`adaptive` 通过独立策略类判断：本请求 pending / active 中没有 Update work，且没有
UPDATE 阶段 GPU 在途批次，同时整体 token 需求和运行名额需求低于各自阈值时才放行。
默认两个阈值均为 1.0；不使用固定等待时间，不把 KV 容量或抢占次数作为策略信号。

压力仅估计当前或队首已放行 work，包含普通请求和未回包批次，排除纯 dormant 与未放行
快照。每次放行后重新计压；Update source finished / PROMPT_FINISHED 越过压力限制，
最终快照与 Question 仍按 FIFO 执行。该实现尚未获得新服务实验结论。

策略类由 `create_update_coalescing_policy()` 根据启动配置创建。Request 维护窗口和 work，
Scheduler 提供只读反馈，EngineCore 在空闲前重评无 deadline 的候选。

### Deadline、dormant 与并发

合并控制器没有独立定时器线程，也不需要给 `pending_work_items` 加锁：

- FastAPI/router 只投递 EngineCore 输入。
- Request、deadline heap、work 和 KV 状态都由 EngineCore/Scheduler 主线程串行修改。
- controller 使用 `(deadline, request_id, generation)` 最小堆；generation 让旧 deadline 惰性失效。
- 所有 PromptStream 在模板完成、没有 active/pending work、不能 decode 且无 GPU 在途步骤时，进入 Scheduler 的 dormant 集合；该机制不依赖 controller 开关。
- EngineCore 空闲时把原本无限阻塞的 `input_queue.get()` 改成最多等待到最近 deadline；超时后同一主线程调用 `flush_due_update_coalescing()`。
- source 事件、deadline 和 work 完成回调都在同一控制线程排序，不会与 GPU 计算中途并发改写同一 work。

功能关闭时不创建 controller、deadline heap 或请求级合并窗口，Update 仍即时 admission；
dormant 是所有 PromptStream 共有的调度机制。停放保留分段 KV，产生 work 后先进入 waiting，
再按运行名额、token budget 和 KV 资源准入 running。Scheduler 单独跟踪在途步骤，保证 worker
结果返回前不会提前停放。显存压力下的 dormant / retained-waiting 驱逐继续走恢复流程。

EngineCore 的 runnable 判定与 shutdown 清理同样覆盖关闭合并的 PromptStream；只有合并
策略的 deadline 查询、反馈候选检查和 flush 由配置控制。详细对象归属、状态转换和验证边界见
[流式请求调度规则](Scheduler.md)。

### 抢占下的统计口径

`PromptStreamTokenStats` 当前使用 schema v2，分开记录：

- policy admission 的 Update work 数、完成数和 recovery 取消数。
- 反事实 eager token、policy admission token、节省 token 和额外 token。
- 实际 execution attempts、被抢占 attempts、recovery works 与重复调度 token。
- Update 快照造成的失效 token。
- PREFIX_APPEND、UPDATE、Question 因抢占丢弃的物理 token。
- running preemption、dormant eviction、retained-waiting eviction、allocation failure 等原因。

这解决了早期“抢占重放被误认为新的合并收益”和“请求已返回但 admission 统计仍 outstanding”的冲突。统计随 P 的 finish 结果返回，proxy 会原样附加到 JSON 或首个 SSE payload；simple test 与 replay 脚本能够解析并校验 schema。

### 启用方式

Update 合并是启动级选项，通过一个 JSON 配置传入 P 节点：

~~~shell
--prompt-stream-update-coalescing-config '{
  "update_policy": "fixed_window",
  "soft_window_ms": 20,
  "hard_deadline_ms": 80,
  "ttft_deadline_ms": null
}'
~~~

不传该参数表示功能完全关闭，不等于传入 `update_policy=immediate`。

## 性能实验结论

### Segmented Prefill TTFT 实验

当前最新性能结论来自重新生成的短序列性能集、Qwen2.5-3B、Triton attention 和 P-only TTFT 路径。每个配置回放 512 个请求，三种模式均成功 512/512：

- **Segmented stream prefill**：Append 和 Update 分段事件逐步 prefill；TTFT 从 finish POST 到 P 首 token。
- **Merged update baseline**：任一来源有新事件时，把“当前 Append 全量 + 当前 Update 快照”拼成一个新的全量 Update 事件；它代表不利用两个 segment 独立增量语义的流式对照。
- **Normal full prompt**：检索结束后一次性提交最终完整 Prompt；TTFT 从 completion POST 到 P 首 token。

固定实验参数包括 Append interval 0.5 s、Update interval 0.2 s、最大并发 8、Poisson 到达。结果如下，单元为毫秒，表格单元为 `p50 / p95`：

| QPS | Segmented | Merged Update | Normal |
| ---: | ---: | ---: | ---: |
| 0.5 | **79.1 / 170.3** | 103.9 / 203.9 | 164.8 / 215.8 |
| 1 | **79.3 / 160.8** | 120.7 / 282.9 | 157.4 / 241.0 |
| 2 | **91.8 / 251.2** | 179.0 / 613.8 | 156.9 / 265.1 |
| 4 | 198.7 / 612.3 | 1210.5 / 1753.4 | **162.7 / 409.6** |

阶段性结论如下：

- QPS=0.5、1、2 时，Segmented 的 p50 和 p95 都优于两个对照。它说明把可稳定的 Append 和可改写的 Update 分段保存，确实能够把相当一部分最终请求计算移到 finish 之前。
- 相对 Merged Update，QPS=1 的 p95 下降约 43%，QPS=2 的 p95 下降约 59%。Merged Update 反复提交全量合并快照，计算放大最严重。
- 相对 Normal，QPS=2 时 Segmented 的 p50 仍下降约 41%，但 p95 只下降约 5%，说明队列尾部开始接近饱和。
- QPS=4 时 Segmented 仍远好于 Merged Update，但落后于 Normal。一个流式请求被拆成多个可调度 prefill work，高压下事件 work 的队列占用、worker refresh 和抢占/恢复成本会淹没提前计算收益。
- QPS=0.5 的 Segmented mean 受单个约 3.84 s 离群点影响，因此主结论应优先使用 survival curve 和 p50/p95，而不是只报告平均值。

这组 Triton 结果没有推翻分段设计的性能结论，同时也明确给出了系统拐点：**分段不是免费并行；事件粒度过细或负载过高时，需要 admission/coalescing 或更好的事件调度策略控制放大。**

### Update coalescing 实验

以下是移除 Append 来源完成约束之前的历史矩阵结果，其中部分实验额外等待 Append 封闭。
这些数值保留作历史探索记录，不能作为当前五种 policy 的新结果。当前实验入口保留
`disabled`、`immediate-controller`、`fixed-window`、`quiet-debounce`、`adaptive` 和
`full-merge`；新背压自适应尚未进入已报告的最终策略对比。

已观察到的主规律是：

- 所有合并策略都能够实质减少 Update admission 与 prefill token；例如短 Update interval 下，fixed-window 的策略 token saving ratio 约为 37%，full-merge 约为 46%。
- **减少 token 不保证降低 TTFT。** 在 Update interval 为 0.2 s 或 0.5 s、QPS 不高时，旧实验的窗口等待和来源完成等待通常大于少算 token 的收益，disabled 往往更快。
- 最清晰的收益区域是 Update interval=0.01 s、QPS=1/2。以 p95 为例：QPS=1 时 disabled 为 276.9 ms、full-merge 为 259.7 ms；QPS=2 时 disabled 为 410.6 ms、fixed-window 为 367.5 ms、full-merge 为 313.4 ms。
- 更高压力下虽然某些配置仍显示数值收益，但 preemption、dormant eviction 和 recovery token 占比明显上升，不能只用一条 TTFT 线归因于 coalescing policy。

因此 Update 合并目前可以作为一个**条件性优化与可解释 tradeoff**：事件越密、Update 快照越长、P 队列越拥塞，跳过中间版本越有价值；事件较稀时应接近 immediate，避免人为等待。Adaptive 的下一步目标不是证明“合并总是更好”，而是在线识别这个收益区域。

## 输出质量与近似正确性实验

### 实验设计

质量实验使用 4980 条与 SQuAD v2 官方答案对齐的混合来源请求：

- Append 取对齐 SQuAD context 的前两个段落。
- Update 保留 ANNS pipeline pool 形成的完整快照序列。
- 模型使用 Meta-Llama-3.1-8B-Instruct，并使用模型自己的 `apply_chat_template()`。
- 温度为 0，最终问题要求只返回最短答案片段，最大输出 32 token。
- Normal 与 Segmented 的最终逻辑 Prompt 文本哈希和 token-id 哈希逐条一致，4980 对全部通过一致性检查。
- 评分使用 SQuAD 官方风格的 Exact Match 与 token F1；一个问题有多个标准答案时取最高分。

### 结果

| 路径 | 完成数 | 严格答对数 | EM | 平均 F1 |
| --- | ---: | ---: | ---: | ---: |
| Normal | 4980 / 4980 | 2580 | 51.81% | 66.36 |
| Segmented | 4980 / 4980 | 2634 | 52.89% | 67.12 |

逐样本配对结果为：

| 配对结果 | 请求数 |
| --- | ---: |
| 两者均正确 | 2459 |
| 两者均错误 | 2225 |
| 仅 Normal 正确 | 121 |
| 仅 Segmented 正确 | 175 |

Segmented 相对 Normal 的总体 EM 增加 1.08 个百分点，平均 F1 增加 0.76，McNemar 检验 $p=0.0020$。但正确解读是：

- 在当前数据集、模型、模板和事件时序下，**没有观察到总体质量下降**。
- 这不是严格等价证明。以 Normal 答对的 2580 条为分母，Segmented 保留 2459 条，保留率为 95.31%；仍有 4.69% 的严格 EM 回归。
- 当前端到端实验没有分别消融 padding 与 Append-after-Update，也无法把单条输出变化唯一归因于近似、GPU 数值差异或执行时机。
- Segmented 分数略高不能解释为近似带来的普遍质量增益；论文表述应同时报告总体指标和配对回归。

## Masked Padding：已实现的可选路径

### 目标与原则

Masked Padding 的目标不是删除物理空洞，而是让它只服务 block 拓扑、不进入模型语义：

~~~text
物理位置：T | A | padding | U | Q
逻辑位置：T | A |           U | Q
~~~

设 padding 半开区间为 $[L,L+r)$。对物理 query 位置 $q_p$ 和 key 位置 $k_p$，目标 attention 条件为：

$$
\operatorname{keep}(q_p,k_p)
= (k_p\le q_p)
\land \neg(L\le k_p<L+r).
$$

padding 后的真实 token 还必须使用修正后的语义位置：

$$
\operatorname{position}_{model}(i)=
\begin{cases}
i, & i<L,\\
i-r, & i\ge L+r.
\end{cases}
$$

因此 **mask 与位置修正缺一不可**：只 mask padding 仍会让 U/Q 的 RoPE 位置右移；只修正位置仍会让 padding KV 进入 softmax 分母。

### 第一版实现

第一版以 Triton unified attention 为后端，保留现有两 segment KV 管理：

1. `PromptStreamExecutionView` 冻结逻辑 PREFIX_APPEND 长度、物理长度和 `[padding_start, padding_end)`。
2. `NewRequestData`、`CachedRequestState` 与 `InputBatch` 传递每请求 padding range；Append 边界变化或阶段切换必须刷新。
3. GPU runner 把当前含义过载的位置拆成：
   - `cache_positions`：物理 slot mapping 与 KV 写入位置。
   - `model_positions`：去掉 padding 偏移后的 RoPE 语义位置。
4. `CommonAttentionMetadata` / Triton metadata 增加形如 `[num_reqs, 2]` 的 `prompt_stream_padding_ranges`。
5. `triton_unified_attention.py` 在现有 `compute_kv_seq_mask()` 后追加 `causal AND not-padding`，并保证后续条件不会重新放开 padding key。
6. P manifest、proxy 状态和 D request 同步传递 padding range 与 schema version，使 D 对 P 拉取的 KV 使用同一 mask 和位置偏移。

这里不能用修正后的 `model_positions` 计算 slot mapping，否则 U/Q 会被写回 PREFIX_APPEND 尾部的 padding slots，重新破坏两个 segment 的 block 所有权边界。

### 第一版边界

当前实现只开放：

- 每请求最多一个中间 padding range。
- decoder-only、full causal attention、普通一维 RoPE。
- 不支持 speculative decode、DCP/context parallel、sliding window、ALiBi、MLA 和复杂多模态位置编码；未支持组合必须 fail closed。
- P/D 两端使用同一个支持 range mask 的 attention backend。

该方案不会省掉最多 $B-1$ 个 padding token 的 forward，也不会减少对应物理 slots；它只消除 padding 的语义近似。未来若要连计算也省掉，需要稀疏 slot mapping、KV compaction 或原生 segmented/cascade attention，改造范围会显著扩大。

### 正确性验证要求

完整模型验证不能只比较自然语言答案，应在相同 Append 上下文下以普通完整 Prompt 为参考，至少覆盖：

- $r=0$、$r=1$、$r=B-1$。
- Update 为空、多个 Update、严格前缀 Update。
- chunked prefill 在 padding 前、padding 内和 padding 后切分。
- Append 增长、抢占恢复和 P/D 拉取后继续 decode。
- 每个真实 token 的模型位置一致，P/proxy/D 的 range 一致。
- 首 token logits 在合理数值误差内一致，确定性完整生成一致。
- 关闭开关时普通请求保持原算子分支；启用时混合 batch 中普通请求使用空区间，额外开销需实测。

最新 Qwen + Triton 性能实验没有开启 Masked Padding，不能当作新路径的正确性或性能结果。当前 range mask 已进入 2D/3D kernel，RoPE 与物理 slot 坐标已经分开；27 项 A30 单算子测试通过，覆盖 NIXL HND 布局及位置转换/RoPE 联合对照，完整模型 logits、质量和性能仍待重测。

启用时在原 P/D sbatch 命令加入 `PROMPT_STREAM_MASKED_PADDING=1` 和
`ATTENTION_BACKEND=TRITON_ATTN`；完整 PD 两端都需设置，TTFT-only 可仅启动 P。
区间通过 `NewRequestData` 的完整 refresh 传给 worker，并以 version=1 的
manifest 经 proxy 传给 D；P-only 也返回语义描述但不申请传输租约。实现和
限制详见 [Masked-Padding.md](Masked-Padding.md) 的“实现状态与启用方式”。

## 当前验证覆盖与工程边界

### 已有测试与实验工具

`tests/v1/test_prompt_stream_request.py` 已覆盖的关键单元场景包括：

- Append/Update 到达顺序、Update 冻结前缀和 source-local LCP。
- 后到 Update 不取消 active/pending work，严格前缀零 forward 不死锁。
- 两个 segment 的独立 block 分配、Update 截断不释放 PREFIX_APPEND。
- Question 复用 UPDATE key，最终 block 顺序正确。
- NIXL exact manifest、D-only 首 token 尾块和本地 APC 隔离。
- running preemption、latest-only recovery、空 Update 快照与 Question recovery barrier。
- coalescing admission、token 指标、绝对 deadline 与 recovery 后统计一致性。

`research/scripts/simple_test` 提供 Append-only、Update-only 与 mixed 三条最小 REST/P/D 链路；`research/scripts/replay/segmented_prefill`、`update_coalescing` 和 `generation_quality` 分别负责性能、合并策略与质量实验。

### 当前限制

- Prompt Stream segment KV 当前要求 full-attention KV groups 与统一 scheduler block size。
- 不支持 speculative lookahead token。
- 当前第一版 Update coalescing 限制一个 Append 源和一个 Update 源。
- REST 客户端必须保证所有要纳入回答的知识事件已经收到 P 的 ACK 后，再提交 `PROMPT_FINISHED`；请求内 FIFO 能保证已接受 work 不被 Question 越过，但不能补回尚未发送到 P 的事件。
- 默认配置及历史实验的 padding 仍参与 attention；可选 Masked Padding 已接入，完整服务验证尚未完成。
- Update 按已冻结 Append 上下文执行，可能看不到后到 Append；如需消除这项时序近似，需在最终上下文确定后重算。
- 高 QPS 下事件拆分会放大全局调度压力；`MAX_NUM_SEQS` 只能改变同时驻留上限，不能从根本消除一个逻辑请求包含多个 prefill work 的服务需求。
- TP=2 + NIXL producer 曾出现启动阶段 KV memory registration / shared-memory broadcast 阻塞，当前仍在 `TODO.md` 中跟踪；TTFT-only 可禁用 KV transfer 规避，但完整 P/D TP 路径仍需继续定位。

## 代码地图

| 主题 | 主要文件 |
| --- | --- |
| 事件枚举与传输结构 | `vllm/v1/engine/prompt_stream.py` |
| 长期 Request、source、work、execution view、统计与恢复 | `vllm/v1/request.py` |
| Update coalescing 启动配置 | `vllm/config/prompt_stream.py`、`vllm/config/scheduler.py`、`vllm/engine/arg_utils.py` |
| Scheduler 准入、deadline、dormant、worker refresh 与 P finish manifest | `vllm/v1/core/sched/scheduler.py` |
| EngineCore deadline-aware 主循环 | `vllm/v1/engine/core.py` |
| 两 segment block 分配、截断、拼接与释放 | `vllm/v1/core/kv_cache_manager.py`、`vllm/v1/core/kv_cache_coordinator.py` |
| P 节点 REST 协议与服务 | `vllm/entrypoints/serve/prompt_stream/protocol.py`、`serving.py`、`api_router.py` |
| NIXL exact transfer | `vllm/distributed/kv_transfer/kv_connector/v1/nixl/metadata.py`、`pull_scheduler.py`、`pull_worker.py` |
| 实验 P/D proxy | `research/scripts/server/proxy_server.py` |
| 请求与 KV 单元测试 | `tests/v1/test_prompt_stream_request.py` |

## 配套研究笔记

- [RESTFUL.md](RESTFUL.md)：REST 协议、P/D proxy 和 tokenization 边界。
- [KV_BLOCK_MANAGEMENT.md](KV_BLOCK_MANAGEMENT.md)：两 segment block 所有权、分配和拼接的详细源码说明。
- [Volatility-Aware_Bounded_Update_Coalescing.md](Volatility-Aware_Bounded_Update_Coalescing.md)：Update 合并的五种策略、执行背压和 deadline、执行依赖与实验设计。
- [Approximate-Calculation-Evaluation.md](Approximate-Calculation-Evaluation.md)：两项近似与 SQuAD EM/F1 质量实验。
- [Masked-Padding.md](Masked-Padding.md)：中间 padding 的 mask、位置修正与 Triton 改造方案。
- [TODO.md](TODO.md)：尚未解决的工程问题和后续排查记录。

## 阶段性判断

当前项目已经完成第一项创新的核心工程闭环：把混合来源 Prompt Stream 从 `EngineCoreRequest` 隔离到长期 `Request` 后，以显式事件、FIFO work 和双 segment KV 管理贯通 Scheduler、worker、NIXL 与 REST proxy；性能实验验证了中低负载下提前分段 prefill 的 TTFT 收益，质量实验也给出了近似影响的可量化边界。

第二项 Update coalescing 已经从设计进入可运行实现，并证明“减少 Update token”与“降低 TTFT”之间存在可研究的条件性权衡。其下一步重点不是继续堆固定参数，而是验证基于执行背压和调度需求的 Adaptive policy。当前不把抢占/recovery 成本纳入
放行决策，实验单独记录恢复指标以识别干扰。

当前的正确性改进 Masked Padding 已进入源码：保留双 segment block 拓扑，在 Triton 路径增加中间 key range mask 与模型位置修正，下一步是完整模型和 P/D 的验证。Append-after-Update 的时序近似仍独立存在，应继续与普通完整 Prefill 做性能和质量对照。
