# 分段 KV 中间 Padding 的 Masked Attention 调研

## 结论摘要

当前分段流式 Prefill 的物理序列为：

$$
X_{\text{physical}}
= T \mathbin\Vert A \mathbin\Vert P_r
\mathbin\Vert U \mathbin\Vert Q,
$$

其中 $T+A$ 属于 `PREFIX_APPEND` KV segment，$U+Q$ 属于 `UPDATE`
KV segment，$P_r$ 是为了让两个 segment 的 block table 能按 block 直接拼接而加入的
中间 padding。

源码调研后的结论是：

- **当前 vLLM 不能直接通过现有请求接口把这段中间 padding mask 掉。** 普通 V1
  decoder 路径没有接收 Hugging Face 风格任意 `attention_mask` 的通用入口；标准
  FlashAttention、FlashInfer 和当前 block table 语义都把有效 KV 看作从位置 0
  开始的连续区间。
- **单独增加 attention mask 仍然不正确。** 当前 worker 使用物理 token 下标同时完成
  token 读取、KV slot 寻址、`seq_lens` 计算和模型 `positions` 构造。即使 padding
  不再参与 attention，$U$ 与 $Q$ 的 RoPE 位置仍会被整体右移 $r$，因此仍不等价于
  $T+A+U+Q$。
- **从注意力原理和现有 kernel 结构看，这件事可以实现。** 需要把“KV 存在哪里”的
  **物理坐标**与“token 在模型序列中的位置”的 **语义坐标** 分开，同时把 padding
  区间传给 attention backend。对全注意力、RoPE decoder 和单个中间空洞，目标语义
  很明确，不需要构造 $O(L^2)$ 的稠密 mask。
- **最合适的第一版不是跳过 padding 的 forward，而是继续生成 padding KV，但禁止任何
  后续真实 token 读取它，并修正 padding 后的模型位置。** 这种方案保留现有两个 segment
  的 block 管理和 P/D block-id 顺序，只消除 padding 的语义污染；padding 本身的少量
  计算和显存开销仍然存在。
- **建议第一版只支持单个 padding 区间、full attention、普通一维 RoPE 模型，并选择
  一个可控 backend 实现。** Triton unified attention 已有按区间修改 `seq_mask` 的结构，
  FlexAttention 已有可编程 `mask_mod`，二者都可作为原型基础。其他 backend 和
  MLA、ALiBi、sliding window、M-RoPE、DCP 等能力应显式拒绝，而不是静默退回当前近似。

因此，这不是简单设置 tokenizer 的 `pad_token_id`，也不是只改一处 causal mask。
它是一项跨越 Request、scheduler 输出、worker 输入、位置编码、attention backend 和
P/D 传输元数据的中等规模改造，但不需要推翻现有 segment KV 管理。

## 目标语义

### 物理布局与逻辑布局

设：

- $L=|T+A|$：当前 Update work 冻结的真实 `PREFIX_APPEND` 长度。
- $B$：KV block size。
- $r=(-L)\bmod B$：为了 block 对齐加入的 padding 长度。
- padding 的物理区间为 $[L,L+r)$。

物理 block table 对应：

$$
[T,A,P_r]_{\text{PREFIX\_APPEND blocks}}
\mathbin\Vert
[U,Q]_{\text{UPDATE blocks}}.
$$

但模型应看到的逻辑序列始终是：

$$
X_{\text{logical}}
=T\mathbin\Vert A\mathbin\Vert U\mathbin\Vert Q.
$$

对任意真实 token，其语义位置应为：

$$
\operatorname{semantic\_pos}(i)=
\begin{cases}
i, & i<L,\\
i-r, & i\ge L+r.
\end{cases}
$$

padding token 自身的模型输出没有语义用途。可以继续为它们执行 forward 并写入 KV，
但这些 KV 必须永远不能成为后续真实 query 的可见 key/value。

### 目标 attention predicate

对使用物理下标 $q_p$、$k_p$ 的普通 causal attention，所需的 key 可见性是：

$$
\operatorname{keep}(q_p,k_p)
= (k_p\le q_p)
\land \neg(L\le k_p<L+r).
$$

这里仍可使用物理下标判断先后顺序。因为当前只有一个空洞，去掉中间 padding 不会改变
任意两个真实 token 的相对顺序。但 RoPE、ALiBi、sliding-window 距离等与“位置差”有关的
机制不能继续使用物理位置，必须使用去掉空洞后的语义位置。

### 为什么 mask 和位置修正必须同时存在

只做 mask：

- 后续 token 不再读取 padding KV；
- 但 $U$ 和 $Q$ 的 RoPE 位置仍从 $L+r$ 开始，而不是从 $L$ 开始；
- 结果仍不等价于无 padding 的完整 Prompt。

只修正位置：

- $U$ 和 $Q$ 获得正确 RoPE 位置；
- 但 padding KV 仍参与 softmax，并改变后续 token 的 attention 输出；
- 结果同样不等价。

只有同时完成下面两件事，才能消除这一项近似：

1. padding key/value 对所有真实 query 不可见；
2. padding 后的真实 token 使用去洞后的语义位置。

## 当前源码中的实际行为

### Request 把 padding 构造成真实 token

[`vllm/v1/request.py`](../vllm/v1/request.py) 中的
`PromptStreamExecutionView` 明确把 `token_ids` 定义为包含
`PREFIX_APPEND` block-aligned padding 的物理 prompt。

`Request._get_prefix_append_physical_token_ids()` 的行为是：

- 取得固定模板和当前 work 冻结的 Append 逻辑前缀；
- 计算距离 block 边界还缺多少 token；
- 使用 `PromptStreamState.padding_token_id` 补齐物理 token 序列；
- `padding_token_id` 当前取 prefix 的最后一个 token id，prefix 为空时取 0。

这说明 padding 不是 block 内没有初始化的槽位，而是进入 forward 的真实 token。
换成 tokenizer 的专用 padding token 只能改变填充值，不能阻止其生成 KV 或参与 attention。

Update 执行视图把以下内容拼成一条物理 token 序列：

$$
[\text{frozen PREFIX\_APPEND}+\text{padding}]
\mathbin\Vert[\text{Update snapshot}].
$$

Question 执行视图继续在其后拼接最终问题。两者的 `num_computed_tokens`、
`prefix_append_physical_len` 和 `segment_offset` 都包含 padding。

### KV 管理只能表达 block 级拼接

[`vllm/v1/core/kv_cache_manager.py`](../vllm/v1/core/kv_cache_manager.py) 中：

- Append 阶段安装 `PREFIX_APPEND` block table；
- Update、Question 和后续阶段安装 `PREFIX_APPEND + UPDATE` block table；
- `get_prompt_stream_blocks()` 最终也按该顺序向 KV connector 暴露 block ids。

源码注释已经明确指出：当前合并只能表示 block 级拼接，不能表示 token 级无缝
compaction。`seq_lens` 只说明连续前缀的总物理长度，无法表达“中间某几项无效”。

### scheduler 输出没有携带空洞信息

[`vllm/v1/core/sched/output.py`](../vllm/v1/core/sched/output.py) 中的
`NewRequestData` 和 `CachedRequestData` 会传递：

- prompt token ids；
- `num_computed_tokens`；
- block ids；
- sampling 和多模态等常规数据。

它们没有传递以下任一信息：

- padding 的起止位置；
- 物理位置到语义位置的映射；
- 需要排除的 KV range；
- 逻辑 prompt 长度与物理 prompt 长度的区别。

因此，即使 Request 已知 $L$ 和 $r$，worker 目前也无法恢复正确 mask。

### worker 把物理位置与模型位置绑定在一起

[`vllm/v1/worker/gpu_model_runner.py`](../vllm/v1/worker/gpu_model_runner.py)
在 `_prepare_inputs()` 中使用：

```python
positions = num_computed_tokens + query_pos
```

同一组位置随后承担多种职责：

- 从物理 prompt token buffer 中定位本轮 token；
- 根据 block table 计算 KV `slot_mapping`；
- 计算本请求的物理 `seq_lens`；
- 作为模型 `positions` 进入 RoPE 等位置编码；
- 放入 `CommonAttentionMetadata.positions`。

对普通连续序列，这些坐标相同是合理的；对存在中间空洞的序列，它们必须被拆开。
KV slot 仍应使用物理位置，模型位置则应使用语义位置。

### 通用 attention metadata 没有 padding range

[`vllm/v1/attention/backend.py`](../vllm/v1/attention/backend.py) 中的
`CommonAttentionMetadata` 提供 `seq_lens`、`block_table_tensor`、
`slot_mapping`、`causal`、`positions` 等字段，但没有任意无效 KV 区间。

当前唯一相近的通用字段是 `mm_req_doc_ranges`，它服务于多模态 PrefixLM 的双向
attention range，语义是**扩大**可见区域，不是从 causal 区域中**删除**padding。
因此不能直接借用这个字段而不改变含义。

## 现有 attention backend 能力

| 路径 | 当前是否能直接处理本需求 | 可复用的能力 | 主要限制 |
| --- | --- | --- | --- |
| 标准 FlashAttention | 否 | paged KV、causal、尾部有效长度 | `seqused_k` 只能描述连续前缀，不能挖掉中间区间。 |
| FlashAttention 4 `mask_mod` | 否，但 kernel 机制可扩展 | 已有 `_make_mm_prefix_mask_mod()` 按请求 range 判断 query/key | 当前只为特定多模态 PrefixLM 路径启用，已有 predicate 是 `causal OR mm_prefix`，且受 FA 版本和硬件约束。 |
| FlexAttention | 否，但很适合做原型 | `logical_mask_mod`、physical-to-logical block 映射、`create_block_mask` | 当前没有文本 padding range 的请求级元数据；动态 per-request mask、编译与 CUDA Graph 行为需要重新验证。 |
| Triton unified attention | 否，但改造路径清楚 | `compute_kv_seq_mask()` 已组合 causal、sliding window 和多模态 range | 需要新增 padding range 的完整数据通路，并保证后续 `OR` 条件不会重新放开 padding。 |
| 当前 FlashInfer 集成 | 否 | paged KV 与连续 sequence length | 当前 vLLM 接入没有暴露适合本需求的 per-request middle-hole mask。 |
| FlashAttention cascade | 否 | 可分别计算 prefix/suffix attention，并通过 LSE 合并结果 | 现有实现面向 batch 共享且 block-aligned 的 common prefix，仍要求 `common_prefix_len % block_size == 0`，不能直接表达本请求的非对齐真实 prefix。 |

### 标准 FlashAttention 为什么不能靠 `seq_lens` 解决

若把物理 `seq_len` 设成 $L$，会同时丢掉位于 padding 后面的 $U+Q$；若设成完整物理
长度 $L+r+|U+Q|$，padding 又会被视为有效 KV。一个标量长度无法表达中间空洞。

同理，简单从 block table 中删掉 padding 所在 block 也不成立，因为这个 block 的前
$L\bmod B$ 个槽位仍保存真实 prefix KV。block table 的最小寻址单位是整个 block，
而本需求的无效区域从 block 中间开始。

### FlashAttention 4 的 `mask_mod`

`flash_attn.py` 已有 `_make_mm_prefix_mask_mod()`。它证明当前代码库中的 FA4 路径能够
把按请求保存的 range 交给 kernel predicate。但现有逻辑计算的是：

$$
(k\le q)\lor \operatorname{same\_mm\_prefix\_range}(q,k),
$$

本需求则是：

$$
(k\le q)\land \neg\operatorname{in\_padding}(k).
$$

两者的数据流和布尔组合都不同。可以参考该实现新增 `prompt_stream_padding_ranges`，
但不能把 padding 冒充 multimodal range。

### FlexAttention 的可编程 mask

[`vllm/v1/attention/backends/flex_attention.py`](../vllm/v1/attention/backends/flex_attention.py)
已经把 paged KV 的物理 block 下标映射为逻辑 KV 下标，再调用 `logical_mask_mod`。
因此从表达能力看，加入下面的条件没有困难：

```python
keep = causal_mask(b, h, q_idx, kv_idx)
keep &= ~is_prompt_stream_padding(b, kv_idx)
```

困难主要在 attention 之外：range 如何进入 batch、如何随 cached request 更新、如何在
CUDA Graph 中保持稳定、如何给模型提供另一套语义 `positions`。所以它是可用的原型
backend，不是现成的请求级功能。

### Triton unified attention 的区间谓词

[`vllm/v1/attention/ops/triton_attention_helpers.py`](../vllm/v1/attention/ops/triton_attention_helpers.py)
中的 `compute_kv_seq_mask()` 已经逐 tile 生成 `seq_mask`，并组合 causal、sliding
window 和多模态区间。对单个 padding range，可在 causal 基础上加入：

```python
not_padding = (seq_offset < pad_start) | (seq_offset >= pad_end)
seq_mask &= not_padding
```

这不需要分配稠密 attention mask，额外元数据也只有每个请求两个整数。但若同时存在
PrefixLM 等会用 `OR` 扩大 mask 的功能，必须定义清楚组合顺序，避免后续逻辑再次放开
padding。更稳妥的最终语义是把 `not_padding` 作为所有可见性规则之外的最外层 `AND`。

### Cascade attention 为什么不是直接答案

[`vllm/v1/attention/backends/flash_attn.py`](../vllm/v1/attention/backends/flash_attn.py)
中的 `cascade_attention()` 会分别计算共享 prefix 和各请求 suffix 的 attention，最后
用 log-sum-exp 状态合并结果。这个思路说明“读取两个独立 KV 区域而不把它们伪装成
连续 token”在数学上可行。

但现有 cascade 路径具有不同目标：

- prefix 必须由 batch 中请求共享；
- `common_prefix_len` 必须按 block 对齐；
- block table 仍按一个对齐切点切成 prefix 和 suffix；
- 当前不支持 ALiBi 和 sliding window。

若把它扩展为每个请求都持有独立的 `PREFIX_APPEND` 与 `UPDATE` block table，并让
attention 对两部分分别计算和合并，就会成为更彻底的 **native segmented attention**。
它有机会连 padding forward 都省掉，但改造范围明显大于本笔记讨论的 masked-padding
第一版。

## 推荐的第一版实现边界

### 保留物理 padding，只消除其语义作用

第一版建议继续维持：

- `PREFIX_APPEND` 独立申请 blocks；
- 其物理长度补齐到 block 边界；
- `UPDATE` 独立申请 blocks；
- worker 与 NIXL 仍按 `PREFIX_APPEND blocks + UPDATE blocks` 传递和安装。

需要改变的是：

- padding KV 仍被计算和保存，但 attention kernel 永远不读取它们；
- padding 后的真实 token 使用逻辑位置；
- usage、最大模型长度检查和用户可见 prompt 长度继续采用逻辑 token 数；
- 内部分配、block table 和 KV slot 使用物理 token 数。

这种方式不会节省最多 $B-1$ 个 padding token 的 forward，也不会释放 padding 占用的
block 槽位；它解决的是正确性近似，而不是这部分资源浪费。

### 第一版仅承诺以下模型与执行模式

- 每个请求最多一个中间 padding range；
- decoder-only causal model；
- full attention；
- 普通一维 RoPE；
- 不开启 speculative decode、DCP、context parallel 或复杂多模态位置编码；
- P 节点和 D 节点使用同一个支持 masked-padding 的 attention backend。

当前 segment KV 检查还允许 `MLA_ATTENTION`。若第一版 backend 没有实现 MLA 对应的
range mask，启动或接收 prompt-stream 请求时必须明确报错；不能继续把 MLA 归入
masked-padding 已支持范围。

## 必须补充的数据与代码通路

### Request 与 execution view

每个 `PromptStreamExecutionView` 至少需要冻结：

- `logical_prefix_append_len=L`；
- `prefix_append_physical_len=L+r`；
- `padding_start=L`；
- `padding_end=L+r`；
- `position_shift=r`。

虽然这些量能够互相推导，仍建议执行视图显式保存逻辑长度和物理长度，并通过一个明确
的 `padding_range` 值对象集中校验以下不变量：

$$
0\le L\le L+r,
\qquad
(L+r)\bmod B=0,
\qquad
0\le r<B.
$$

Update work 必须使用该 work 冻结 Append 快照时对应的 range，不能直接读取请求后来增长
后的最新 Append 长度。

### scheduler 到 worker 的增量协议

`NewRequestData` 和 `CachedRequestData` 需要携带 padding metadata。因为 Append work
会改变 $L$ 和 $r$，cached request 更新不能只发送新的 `num_computed_tokens`。

worker 的 `CachedRequestState` 与 `InputBatch` 也需要保存每个请求的 padding range，并在：

- 新请求加入；
- prompt-stream work 切换；
- Append 边界变化；
- preemption 后恢复；
- worker refresh；
- 请求移除和 batch row 交换；

这些路径上同步更新或清理。

### worker 内拆分两套位置

建议把当前含义过载的 `positions` 拆成：

- `cache_positions`：物理位置，用于 token buffer 下标、block table、slot mapping、
  物理 `seq_lens` 和 KV 写入位置；
- `model_positions`：语义位置，传给模型的 RoPE；
- `padding_ranges`：每个请求在物理 KV 坐标中的无效区间，传给 attention metadata。

对于本轮真实 query：

```python
model_position = cache_position
if cache_position >= padding_end:
    model_position -= padding_end - padding_start
```

不能直接用修正后的 `model_positions` 计算 `slot_mapping`，否则 $U/Q$ KV 会被写进
`PREFIX_APPEND` 最后一个 block 的 padding 槽位，重新破坏两个 segment 的所有权边界。

### attention metadata 与 backend

`CommonAttentionMetadata` 可新增语义明确的字段，例如：

```python
prompt_stream_padding_ranges: torch.Tensor | None
```

其形状可以是 `[num_reqs, 2]`，使用半开区间 `[start, end)`；普通请求用 `start == end`
表示无 padding。若以后允许多 segment 或多个空洞，再扩展为 `[num_reqs, max_ranges, 2]`。

backend 必须对所有 attention 层使用一致 predicate。若某个 layer 选择了不支持 range
mask 的 backend，应在调度前失败，不能只让部分层忽略 padding。

### P/D 传输

只在 P 节点 mask padding 不够。D 节点拉取已经生成的 KV 后仍会继续对 Question 末尾
或 P 节点产生的首 token 执行 attention；它必须知道相同的 padding range 和语义位置偏移。

因此最终 KV transfer/request metadata 至少需要传递：

- 物理 prompt token 数；
- 逻辑 prompt token 数；
- 最终 `padding_start` 与 `padding_end`；
- block ids 的 segment 顺序；
- masked-padding schema/version，避免新旧 P/D 节点误配。

这些信息必须沿着 P 节点 `request_finished()`、proxy 的内部状态和 D 节点
`EngineCoreRequest` 一直保留到 decode 结束。D 节点不能只凭最终 token ids 猜出哪些重复
token 是 padding，因为相同 token id 完全可能是用户真实输入。

### 长度口径

实现后至少存在三种长度，命名必须明确：

- **逻辑 prompt 长度**：$|T+A+U+Q|$，用于 API usage 和模型最大上下文语义检查；
- **物理 prompt 长度**：逻辑长度加 $r$，用于 block table、slot mapping 和传输描述；
- **物理已计算长度**：scheduler/worker 的 KV 推进量，包含已经 forward 的 padding。

不能继续用一个 `num_prompt_tokens` 同时隐式表达三种含义，否则 preemption、P/D matched
token 数和生成位置很容易再次错位。

## 与 Append/Update 时序近似的关系

Masked padding 只能消除
[`research/Approximate-Calculation-Evaluation.md`](Approximate-Calculation-Evaluation.md)
中的第一类近似，不能消除“Update 可能没有关注后到 Append”的第二类近似。

例如 Update work 在 Append 长度 $L_1$ 时完成，随后 Append 增长到 $L_2$：

- Update KV 已按 $L_1$ 对应的上下文和语义位置生成；
- 最终 Question 使用 $L_2$ 对应的 prefix；
- mask 无法把缺失的 Append 信息补进已经生成的 Update KV，也不能无损修改已旋转的
  Update key 位置。

只有严格的 Append gate 保证 Update 在最终 Append 后计算，或在 Append 增长后重新计算
相应 Update，最终结果才可能与普通完整 prefill 严格对齐。因此后续实验仍需分别消融：

- padding 近似；
- Append-after-Update 时序近似。

## 不建议的简化方案

### 更换专用 pad token

`pad_token_id` 只决定 embedding 内容。只要它进入 self-attention 并生成 KV，就仍会改变
softmax 分母和输出；特殊 token 不会自动获得“不可见”语义。

### 把 padding token id 设为 0

token id 0 仍是词表中的正常 embedding，并不是全零 KV。即使 embedding 恰好为零，经过
残差、归一化、位置编码和线性层后也不能保证 K/V 为零；而全零 K/V 也仍会进入 softmax。

### 只缩短 `seq_lens`

一个 `seq_len` 只能裁掉尾部，不能保留 padding 后的 Update/Question 同时删除中间区域。

### 只修正 RoPE positions

这只能消除位置偏移，不能阻止 padding KV 参与 attention。

### 完全不计算 padding 并跳过物理位置

这是可行的更高级方案，但当前调度进度是一个连续标量，slot mapping 也假设本轮位置连续。
跳过区间需要稀疏的计算位置、单独的 query/KV 映射，并重新审查 chunked prefill、
preemption 和 P/D matched-token 逻辑，改造量比“计算但 mask”明显更大。

### 为每个请求创建稠密二维 mask

稠密 $[Q,K]$ mask 会带来 $O(QK)$ 的构造、显存和传输开销，而本需求只需要两个整数描述
一个区间。应把范围判断放进 kernel predicate 或 block-mask 构建过程。

## 可选的后续方案

| 方案 | 是否消除语义近似 | 是否省去 padding forward | 对现有 segment 管理的影响 | 综合判断 |
| --- | --- | --- | --- | --- |
| 计算 padding + range mask + 位置修正 | 是 | 否 | 较小 | **推荐第一版**。 |
| 稀疏跳过 padding 位置 | 是 | 是 | 中等到大 | 需重构调度进度和 slot mapping。 |
| finish 时 KV compaction | 最终阶段可做到 | 部分 | 大 | 涉及跨 block KV copy、所有权和 NIXL 描述。 |
| native segmented/cascade attention | 是 | 是 | 很大 | 长期最干净，但需要按请求双 block table 与 attention-state 合并。 |

## 正确性验证要求

若后续实现，不能只看最终回答“是否像人”。应把普通完整 prompt 作为严格参考，对同一组
token 做逐层或至少逐 logits 验证。

### 基础用例

- $r=0$：已经 block 对齐，masked path 必须退化为普通 causal path；
- $r=1$；
- $r=B-1$；
- Update 为空，仅有 Question；
- 多轮 Update 和 source-local LCP；
- Append 增长但尚未跨 block，以及跨越新 block；
- chunked prefill 在 padding 边界前、边界内和边界后切分；
- preemption 后恢复；
- P/D 拉取后继续 decode。

### 比较内容

- 最终逻辑 token ids 完全一致；
- 每个真实 token 的 `model_positions` 与普通完整 prefill 一致；
- padding range 在 P、proxy、D 三端一致；
- 相同模型、确定性采样下首 token logits 在合理数值误差内一致；
- 完整生成 token 序列一致；
- 普通非 prompt-stream 请求不产生额外 mask，性能和结果不变。

### 特性组合

应分别验证或明确拒绝：

- FlashAttention、Triton、FlexAttention、FlashInfer；
- full attention、sliding window、ALiBi；
- MLA；
- M-RoPE/XD-RoPE；
- CUDA Graph；
- prefix caching 与 cascade attention；
- DCP/context parallel；
- speculative decode。

## 最终建议

可以实现中间 padding mask，并且在当前单空洞、full-attention、RoPE decoder 的约束下，
数学语义清楚，现有 Triton/FlexAttention 机制也足以承载原型。但应把它定义为一个显式、
受能力检查保护的 prompt-stream 执行模式，而不是给所有 backend 偷偷增加一个可选字段。

推荐实施顺序是：

1. 在 Request 和 scheduler/worker 协议中明确逻辑长度、物理长度与 padding range；
2. 在 GPU model runner 中拆分 `cache_positions` 与 `model_positions`；
3. 先为一个 backend 实现 `causal AND not-padding`；
4. 打通 P/proxy/D 的 range metadata；
5. 与普通完整 prefill 做 logits 级等价验证；
6. 验证后再扩展其他 backend 与位置编码类型。

完成以上路径后，当前近似评测中的 **PREFIX_APPEND block 对齐 padding** 可以从“有意的
近似”改为“仅有物理占位、对模型语义不可见”。另一项 Append/Update 到达时序近似仍然
独立存在，不能一并宣称被消除。
