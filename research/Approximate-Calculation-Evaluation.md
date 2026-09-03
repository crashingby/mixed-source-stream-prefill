# 分段流式 Prefill 的近似计算质量评测

## 结论摘要

当前分段流式 Prefill 为了保持 Append 与 Update 两个 KV 域可独立管理，并能把它们的 block table 直接拼接给 D 节点，接受了两类**有意的近似**。因此，它不应被表述为对最终完整 Prompt 的严格等价 prefill。

在 4980 条带 SQuAD v2 官方答案的混合来源请求上，分段路径与 Normal 路径的最终逻辑 Prompt 文本、token 序列均完全相同；两种路径均成功完成全部请求。分段路径的总体 EM/F1 未低于 Normal，反而略高；但逐样本输出并不完全相同，仍有一部分原本由 Normal 答对的样本会在分段路径中答错。因此，这一实验支持“当前近似在该数据集上没有观察到总体质量退化”，不支持“近似严格无误差”或“分段计算必然提升质量”。

## 评测对象与严格参考

设最终逻辑 Prompt 为：

$$
X = T \mathbin\Vert A \mathbin\Vert U \mathbin\Vert Q.
$$

其中：

- $T$：模型 chat template、系统提示和请求启动时的公共模板。
- $A$：Append 源最终单调累积得到的 token 序列。
- $U$：Update 源最终版本的完整快照。
- $Q$：最终用户问题及答案起始标记。

**Normal** 参考路径把 $X$ 一次性交给普通 vLLM 请求进行 prefill 和生成。它是本实验的最终上下文语义参考。

**Segmented** 路径把 $T+A$ 维护在 **PREFIX_APPEND** KV 域，把 $U+Q$ 维护在 **UPDATE** KV 域；Append 与 Update 事件在到达时分别触发流式 prefill，最后再拼接两个物理 KV 域进行 Question prefill、P/D 交接和生成。

两条路径的差异只应来自分段物理执行与事件到达时序，而不是最终问题、模板或分词差异。

## 当前近似点

### PREFIX_APPEND 的 block 对齐 padding

**PREFIX_APPEND** 域必须在物理 block 边界结束，才能与后续 **UPDATE** 域的 block table 直接拼接。设 block size 为 $B$，则在 $T+A$ 后补齐：

$$
r = (B - |T+A| \bmod B) \bmod B.
$$

当 $r>0$ 时，物理执行序列是：

$$
X_{\mathrm{physical}}
= T \mathbin\Vert A \mathbin\Vert P_r \mathbin\Vert U \mathbin\Vert Q,
$$

其中 $P_r$ 是长度为 $r$ 的填充序列。当前实现使用 **prefix 最后一个 token id 的重复值**作为 padding；若 prefix 为空，则使用 token id 0。

这个 padding 不是只存在于 block 元数据中：它会真正执行 prefill、占据位置，并参与后续 Update / Question token 的 causal attention。当前没有为这些 token 安装专用 attention mask，也没有对位置编码做补偿。因此，$P_r$ 是第一类近似来源：后续 token 看到的是“真实上下文加无语义 padding”，而不是严格的 $T+A$。

Append 继续增长时，新 Append token 会从原逻辑尾部开始覆盖旧 padding，并在新的物理尾部重新补齐 padding。已经生成的 Update KV 不会仅因这一覆盖而回滚或重算。

### Update 可能缺少后到 Append 的上下文

Update work 被 scheduler 选中时，会冻结当时已经物理提交的 Append 快照 $A_t$，并在该前缀上计算 Update KV。若之后才到达新的 Append chunk，Append 域会增长为 $A_{t+1}$，但已完成的 Update KV 保持不变。

因此，最终 Question 阶段会安装新的 Append blocks 与旧 Update blocks：

$$
\operatorname{KV}(U_t \mid T, A_t)
\quad \text{与} \quad
T \mathbin\Vert A_{t+1} \mathbin\Vert U_t \mathbin\Vert Q
\quad \text{共同参与后续计算，}
$$

其中 $A_t \ne A_{t+1}$。严格的最终 Prompt prefill 则应重新计算 $\operatorname{KV}(U_t \mid T, A_{t+1})$。当前路径有意不做这次回滚和重算，以保留流式 overlap 带来的计算节省；这就是第二类近似。

只有 Append 已经稳定、或 Update work 在最后一个相关 Append chunk 提交之后才执行时，这一项近似才不会发生。它由真实调度顺序决定，而不是由 Update source-local LCP 决定。

## 实验设计

### 数据与样本

- 数据集使用混合来源请求：问题与标准答案来自 SQuAD v2；Append 源取对齐 SQuAD context 的前两个段落；Update 源保留 ANNS pipeline pool 形成的全量快照序列。
- 仅纳入具有完整请求文件、且能与官方 SQuAD 答案可靠对齐的样本，共 **4980** 条。
- 每条请求均包含一个 Append 源和一个 Update 源，最终问题与标准答案在两条路径中相同。

### 生成与可比性控制

- 模型为 Meta-Llama-3.1-8B-Instruct，使用模型自己的 **apply_chat_template()** 构造 prompt；温度为 0。
- 最终问题要求模型仅返回最短答案片段；最大生成长度为 32 token，并以换行停止。
- Normal 路径一次性发送最终 $T+A+U+Q$；Segmented 路径按 Append / Update 事件流发送，再以最终问题触发完成。
- 对每一对结果记录最终逻辑 Prompt 的文本哈希和 token-id 哈希。4980 对样本均满足两个哈希一致，故评分比较的不是不同输入，而是不同 prefill / KV 执行路径。
- 两条路径均成功完成 4980/4980 条请求；没有把 HTTP、服务端或不完整结果计入质量分数。

### 指标

采用 SQuAD 官方风格的回答级指标，并对一个问题的多个标准答案取最高分：

- **Exact Match（EM）**：小写化、移除标点和冠词、压缩空白后，预测与任一标准答案完全一致。它很严格，例如少写 Saint 也会记为 0。
- **Token F1**：按同一规范化规则计算预测与标准答案之间的 token overlap F1，再对多个标准答案取最大值。
- **配对统计**：对每条请求比较 Normal / Segmented 的 EM 结果，并用 McNemar 检验评估两条路径的总体 EM 差异；同时报告“Normal 对而 Segmented 错”的逐样本回归比例。

## 全量结果

| 路径 | 完成数 | 严格答对数 | EM | 平均 F1 |
| --- | ---: | ---: | ---: | ---: |
| Normal | 4980 / 4980 | 2580 | 51.81% | 66.36 |
| Segmented | 4980 / 4980 | 2634 | 52.89% | 67.12 |

相对 Normal，Segmented 的总体结果为：

- EM 增加 **1.08 个百分点**，即净多答对 **54** 条。
- 平均 F1 增加 **0.76 分**。
- EM 的配对 McNemar 检验为 $p=0.0020$。

逐样本配对关系如下：

| 配对结果 | 请求数 | 含义 |
| --- | ---: | --- |
| 两者均严格答对 | 2459 | 两条路径都得到官方完全匹配答案。 |
| 两者均不严格答对 | 2225 | 两条路径都没有得到官方完全匹配答案。 |
| Normal-only | 121 | Normal 严格答对、Segmented 未严格答对。 |
| Segmented-only | 175 | Segmented 严格答对、Normal 未严格答对。 |

以 Normal 严格答对的 2580 条为分母，Segmented 保留其中 2459 条，严格答对保留率为 **95.31%**；也就是说有 **4.69%** 的 Normal 正确样本在 Segmented 路径中发生了严格 EM 回归。

## 如何解读这一结果

- **总体质量**：在这个混合来源 SQuAD 对齐数据集、这个模型和这组事件时间参数下，当前两类近似没有造成总体 EM/F1 下降；分段路径的总体分数略高。
- **逐样本一致性**：这不是严格等价实现。121 条 Normal 正确样本在 Segmented 中变错，说明 padding 或 Append-after-Update 的历史上下文近似确实可能改变输出。
- **不能作因果归因**：这组端到端结果没有将两类近似分别消融，也不能区分 padding、事件时序、GPU 数值非确定性或 P/D 执行时机各自对单条输出的贡献。因此不能把 0.76 F1 的差值解释为“近似带来的质量增益”。
- **可用于论文的表述边界**：可以报告“在 4980 条 SQuAD 对齐混合来源请求上，分段近似路径没有观察到总体输出质量退化”；同时应报告 4.69% 的严格 EM 回归率，而不应宣称输出完全一致。

## 后续需要补充的消融

若要明确两类近似各自的影响，应至少增加以下对照：

- **严格 Append-then-Update**：等待所有 Append 完成后才计算 Update，消除 Append-after-Update 近似，保留 block padding。
- **无语义 padding 对照**：通过 attention mask、KV compaction 或原生 segment offset 让 padding 不参与 attention，隔离 block 对齐近似。
- **事件时序扫描**：改变 Append / Update chunk 的相对到达间隔，统计 Update 先于后续 Append 提交的比例，以及相应的 EM/F1 变化。
- **多模型与更多数据集**：本笔记的结论只覆盖当前 Llama 3.1 8B、SQuAD 对齐混合来源数据和当前 Prompt 模板，不能外推为普遍结论。
