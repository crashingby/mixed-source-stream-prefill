# Volatility-Aware Bounded Update Coalescing

> **研究定位**：混合来源流式 Prefill 的第二个创新点。
>
> **范围**：每个逻辑请求只有一个 Update 源。本文暂不处理多 Update 源、跨源 KV 排布与跨源公平性。
>
> **状态**：设计与实验方案，尚未接入 scheduler。文中的 **UpdateCoalescingState** 是建议的数据模型。

## 核心思想

第一个创新点将稳定的 prefix + append 与可替换的 update 分段管理，使 Update 变化时只失效 Update 源内不匹配的 KV。

但若每个 Update 快照一到达就立刻 Prefill，某些中间版本会在刚计算后立刻被新快照覆盖。它们已经消耗 GPU 时间和 scheduler 配额，却从未参与最终回答。

本机制把 Update Prefill 看成一个**有界在线合并决策**：

- 稀疏、稳定的 Update 立即 Prefill，避免无收益等待。
- 短时间连续到达的 Update burst 暂缓执行，只计算其中最新的完整快照。
- 是否等待和等待多久取决于更新频率、预计合并收益、P 节点负载与 TTFT 剩余预算，而不是固定等待时间。
- 一轮等待具有不可重置的硬 deadline，持续的高频事件不能导致无限等待。

它不改变 Update 的快照语义：最终仍须使用最新完整快照；它只决定已经到达的中间版本是否值得实际提交 Prefill scheduler。

## 与当前分段 KV 设计的关系

逻辑 prompt 顺序为：

$$
\underbrace{\text{template}+\text{append}}_{\text{稳定 prefix 段}}
+\underbrace{\text{update snapshot}}_{\text{可替换 Update 段}}
+\underbrace{\text{final question}}_{\text{finish 时加入}}.
$$

- Append 是单调增长的，不参与本机制的版本合并，仍按既有 scheduler 的竞争语义执行。
- 一个 Update 事件表示唯一 Update 源的**全量快照**。一个等待窗口只保留最新 pending 快照，较旧 pending 快照不会实际计算。
- Update flush 时，现有计算可见已完成的 template + append KV 和 Update 源内 LCP KV；实际重算、失效和新申请 KV block 的仅是 Update 失配后缀。
- 收到 prompt_finished 时，必须先 flush 最新 pending Update 并完成其 Prefill，随后才允许最终用户问题进入最后一次 Prefill / decode。

第一创新点回答“KV 属于哪里、快照变化后失效哪里”；本创新点回答“一个已到达 Update 快照现在是否值得计算”。二者互补。

## 记号与精确 token 收益

设唯一 Update 源产生全量 token 快照：

$$
U_0,U_1,\ldots,U_i,\ldots
$$

最近一次真正完成 Update Prefill 的快照是 $U_c$，称为 committed snapshot；当前等待窗口的最新快照是 $U_p$，称为 latest pending snapshot。

令 $\operatorname{LCP}(U_a,U_b)$ 为两个快照的最长公共前缀。若 KV block 大小为 $B$，且复用与释放按 block 对齐，则：

$$
\operatorname{LCP}_B(U_a,U_b)
=B\left\lfloor
\frac{\operatorname{LCP}(U_a,U_b)}{B}
\right\rfloor.
$$

从 $U_a$ 更新到 $U_b$ 的 Update 段重算 token 数为：

$$
R(U_a,U_b)=|U_b|-\operatorname{LCP}_B(U_a,U_b).
$$

$R$ 只统计 Update 段失效 token；template + append 是可见 attention 前缀，但不是 Update KV 的失效范围。论文统计必须和真实的 KV invalidation 粒度一致；若后续能够精确处理未满 block，可用真实可复用长度替换 $\operatorname{LCP}_B$。

若已经提交 $U_c$ 后依次到达 $U_i,U_{i+1}$，每次立即执行的成本为：

$$
C_{\text{eager}}=R(U_c,U_i)+R(U_i,U_{i+1}).
$$

跳过 $U_i$、只执行最新版本的成本为：

$$
C_{\text{coalesce}}=R(U_c,U_{i+1}).
$$

因此跳过短命中间版本的精确收益是：

$$
G_i=\max\left(
0,
R(U_c,U_i)+R(U_i,U_{i+1})-R(U_c,U_{i+1})
\right).
$$

对整个窗口 $U_c\rightarrow U_{c+1}\rightarrow\cdots\rightarrow U_j$：

$$
\begin{aligned}
C_{\text{eager}}&=\sum_{m=c+1}^{j}R(U_{m-1},U_m),\\
C_{\text{coalesce}}&=R(U_c,U_j),\\
S_{\text{token}}&=C_{\text{eager}}-C_{\text{coalesce}}.
\end{aligned}
$$

虽然中间快照没有真正执行，仍可在事件到达时累计假想 eager 成本，从而准确记录 $S_{\text{token}}$。

## 在线决策模型

这一节要回答一个很具体的问题：**当前 Update 快照已经到达，但还不知道会不会很快被下一快照覆盖时，应该立刻算，还是最多等一小段时间？**

在线决策分为四步：

1. 从已观察到的事件间隔，估计下一次 Update 很快到达的可能性。
2. 从已发生的版本覆盖，估计跳过一个中间版本通常能少算多少 token。
3. 将“预计节省的计算价值”与“额外等待造成的时延代价”比较。
4. 若值得等待，计算一个软等待窗口；硬 deadline 和 TTFT slack 始终限制这个窗口。

### EWMA 是什么

EWMA 是 **Exponentially Weighted Moving Average（指数加权移动平均）**。它是一种只保留一个历史摘要、但让近期观测影响更大的平滑方法。

对连续观测值 $x_i$，EWMA 写作：

$$
\bar{x}_i=(1-\alpha)\bar{x}_{i-1}+\alpha x_i,
\qquad 0<\alpha\leq1.
$$

- $x_i$：这一次刚观察到的值。
- $\bar{x}_{i-1}$：此前保存的平滑历史值。
- $\bar{x}_i$：更新后的平滑估计。
- $\alpha$：近期样本权重。$\alpha$ 越大，系统越快速响应新变化；$\alpha$ 越小，估计越稳定但反应更慢。

例如，先前估计的 Update 间隔为 $100$ ms，本次实际间隔为 $20$ ms，取 $\alpha=0.2$：

$$
\bar{\delta}_i=0.8\times100+0.2\times20=84\text{ ms}.
$$

因此系统不会把一次短间隔直接视为长期规律，但会认为 Update 正在变得更频繁。本文用 EWMA 分别估计 Update 间隔和实际合并收益。

相邻 Update 的到达间隔为：

$$
\delta_i=t_i-t_{i-1}.
$$

第一版对间隔做 EWMA：

$$
\bar{\delta}_i=(1-\alpha)\bar{\delta}_{i-1}+\alpha\delta_i,
\qquad
\hat{\lambda}_i=\frac{1}{\bar{\delta}_i+\epsilon}.
$$

对间隔平滑比直接平滑 $1/\delta_i$ 更稳定。初始时近似下一事件服从指数到达：

$$
F_i(w)
=P(t_{i+1}-t_i\leq w\mid H_i)
=1-e^{-\hat{\lambda}_iw}.
$$

真实检索更新可能呈 bursty 模式；后续可用最近 $N$ 个间隔的经验 CDF 替换 $F_i$，不改变框架。

新事件到达后，前一个 pending 快照已确定被覆盖，可计算 $G_i$ 并更新：

$$
\hat{G}_i=(1-\gamma)\hat{G}_{i-1}+\gamma G_i.
$$

请求刚开始时样本很少，可采用全局先验：

$$
\hat{x}
=\frac{n}{n+\kappa}\hat{x}_{\text{request}}
+\frac{\kappa}{n+\kappa}\hat{x}_{\text{prior}},
\qquad x\in\{G,\lambda\}.
$$

这里 request 级统计就是唯一 Update 源的统计，不需要多源状态表。

等待 $w$ 的预计收益为：

$$
B_i(w)=\mu(t)\hat{G}_iF_i(w),
$$

其中 $\mu(t)$ 是 P 节点资源价格。第一版可取：

$$
\mu(t)=1+\eta Q_P(t),
$$

$Q_P(t)$ 为 Prefill 等待队列长度。系统越拥塞，避免无效 Prefill 的价值越大。

等待的延迟代价为：

$$
D_i(w)=\beta_iw.
$$

$\beta_i$ 是请求对额外延迟的价格；越接近 TTFT deadline，$\beta_i$ 应越大。最终优化：

$$
w_i^*
=\arg\max_{0\leq w\leq W_i^{\max}}
\left[
\mu(t)\hat{G}_i(1-e^{-\hat{\lambda}_iw})-\beta_iw
\right].
$$

在 $w=0$ 处，等待的边际收益为：

$$
\left.\frac{dB_i(w)}{dw}\right|_{w=0}
=\mu(t)\hat{G}_i\hat{\lambda}_i.
$$

故进入等待队列的阈值为：

$$
\mu(t)\hat{G}_i\hat{\lambda}_i>\beta_i.
$$

高频更新、更大合并收益、更高系统负载会提高等待的价值；更紧的 TTFT 则会使系统偏向立即执行。

在指数到达模型下：

$$
w_i^*
=\left[
\frac{1}{\hat{\lambda}_i}
\ln\left(
\frac{\mu(t)\hat{G}_i\hat{\lambda}_i}{\beta_i}
\right)
\right]_0^{W_i^{\max}},
$$

其中 $[x]_0^W=\min(\max(x,0),W)$。若阈值不成立，则 $w_i^*=0$。

### 符号总表

| 符号 | 含义 | 单位 / 取值 |
| --- | --- | --- |
| $i$ | 当前 Update 事件的版本下标；$i+1$ 是随后到达的新版本。 | 非负整数 |
| $t_i$ | $U_i$ 到达 P 节点的时间。 | ms 或 s |
| $\delta_i=t_i-t_{i-1}$ | 相邻两次 Update 的实际到达间隔。 | ms 或 s |
| $\bar{\delta}_i$ | 经 EWMA 平滑后的平均到达间隔估计。 | ms 或 s |
| $\alpha$ | 间隔 EWMA 的近期样本权重。 | $(0,1]$ |
| $\epsilon$ | 防止除零的小正数。 | 与时间同单位 |
| $\hat{\lambda}_i$ | 下一事件的预测到达率，近似为 $1/(\bar{\delta}_i+\epsilon)$。 | 次 / ms 或次 / s |
| $w$ | 当前尝试等待的时间长度。 | ms 或 s |
| $F_i(w)$ | 从现在起 $w$ 内至少到达下一 Update 的预测概率。 | $[0,1]$ |
| $H_i$ | 当前事件前已经观察到的单源 Update 历史。 | 历史状态 |
| $G_i$ | 新版本到达后才能精确观测到的实际合并收益，即跳过前一个中间版本少算的 token 数。 | tokens |
| $\hat{G}_i$ | 根据历史 $G_i$ 预测的未来合并收益，用于当前是否等待的决策。 | tokens |
| $\gamma$ | 合并收益 EWMA 的近期样本权重。 | $(0,1]$ |
| $n$ | 当前请求已观察到的样本数量。 | 非负整数 |
| $\kappa$ | 冷启动时对全局先验的信任强度。 | 非负常数 |
| $\hat{x}_{\text{request}}$ | 仅由当前请求唯一 Update 源历史得到的估计。 | 依 $x$ 而定 |
| $\hat{x}_{\text{prior}}$ | 从历史请求汇总得到的全局先验。 | 依 $x$ 而定 |
| $\mu(t)$ | 时刻 $t$ 的 P 节点资源价格；负载高时更大。 | 无量纲权重 |
| $Q_P(t)$ | 时刻 $t$ 的 Prefill 等待队列长度。 | 请求数 |
| $\eta$ | 队列长度映射为资源价格的系数。 | 非负常数 |
| $B_i(w)$ | 等待 $w$ 的预计计算收益。 | token-equivalent |
| $\beta_i$ | 当前请求每单位等待时间的延迟价格；TTFT 越紧越大。 | token-equivalent / ms |
| $D_i(w)$ | 等待 $w$ 的延迟代价。 | token-equivalent |
| $W_i^{\max}$ | 当前事件允许等待的上限，受 hard deadline 与 TTFT slack 约束。 | ms 或 s |
| $w_i^*$ | 策略选择的最优软等待时间；星号表示优化问题的最优解。 | ms 或 s |
| $\hat{\cdot}$ | 帽子表示在线预测值或平滑估计值，而不是已经观测到的精确真值。 | 记号 |

需要特别区分 $G_i$ 和 $\hat{G}_i$：

- $G_i$ 是事后量。只有下一快照到达并覆盖了前一个 pending 版本后，才能精确算出它。
- $\hat{G}_i$ 是事前预测量。当前还不知道下一快照会不会到达，因此只能依赖历史 $G_i$ 的统计来决定是否等待。


## 有界等待

纯 debounce 会在每次新事件后重置窗口，高频事件可能使请求永久不执行。每轮 pending Update 必须维护两类计时器：

| 计时器 | 定义 | 新事件到达时 | 作用 |
| --- | --- | --- | --- |
| soft_deadline | $t_{\text{last}}+w_i^*$ | 可以重置 | 等待短 burst 自然安静 |
| hard_deadline | $t_{\text{first}}+D_{\text{hard}}$ | **不能重置** | 限制额外陈旧度，防止饥饿 |

标准 P/D 路径的 TTFT 剩余 slack 可估计为：

$$
S_i=D_{\text{TTFT}}-t_{\text{now}}
-\hat{T}_{\text{prefill}}
-\hat{T}_{\text{transfer}}
-\hat{T}_{\text{decode-first}}.
$$

TTFT-only 实验不含 KV transfer 与 D 首 token，应去掉对应项。等待上限为：

$$
W_i^{\max}
=\max\left(0,\min(D_{\text{hard}},S_i)\right).
$$

最终 flush 时刻为：

$$
t_{\text{flush}}
=\min(
\text{soft\_deadline},
\text{hard\_deadline},
t_{\text{now}}+S_i).
$$

若 $S_i\leq0$，必须立即 flush 最新 Update。

## 单 Update 源状态与算法

每个流式 Request 只维护一份长期 **UpdateCoalescingState**：

| 状态 | 含义 |
| --- | --- |
| committed_snapshot | 最近一次真正完成 Update Prefill 的 $U_c$，是 LCP 与 KV 复用基准。 |
| previous_event_snapshot | 最近到达快照，用于累计假想 eager 成本。 |
| latest_pending_snapshot | 当前窗口最新 $U_p$，flush 时唯一会执行的版本。 |
| first_pending_time / last_update_time | 当前窗口首事件与最近事件时间。 |
| soft_deadline / hard_deadline | 可重置静默截止点和不可重置最长等待截止点。 |
| interval_ewma / arrival_rate / merge_gain_ewma | 在线频率与收益预测。 |
| hypothetical_eager_tokens | 本窗口若每个快照立即执行，本应重算的 token 数。 |
| pending_event_count / source_finished | 窗口事件数与唯一 Update 源结束状态。 |

状态只保存 committed、前一个事件和最新 pending 三个快照；旧快照完成 LCP / 收益统计后即可释放。

### 事件到达

1. 校验事件属于唯一 Update 源，且版本号单调递增。
2. 若没有 pending 快照，设置 first_pending_time 和 hard_deadline，并初始化：
   $$
   \text{hypothetical\_eager\_tokens}=R(U_c,U_{\text{new}}).
   $$
3. 若已经存在 pending 快照，累计：
   $$
   \text{hypothetical\_eager\_tokens}
   \mathrel{+}=R(U_{\text{previous}},U_{\text{new}}),
   $$
   同时计算 $G_i$ 并更新收益预测。
4. 更新 previous_event_snapshot、latest_pending_snapshot、间隔 EWMA 与到达率。
5. 按入队阈值和 $w_i^*$ 决定立即 flush，或刷新 soft_deadline 并登记到全局等待队列。

新事件只能刷新 soft_deadline，不能刷新 hard_deadline。

### Flush 最新 Update

1. 以 latest_pending_snapshot 作为本轮唯一执行的目标快照。
2. 仅比较 committed_snapshot 与目标快照，计算 Update 源内 LCP 与 $R(U_c,U_p)$。
3. 按当前分段 KV 语义进行 Update Prefill：可见前缀为 template + append KV 和 Update LCP KV；仅 Update 失配后缀失效、申请新块并计算。
4. 记录：
   $$
   \begin{aligned}
   \text{actual\_merged\_tokens}&=R(U_c,U_p),\\
   \text{saved\_tokens}&=\text{hypothetical\_eager\_tokens}
   -\text{actual\_merged\_tokens}.
   \end{aligned}
   $$
5. 更新 committed_snapshot，并清空 pending 状态。

必须 flush 的情形：

- soft_deadline 到期。
- hard_deadline 到期。
- TTFT slack 耗尽。
- Update 源结束，例如 UPDATE_FINISHED 或 SOURCE_EOS。
- 收到 prompt_finished；先 flush 最新 Update，再调度最终用户问题。
- P 节点空闲且重新评估后立即执行更合算；这是后续优化，不能突破 deadline。

全局 scheduler 维护等待请求的最小堆：

$$
\text{heap key}=
\min(
\text{soft\_deadline},
\text{hard\_deadline},
\text{ttft\_deadline}).
$$

堆顶到期的请求必须转为可调度的 Update Prefill。多个 ready 的 Append / Update Prefill 继续按现有竞争语义调度；本机制不应重新硬编码 Append 优先。

## 正确性与限制

- **快照语义是前提**：Update 必须是全量替代快照。若前端传增量 patch，必须先在长期请求状态中物化完整快照。
- **最终版本不丢失**：只跳过已经被更新版本覆盖的 pending 快照；源结束或 prompt_finished 时，最新 pending 快照必定 flush。
- **陈旧度有界**：从本轮第一个 pending Update 到实际执行的额外等待不超过 $D_{\text{hard}}$，也不应超过 TTFT slack。
- **Update LCP 始终是源内 LCP**：不得将模板、Append token 或最终用户问题放进 $R(U_c,U_p)$。
- **可见上下文不等于可复用范围**：Update token 可注意到 template + append + update-LCP，但可复用 Update KV 只能来自 Update 源内 LCP。
- **当前不支持多 Update 源**：未来若加入多个源，每个源需要独立的 committed / pending 状态与 deadline，并需重新定义最终顺序和 KV 排布。

## 从 token 收益到 GPU 时间收益

相同的 Update 重算 token 在不同可见前缀长度下 GPU 成本不同。可用离线 profiling 拟合：

$$
\hat{T}_{\text{prefill}}(P,R)=aR+bPR+cR^2+d,
$$

其中 $P$ 是可见前缀长度，$R$ 是 Update 重算 token 数；$aR$ 近似线性层开销，$bPR$ 近似新 token 对历史 KV 的 attention，$cR^2$ 表示重算区域内部 attention，$d$ 是 scheduler 与 kernel 固定开销。

后续可用 GPU 时间收益替代纯 token 收益：

$$
\hat{G}^{\text{GPU}}_i
=\hat{T}_{\text{eager}}-\hat{T}_{\text{coalesce}}.
$$

研究应先验证 saved_tokens，再验证这些节省在高负载下是否转化为更低 TTFT 和更高 goodput。

## 离线 Oracle 与实验

给定完整 Update trace 和最大等待 $D$，可计算仅优化重算 token 的离线下界。令 DP[i] 表示 $U_i$ 已提交时的最小累计重算量：

$$
DP[i]
=
\min_{\substack{0\leq j<i\\
t_i-t_{j+1}\leq D}}
\left[
DP[j]+R(U_j,U_i)
\right].
$$

Oracle 不要求在线实现，但可以衡量在线策略与理论下界的距离。

实验应比较：

| 策略 | 行为 |
| --- | --- |
| Immediate | 每个 Update 一到达就执行。 |
| Fixed window | 固定等待 10 / 20 / 40 / 80 ms，只执行窗口末最新快照。 |
| Quiet debounce | 固定静默窗口，但使用同一 hard deadline。 |
| Adaptive | 使用 $\hat{\lambda}$、$\hat{G}$、$\mu(t)$ 和 TTFT slack。 |
| Offline Oracle | 已知完整 trace，在同一等待约束下最小化重算 token。 |

主要自变量为 Update 间隔、burst 长度、快照长度、LCP 比例、Append 长度、QPS、P 节点利用率和 $D_{\text{hard}}$。主要指标为：

- recomputed_update_tokens、saved_tokens、Token Saving Ratio。
- executed_update_versions、Execution Reduction Ratio。
- 等待时间、最大额外陈旧度、hard-deadline flush 比例。
- P 节点 GPU 时间、无效 Prefill 时间、队列长度、SLO goodput。
- p50 / p95 / p99 TTFT；标准 P/D 路径还应拆分 P Prefill、KV transfer 和 D 首 token 时延。

最关键的图包括：额外等待与 Token Saving Ratio 的关系，以及不同 QPS 下的 TTFT 分布。若 Adaptive 在相同硬等待约束下更接近 Oracle，并改善高负载 TTFT tail，就能支撑该创新点。

## 分阶段实现

1. **V1：固定有界窗口**。实现单源 pending 快照、软/硬 deadline 与精确 saved_tokens 统计，先验证跳过中间版本的收益。
2. **V2：请求级自适应**。加入 interval_ewma、merge_gain_ewma、入队阈值与闭式 $w^*$；全局先验只用于冷启动。
3. **V3：负载与时延感知**。加入 $\mu(t)$、TTFT slack 和基于 $(P,R)$ 的 GPU 时间预测，并与离线 Oracle 比较差距。

每一版都应先通过 trace 回放验证：最终提交的 Update 快照与 Immediate 完全一致，不会遗漏 prompt_finished 前最后一次更新，且等待从不超过 $D_{\text{hard}}$。在语义正确的前提下，再比较 token、GPU 时间和 TTFT。