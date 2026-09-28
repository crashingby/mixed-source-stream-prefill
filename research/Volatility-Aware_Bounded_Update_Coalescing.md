# Update 合并：执行背压自适应与有界时间窗口

> **研究定位**：混合来源流式 Prefill 的第二个创新点。
> **范围**：一个逻辑请求至多包含一个单调 Append 源和一个全量快照语义的 Update 源；暂不处理多 Append / Update 源、跨源 KV 排布与跨源公平性。
> **状态**：配置、独立策略类、Request、Scheduler、EngineCore 与 schema v2 统计已接入。当前 `adaptive` 使用执行背压与计算压力。新策略的 GPU 端到端性能尚需用户重新实验验证。

## 当前控制面：只按 Update 合并策略放行

Update 合并只决定“当前快照是否值得现在创建 work”。所有合法 Update 始终维护唯一窗口中的
最新快照与 source-local LCP 统计；仅相应时间策略使用 deadline 决策。一旦满足 policy 的 flush 条件，就直接
创建不可变的 Update work，进入 `pending_work_items`。Append source 是否结束不再作为准入条件。

这与分段流式 Prefill 的近似设计一致：Update 使用该 work 冻结的 Append 上下文；后来到达的
Append 不要求已完成的 Update 自动回滚。**work 准入与 work 可执行仍是两件事**：固定模板必须
先计算，work 依赖的 Append epoch 必须物理提交，最终 Question 仍需等待最终来源工作完成。

### 五种 Update 策略

| 策略 | 放行规则 |
| --- | --- |
| `immediate` | 每条有效 Update 都立即 flush。 |
| `fixed_window` | 从窗口第一条 Update 起等待固定 soft window。 |
| `quiet_debounce` | 每条有效 Update 刷新 soft deadline，等待一段静默时间。 |
| `adaptive` | 本请求无未完成 Update，且全局 token 与序列压力都有余量时放行；没有时间 deadline。 |
| `full_merge` | 等到 Update source 结束或 `PROMPT_FINISHED` 才 flush 最终快照。 |

两种有界等待策略 `fixed_window`、`quiet_debounce` 继续受 hard deadline 和
可选 TTFT deadline 约束。`full_merge` 只等待 Update 的终止条件；Update 先结束时无需等待 Append
结束。它是合并程度的对照，不保证已计算 Update 最终看到后来到达的全部 Append。

`adaptive` 不使用 soft / hard / TTFT deadline。Update source finished / Question 是
强制收尾出口；正常阶段根据执行完成和计算余量放行。已有时间策略、FIFO、LCP 与分段 KV
保持各自语义。所有策略都不设置 Append 来源完成门槛。

## 当前主策略：执行背压自适应

### 优化目标与边界

目标是减少中间 Update 对队列的占用，同时利用 Question 到达前的计算机会。固定毫秒
窗口不能直接控制这两者：连续 deadline 可能提前承诺一长串旧 work，完全等到结束又可能
把全部最终 Update 计算推迟到 Question 后。

`adaptive` 依据执行状态与可执行需求决策，不假设 ANNS 快照必然逐渐稳定。重排序的检索
结果、语音识别修订和其他完整快照来源可使用相同规则。KV 使用率、抢占次数、Question
来源到达率预测均不参与当前 adaptive 决策。

### 放行规则

| 当前状态 | 行为 |
| --- | --- |
| 本请求还有已放行、未完成的 Update | 保留唯一可覆盖快照，不创建新的普通 Update work。 |
| 无未完成 Update，且 token / slot 两类压力均低于阈值 | 冻结最新快照为 work，进入普通 FIFO。 |
| 无未完成 Update，但任一种压力达到阈值 | 继续合并，等待执行进度或资源变化。 |
| Update source finished 或合法 `PROMPT_FINISHED` | 强制放行最后快照，Question 排在最后来源 work 后面。 |

普通阶段通常最多有一个未完成 Update；终止事件可以额外放行最终版本。已放行 work 不再
覆盖、取消或插队。`adaptive` **没有固定等待上限**：`first_pending_time` 可用于观测，
不会生成 soft / hard / TTFT deadline。持续拥堵时可能一直合并到终止事件，因此不承诺
毫秒级 admission 或最优 TTFT，需要分别检验低负载提前计算与高负载少算之间的取舍。

### 真正完成与已经提交 GPU 的区别

`Request.has_outstanding_prompt_stream_update()` 检查以下任意条件：

- `pending_work_items` 中存在 `UPDATE`。
- `active_work_item` 是 `UPDATE`。
- `PromptStreamState.num_update_inflight_steps > 0`。

最后一条不能省略：scheduler 在提交计算时就会推进 token 计数，最后一块调度出去后
active work 可能已经清空，GPU 却还没有回包。每批 `SchedulerOutput` 冻结
`prompt_stream_update_req_ids`，登记真正的 UPDATE 在途步骤，并在对应输出处理时解除。
不会因为随后 Request 切到 Append / Question 阶段就错认这一批的类型。

不额外建立一份 admission ID 集合，也不改变 schema v2 的逻辑 work 结算时机。物理在途
标记专门保护准入判定；原统计、FIFO 和恢复账本仍各自维护。零 forward 的 Update 在状态
转换确认无需计算后解除背压，恢复 UPDATE 同样按 pending / active / inflight 跟踪。

### token budget 与运行名额的双维压力

单轮 token budget 记为 $B$，对应 `max_num_scheduled_tokens`；运行序列容量记为 $S$，
对应 `max_num_seqs`。Scheduler 为已有可执行请求估计下一轮需求（它不是上一轮 budget 用满比例）：

1. 读取当前 work，或 FIFO 队首已放行 work 的剩余 token。PREFIX_APPEND 包括实际 padding
   计算，UPDATE 扣除冻结的可复用 LCP，QUESTION 按未完成部分计算。零 forward 状态转换暂记
   一个需求单位，保证它能得到一次处理机会，而不是把未来整条 FIFO 的工作一次性求和。
2. 新需求先按启用的 `long_prefill_token_threshold` 截断，再与该请求所有已提交、未回包的
   `inflight_tokens` 取较大值，最后上限为 $B$，得到该请求需求 $d_r$。
3. 所有 $d_r$ 的总和记为 $D$，具有正需求的请求数记为 $N$，归一化压力为
   $P_T=D/\max(1,B)$、$P_S=N/\max(1,S)$。

两个阈值默认均为 1.0。普通放行判定等价于下面的表达式：

```python
should_flush = (
    not has_outstanding_update
    and token_pressure < config.adaptive_token_pressure_threshold
    and slot_pressure < config.adaptive_slot_pressure_threshold
)
```

等于阈值也继续合并。两个维度分别处理“长 work 消耗预算”和“短 work 占满序列名额”，
避免仅看 `len(running)` 或单看已调度 token 导致偏差。候选自身已有的 Append 同样计入
压力，例如 `MAX_NUM_SEQS=1` 时可能先完成当前 Append work 再放行 Update；不要求 Append
source finished，后续 Append 仍可到达并与 Update 竞争。

纯等输入的 dormant、尚未放行的合并缓冲与纯外部依赖阻塞请求不计入已有计算需求。
普通请求和流式请求都计入；即使已 abort 或移出队列，只要 GPU 尚未回包，仍保留其在途压力。
候选自身尚未放行的 work 不预先加入压力，否则一个大于单轮 budget 的快照可能永远无法准入。

Controller 的 `adaptive_request_ids` 是有序候选索引，只保存有未放行窗口的请求。
本请求有未完成 Update 且非终止事件时直接 hold，无需扫描全局。批量重评期间，失败候选
复用同一份压力上下文；只有成功放行后才刷新压力，使后续候选看到新增 work。最后一次可能让压力越过
阈值。这是**软准入阈值**，真正 GPU 调度仍严格受 token budget、序列名额和 KV 分配约束。

第一版不对压力做历史平滑，不估计 GPU 毫秒数。该压力也不等于 GPU 利用率。
`MAX_NUM_SEQS`、token budget、长 prefill 截断配置和两个阈值都应随实验记录，之后再调整模型。

### 没有新事件时如何继续推进

Scheduler 在事件处理、batch / 调度入口和回包后重新评估候选。前驱 Update 完成会解除
本请求背压，其他请求结束或进入 dormant 会释放全局计算机会。被放行的 dormant 请求先
回到 waiting，按普通规则进入 running。

EngineCore 无限阻塞等输入之前也要检查候选。若只有合并快照而没有其他可执行 work，压力
已为零，应放行符合依赖的候选继续计算。不能因为 adaptive 没有 deadline 就永久等下一条
REST 消息；也不需要新建定时器线程或每毫秒空轮询。

### 策略类与职责边界

`vllm/v1/core/sched/update_coalescing.py` 提供独立策略类及工厂，按 `update_policy`
创建实例。Request 持有策略与唯一权威窗口，Scheduler 提供不可变决策上下文。

| 层级 | 职责 |
| --- | --- |
| 策略类 | 根据上下文决定是否放行、是否使用时间 deadline，不直接写 KV 或队列。 |
| Request | 校验事件、覆盖快照、计算 LCP、冻结 work、维护原统计与 FIFO。 |
| Scheduler / controller | 计算压力、跟踪在途步骤、重评候选、维护 deadline 及 dormant / waiting。 |
| EngineCore | 驱动调度、消费输出、在空闲阻塞前兑现可运行机会。 |

时间策略继续复用最小堆；adaptive 与 full-merge 不登记时间 deadline。新增策略无需复制
REST、source version、KV 分配或 Question barrier，也无需新增客户端协议字段。

## 已放行 work 的不可变边界


Update 合并必须有清晰的准入边界：

> **合并只发生在 UpdateCoalescingState 中。Update 一旦 flush 放行，就创建完整、冻结且不可取消的 PromptStreamWorkItem；pending_work_items 此后只负责 FIFO 执行，不再做合并、替换或取消。**

因此，合并不是从 pending_work_items 删除旧 Update work，而是在 work 产生之前保留一个唯一可替换的 pending 快照：

~~~text
Update 事件
    │
    ├── 更新最新逻辑快照 source.token_ids
    │
    ├── 在 UpdateCoalescingState 中替换 latest pending snapshot
    │
    └── 满足 flush 条件后，才创建一个不可变 Update work
             │
             ▼
      pending_work_items → active_work_item → UPDATE KV committed
~~~

这保证：

- 高频 burst 中被后续版本覆盖的中间快照不必进入 GPU Prefill。
- 已放行的 work 必定保留并计算；后续 Update 绝不改写、取消或合并它。
- 当前 active_work_item、延后 KV 截断、worker refresh 和 FIFO 语义可以继续成立。
- “已经放行”表示不再受合并策略影响，不表示立即获得 GPU；它仍与其他请求竞争 scheduler token budget 和 KV 资源。

## 分段 KV 前提与 Append-context 风险

混合来源 prompt 的逻辑顺序为：

$$
\underbrace{\mathrm{template}+\mathrm{append}}_{\mathrm{PREFIX\_APPEND}}
+
\underbrace{\mathrm{update\ snapshot}}_{\mathrm{UPDATE}}
+
\underbrace{\mathrm{final\ question}}_{\mathrm{suffix}}.
$$

第一项创新已经规定：

- template 与 Append 组成稳定的 PREFIX_APPEND KV 段；Append 只会单调追加。
- Update 使用独立的 UPDATE KV 段；Update 的失效、重算和 block 申请只影响该段。
- Update Prefill 可以注意到已提交的 PREFIX_APPEND KV，但 PREFIX_APPEND 不属于 Update 源内 LCP，也不因 Update 改写而失效。
- 最终 question 只能在所有已确认的 source work 完成后进入最后一次 Prefill；之后 P 节点产生首 token，并按既有 P/D 路径处理。

本文决定的是 Update 从逻辑快照到不可变 scheduler work 的准入时机；它不改变最终 prompt
顺序、既有分段 KV 排布或 P/D 传输路径。Append epoch 用于保留物理执行依赖和已有的 KV
复用检查，不代表等待 Append source 完成。

## 版本层次

同一个 Update 源在任意时刻维护四个版本，按“输入 → 合并窗口 → 已承诺执行 → 调度侧提交”
前进。调度侧会乐观推进；真实 GPU 完成还要检查在途步骤：

| 记号 | 建议状态 | 含义 | 是否允许后续事件覆盖 |
| --- | --- | --- | --- |
| $U_o$ | `PromptSourceState.token_ids` | 最近接受的最新逻辑快照。 | 是。 |
| $U_p$ | `UpdateCoalescingState.latest_pending_token_ids` | 当前窗口内最新、尚未 flush 的目标快照。 | 是。 |
| $U_e$ | `UpdateCoalescingState.last_admitted_token_ids` | 最近一次已 flush 的完整 Update；它是已承诺执行链的尾部，可能尚未物化到 KV。 | 仅在下一次 flush 时前移。 |
| $U_c$ | `PromptStreamState.update_committed_token_ids` | 调度侧认定已完成整份 work 的快照；最后步骤可能尚在 GPU 中执行。 | work 调度进度完成时前移；不能单独证明 GPU 已回包。 |

此外，`PromptStreamSourceState.append_epoch` 记为 $E_A$，表示当前 Append 上下文版本。
Update work 绑定它 admission 时已经接受的 $E_A$，并在真正执行前等待该 epoch 物理提交。
来源状态保存在 `PromptStreamState.coalescing_sources`，不参与 policy 的 flush 条件。


### 快照必须完整保存

不能只保存 LCP 长度或 LCP 的 token。它们既不能提供当前 work 的待 Prefill 后缀，也不能支持下一次 LCP 比较。

$$
B=[1,2,7],\qquad
P_1=[1,2,8],\qquad
P_2=[1,2,9].
$$

$P_1$ 与 $P_2$ 相对 $B$ 的 LCP 都是 $[1,2]$。若新快照为 $N=[1,2,8]$：

$$
\operatorname{LCP}(P_1,N)=3,\qquad
\operatorname{LCP}(P_2,N)=2.
$$

只保存旧 LCP 无法判断应复用多少。因此，合并层和每个已放行 Update work 都要保存完整冻结快照，或采用“父快照引用 + LCP 长度 + 后缀 token”这种可以完整还原快照的持久化表示。第一版用不可变 tuple[int, ...] 最清楚。

## 数据模型

长期 Request 保留一个 Update 窗口状态和一个轻量来源状态。前者保存 Update 合并统计；后者
保存 source id、完成状态和 Append epoch，不保存第二份 Update token 快照。

下面只展示核心关系。源码 dataclass 还包含 `committed_append_epoch`、
`update_committed_append_epoch`、配置、请求开始时间、TTFT deadline、窗口 token 统计和
`last_admitted_append_epoch` 等执行与恢复字段。

```python
@dataclass
class PromptStreamSourceState:
    # 来源封闭状态用于协议校验和最终 Question barrier，不限制 Update 准入。
    append_source_finished: bool = False
    update_source_finished: bool = False

    # 每次 Append 真实增长递增；work 冻结此版本作为物理执行依赖。
    append_epoch: int = 0
    final_append_epoch: int | None = None


@dataclass
class UpdateCoalescingState:
    # 唯一可被后续 Update 覆盖、尚未进入 pending_work_items 的完整快照。
    latest_pending_token_ids: tuple[int, ...] | None = None

    # 已 flush 的逻辑执行链尾部；新 work 以它作为 execution base。
    last_admitted_token_ids: tuple[int, ...] = ()

    # 时间策略从第一条 Update 起计时；adaptive 不创建下列 deadline。
    first_pending_time: float | None = None
    last_update_time: float | None = None
    soft_deadline: float | None = None
    hard_deadline: float | None = None
    generation: int = 0

    pending_event_count: int = 0
    hypothetical_eager_tokens: int = 0
```

约束如下：

- 合法 Update 先更新 `PromptSourceState.token_ids`，再覆盖 `latest_pending_token_ids`，
  并累计事件数与假想 eager token 成本。
- 未 flush 的窗口期间，`last_admitted_token_ids` 不前移；flush 时读取它并冻结进 work 的
  `execution_base_token_ids`。
- 执行机会、适用 deadline 或终止事件满足 policy 时立即 flush；实际 GPU 执行仍等待 FIFO、Append epoch
  和 scheduler 资源。
- flush 后清空 pending 快照、窗口 deadline 和窗口统计。
- `update_committed_token_ids` 在调度侧整份 work 进度完成时前移；真实 GPU 完成依赖
  `num_update_inflight_steps` 等在途标记。不能用 $U_c$ 或 active 为空替代物理完成判断。

## LCP、重算 token 与 KV block

定义两个 Update 完整快照的 source-local LCP：

$$
L(A,B)=\operatorname{LCP}(A,B).
$$

template、Append 和 question 都不能出现在 $L$ 的输入中。

当前实现按 token 级 LCP 设置 Update work 的已计算进度。因此，实际需要 GPU Prefill 的 Update token 数是：

$$
R(A,B)=|B|-L(A,B).
$$

这应作为论文中的 recomputed_update_tokens。

KV block 仍按 block 管理。若 block 大小为 $B_{\mathrm{blk}}$，安全保留的 UPDATE block 数为：

$$
K(A,B)=
\begin{cases}
0, & L(A,B)=0,\\
\left\lceil \dfrac{L(A,B)}{B_{\mathrm{blk}}}\right\rceil,
& L(A,B)>0.
\end{cases}
$$

当 LCP 落在未满 block 中时，系统保留包含该前缀的 block，但从 token 级 LCP 位置开始重新 Prefill，并覆写该 block 的后缀。因此：

- 逻辑可复用 token 数是 $L(A,B)$。
- 实际重算 token 数是 $R(A,B)$。
- 保留的 block 数是 $K(A,B)$。

它们不是同一个量。旧版笔记使用向下 block 对齐的 LCP 来估算重算 token，会高估当前实现实际需要重新运行的 GPU token。实验应分别报告 token 重算和 block 保留/释放。

Update work 的 attention 可见范围与 Update KV 可复用范围也不同：

~~~text
可见 attention：
    已冻结 PREFIX_APPEND KV + Update LCP KV

新 KV 写入：
    UPDATE segment

允许复用：
    仅限 Update 源内 LCP；PREFIX_APPEND 只是 attention 前缀
~~~


### Update LCP 不变，但 KV 复用需要 Append epoch 一致

source-local LCP 的定义保持不变：$L(U_i,U_j)$ 仍然只比较 Update token。它回答的是
“同一个 Append-context 下，Update 的哪一段 token 可以复用”。

但在 Append-aware 设计中，一个物理 UPDATE KV 还能复用，当且仅当同时满足：

$$
\mathrm{reusable}(U_i,E_i \rightarrow U_j,E_j)
=
\bigl(E_i=E_j\bigr)
\land
\bigl(\text{Update 前缀由 }L(U_i,U_j)\text{ 决定}\bigr).
$$

其中 $E_i$、$E_j$ 是 work 绑定的 Append epoch。启用 controller 后，若下一代 Update
绑定了新的 epoch，已有执行检查仍将可复用 LCP 视为零并重算该快照；本次不修改这条规则。

这不等于每次 Append 到达都会重算 Update：若 Update 已结束，后续只有 Append 增长而没有
新 Update work，已有 UPDATE KV 会保留。最终 Question 可读取这份基于历史 Append 上下文
生成的 KV；这仍是项目接受的时序近似。


## 统一 Update 窗口：观测、合并与 admission

一个合法 Update 从到达到执行分为三步，始终只有一个可覆盖窗口：

| 步骤 | 快照 / 状态 | 用途 | 是否创建 work |
| --- | --- | --- | --- |
| 到达与合并 | 相邻原始快照 $P,N$，并覆盖 $U_p$ | 累计假想 eager 成本和事件数；只按对应时间策略登记 deadline。 | 否。 |
| admission 判定 | $U_p$、执行背压 / 调度压力 / 适用 deadline / source-finished 状态 | 判断 policy 是否要求立即 flush。 | 条件满足时。 |
| flush 执行 | 当前 $U_e$ 与 $U_p$ | 创建 work，work 冻结 `execution_base_token_ids=U_e`。 | 是。 |

### 到达与合并统计

假设新快照 $N$ 到达，上一原始事件快照为 $P$。不论 Append 是否结束，都更新：

$$
\mathrm{hypothetical\_eager\_tokens}\mathrel{+}=R(P,N).
$$

窗口还累计有效事件数。最终省算量用假想 eager 成本减去实际 policy admission 成本，
统计基准跨 recovery 保留；不根据历史事件预测下一次 Update 的时间或收益。

### Flush 与 admission 判定

当新事件、执行进度或调度压力变化、适用的时间 deadline 到期，以及 source-finished /
`PROMPT_FINISHED` 到达时，检查当前 policy 的 flush 条件：

- 条件尚未满足：保留 $U_p$；adaptive 等执行进度，时间策略等 deadline，full-merge 等终止事件。
- 条件已经满足：以 $U_e$ 为前驱、$U_p$ 为目标立即创建冻结 work，推进 $U_e$ 并清空窗口。
- Append source-finished 仅更新来源封闭状态，不会重新计时，也不为 Update 增加额外等待。

这里的“立即”指进入 `pending_work_items`，并不保证该轮马上获得 GPU token budget。

### Flush 时的 LCP

flush 时读取：

$$
L_{\mathrm{execute}}=L(U_e,U_p),
\qquad
R_{\mathrm{execute}}=R(U_e,U_p).
$$

上述是 source-local token LCP；controller 沿用现有 epoch 检查，若前驱与目标 work 的 Append
epoch 不同，则将实际复用 LCP 置零。新 work 冻结完整 $U_p$、`execution_base_token_ids=U_e`、
执行 LCP 和所需 Append epoch。随后立即令
$U_e\leftarrow U_p$，清空 $U_p$ 与当前窗口状态。

## Update 合并策略

所有策略都从窗口第一条 Update 开始维护统计；只有 `fixed_window` 与
`quiet_debounce` 创建 soft / hard / 可选 TTFT deadline。策略满足时直接创建 work。

| 策略 | Update 到达时 | 正常 flush 条件 |
| --- | --- | --- |
| `immediate` | 更新 $U_p$。 | 每次有效到达。 |
| `fixed_window` | 覆盖 $U_p$，不刷新 soft deadline。 | 窗口首事件加 `soft_window_ms`。 |
| `quiet_debounce` | 覆盖 $U_p$，刷新 `last_update_time` 和 soft deadline。 | 最近事件后静默 `soft_window_ms`。 |
| `adaptive` | 覆盖 $U_p$，等待 Scheduler 的最新上下文。 | 无未完成 Update 且 token / slot 都有余量；终止事件强制收尾。 |
| `full_merge` | 始终覆盖 $U_p$。 | Update source finished 或合法 `PROMPT_FINISHED`。 |

`quiet_debounce` 每次有效 Update 都刷新 `last_update_time` 和 soft deadline，但从不刷新
`first_pending_time` 或 hard deadline。重复事件和内容未变化的快照沿用既有去重语义。

## 时间策略的有界等待

`fixed_window` 与 `quiet_debounce` 的 hard deadline 固定锚定窗口第一条 Update；
soft deadline 分别采用固定窗口或最近有效事件的静默窗口规则。`immediate` 与
`adaptive` 没有 timer；`full_merge` 只在 Update source finished 或合法 `PROMPT_FINISHED` 时 flush。

到达有效 deadline 时直接创建 work。**有界的是合并窗口的等待，不是排队到 GPU 执行或完成的
时长**；FIFO、已冻结 Append epoch 和 token / KV 资源仍可能产生额外等待。

当前第一版不在线预测剩余 P 计算、KV 传输和 D 首 token 时间。配置 `ttft_deadline_ms` 后，
只为两种有界 policy 建立绝对 deadline：

$$
D_{\mathrm{abs}}=t_{\mathrm{request\_start}}+D_{\mathrm{TTFT}}.
$$

当 EngineCore 单调时钟到达 $D_{\mathrm{abs}}$ 时 flush 当前窗口。
### 具体例子：Append 未结束时窗口到期

设 fixed window 为 $40\,\mathrm{ms}$，初始执行链尾为 $U_e=U_0$：

| 时刻 | 事件 | Update 窗口状态 | admission |
| --- | --- | --- | --- |
| $0\,\mathrm{ms}$ | $U_1$ 到达 | $U_p=U_1$，soft deadline 为 40 ms。 | 暂不建 work。 |
| $18\,\mathrm{ms}$ | $U_2$ 到达 | $U_p=U_2$，不刷新 soft deadline。 | 暂不建 work。 |
| $40\,\mathrm{ms}$ | soft deadline 到期，Append 仍未结束 | 清空窗口，推进 $U_e=U_2$。 | 创建 `work(U_0 -> U_2)`。 |
| $55\,\mathrm{ms}$ | $U_3$ 到达 | 新窗口 $U_p=U_3$，soft deadline 为 95 ms。 | 暂不建新 work。 |
| $70\,\mathrm{ms}$ | Append source finished | 保留原窗口与 95 ms deadline。 | 不因 Append 结束更改 policy。 |
| $95\,\mathrm{ms}$ | 下一窗口到期 | 清空窗口，推进 $U_e=U_3$。 | 创建 `work(U_2 -> U_3)`。 |

两个 work 均冻结 admission 时已接受的 Append epoch，真正成为 active 前等待该 epoch 物理提交。

## 状态机与事件顺序

先维护逻辑快照与合并窗口，达到 policy 条件后创建冻结 work；物理执行另行接受依赖检查。

### 合法 Update 事件到达

1. `Request` 先完成 version、sequence、去重和乱序校验，更新 $U_o$。
2. 更新唯一的 $U_p$：首次到达初始化 `first_pending_time`；两种有界 policy 同时创建 soft / hard /
   可选 TTFT deadline，后续到达覆盖 $U_p$ 并按 policy 决定是否刷新 soft deadline。
3. 计算相邻 Update 的统计与 `hypothetical_eager_tokens`。
4. 按 policy 判断是否应 flush；满足时立即创建冻结 work。

### Append 与 Update source finished

- **Append event**：照常单调追加并递增 `append_epoch`，不改写 $U_p$、窗口统计或 deadline。
- **Append source finished**：验证前序 sequence、记录 `final_append_epoch` 和来源封闭状态。
  该标记用于后续 Question barrier，不改变 Update 窗口起点或放行规则。adaptive 仍由
  Scheduler 携带真实压力上下文重评，不能在 Request 中用默认空闲上下文绕过全局条件。
- **Update source finished**：标记 source sealed，并立即 flush 尚未放行的最新快照；即使 Append
  尚未结束也创建 work，work 只依赖 admission 时已经接受的 Append epoch。

### Flush 与 Question barrier

flush 时读取 `base = last_admitted_token_ids` 与 `target = latest_pending_token_ids`，创建冻结的
`work(base -> target)`，并将 `base` 写入 `work.execution_base_token_ids`。随后令
`last_admitted_token_ids = target`，清空窗口状态。

启用合并器时，`PROMPT_FINISHED` 是最终请求级 barrier。它会原子封闭尚未单独 sealed、且
没有乱序缺口的现有 source，并把未出现的可选 source 视为不会再到达；随后
强制 flush 尚未放行的最终 Update，再处理 Question 入队。

`finished=True` 只表示最终用户问题已经到达，**不表示 question 已经可调度**。
当前实现将 question 建成显式的 `QUESTION` work，并按以下步骤处理：

1. `PROMPT_FINISHED` 先保存 suffix，标记 `finished=True`，并强制最终 Update 窗口进入应 flush 状态。
2. controller 在最终 Update 已 admission 后调用 `enqueue_question_if_ready()`；它要求
   所有存在或隐式关闭的 source 已 sealed、没有 $U_p$，且最终
   Update work 已在 FIFO 中排定。
3. 该函数只追加一个 `QUESTION` work 到 `pending_work_items` 尾部，并用
   `question_work_enqueued` 防止重复入队。FIFO 保证它不会越过已放行但尚未完成的 Append / Update
   work；Update work 仍先等待 `required_append_epoch` 的物理提交。

因此，question 的真实执行仍由普通 scheduler 决定；它只是不能再依靠 `finished=True` 和
`pending_work_items` 暂时为空这一隐式条件直接开始。

### 准入与物理执行的边界

Update work 在 policy 满足时 admission；固定模板和冻结 Append epoch 是执行依赖，不是
来源完成门槛。不能因为删除了策略约束，就让 Update 读取尚未写入的 Prefix/Append blocks。
Question 的来源封闭、最终 Append 提交与 FIFO barrier 同样保留。

## 启动级功能开关与接入路径

### 目标：合并可选，停放共用

Update 合并是本项目的实验功能，而不是重新定义 vLLM 的通用 scheduler。因此必须满足：

- **默认关闭**：普通 `/v1/completions`、普通 streaming input，以及已有 PromptStream 的
  “每个 Update 事件立即生成 work”语义都保持不变。
- **仅启动 P 节点时开启**：D 节点不接收 PromptStream Update，实验中的 D 节点不创建
  coalescer。普通服务也不传该启动参数。
- **API 协议不变**：`/v1/prompt_stream/start`、`event`、`finish` 的 payload 不增加
  合并控制字段；这是 engine 启动级策略，而不是由单个客户端请求临时改变的策略。
- **启用时只拦截 Update**：Append、最终 question、普通请求和已 flush 的 work 继续走
  当前路径。合并层只在 Update 快照尚未放行的缓冲期间生效。

合并关闭时保留 Update 的即时 admission 语义，不分配窗口或 deadline heap。
流式请求的空闲停放独立于此开关：所有 PromptStream 都在没有 work 时进入 dormant，
有 work 后经 waiting 重新准入。EngineCore 统一区分逻辑未完成请求和当前 runnable work；
只有合并启用时才查询 deadline，关闭时纯等输入仍使用无限期阻塞。
完整共同调度规则见 [Scheduler 笔记](Scheduler.md)。

### 配置对象与命令行路径

新增独立配置对象，避免把每个研究参数平铺到通用 Scheduler 字段中。它采用与
`--kv-transfer-config` 相同的 JSON dict 命令行形式：**不传该参数即关闭；传入合法 dict 即启用**。
因此不再额外设置冗余的 `--enable-*` 布尔开关。

~~~python
@config
class PromptStreamUpdateCoalescingConfig:
    # 所有合法 Update 使用同一窗口策略；满足条件即放行。
    update_policy: Literal[
        "immediate",
        "fixed_window",
        "quiet_debounce",
        "adaptive",
        "full_merge",
    ] = "fixed_window"

    # 仅 fixed_window / quiet_debounce 使用。
    soft_window_ms: float = 20.0
    hard_deadline_ms: float = 80.0
    ttft_deadline_ms: float | None = None

    # 仅 adaptive 使用；是归一化压力阈值，不是毫秒窗口。
    adaptive_token_pressure_threshold: float = 1.0
    adaptive_slot_pressure_threshold: float = 1.0

~~~

| 字段 | 取值 / 使用范围 | 含义 |
| --- | --- | --- |
| `update_policy` | `immediate`、`fixed_window`、`quiet_debounce`、`adaptive`、`full_merge` | 所有 Update 持续采用的窗口 / flush 规则。 |
| `soft_window_ms` | fixed、debounce | fixed window 长度、debounce 静默时长；单位 ms。 |
| `hard_deadline_ms` | fixed、debounce | Update 窗口的最大等待，单位 ms；到期直接创建 work。 |
| `ttft_deadline_ms` | fixed、debounce | 从 Request start 计的绝对放行 deadline；不保证 GPU 排队后的实际 TTFT。 |
| `adaptive_token_pressure_threshold`、`adaptive_slot_pressure_threshold` | adaptive | 有限正数，默认均为 1.0；两类压力均低于阈值时普通放行。 |

`soft_deadline`、`hard_deadline`、token / slot 压力、
`append_epoch` 和来源封闭状态都是每个请求运行时计算出的状态，不放入 JSON。

当前实现的文件和传递路径为：

~~~text
vllm/config/prompt_stream.py
    PromptStreamUpdateCoalescingConfig

vllm/engine/arg_utils.py
    EngineArgs.prompt_stream_update_coalescing_config
    --prompt-stream-update-coalescing-config '<JSON dict>'
        → 解析 PromptStreamUpdateCoalescingConfig | None

vllm/config/scheduler.py
    SchedulerConfig.prompt_stream_update_coalescing_config
        → 随 VllmConfig 传到 P 节点 EngineCore

vllm/v1/core/sched/update_coalescing.py
    根据 update_policy 创建独立策略，接收不可变决策上下文

vllm/v1/core/sched/scheduler.py
    config is not None 时创建 PromptStreamUpdateCoalescingController
    adaptive 读取工作需求与在途状态；时间策略继续维护 deadline

vllm/v1/request.py
    保存来源状态、单窗口、冻结 work、显式 QUESTION 与 Append epoch

vllm/v1/engine/core.py
    所有 PromptStream 使用 runnable 判定；controller 开启时再使用 deadline-aware queue wait
~~~

固定窗口示例：

~~~shell
--prompt-stream-update-coalescing-config \
  '{
    "update_policy":"fixed_window",
    "soft_window_ms":20,
    "hard_deadline_ms":80
  }'
~~~

完全合并基线示例；Update source 结束或 `PROMPT_FINISHED` 时计算最新快照：

~~~shell
--prompt-stream-update-coalescing-config '{"update_policy":"full_merge"}'
~~~

当前 adaptive 的配置只需策略名；显式阈值示例如下：

```shell
--prompt-stream-update-coalescing-config \
  '{"update_policy":"adaptive","adaptive_token_pressure_threshold":1.0,"adaptive_slot_pressure_threshold":1.0}'
```

参数未传时，`EngineArgs.prompt_stream_update_coalescing_config` 与
`SchedulerConfig.prompt_stream_update_coalescing_config` 都是 `None`。这比创建 no-op
controller 更清楚：请求不分配 deadline heap 或合并窗口，Update 仍走即时路径。
Scheduler 的 dormant / inflight 状态独立存在，为所有 PromptStream 提供共同的停放与唤醒；
EngineCore 也统一使用 runnable 判定，避免关闭合并时空闲请求占据 running 或引起忙转。

### 实际开启方式

该参数只传给 P 节点。直接启动 P 节点时，在原命令后追加一段 JSON：

~~~shell
.venv/bin/python -m vllm.entrypoints.openai.api_server \
  --model <MODEL_PATH> \
  --served-model-name <SERVED_MODEL_NAME> \
  --host 0.0.0.0 \
  --port 8100 \
  --enforce-eager \
  --no-async-scheduling \
  --prompt-stream-update-coalescing-config \
  '{"update_policy":"fixed_window","soft_window_ms":20,"hard_deadline_ms":80}' \
  --kv-transfer-config \
  '{"kv_connector":"NixlConnector","kv_role":"kv_producer","kv_load_failure_policy":"fail"}'
~~~

使用仓库内 Slurm 脚本时，通过环境变量传入相同 JSON：

~~~shell
PROMPT_STREAM_UPDATE_COALESCING_CONFIG='{"update_policy":"adaptive","adaptive_token_pressure_threshold":1.0,"adaptive_slot_pressure_threshold":1.0}' \
MODEL_PATH=<MODEL_PATH> \
SERVED_MODEL_NAME=<SERVED_MODEL_NAME> \
sbatch --nodelist=<P_HOST> research/scripts/server/start_p_node.sbatch
~~~

不设置 `PROMPT_STREAM_UPDATE_COALESCING_CONFIG` 就是关闭。D 节点的启动命令
不追加该参数；REST payload 也无需增加合并字段。

### 启动拓扑与第一版兼容性边界

| 进程角色 | 是否启用 | 行为 |
| --- | --- | --- |
| 普通 vLLM server | 否 | 当前 scheduler 与普通 REST 行为完全不变。 |
| P 节点 | 是 | 接收 Update 事件，并在本地合并、Prefill 和保留 P 侧 KV。 |
| D 节点 | 否 | 只接收 P 传来的最终 KV 与首 token，不维护 Update 合并窗口。 |

当前源码会在配置层明确拒绝 DP、多在途 batch，以及已配置但不是 NIXL pull producer 的
KV transfer 组合；source 数量则在 Request 接受事件时 fail closed。运维上仍只应给 P 节点传入该参数：

- 至多支持一个 Update source 和一个真实 Append source；省略的来源会在
  `PROMPT_FINISHED` 时被请求级 barrier 封闭。
- 仅支持单 EngineCore、DP=1、PP=1 的 P 路径。
- 启用时要求 `max_concurrent_batches == 1`；这同时排除 async scheduling 与
  pipeline parallel 的多在途 batch 生命周期。
- 配置 KV transfer 时，仅支持 `NixlConnector` / `NixlPullConnector` 的
  `kv_producer`；D 节点或其它 connector 若误传该开关会在配置阶段被拒绝。

这些限制只在传入 coalescing JSON config 时校验；参数未传时不得改变现有配置组合的合法性。

### 运行时对象与事件接入边界

启用后，Scheduler 创建一个组合式 controller，而不是把合并判断散落到通用队列代码中：

~~~python
self._prompt_stream_update_coalescer = (
    PromptStreamUpdateCoalescingController(config)
    if config is not None
    else None
)
~~~

建议的职责划分如下：

| 位置 | 关闭时 | 启用时 |
| --- | --- | --- |
| `Request` / `PromptSourceState` | 当前事件校验、tokenize 后快照更新、立即创建 Update work。 | 仍是 version、sequence、去重与 source 状态的唯一权威；持有 `PromptStreamSourceState` 和 `UpdateCoalescingState`。 |
| 策略对象与 controller | 不存在。 | 工厂为 Request 创建策略；controller 不复制 token，通过 Request 窄接口处理 timer 与调度重评。 |
| Scheduler 的 dormant / inflight 状态 | 模板完成、没有 work 且无在途步骤时停放；新 work 到达后进入 waiting。 | 采用完全相同的停放与唤醒规则，合并窗口本身不算可运行 work。 |
| `Scheduler._apply_prompt_stream_request()` | 即时创建 work，再执行共同 dormant reconcile。 | 先更新权威 source / window 状态，再执行 flush、Question admission 与共同 dormant reconcile。 |
| `Scheduler.schedule()` | 当前 work 队列调度。 | 只消费已冻结 work；选择 Update work 前还检查 `required_append_epoch` 是否已物理提交。 |

这要求把 Request 内“更新 source 快照”和“创建 Update work”拆成两个内部动作，但不改变
外部 REST 事件或 `EngineCoreRequest` 结构。controller 不能自行绕过 Request 更新 token，
否则会复制 version、sequence number、乱序和去重逻辑。

### dormant 请求独立于合并器

Scheduler 对所有 PromptStream 使用共同停放规则：模板 bootstrap 完成，没有 active / pending
work，当前不能 decode，且 worker 没有该请求的在途步骤。controller 不再拥有 dormant id。
源码把停放原因归并成三个实际字符串；它们是诊断标签，不改变 Request 内的 policy 状态：

| dormant 原因 | 条件 | 是否登记 deadline | 唤醒方式 |
| --- | --- | --- | --- |
| `stream_input` | bootstrap 已完成，当前没有 Update 窗口或可运行 work。 | 否。 | 新 source event、finish 或 abort。 |
| `update_deadline` | 有界窗口尚未到期。 | 是。 | 最近 deadline 或新输入。 |
| `update_policy` | adaptive 等待执行机会，或 full-merge 等待终止事件。 | 否。 | 调度重评、source-finished、finish 或 abort。 |

等待 Update 合并 **不等于**整个 Request 都 dormant：只要还有 PREFIX_APPEND work，request
仍是 runnable，Append 会照常占用 scheduler。所有 work 耗尽后，请求可能只等外部输入，
也可能等待窗口策略；两种情况都可停放。`_prompt_stream_inflight_steps` 保护尚未返回的
GPU 步骤，不能只根据前移的 token 计数提前移除 worker row。

1. Request 仍保留在 `Scheduler.requests[request_id]`，因此后续 event、finish、abort 能按
   request id 找到同一个长期 Request。
2. Scheduler 记录 `_prompt_stream_dormant_reasons[request_id] = reason`；它不复制 Request
   的 token 或 KV 状态，也不要求存在 controller。
3. `_wake_prompt_stream_request(request)` 是唯一的恢复入口：删除 dormant 标记后，若已经有冻结 work，或
   deadline 到期使 controller 新建了 work，则用现有 `_enqueue_waiting_request()` 放回普通队列。
   请求从 dormant 醒来到真正进入 running 之前仍可能保留 segment KV，因此 Scheduler 会把它登记进
   `_prompt_stream_retained_waiting_req_ids` 并设置 `prompt_stream_needs_worker_refresh`。该状态仍是
   runnable waiting，但在 KV 压力和强制 cache reset 下可被回收并 replay。真正 admission 到 running
   后删除 retained 标记。
4. parking 同时把 request id 放入 `_prompt_stream_worker_removal_req_ids`；下一次
   `SchedulerOutput.preempted_req_ids` 先删除 worker persistent row，但 Scheduler 持有的 segment KV
   继续保留。worker removal 自身也算 maintenance work，纯 dormant 时不会无限延后。
5. abort、engine error 与正常完成会清理 dormant、retained-waiting 和 deadline 的逻辑所有权；heap
   中的遗留条目仍可通过 generation / request-id 校验惰性丢弃。

dormant id 仍参与逻辑未完成请求计数，并在 connector 的 accounted-request
口径中占位；只有 EngineCore 的即时 runnable 判定排除它。Scheduler 提供：

~~~python
def has_runnable_engine_work(self) -> bool:
    return bool(
        waiting_or_running
        or pending_worker_removal
        or finished_connector_cleanup
        or connector_pending_push
    )
~~~

上述是语义伪码。实际实现保留真正的 `waiting` / `running` 请求、待发送的 worker-row removal、已完成请求的 connector 后处理以及
connector pending push work。它只排除纯粹等待外部 source 或 admission deadline 的
request id，无论 controller 是否存在；不能把 dormant Request 误判成 finished connector cleanup。
在途计数阻止请求提前离开 running，EngineCore 另外保留 `batch_queue` 检查以消费已提交输出。

### EngineCore 的可选唤醒接口

EngineCore 不应了解 Update 快照、LCP 或策略内部判断，只识别“当前是否存在一个可选的 idle
wakeup provider”。建议 controller 经 Scheduler 暴露三个窄接口：

~~~python
def has_prompt_stream_update_coalescer(self) -> bool: ...
def get_next_prompt_stream_deadline(self) -> float | None: ...
def flush_due_update_coalescing(self, now: float | None = None) -> int: ...
~~~

其中 `get_next_prompt_stream_deadline()` 返回最近的有效有界 Update deadline。
`adaptive` 和 `full_merge` 均返回 `None`；adaptive 通过调度进度及空闲前重评继续推进。

`flush_due_update_coalescing()` 同时处理已经到期的有界 Update window，以及已有计算机会的
adaptive 候选。它返回本次实际放行数；成功时冻结 work，并让因此可运行的 dormant 请求回到 waiting。
Update source-finished 与 `PROMPT_FINISHED` 触发最终窗口 flush；Append work 的物理提交
只解除 `required_append_epoch` 执行 barrier。

EngineCore 将共同停放与可选 deadline 分开处理：

| 调用点 | `enabled=False` | `enabled=True` |
| --- | --- | --- |
| `has_work()` / `step()` / 无模型执行时的 sleep 判定 | 使用 `scheduler.has_runnable_engine_work()`，排除纯 dormant id。 | 使用同一 runnable 判定。 |
| `_process_input_queue()` | 保持 `input_queue.get(block=block)`。 | 先重评 adaptive；仍没有 runnable work 时才按最近 deadline 设置阻塞 timeout。 |
| 输入处理后、调用 `step()` 前 | 不调用合并逻辑。 | 在 EngineCore 主线程调用 `flush_due_update_coalescing(time.monotonic())`。 |

关闭时不应为了“统一接口”而每轮调用 no-op `flush`，也不应创建 timer thread。只有 controller
存在时才查询 deadline；dormant 停放、在途步骤保护与 shutdown 清理则始终适用于
PromptStream。普通非 PromptStream 请求的 token budget、KV 管理和 REST 行为不变。

### 完整的 enabled / disabled 数据流

~~~text
关闭：Update event
    → Request 更新 Update source
    → 立即创建 Update work
    → 现有 Scheduler / Worker 路径

开启：Update event
    → Request 校验并更新 Update source
    → 始终覆盖 U_p，按策略更新适用的观测与 deadline
    → 暂不放行：保留窗口，等待执行机会 / deadline / 终止事件
    → policy 要求 flush：创建绑定 required_append_epoch 的冻结 Update work
    → 现有 Scheduler / Worker 路径
~~~

Append token 仍走现有 `PREFIX_APPEND` work 路径，每次真实增长递增 `append_epoch`。
Update work 绑定 admission 时已接受的 epoch，在该版本物理提交前不能 active；completion
callback 解除这一执行依赖。`PROMPT_FINISHED` 还负责封闭来源、强制 flush 最终 Update 并将
Question 排到 FIFO 尾部；Question 执行前必须确认最终 Append 和 Update 已物理完成。
普通请求不进入上述控制面。

### 关闭开关的回归保证

后续验证必须覆盖：

- 不传开关时，普通 scheduler 行为和 PromptStream 即时 Update admission 规则不变；
  PromptStream 无 work 时也必须停放，唤醒后先经过 waiting。
- 关闭时不创建 `PromptStreamUpdateCoalescingController`、deadline heap 或
  `UpdateCoalescingState`；Scheduler 的 dormant / inflight 状态不受该开关限制。
- 开启但没有 Update 事件时，Append-only 流式 Prefill 的行为与当前版本一致。
- 开启后只有 Update source 的 work 数、等待时间和 Update KV 截断时机发生变化；最终
  prompt token 序列和 D 节点接收的 KV 排布必须与相同输入的 `immediate` 对照一致；不同
  interleaving 下 KV 数值可以因时序近似而不同。

## Scheduler、并发与唤醒

本章的 deadline heap 和 EngineCore deadline timeout 只在
`prompt_stream_update_coalescing_config is not None` 时启用。dormant 集合、在途步骤保护、
唤醒进入 waiting、逻辑请求计数和 shutdown 清理属于所有 PromptStream 的共同机制；
关闭合并不再让无 work 的请求滞留 running。

### 不需要锁

FastAPI/router 不应直接修改长期 Request。外部 REST 事件进入 EngineCore 输入队列后，由同一个 EngineCore 主线程顺序执行：

~~~text
Scheduler.add_request
    → _apply_prompt_stream_request
    → [仅时间策略] 先兑现同一 monotonic 时刻之前已到期的旧窗口
    → Request.apply_prompt_stream_event，先覆盖本次最新快照
    → [adaptive] 使用真实背压与负载 context 判断新快照
    → [其他已启用策略] 按策略条件 flush
    → 所有 PromptStream 执行 dormant reconcile
    → Scheduler.schedule
~~~

PromptStreamState、pending_work_items、active_work_item 和 KV block table 都由该线程修改；开启
controller 时，PromptStreamSourceState、UpdateCoalescingState 与 deadline heap 也由同一线程修改。因此：

- 不需要把 pending_work_items 改成带锁队列。
- 不需要 asyncio.Lock 或 threading.Lock；它们不能跨 API 进程保护 EngineCore 状态。
- 第一版不创建定时器线程；若未来增加外部 timer，也只能投递 wake-up / deadline hint，绝不能
  直接 flush Request 或修改快照。

### deadline 最小堆

启用 controller 后，`PromptStreamUpdateCoalescingController` 维护一个最小堆，条目只包含：

~~~text
(deadline, request_id, generation)
~~~

- 所有尚未到期的有界 window 都可入堆；到期即 flush，清空该窗口的有效 deadline。
  `adaptive` 和 `full_merge` 不创建时间 deadline。
- 每个新窗口和每次允许刷新的 soft deadline 刷新都递增 generation。
- 堆不保存 token；快照始终归 Request 的 UpdateCoalescingState 所有。
- 每条 PromptStream 事件应用前，先以同一 EngineCore monotonic 时刻兑现已经到期的旧窗口，
  因而越过 deadline 的新事件不会反向刷新旧窗口；输入队列 drain 完成后、schedule 前还会再统一
  扫描一次到期条目。
- generation 不匹配的条目是过期 deadline，直接跳过。
- 有效条目在同一 EngineCore 主线程中处理：到期窗口直接创建不可变 work。

这样，deadline 附近若新事件先到达，它会使旧条目失效；若 deadline 先被处理，则旧窗口先 flush，新事件进入下一窗口。两者都满足有界等待和不可变 work 语义。

### dormant 请求

所有 PromptStream 都可能在模板和当前 work 完成后只等待新事件；开启 controller 时还可能
等待尚未 flush 的 latest Update。这些请求不应持续留在 running 集合做空 GPU 调度。

- 已有 active 或已放行 Append/Update work 不受影响，继续正常调度。
- 若 adaptive 暂无执行机会，或 full-merge 等 source 终止事件，则进入无 deadline 的 dormant。
  adaptive 在工作完成与空闲前主动重评，不依赖固定 timeout。
- 若仅等待有界 deadline，则标为带 deadline 的 dormant；EngineCore 在最近
  deadline 到达时被唤醒，flush 后再把请求加入普通等待队列。
- dormant 仍是**逻辑未完成请求**，计入 `get_num_unfinished_requests()`；
  它只从 `has_runnable_engine_work()` 中排除，避免 shutdown、统计或 connector
  生命周期把长期请求误判为已经完成。
- dormant 保留分段 KV。KV 分配失败时，Scheduler 可以驱逐最早的 dormant
  请求块，并从 source 权威状态重建其物理执行链。

第一版应采用下文的 **deadline-aware 阻塞等待**，不能每毫秒空 `schedule()` 检查时间；
后者会引入 CPU 空转并污染实验。

### EngineCore 的 deadline-aware 阻塞等待

第一版已经在 `vllm/v1/engine/core.py` 接入该能力；下面保留问题背景与
实现边界，作为后续审查基准。

#### 为什么空闲等待需要感知 deadline

当前 `EngineCoreProc.run_busy_loop()` 的主循环是：

~~~python
while self._handle_shutdown():
    self._process_input_queue()
    self._process_engine_step()
~~~

原始循环把 `scheduler.has_requests()` 视为有工作；当前实现已经改用 runnable 判定，
以排除纯 dormant 请求。先重评 adaptive 候选，仍没有 runnable work 且没有有效合并 deadline 时，
`_process_input_queue()` 执行 `input_queue.get(block=True)`，无限期阻塞，直到
IO 线程从 REST/ZMQ 收到新的 ADD、ABORT 或其他控制消息并写入 `input_queue`。

如果所有空闲情况都无限等输入，就无法处理**有界 Update 窗口**：一个 dormant 请求可能
拥有 `latest_pending_token_ids`，却没有 active work 或已放行 work。没有新的 Update、
`PROMPT_FINISHED` 或 abort 时，必须依靠 deadline timeout 自行 flush。当前实现已经加入
这个唤醒机制；`full_merge` 继续等待终止事件。adaptive 也没有 timer，但最后一个竞争者
完成、系统准备空闲时必须重评候选，不能遗漏可放行的最新快照。

反过来，不能让这种 dormant 请求继续被当作普通 scheduler work。否则
`scheduler.has_requests()` 始终为真，EngineCore 会不断进入 `step()` 和
`scheduler.schedule()`；没有 GPU work 时当前路径只短暂 sleep，形成低效忙转，而不是
真正等待 deadline。

#### Scheduler 与 EngineCore 的职责边界

Scheduler 独立拥有 dormant 集合与请求在途步骤计数；可选 controller 拥有 deadline
最小堆，以及 adaptive 候选索引 `adaptive_request_ids` 与全局在途 token 的
`inflight_tokens`。Request 拥有策略和 Update 专用在途计数。
当前实现通过 `has_prompt_stream_update_coalescer()`、
`has_runnable_engine_work()` 以及下面的 deadline 查询、统一放行接口与 EngineCore 协作：

~~~python
def get_next_prompt_stream_deadline(self) -> float | None:
    """返回时间策略的最近有效 deadline；adaptive/full-merge 返回 None。"""

def flush_due_update_coalescing(self, now: float | None = None) -> int:
    """处理到期窗口和 adaptive 候选，唤醒可运行请求并返回放行数。"""
~~~

- dormant 请求不应被当作可运行 GPU work。当前
  `Scheduler.has_runnable_engine_work()` 供 `EngineCore.has_work()`、`step()` 中
  “是否进入 `scheduler.schedule()`”的早退判断，以及“无模型执行时短暂 sleep”的判断
  统一使用 runnable 语义，与合并开关无关；不能只改 `has_work()`，否则仍可能进入空转路径。
  `has_requests()` 若还承担请求生命周期统计，可保留其原语义。
- dormant 集合不混入普通 `waiting` / `running` 队列；请求对象仍可保留在按
  request id 索引的长期状态表中，以接收后续 Update、finish 或 abort。
- `get_next_prompt_stream_deadline()` 只读最小堆，跳过 generation 失效、请求已取消、策略已变为
  `full_merge` 或已无 pending Update 的陈旧条目。
- 所有 deadline、事件与 recovery 路径最终都调用 Request 的
  `flush_prompt_stream_update()`，它是将 `latest_pending_token_ids` 物化为
  不可变 `pending_work_items` 的唯一状态转换接口。

#### 主循环的目标顺序

第一版不创建定时器线程。空闲阻塞前先重评 adaptive，再为时间策略把 queue wait
设为“最多等待到最近 deadline”。下面展示时间路径，adaptive 重评还在空闲判定与回包后执行：

~~~python
while self._handle_shutdown():
    # 处理已经到达的 REST/ZMQ 输入；可能创建、刷新或取消 deadline。
    self._process_input_queue()

    if self.scheduler.has_prompt_stream_update_coalescer():
        # 同一主线程检查到期窗口及 adaptive 候选，必要时唤醒 dormant。
        self.scheduler.flush_due_update_coalescing(time.monotonic())

    # 只调度已经有实际 GPU work 的普通请求。
    self._process_engine_step()
~~~

`_process_input_queue()` 在 `not self.has_work()` 时的等待逻辑改为概念上的：

~~~python
if self.scheduler.has_prompt_stream_update_coalescer():
    # 输入队列阻塞前检查执行机会；adaptive 即使没有新事件也可以放行。
    self.scheduler.flush_due_update_coalescing(time.monotonic())
    if self.has_work():
        return  # 回到 EngineCore 调度执行，不能继续阻塞等 REST 输入。

if not self.scheduler.has_prompt_stream_update_coalescer():
    # 关闭时保留当前的无限期阻塞语义。
    request = self.input_queue.get(block=block)
else:
    deadline = self.scheduler.get_next_prompt_stream_deadline()
    timeout = None if deadline is None else max(0.0, deadline - time.monotonic())

    try:
        request = self.input_queue.get(block=True, timeout=timeout)
    except queue.Empty:
        # 不是错误；最近的 coalescing deadline 到期。
        return

self._handle_client_request(*request)
~~~

开启时，`queue.Empty` 返回主循环后，紧随其后的 `flush_due_update_coalescing()` 会把到期的
Update 放行；adaptive 候选已重评、仍无 runnable work 且无有效 deadline 时，
`timeout=None`，才允许无限等待。关闭时不会出现
这个 timeout 路径。因此“EngineCore 因 deadline 醒来”并不是新增一个线程或主动中断 GPU；它只是
启用 controller 后，原有 EngineCore 主线程从带 timeout 的阻塞 `Queue.get()` 返回。

#### 输入事件与 deadline 同时发生

下面的 deadline 先后规则仅适用于时间策略。应用每个 PromptStream 事件前，Scheduler
先按当前 monotonic 时间兑现已经到期的旧窗口；处理完输入队列后，主循环再检查到期窗口与
adaptive 候选。adaptive 在单个事件处理中先应用最新快照，再使用真实 context 判断，
不会在新版本已经到达时先抢着放行旧版本。

- 事件处理时刻仍早于 deadline，它可以刷新 soft deadline 并递增 generation；
  堆内旧条目随后惰性失效。
- 事件处理时刻已经达到 deadline，旧窗口先冻结成 work；
  本事件进入下一窗口，不能把已到期边界向后刷新。
- queue wait 先 timeout 时，主线程先 flush 旧窗口；稍后到达的新 Update 进入下一窗口。
- 无论哪一种先发生，hard deadline 不会被刷新，且已放行 work 永远不可变，因此不需要
  锁来保护 `Request`、`pending_work_items` 或 KV block table。Append / Update source-finished
  事件与最后一个 Append work 的 completion callback 也在这个主线程串行化：只有
  `required_append_epoch` 已提交，Update work 才可变为 runnable。

`process_input_queue_block=False` 的既有非阻塞路径应保持原样；deadline 检查仍在每轮
主循环发生，但它不应向该特殊路径额外引入阻塞。

## flush 后的 KV 执行与 Append epoch barrier

本节定义 policy 放行后，冻结 Update work 如何与既有分段 KV 路径衔接。合并器只决定**是否和何时**
创建 work；它不直接持有 block id，也不直接调用 worker。

### 冻结 work 的语义

启用 controller 时，每个已放行的 Update work 至少冻结以下逻辑字段：

| 字段 | 含义 | 写入时刻 |
| --- | --- | --- |
| `update_token_ids` | 本 work 对应的完整 Update 快照 $U_p$。 | flush。 |
| `execution_base_token_ids` | flush 时读取的 $U_e$；它仅在该 work 内冻结为执行前驱。 | flush。 |
| `update_common_prefix_len` | $L(U_e,U_p)$，只比较 Update token。 | flush。 |
| `required_append_epoch` | 本 work 所需的 `PREFIX_APPEND` 物理版本，为 admission 时已接受的 epoch。 | flush。 |
| `prefix_append_logical_len` | 该 epoch 下可见 Prefix+Append 的逻辑 token 长度。 | flush 或由 epoch 查表。 |

`required_append_epoch` 是 **scheduler dependency**，不是让 worker 临时拼接任意 block id 的捷径。
Request 的 `PromptStreamState` 仍是 block table 的唯一权威：它根据已提交的 prefix / update
segment、各段逻辑长度和既有 padding 规则，构造 worker 所需的 slot mapping / block table。

### 从放行到提交的顺序

1. Policy 满足条件，冻结目标快照和 `required_append_epoch`。
2. 前序 `PREFIX_APPEND` work 提交该 epoch。
3. Scheduler 校验 `committed_append_epoch == required_append_epoch`，允许 Update active。
4. 校验 Update 执行前驱，按 source-local LCP 与 epoch 规则截断、计算并提交 UPDATE KV。
5. 若最终 Question 已入队，继续按 FIFO 等待其余来源工作完成。

以下检查在这条链路中始终保留：

1. **flush 只创建不可变 work。** 它不截断 UPDATE KV、不释放 block，也不把 work 直接塞给 worker。
2. **Scheduler 先检查 Append dependency。** 若 `committed_append_epoch < required_append_epoch`，该
   Update work 不能成为 active；若前者大于后者，则说明 FIFO / 冻结依赖已被破坏并立即 fail closed。
   只有二者严格相等时才可执行。
3. **dependency 满足后才计算执行 LCP。** 沿用已有 controller 规则，UPDATE KV 当前 epoch 与
   `required_append_epoch` 相同，才可按 work 内冻结的
   `L(execution_base_token_ids, update_token_ids)` 延后截断并复用
   Update 前缀；epoch 不同则必须视为 LCP 为零并重建，不能错误复用 token LCP。
4. **worker 的 attention 视图**是当前已提交的 `PREFIX_APPEND` 加上 Update LCP KV；新计算 token
   只写入独立 `UPDATE` segment。Append block 不因 Update 被截断或覆写。
5. **完成提交**后才令 `update_committed_token_ids = U_p`，同时记录
   `update_committed_append_epoch = required_append_epoch`。work 对象可回收，但 committed snapshot
   与对应 UPDATE KV 继续保留给下一代 Update 比较。

逻辑 token 与物理 block 边界必须继续分开看：`PREFIX_APPEND` 末 block 未满而 UPDATE segment 从
独立 block 起始时，现有分段实现可能含有 segment 间 padding。该 padding / attention-mask /
position 处理由现有 Masked Padding 路径负责；本合并器不得通过改变 `update_common_prefix_len`
或伪造 Append token 来处理它。

若目标 $U_p$ 是其 work 内执行基准 $U_e$ 的严格前缀，$R(U_e,U_p)=0$，仍要创建零 token
状态转换 work：它负责截断多余 UPDATE KV、更新 `(U_c,E_A)` 和解除后续依赖，不能简单丢弃。

### 执行语义的近似边界

- Append 与 Update work 按实际到达、policy 放行和 FIFO 调度顺序竞争，Update 不等待 Append
  source 结束；它只等待已经冻结的 Append epoch 物理提交。
- 开启 controller 后，下一代 Update 若绑定了不同 Append epoch，继续使用现有 LCP 清零重算
  规则；该检查与本次删除的来源完成约束无关。
- 已完成的 Update 不会仅因后续 Append 到达就自动创建新 work。若没有下一代 Update，最终
  Question 仍可读取基于较短历史 Append 前缀生成的 UPDATE KV。
- 合并改变计算时机，也可能改变各版 Update 所见的 Append 上下文。性能对照共享相同的输入和
  近似规则，但不承诺不同 policy 的 KV 数值或模型输出逐位一致。

## 预抢占、取消、重试与协议异常

### 预抢占 / 物理 KV 丢失

preemption 释放的是物理 KV，不是逻辑 source 状态。正常新 Update 到达绝不删除已放行 work；但
preemption 使其物理前驱不存在时，允许清空受影响的 active / pending 物化 work，并从 Request 的
权威逻辑状态确定性 replay。这是故障恢复，不是合并替换。

1. 记录丢失的 PREFIX_APPEND、UPDATE 与 Question 物理 token，释放 blocks。未完成 admission
   的取消或继承写入 ledger，不能将 recovery attempt 重新记作 policy 收益。
2. 清空物理 active / pending work、segment 计算进度和 committed 快照；保留权威 source、
   source finished 状态、Append epoch、未放行窗口、原绝对 deadline 与窗口 token 统计。
3. 先按最新逻辑 Append 重建 PREFIX_APPEND，物理 committed epoch 从零重新推进。
4. 若存在未放行窗口，保留其原计时和 policy，重置已失效的执行链基准，待原条件满足后 flush；
   repeated preemption 不能延长合并等待。
5. 若没有未放行窗口，则为最新已接受 Update 建立 recovery work，从 LCP 为零开始恢复；空快照
   沿用零 token 状态转换规则，不强行发起空 forward。
6. Question 可以排在 recovery work 后面，执行时仍受 FIFO、最终 Append epoch 和最新 Update
   committed 状态约束。

恢复 work 的 `execution_base_token_ids` 和 `required_append_epoch` 指向重建后的执行链，
不能引用已经释放的旧 blocks。

### 取消、重复、乱序与 finish 协议

- abort、engine error 或 Request 完成后，Scheduler 清理 request id 的 dormant / retained
  状态；存在 controller 时同时取消有效 deadline，heap 遗留条目通过 request 不存在或
  generation 不匹配自然失效。在途输出的处理仍遵守已有 worker 生命周期。
- 第一版 REST 事件采用严格 source-local 顺序：每个新 source 的首事件必须是 `sequence_no=0`，
  后续 sequence 必须正好等于 `next_sequence_no`；发现 gap、回退或跳号立即 fail closed，不先返回
  ACK 再把事件留在未消费的 pending buffer。`SOURCE_FINISHED` 同样必须与下一期待 sequence 和
  当前 source version 一致。
- 重复 chunk 或 token 完全相同的 Update snapshot 不创建 work，也不刷新 `quiet_debounce` deadline。
- Update 的相同 version 若携带不同 token 内容仍为协议错误；Append 的 version 不承担快照
  替换语义，允许同一 version 下追加不同 chunk，但 version 也必须保持单调且不能回退。
- 启用合并器时，`PROMPT_FINISHED` 是最终请求级 barrier：已单独发送 `SOURCE_FINISHED` 的
  source 保持 sealed；尚未单独封闭但没有乱序缺口的现有 source，以及根本未出现
  的可选 source，会被原子地封闭。此后任何 source event 均被拒绝。

## 正确性不变量

- **最终快照不丢失**：每个合法 Update 更新 $U_o$ 并进入唯一未放行窗口 $U_p$；合法结束后，
  最终 $U_o$ 必须可追溯到最终 Update work。
- **准入不等于物理提交**：policy 满足即可创建 work；只有
  `committed_append_epoch == required_append_epoch` 才允许它 active。大于所需 epoch 表示 FIFO /
  冻结依赖已被破坏，必须 fail closed。
- **保留既有 epoch 复用规则**：Update LCP 只比较 Update token；启用 controller 时，跨 Append
  epoch 的下一代 Update 仍按零 LCP 重算。该规则不要求因 Append 增长自动重放已完成 Update。
- **放行后不可变**：除 preemption / error recovery 外，新 Update 只能覆盖未放行的 $U_p$，
  不能删除、覆盖或缩短已放行 work。
- **KV 截断安全**：event 到达、窗口合并和 deadline 到期不触碰 UPDATE KV；仅在 work active、
  Append dependency 与 FIFO 前驱满足后，才执行延后截断。
- **时间策略的有界等待范围**：fixed/debounce 的 hard deadline 锚定窗口首事件；到期放行不能被后续事件推迟，但 GPU 排队
  和恢复耗时不受窗口 deadline 保证。adaptive / full-merge 不承诺时间上限。
- **Question barrier 完整**：Question 执行前来源已封闭、最终 Append epoch 已提交，窗口和
  active / pending 中不存在尚未完成的最终 Update 状态转换。
- **单线程控制**：所有 source、window、deadline、work 与 KV 状态仅由 P 节点 EngineCore 主线程
  变更；router / FastAPI 线程只投递输入。
- **合并可选、停放共用**：config 为 `None` 时不创建 controller / heap / 合并窗口，Update
  仍即时 admission；所有 PromptStream 共享 dormant / waiting / running 生命周期，普通
  非 PromptStream 请求保持原有调度语义。

## 统计、对照与实验

### 公平的 token 成本口径

“少算 token”首先明确对照：`disabled` 表示原即时路径；`immediate-controller` 表示开启合并
控制器、每次有效 Update 都放行。两者现在共享 dormant 停放规则，但 controller 仍保留
自己的 epoch 检查，后者是比较其余 policy 的直接控制面对照；不能未经验证就把两条代码
路径当成完全相同的执行成本。旧版 disabled 与 controller 的停放行为不同，新实验应统一版本。

对于相同 Append epoch、物理前驱仍有效的一次 flush：

$$
C_{\mathrm{actual}}=R(B,W),
\qquad
B=\text{work.execution\_base\_token\_ids},\quad
W=\text{work.target\_update\_token\_ids}.
$$

若 epoch 改变导致零 LCP，实际 work 成本改用完整目标长度；物理抢占导致的恢复另计。
策略比较使用相同 source 输入和近似执行规则：

$$
S_{\mathrm{token}}=C_{\mathrm{immediate}}-C_{\pi},
\qquad
\mathrm{saving\_ratio}
=\frac{S_{\mathrm{token}}}{\max(C_{\mathrm{immediate}},1)}.
$$

现有 schema v2 区分反事实 eager、policy admission 与 recovery execution。上述跨实验对照
和服务器返回的窗口反事实统计应分别标注，不能把抢占重算忽略后声称是总 GPU token 节省。

### 必要的运行时统计

下列为诊断与后续实验需要观察的指标，部分是建议补充项，不表示当前返回 schema 已全部导出。

- `update_window_events`、`replaced_pending_snapshots`、`window_wait_ms`：衡量单一 Update
  窗口吸收了多少 Update 波动及其等待。
- `admitted_update_versions`、`execution_reduction_ratio`：衡量合并后真正进入执行链的版本。
- `recomputed_update_tokens`、`saved_update_tokens`、`token_saving_ratio`：区分 policy 与 recovery 口径统计。
- `append_epoch_at_admission`、`required_append_epoch`、`update_committed_append_epoch` 与 dependency
  wait time：验证无跨 epoch 误复用。
- `flush_reason`：`immediate`、`soft_deadline`、`hard_deadline`、`ttft_slack`、`update_finished`、
  `prompt_finished`、`recovery_replay`。
- retained / released Update blocks、partial-block overwrite 次数和 segment padding 的逻辑长度：区分
  token 节省与 block churn。
- P 节点 GPU 时间、队列长度、GPU 利用率、TTFT p50/p95/p99、goodput；将 Update window waiting、
  scheduler queueing 和 recovery 分开报告。
- 如需解释质量变化，可额外记录 work 完成后到达的 Append token 数，单独评估时序近似影响。

不同可见 PREFIX_APPEND 长度下，相同 $R$ 的 GPU 时间并不相同。后续可拟合：

$$
\hat{T}_{\mathrm{prefill}}(P,R)=aR+bPR+cR^2+d,
$$

其中 $P$ 是 work 绑定 epoch 下可见的 Prefix+Append 逻辑长度，$R$ 是 Update 重算 token；于是

$$
\hat{G}^{\mathrm{GPU}}
=\hat{T}_{\mathrm{immediate}}-\hat{T}_{\pi}.
$$

### 实验臂与 Oracle

| 实验臂 | Update policy | 作用 |
| --- | --- | --- |
| `disabled` | 配置为 `None` | 原始逐事件即时路径。 |
| `immediate-controller` | `immediate` | controller 开启后的逐事件对照。 |
| `fixed-window` | `fixed_window`，10 / 20 / 40 / 80 ms | 固定等待。 |
| `quiet-debounce` | `quiet_debounce` + 同一 hard deadline | 吸收 burst。 |
| `adaptive` | `adaptive` | 执行背压与双维计算压力，当前主自适应方案。 |
| `full-merge` | `full_merge` | 只计算终止时最新快照的合并基线。 |

下述 `Offline Oracle` 是有界时间策略的历史参考，沿用真实到达时间、最大窗口等待约束和
执行 epoch 规则；adaptive 没有时间上限，不能直接当作同约束的比较对象。
当一个比较区间的 Append epoch 固定时，若 $U_0$ 是执行基准、$U_i$ 是窗口内快照、$D$ 是最大
等待，令 `DP[i]` 为最终提交 $U_i$ 时的最小累计重算量：

$$
\mathrm{DP}[i]
=\min_{\substack{0\leq j<i\\
t_i-t_{j+1}\leq D}}
\left[\mathrm{DP}[j]+R(U_j,U_i)\right].
$$

若跨 epoch，则还需纳入当前实现的零 LCP 成本及执行依赖；不能让 Oracle 免费使用未来才到达的
Append context。上式是 token 成本参考，不是包含 GPU 排队与抢占的端到端最优调度器。

自变量包括 Append / Update 相对结束顺序、Update 间隔、burst 长度、快照长度、LCP 比例、
Append 长度、QPS、P 节点利用率和 hard deadline。重点图包括各策略相对 `full-merge` 的 TTFT
survival curve、相对即时路径的 token 节省与额外等待，以及不同 QPS 的 p50/p95/p99、成功率、
goodput 和 recovery 开销。

旧实验曾额外限制 Update 必须等待 Append 来源封闭。其已有结果只能作为历史探索记录，不能
直接当作当前 policy 的新结果；移除这一限制后的性能结论应重新采集。

## 实现进度与后续阶段

1. **配置与 Request 状态机已接入**：JSON config、`None` 默认路径、`PromptStreamSourceState`、
   单一 Update 窗口、Append epoch、冻结 Update / QUESTION work 和执行依赖仍保留。
2. **Scheduler / EngineCore 控制面已接入**：五种策略由工厂创建；adaptive 按执行机会重评，时间策略的 deadline heap 和
   deadline-aware `Queue.get(timeout=...)` 由启动开关隔离；达到 policy 条件直接 admission。
   dormant 停放 / waiting 唤醒、在途步骤保护和 shutdown 清理已扩展到所有 PromptStream。
3. **恢复和协议保护已接入**：preemption / dormant KV 驱逐从 source 权威状态重建，保留未放行
   窗口的原 deadline / token 统计；abort、sequence gap、Append epoch 和 committed 快照检查保留。
4. **当前自适应机制**：无硬时间上限，使用本请求未完成 Update 与全局 token / slot 压力；
   source-finished / Question 强制收尾。不依赖来源更新趋势或到达率预测。
5. **实验脚本已配套**：profile helper 同时生成 P JSON 与 replay 审计值，汇总保留压力阈值；
   启动和 sweep 命令见 [实验 README](scripts/replay/update_coalescing/README.md)。
6. **待验证**：用户运行新策略 GPU 实验，比较 TTFT、work 与 token 节省；阈值只是第一版
   启发式，尚无性能结论。主实验尽量避免抢占，保留恢复指标监测干扰，不在本次优化 victim。

所有策略必须保留最终逻辑 Append、最新 Update 和 Question，并正确维护其物理执行顺序。
合并只能覆盖尚未放行的窗口，不能改写已经承诺执行的 work；不同 policy 的执行时机可能改变
历史 Update 所见的 Append 上下文，其近似质量应与性能指标分别评估。

## 本次实现验证

当前 99 项 CPU 单元测试通过，覆盖策略工厂与配置、Update 在途背压、
token / slot 压力、终止事件收尾、dormant 唤醒、普通与异步 scheduler 相关回归路径。
实验目录的 shell 语法、Python 语法、profile JSON 和 sweep 参数转发也完成本地检查。

没有启动 vLLM 服务、GPU 端到端测试或性能回放。上述检查说明状态机与配套接口通过了
本地验证，不代表已得到 adaptive 的 TTFT 收益结论；新性能数据仍需用户实际采集。
