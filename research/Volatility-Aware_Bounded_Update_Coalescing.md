# Volatility-Aware Bounded Update Coalescing

> **研究定位**：混合来源流式 Prefill 的第二个创新点。
> **范围**：一个逻辑请求只允许一个单调 Append 源和一个全量快照语义的 Update 源；暂不处理多 Append / Update 源、跨源 KV 排布与跨源公平性。
> **状态**：完整设计，尚未接入 scheduler。现有系统已经具备分段 KV、冻结 Update work、FIFO 执行与延后 UPDATE KV 截断；本文在此基础上设计 Append-aware（感知 Append 完成）的两阶段 Update 合并层。

## 重构说明：Append-aware 两阶段控制面

本文原先只讨论“一个 Update 快照进入 UpdateCoalescingState 后，何时 flush”。现在将其重构为两个职责分离、但每条 Update 都会经过的控制面：

1. **Update 合并策略**始终维护唯一的 Update 窗口：最新快照、LCP 统计、EWMA 与 deadline。
2. **Append 完成门控（Append completion gate）**只决定一个已满足 flush 条件的窗口，此刻能否 admission 成 scheduler work。

这个拆分来自一个关键观察：Update KV 虽然只用 Update 源内 LCP 管理失效，但它的 attention
仍依赖当时的 `PREFIX_APPEND`。因此 Update 在 Append 仍会增长时先执行，看到的是**执行时刻
可见的 Append 前缀**；这属于本项目已接受的流式近似，而不是运行时错误。

### 两阶段总览

```text
合法 Update 事件
    │
    ├── 始终更新 PromptSourceState.token_ids（最新逻辑快照）
    │
    ├── 始终更新 UpdateCoalescingState
    │   │      ├── 覆盖 latest_pending_token_ids
    │   │      ├── 更新 LCP 统计、EWMA、soft / hard deadline
    │   │      └── 记录当前窗口的 flush 条件
    │   │
    ├── 当前 Update policy 是否要求 flush？
    │   │
    │   ├── 否：保留窗口状态，等待下一条 Update 或 policy 触发
    │   │
    │   └── 是：检查 Append gate 是否允许 admission
    │          ├── 否：保留窗口状态；置 flush_ready，不创建 work
    │          └── 是：将 latest pending flush 成冻结 work
    │
    └── pending_work_items → active_work_item → UPDATE KV committed
```

### 两条正交策略轴

| 轴 | 选项 | 含义 |
| --- | --- | --- |
| Append gate | `none` | 不等 Append；这是旧的纯 Update 合并语义与消融基线。 |
| Append gate | `hold_until_append_finished` | 严格模式：Append 未封闭前不物化 Update work。主方案与正确性对照使用它。 |
| Append gate | `hold_until_append_or_update_finished` | Update source 先结束时也打开 gate；将此前 Update 合并结果按当时可见的 Append 前缀执行。它是近似早放行模式。 |
| Update policy | `immediate` | 每条 Update 都要求 flush。 |
| Update policy | `fixed_window` | 固定窗口。 |
| Update policy | `quiet_debounce` | 静默防抖窗口。 |
| Update policy | `adaptive` | EWMA 驱动的在线等待。 |
| Update policy | `full_merge` | 等到 Update source 结束或 `PROMPT_FINISHED` 才要求 flush。 |

### 研究对照必须区分的三类方案

- **无 gate 的纯 Update 完全合并**：不管 Append 状态，所有 Update 都等到终止事件才计算。
  它保留旧设计的假设，作为消融基线。
- **严格 joint-finish merge**：同时等待 Append 与 Update 都封闭，再仅计算最终 Update。
  这是在最终上下文稳定条件下的“完全合并”基线，应记为 `joint_finish_merge`。
- **严格 Append gate + Update policy**：Update 从第一条事件起就按 `immediate`、两种固定窗口、
  `adaptive` 或 `full_merge` 维护窗口；Append 一结束即允许已到期窗口 admission。

严格模式中，Append gate 的等待不是普通 Update window：其上限由上游 source 的完成、请求
取消决定；source 已封闭后的 `PROMPT_FINISHED` 只承担最终 question barrier 的作用，而不是由
Update 的 `hard_deadline_ms` 强行打破。否则会重新引入尚未稳定的 Append-context Update KV。

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

本文决定的是 Update 从逻辑快照到不可变 scheduler work 的两阶段准入时机；它不改变最终 prompt
语义、既有分段 KV 排布或 P/D 传输路径。新增的 Append epoch 只用于判定某份 UPDATE KV 是否仍能
在当前 `PREFIX_APPEND` 上下文中安全复用。

## 版本层次

同一个 Update 源在任意时刻只需要维护四个版本。它们按“输入 → 合并窗口 → 已承诺执行 → GPU
已完成”的顺序前进：

| 记号 | 建议状态 | 含义 | 是否允许后续事件覆盖 |
| --- | --- | --- | --- |
| $U_o$ | `PromptSourceState.token_ids` | 最近接受的最新逻辑快照。 | 是。 |
| $U_p$ | `UpdateCoalescingState.latest_pending_token_ids` | 当前窗口内最新、尚未 flush 的目标快照。 | 是。 |
| $U_e$ | `UpdateCoalescingState.last_admitted_token_ids` | 最近一次已 flush 的完整 Update；它是已承诺执行链的尾部，可能尚未物化到 KV。 | 仅在下一次 flush 时前移。 |
| $U_c$ | `PromptStreamState.update_committed_token_ids` | GPU 中 UPDATE KV 当前真实对应的完整快照。 | 仅在对应 work 完成时前移。 |

此外，`AppendUpdateGateState.append_epoch` 记为 $E_A$，表示当前 Append 上下文版本。严格
`hold_until_append_finished` 下，最终 Update work 绑定最终 $E_A$；近似早放行模式中，work 则绑定
它实际开始时可见的 $E_A$。


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

## 建议的数据模型

长期 Request 中保留一个 Update 窗口状态，以及一个轻量 Append gate 状态。前者保存所有 Update
合并统计；后者只保存 source 完成状态和 Append epoch，绝不保存第二份 Update token 快照。

```python
class AppendGatePhase(enum.Enum):
    # Append 尚未结束；Update 窗口照常累积，但 work admission 被拦截。
    WAITING_FOR_APPEND = "waiting_for_append"
    # Append 已结束；允许严格路径的 Update work admission。
    APPEND_FINISHED = "append_finished"
    # Update 先结束触发的近似早放行；允许当前窗口 admission。
    EARLY_RELEASE = "early_release"


@dataclass
class AppendUpdateGateState:
    phase: AppendGatePhase = AppendGatePhase.WAITING_FOR_APPEND
    append_source_finished: bool = False
    update_source_finished: bool = False

    # 每次 Append 真实增长递增；只保存上下文版本，不保存 Update 快照。
    append_epoch: int = 0
    final_append_epoch: int | None = None


@dataclass
class UpdateCoalescingState:
    # 唯一可被后续 Update 覆盖、尚未进入 pending_work_items 的完整快照。
    latest_pending_token_ids: tuple[int, ...] | None = None

    # 已 flush 的逻辑执行链尾部；新 work 在 flush 时以它作为 execution base。
    last_admitted_token_ids: tuple[int, ...] = ()

    # 窗口与统计从第一条 Update 起连续维护，与 gate 是否打开无关。
    first_pending_time: float | None = None
    last_update_time: float | None = None
    soft_deadline: float | None = None
    hard_deadline: float | None = None
    flush_ready: bool = False
    generation: int = 0

    pending_event_count: int = 0
    hypothetical_eager_tokens: int = 0
    interval_ewma: float | None = None
    merge_gain_ewma: float | None = None
```

约束如下：

- 所有合法 Update 都先更新 `PromptSourceState.token_ids`，再覆盖
  `latest_pending_token_ids`，并更新到达统计和 EWMA。
- 一个未 flush 窗口期间，`last_admitted_token_ids` 不前移；flush 时读取它并冻结进 work 的
  `execution_base_token_ids`。
- deadline 到期但 gate 不允许 admission 时，仅令 `flush_ready=True`，不创建 work；Append source
  结束后立即重新判断该标记。
- `flush_ready` 是**黏性状态**：后续 Update 可以继续覆盖
  `latest_pending_token_ids` 并更新统计，但不得把 `flush_ready` 改回 `False`；只有成功 flush、
  abort 或 preemption recovery 才能清除它。
- flush 后清空 `latest_pending_token_ids`、窗口 deadline、`flush_ready` 与窗口统计；EWMA 可跨窗口
  保留。
- `update_committed_token_ids` 只在 work 完成后更新，表示物理 UPDATE KV 的真实内容。

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

其中 $E_i$、$E_j$ 是计算该 Update work 时绑定的 Append epoch。严格 Append gate 让所有
最终 Update work 都绑定最终 epoch $E_f$，于是第二个条件退化回当前已经实现的
source-local LCP 规则。

反过来，若 Update source 已结束但 Append 后来继续增长，则 $E_i\ne E_f$。在**严格 gate**
中，这表示旧 UPDATE KV 不能作为最终上下文的严格复用结果；严格路径因此等待 $E_f$。在
`none` 与近似早放行模式中，系统有意保留 $E_i$ 的结果，把它解释为“Update 按执行时已见
Append 前缀计算”的近似，而不在后续 Append 到达时回滚或重建。


## 统一 Update 窗口：观测、合并与 admission

Append gate 不再把 Update 分为两套 buffer。一个合法 Update 从到达到执行分为三步：

| 步骤 | 快照 / 状态 | 用途 | 是否创建 work |
| --- | --- | --- | --- |
| 到达与合并 | 相邻原始快照 $P,N$，并覆盖 $U_p$ | 更新假想 eager 成本、覆盖收益、EWMA 和 deadline。 | 否。 |
| admission 判定 | $U_p$、deadline / source-finished 状态、Append gate | 判断窗口是否应立即 flush，或仅设为 `flush_ready`。 | 仅 gate 允许时。 |
| flush 执行 | 当前 $U_e$ 与 $U_p$ | 创建 work，work 冻结 `execution_base_token_ids=U_e`。 | 是。 |

### 到达与合并统计

假设新快照 $N$ 到达，上一原始事件快照为 $P$。不论 Append 是否结束，都更新：

$$
\mathrm{hypothetical\_eager\_tokens}\mathrel{+}=R(P,N).
$$

并以覆盖收益

$$
G_i=
\max\left(0,
R(P,N)-R(U_e,N)
\right)
$$

更新窗口统计、间隔 EWMA 与收益 EWMA。这里 $U_e$ 是当前已承诺执行链的尾部；在一个未 flush
窗口内它保持稳定，因此无需额外保存 $U_b$。

### Flush 与 admission 判定

当 soft deadline、hard deadline、TTFT slack、source-finished 或 `PROMPT_FINISHED` 触发时，先判断：

```text
should_flush_by_policy = 当前 policy 的正常 flush 条件
admission_allowed =
    append_source_finished
    or append_gate == "none"
    or (
        append_gate == "hold_until_append_or_update_finished"
        and update_source_finished
    )
```

- `should_flush_by_policy=False`：继续等待后续 Update。
- `should_flush_by_policy=True` 且 `admission_allowed=True`：立刻创建 work。
- `should_flush_by_policy=True` 且 `admission_allowed=False`：保留 $U_p$ 和全部统计，设置
  `flush_ready=True`；不触碰 UPDATE KV，也不重复创建 deadline work。

Append source finished 后，controller 立即再次检查 `flush_ready`、hard deadline、TTFT slack 与
当前 policy；若已满足，直接 flush 原有窗口，而不是重新创建一个从零计时的窗口。

### Flush 时的 LCP

flush 时读取：

$$
L_{\mathrm{execute}}=L(U_e,U_p),
\qquad
R_{\mathrm{execute}}=R(U_e,U_p).
$$

新 work 冻结完整 $U_p$、`execution_base_token_ids=U_e`、该 LCP 和所需 Append epoch。随后立即令
$U_e\leftarrow U_p$，清空 $U_p$ 与当前窗口状态。

## Update 策略与 Append admission gate

所有策略都从窗口第一条 Update 开始维护统计和 deadline；Append gate 只决定策略希望 flush 时能否
真的创建 work。`flush_ready=True` 表示策略已经要求执行、但严格 Append gate 尚未允许 admission。

| 策略 | Update 到达时 | 正常 flush 条件 | gate 未开但条件满足时 |
| --- | --- | --- | --- |
| `immediate` | 更新 $U_p$。 | 每次到达。 | 保留最新 $U_p$ 并置 `flush_ready`。 |
| `fixed_window` | 覆盖 $U_p$，不刷新 soft deadline。 | 窗口首事件加 `soft_window_ms`。 | 保留状态，Append 结束后立即复查。 |
| `quiet_debounce` | 覆盖 $U_p$，刷新 `last_update_time`。 | 最近事件后静默 `soft_window_ms`。 | 同上。 |
| `adaptive` | 覆盖 $U_p$，更新 EWMA 与 $w_i^\star$。 | 自适应 soft deadline、hard deadline 或 slack。 | 同上。 |
| `full_merge` | 始终覆盖 $U_p$。 | Update source finished 或合法 `PROMPT_FINISHED`。 | 严格路径继续等待 Append；近似早放行可在 Update finished 时 admission。 |

`quiet_debounce` 每次有效 Update 都刷新 `last_update_time` 和 soft deadline，但从不刷新
`first_pending_time` 或 hard deadline。Append gate 的开闭不重置这些时间。

### 到达率与收益 EWMA

EWMA 是指数加权移动平均：

$$
\bar{x}_i=(1-\alpha)\bar{x}_{i-1}+\alpha x_i,
\qquad 0<\alpha\leq1.
$$

对相邻 Update 到达间隔：

$$
\delta_i=t_i-t_{i-1},
$$

$$
\bar{\delta}_i
=
(1-\alpha)\bar{\delta}_{i-1}+\alpha\delta_i,
\qquad
\hat{\lambda}_i
=
\frac{1}{\bar{\delta}_i+\epsilon}.
$$

$\hat{\lambda}_i$ 估计下一事件到达率。第一版可使用指数到达近似：

$$
F_i(w)=1-\exp(-\hat{\lambda}_i w),
$$

即未来 $w$ 时间内再到达一个 Update 的预测概率。后续可以以经验 CDF 替换。

合并收益用：

$$
\hat{G}_i
=
(1-\gamma)\hat{G}_{i-1}+\gamma G_i.
$$

冷启动时，可将请求级统计与全局先验混合：

$$
\hat{x}
=
\frac{n}{n+\kappa}\hat{x}_{\mathrm{request}}
+
\frac{\kappa}{n+\kappa}\hat{x}_{\mathrm{prior}},
\qquad
x\in\{G,\lambda\}.
$$

### 价值、代价和软窗口

令 $Q_P(t)$ 为 P 节点可运行 Prefill 请求数或归一化等待工作量。资源价格可取：

$$
\mu(t)=1+\eta Q_P(t).
$$

等待 $w$ 的预计收益：

$$
\mathcal{B}_i(w)
=
\mu(t)\hat{G}_i
\left(1-\exp(-\hat{\lambda}_i w)\right).
$$

等待的时延代价：

$$
\mathcal{D}_i(w)=\beta_i w.
$$

$\beta_i$ 是请求每单位等待的延迟价格；越接近 TTFT deadline，$\beta_i$ 越大。

最优软等待时间：

$$
w_i^\star
=
\arg\max_{0\leq w\leq W_i^{\max}}
\left[
\mu(t)\hat{G}_i
\left(1-\exp(-\hat{\lambda}_i w)\right)
-\beta_i w
\right].
$$

在 $w=0$ 的边际放行阈值为：

$$
\mu(t)\hat{G}_i\hat{\lambda}_i>\beta_i.
$$

不满足阈值、没有足够冷启动统计且不采用先验、或无剩余 TTFT slack 时，取 $w_i^\star=0$ 并立即 flush。指数到达近似下：

$$
w_i^\star
=
\left[
\frac{1}{\hat{\lambda}_i}
\ln\left(
\frac{\mu(t)\hat{G}_i\hat{\lambda}_i}{\beta_i}
\right)
\right]_0^{W_i^{\max}},
$$

其中 $[x]_0^W=\min(\max(x,0),W)$。

## 有界等待与被 gate 阻挡的 flush

`fixed_window`、`quiet_debounce` 与 `adaptive` 的 soft / hard deadline 从**窗口第一条 Update**
开始计算，不因 Append gate 关闭而暂停或重置。`immediate` 没有 timer；`full_merge` 只在 Update
source finished 或合法 `PROMPT_FINISHED` 时要求 flush。

| 状态 | 到达 deadline 时的动作 |
| --- | --- |
| `admission_allowed=True` | 立刻 flush 当前 $U_p$。 |
| `admission_allowed=False` | 设置 `flush_ready=True`，保留 $U_p$、deadline、EWMA 与统计；从 deadline heap 移除该条目，避免重复空唤醒。 |
| Append source finished | 更新 gate，然后立即复查 `flush_ready`、当前 deadline 与 policy；条件满足则直接 flush 原窗口。 |
| Update source finished + 近似早放行 gate | 更新 gate，然后立即复查并按当前可见 Append epoch admission。 |

标准 P/D 路径的 TTFT 剩余 slack 为：

$$
S_i
=
D_{\mathrm{TTFT}}
-
(t_{\mathrm{now}}-t_{\mathrm{request\_start}})
-
\hat{T}_{\mathrm{P,remain}}
-
\hat{T}_{\mathrm{transfer}}
-
\hat{T}_{\mathrm{D1}}.
$$

当 $S_i\leq0$ 时，policy 已经要求立即 flush；若严格 Append gate 仍关闭，只能记录
`flush_ready=True`，不能违反 gate 直接物化 work。Append 结束后，这份已过期的窗口必须优先
admission。这样 TTFT slack 仍被完整统计，且其等待来源会被准确记为“Append gate blocking”。

### 具体例子：窗口先到期、随后 Append 才结束

设第一条 Update 在 $t=0$ 到达，Append 仍 open，初始执行链尾为 $U_e=U_0$：

| 时刻 | 事件 | Update 窗口状态 | admission |
| --- | --- | --- | --- |
| $0\,\mathrm{ms}$ | $U_1$ 到达 | $U_p=U_1$；记录 first time、EWMA 和 deadline。 | gate 关闭，不建 work。 |
| $18\,\mathrm{ms}$ | $U_2$ 到达 | $U_p=U_2$；正常更新 LCP 统计。 | gate 关闭。 |
| $40\,\mathrm{ms}$ | fixed soft deadline 到期 | $U_p=U_2$ 保留，设 `flush_ready=True`。 | 仍不建 work。 |
| $55\,\mathrm{ms}$ | $U_3$ 到达 | $U_p=U_3$；继续更新 EWMA / 覆盖收益；hard deadline 不刷新。 | gate 关闭。 |
| $70\,\mathrm{ms}$ | Append source finished | 复查既已到期的窗口。 | 立即创建 `work(U_0 -> U_3)`。 |

这里不会把 $70\,\mathrm{ms}$ 当作新的窗口起点：$U_1\rightarrow U_3$ 的到达过程、统计与
等待都已经发生。Append gate 只解释为什么窗口到期后尚未物化 work。

## 两阶段状态机与事件顺序

这里的“两阶段”只指 **Update 窗口策略** 与 **Append admission** 两个判定，而不是两套 Update
buffer。

### 合法 Update 事件到达

1. `Request` 先完成 version、sequence、去重和乱序校验，更新 $U_o$。
2. 更新唯一的 $U_p$：首次到达初始化 `first_pending_time`、`soft_deadline`、`hard_deadline`；后续
   到达覆盖 $U_p$ 并按 policy 更新 soft deadline。
3. 无论 gate 是否打开，都计算相邻 Update 的统计、更新 EWMA 与 `hypothetical_eager_tokens`。
4. 按 policy 判断是否应 flush；若 gate 未允许，设置 `flush_ready=True` 而非建 work。

### Append 与 Update source finished

- **Append event**：照常单调追加、递增 `append_epoch`，不改写 $U_p$、EWMA 或 deadline。
- **Append source finished**：验证前序 sequence，记录 `final_append_epoch`，打开严格 admission；立即
  复查当前窗口。若 `flush_ready=True`、deadline 已过、Update source 已结束或 policy 立即执行，直接
  flush 原窗口；否则继续沿用原有 deadline。
- **Update source finished**：标记 source sealed。严格 `hold_until_append_finished` 下，窗口继续存在，
  但若 policy 已要求 flush 仅记 `flush_ready`；近似
  `hold_until_append_or_update_finished` 下，此事件同时打开 admission，按当前可见 Append epoch flush。

### Flush 与 Question barrier

flush 时读取 `base = last_admitted_token_ids` 与 `target = latest_pending_token_ids`，创建冻结的
`work(base -> target)`，并将 `base` 写入 `work.execution_base_token_ids`。随后令
`last_admitted_token_ids = target`，清空窗口状态。

`PROMPT_FINISHED` 仍要求两个 source 已按协议 sealed。它强制 policy 进入“应 flush”状态；若严格
Append gate 尚未打开，则请求协议有问题，不能以 finish 穿透 gate。

`finished=True` 只表示最终用户问题已经到达，**不表示 question 已经可调度**。第一版应将 question
建成显式的 `QUESTION` work，并按以下步骤处理：

1. `PROMPT_FINISHED` 先保存 suffix，标记 `finished=True`，并强制最终 Update 窗口进入应 flush 状态。
2. controller 在最终 Update 已 admission 后，调用 `enqueue_question_if_ready()`；它要求两个 source
   sealed、没有 `$U_p$` 或 `flush_ready`，且所有最终 Update work 已在 FIFO 中排定。
3. 该函数只追加一个 `QUESTION` work 到 `pending_work_items` 尾部，并用
   `question_work_enqueued` 防止重复入队。FIFO 保证它不会越过已放行但尚未完成的 Append / Update
   work；严格 Update work 仍先等待 `required_append_epoch` 的物理提交。

因此，question 的真实执行仍由普通 scheduler 决定；它只是不能再依靠 `finished=True` 和
`pending_work_items` 暂时为空这一隐式条件直接开始。

### 严格与近似 admission

- `append_gate=none`：每次 policy 想 flush 时均可 admission；Update 使用执行时 Append 前缀。
- `hold_until_append_finished`：只有 Append source finished 后才可 admission；严格路径的最终 Update
  work 绑定 $E_f$。
- `hold_until_append_or_update_finished`：Append 或 Update source 任一结束都可 admission。Update 先
  结束时，这是合并后的近似早放行：work 绑定当前 $E_s$，后续 Append 不回滚它。

## 启动级功能开关与接入路径

### 目标：默认路径零语义变化

Update 合并是本项目的实验功能，而不是重新定义 vLLM 的通用 scheduler。因此必须满足：

- **默认关闭**：普通 `/v1/completions`、普通 streaming input，以及已有 PromptStream 的
  “每个 Update 事件立即生成 work”语义都保持不变。
- **仅启动 P 节点时开启**：D 节点不接收 PromptStream Update，实验中的 D 节点不创建
  coalescer。普通服务也不传该启动参数。
- **API 协议不变**：`/v1/prompt_stream/start`、`event`、`finish` 的 payload 不增加
  合并控制字段；这是 engine 启动级策略，而不是由单个客户端请求临时改变的策略。
- **启用时只拦截 Update**：Append、最终 question、普通请求和已 flush 的 work 继续走
  当前路径。合并层只在 Update 快照尚未放行的短窗口内生效。

因此，“关闭时完全没有额外语义”比“源码中完全没有一个 if”更重要。为满足 deadline，
EngineCore 必须拥有一个可选唤醒 hook；但当开关关闭时，该 hook 必须走现有的无限期
`input_queue.get(block=...)` 分支，不能改变普通请求的调度或等待行为。

### 配置对象与命令行路径

新增独立配置对象，避免把每个研究参数平铺到通用 Scheduler 字段中。它采用与
`--kv-transfer-config` 相同的 JSON dict 命令行形式：**不传该参数即关闭；传入合法 dict 即启用**。
因此不再额外设置冗余的 `--enable-*` 布尔开关。

~~~python
@config
class PromptStreamUpdateCoalescingConfig:
    # 第一轴：Append 未封闭时是否允许把已满足策略条件的窗口 admission 为 work。
    append_gate: Literal[
        "none",
        "hold_until_append_finished",
        "hold_until_append_or_update_finished",
    ] = "hold_until_append_finished"

    # 第二轴：所有 Update 都持续运行的窗口策略；gate 只阻止 admission。
    update_policy: Literal[
        "immediate",
        "fixed_window",
        "quiet_debounce",
        "adaptive",
        "full_merge",
    ] = "fixed_window"

    # 仅 fixed_window / quiet_debounce / adaptive 使用。
    soft_window_ms: float = 20.0
    hard_deadline_ms: float = 80.0
    ttft_deadline_ms: float | None = None

    # 仅 update_policy="adaptive" 使用。
    interval_ewma_alpha: float = 0.25
    gain_ewma_alpha: float = 0.25
    cold_start_prior_weight: float = 8.0
    arrival_rate_prior_per_s: float | None = None
    merge_gain_prior_tokens: float | None = None
    delay_price: float = 1.0
    load_price_scale: float = 0.0
~~~

| 字段 | 取值 / 使用范围 | 含义 |
| --- | --- | --- |
| `append_gate` | `none`、`hold_until_append_finished`、`hold_until_append_or_update_finished` | 决定 Update 是否先被 Append completion gate 阻挡。第三项是近似早放行模式，使用执行时可见的 Append 前缀。 |
| `update_policy` | `immediate`、`fixed_window`、`quiet_debounce`、`adaptive`、`full_merge` | 所有 Update 持续采用的窗口 / flush 规则；gate 只决定此刻能否 admission。 |
| `soft_window_ms` | fixed、debounce、adaptive | fixed window 长度、debounce 静默时长，以及 adaptive 的配置最大候选等待；单位 ms。 |
| `hard_deadline_ms` | fixed、debounce、adaptive | Update 窗口的最大等待，单位 ms；gate 关闭时到期只置 `flush_ready`。 |
| `ttft_deadline_ms` | fixed、debounce、adaptive | 从 Request start 计的总 TTFT 预算；无论 gate 是否打开均参与 policy 判断，gate 阻挡时结果记为 `flush_ready`。 |
| `interval_ewma_alpha`、`gain_ewma_alpha` | adaptive | 两个 EWMA 的更新系数。 |
| `cold_start_prior_weight`、`arrival_rate_prior_per_s`、`merge_gain_prior_tokens` | adaptive | 冷启动先验。 |
| `delay_price`、`load_price_scale` | adaptive | 等待的基础时延价格与 P 节点负载价格系数。 |

`append_gate=none` 在 controller 构造时直接视为 admission 已允许；它不需要特殊 Update buffer，
只让每次 policy 要求的 flush 立刻创建 work。

`epsilon` 仅用于避免到达率计算除零，应在实现内部固定，不放入实验配置。
`soft_deadline`、`hard_deadline`、$\hat{\lambda}$、$\hat{G}$、$w^\star$、
`append_epoch` 和 gate 状态都是每个请求运行时计算出的状态，不放入 JSON。

建议的文件和传递路径为：

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

vllm/v1/core/sched/scheduler.py
    仅当 config is not None 时创建 Append-aware controller
~~~

严格 Append gate + 固定窗口的主方案示例：

~~~shell
--prompt-stream-update-coalescing-config \
  '{
    "append_gate":"hold_until_append_finished",
    "update_policy":"fixed_window",
    "soft_window_ms":20,
    "hard_deadline_ms":80
  }'
~~~

严格 Append gate + 完全合并基线（joint-finish merge）的示例：

~~~shell
--prompt-stream-update-coalescing-config \
  '{
    "append_gate":"hold_until_append_finished",
    "update_policy":"full_merge"
  }'
~~~

无 gate 的纯 Update 完全合并消融基线：

~~~shell
--prompt-stream-update-coalescing-config \
  '{
    "append_gate":"none",
    "update_policy":"full_merge"
  }'
~~~

Adaptive 的完整示例如下。这里的先验值只是一组起始实验配置，最终应写入实验记录，不应
当作跨模型、跨 workload 的通用默认值：

~~~shell
--prompt-stream-update-coalescing-config \
  '{
    "append_gate":"hold_until_append_finished",
    "update_policy":"adaptive",
    "hard_deadline_ms":80,
    "ttft_deadline_ms":400,
    "interval_ewma_alpha":0.25,
    "gain_ewma_alpha":0.25,
    "cold_start_prior_weight":8.0,
    "arrival_rate_prior_per_s":10.0,
    "merge_gain_prior_tokens":128.0,
    "delay_price":1.0,
    "load_price_scale":0.0
  }'
~~~


参数未传时，`EngineArgs.prompt_stream_update_coalescing_config` 与
`SchedulerConfig.prompt_stream_update_coalescing_config` 都是 `None`。这比创建 no-op
controller 更清楚：Scheduler、EngineCore 与 Request 都通过一次 `is None` 判定回到原始
路径，也不会分配 deadline heap、dormant 集合或请求级合并状态。

### 启动拓扑与第一版兼容性边界

| 进程角色 | 是否启用 | 行为 |
| --- | --- | --- |
| 普通 vLLM server | 否 | 当前 scheduler 与普通 REST 行为完全不变。 |
| P 节点 | 是 | 接收 Update 事件，并在本地合并、Prefill 和保留 P 侧 KV。 |
| D 节点 | 否 | 只接收 P 传来的最终 KV 与首 token，不维护 Update 合并窗口。 |

第一版启动校验应明确拒绝尚未实现的组合，而不是静默改变行为：

- 仅支持一个 Update source 和一个 Append source；多 Append source 需要先定义“全部封闭”的
  source barrier。
- V0 支持 `append_gate=none`、严格 `hold_until_append_finished` 与近似
  `hold_until_append_or_update_finished`；第三项只需记录执行时 Append epoch，不要求失效 / 重建机制。
- 仅支持常规单 EngineCore P 路径；暂不支持 data-parallel EngineCore 的独立 busy loop。
- 启用时要求 `async_scheduling=False`。异步调度的并行 batch 生命周期需要单独验证后再
  放开。

这些限制只在传入 coalescing JSON config 时校验；参数未传时不得改变现有配置组合的合法性。

### 运行时对象与事件接入边界

启用后，Scheduler 创建一个组合式 controller，而不是把合并判断散落到通用队列代码中：

~~~python
self.prompt_stream_update_coalescer = (
    PromptStreamUpdateCoalescer(config)
    if config is not None
    else None
)
~~~

建议的职责划分如下：

| 位置 | 关闭时 | 启用时 |
| --- | --- | --- |
| `Request` / `PromptSourceState` | 当前事件校验、tokenize 后快照更新、立即创建 Update work。 | 仍是 version、sequence、去重与 source 状态的唯一权威；持有 `AppendUpdateGateState` 和 `UpdateCoalescingState`。 |
| `PromptStreamUpdateCoalescer` | 不存在。 | 不复制 token 快照；只维护 controller 的 dormant id、deadline heap 和 gate / window 索引，并通过 Request 的窄接口移动状态。 |
| `Scheduler._apply_prompt_stream_request()` | 保持当前即时语义。 | 对 Update 调 `on_update_snapshot(...)`，对 Append 记录 epoch，对 source-finished 调 `on_source_finished(...)`，对 `PROMPT_FINISHED` 调 `force_final_flush(...)`；Append work completion 则调 `on_append_epoch_committed(...)`。 |
| `Scheduler.schedule()` | 当前 work 队列调度。 | 只消费已冻结 work；选择 Update work 前还检查 `required_append_epoch` 是否已物理提交。 |

这要求把 Request 内“更新 source 快照”和“创建 Update work”拆成两个内部动作，但不改变
外部 REST 事件或 `EngineCoreRequest` 结构。controller 不能自行绕过 Request 更新 token，
否则会复制 version、sequence number、乱序和去重逻辑。

### dormant 请求的局部化

controller 只能在请求当前**没有任何 runnable work**时把它标为 dormant。Append-aware 设计中至少
存在三类 dormant 原因：

| dormant 原因 | 条件 | 是否登记 deadline | 唤醒方式 |
| --- | --- | --- | --- |
| `append_gate_blocked` | Update window 已要求 flush，但 Append 尚未结束，且没有其他 work。 | 否。 | 等 Append source-finished、abort 或合法 finish。 |
| `admission_deadline` | admission 已允许、有界 Update window 尚未到期，且当前没有可运行 work。 | 是。 | EngineCore 等到最近 deadline 或接收新输入。 |
| `full_merge` | policy 为 `full_merge`，只保存最新 Update。 | 否。 | 等 Update source-finished、Append source-finished、finish 或 abort 输入。 |

因此，gate 阻挡 **不等于**整个 Request 都 dormant：只要还有 PREFIX_APPEND work，request 仍是
runnable，Append 会照常占用 scheduler。只有所有 runnable work 都耗尽、controller 又无法 admission
当前 Update 窗口时，才从普通 `waiting` / `skipped_waiting` / `running` 集合中移出。

1. Request 仍保留在 `Scheduler.requests[request_id]`，因此后续 event、finish、abort 能按
   request id 找到同一个长期 Request。
2. controller 记录 `dormant_request_ids[request_id] = reason`；它不复制 Request 的 token
   或 KV 状态。
3. `wake_dormant(request_id)` 是唯一的恢复入口：删除 dormant 标记后，若已经有冻结 work，或
   deadline 到期使 controller 新建了 work，则用现有 `_enqueue_waiting_request()` 放回普通队列。
   `append_gate_blocked` 保留已到期状态但不登记 timer，收到 source 终止事件后才重新判断。
4. abort、engine error 与正常完成必须执行 `remove_dormant(request_id)`，并同时清理 deadline heap
   的逻辑所有权；堆中的遗留条目仍可通过 generation / request-id 校验惰性丢弃。

不能让 dormant id 参与普通未完成请求计数或 connector cleanup 计数。启用 controller 时，
Scheduler 需提供只给 EngineCore 用的 runnable 判定，例如：

~~~python
def has_runnable_engine_work(self) -> bool:
    if self.prompt_stream_update_coalescer is None:
        return self.has_requests()
    return self._has_requests_excluding_coalescing_dormant()
~~~

后者必须保留真正的 `waiting` / `running` 请求、已完成请求的 connector 后处理以及
connector pending push work。它只排除 controller 纯粹等待外部 source 或 admission deadline 的
request id，不能把 dormant Request 误判成 finished connector cleanup。

### EngineCore 的最小 gated hook

EngineCore 不应了解 Update 快照、LCP 或 EWMA，只识别“当前是否存在一个可选的 idle
wakeup provider”。建议 controller 经 Scheduler 暴露三个窄接口：

~~~python
def has_prompt_stream_coalescer(self) -> bool: ...
def get_next_prompt_stream_deadline(self) -> float | None: ...
def flush_due_update_coalescing(self, now: float) -> int: ...
~~~

其中 `get_next_prompt_stream_deadline()` 只返回 **admission 已允许、且尚未到期** 的有界 Update
窗口的最近 deadline。gate 关闭而已 `flush_ready` 的窗口，以及 `full_merge` 等待期都返回 `None`；
它们应等待下一条外部 source 事件，而不应反复唤醒 EngineCore。

`flush_due_update_coalescing()` 只处理已经到期且 admission 已允许的有界 Update window。Append /
Update source-finished 与 `PROMPT_FINISHED` 是输入事件，它们会更新 admission 条件并重新判定
`flush_ready`；最后一个 Append work 的物理提交则是 scheduler / worker completion callback，它通过
`on_append_epoch_committed()` **解除既有 work 的执行 barrier**，而不是把 completion 伪装成 deadline
或重新决定 window policy。

对 `EngineCore` 的改动应限于以下三处，并且关闭时逐字保留旧分支的关键行为：

| 调用点 | `enabled=False` | `enabled=True` |
| --- | --- | --- |
| `has_work()` / `step()` / 无模型执行时的 sleep 判定 | 使用当前 `scheduler.has_requests()`。 | 使用 `scheduler.has_runnable_engine_work()`，排除 dormant id。 |
| `_process_input_queue()` | 保持 `input_queue.get(block=block)`。 | 若没有 runnable work，使用最近 deadline 计算 `Queue.get(..., timeout=...)`。 |
| 输入处理后、调用 `step()` 前 | 不调用合并逻辑。 | 在 EngineCore 主线程调用 `flush_due_update_coalescing(time.monotonic())`。 |

关闭时不应为了“统一接口”而每轮调用 no-op `flush`，也不应创建 timer thread。只有 controller
存在时才查询 deadline；普通 server 的调度循环、token budget、KV 管理和 REST 行为均保持当前
路径。

### 完整的 enabled / disabled 数据流

~~~text
关闭：Update event
    → Request 更新 Update source
    → 立即创建 Update work
    → 现有 Scheduler / Worker 路径

开启：Update event
    → Request 校验并更新 Update source
    → 始终覆盖 U_p、更新 EWMA 与 deadline
    → policy 要求 flush 时检查 Append admission
    → 不允许：保留 U_p 并置 flush_ready
    → 允许：创建绑定 required_append_epoch 的冻结 Update work
    → 现有 Scheduler / Worker 路径
~~~

Append token 仍走现有 `PREFIX_APPEND` work 路径，但每次真实增长都递增 `append_epoch`。Append
source-finished 在逻辑封闭时立刻打开严格 admission，并复查已经存在的 $U_p$ 与 `flush_ready`；它
不会重置 fixed window / debounce 的计时。严格模式的 work 绑定 `required_append_epoch=E_f`，在最后
一个 Append work 物理提交前保持不可 active。completion callback 只解除这个 barrier。`PROMPT_FINISHED` 不是简单的 `force_flush`：它还必须满足 Append 与 Update 都
逻辑封闭、最终 Append epoch 已物理提交，之后 question barrier 才能等待最后一份已放行 Update
work。普通请求不进入上述任何一步。

### 关闭开关的回归保证

实现完成后必须覆盖：

- 不传开关时，现有普通 scheduler 测试与 PromptStream 即时 Update 测试的结果不变。
- 关闭时不创建 `PromptStreamUpdateCoalescer`、deadline heap、dormant 集合或
  `UpdateCoalescingState`。
- 开启但没有 Update 事件时，Append-only 流式 Prefill 的行为与当前版本一致。
- 开启后只有 Update source 的 work 数、等待时间和 Update KV 截断时机发生变化；最终
  prompt token 序列和 D 节点接收的 KV 排布必须与同一 Append gate 下的 `immediate` 对照一致。

## Scheduler、并发与唤醒

本章的 deadline heap、dormant 集合和 EngineCore timeout 改动只在
`prompt_stream_update_coalescing_config is not None` 时启用。关闭时完全绕过本章新增的
controller 路径，并保持已有 scheduler / EngineCore 行为。

### 不需要锁

FastAPI/router 不应直接修改长期 Request。外部 REST 事件进入 EngineCore 输入队列后，由同一个 EngineCore 主线程顺序执行：

~~~text
Scheduler.add_request
    → _apply_prompt_stream_request
    → Request.apply_prompt_stream_event
    → [仅启用 controller 时] Scheduler.flush_due_update_coalescing
    → Scheduler.schedule
~~~

PromptStreamState、pending_work_items、active_work_item 和 KV block table 都由该线程修改；开启
controller 时，AppendUpdateGateState、UpdateCoalescingState 与 deadline heap 也由同一线程修改。因此：

- 不需要把 pending_work_items 改成带锁队列。
- 不需要 asyncio.Lock 或 threading.Lock；它们不能跨 API 进程保护 EngineCore 状态。
- 定时器线程最多向 EngineCore 投递 wake-up 或 deadline hint，绝不能直接 flush Request 或修改快照。

### deadline 最小堆

启用 controller 后，PromptStreamUpdateCoalescer 应维护一个最小堆，条目只包含：

~~~text
(deadline, request_id, generation)
~~~

- 只给 admission 已允许且尚未到期的有界 window 入堆；gate 阻挡的 `flush_ready` 和 `full_merge` 不入堆。
- 每个新窗口和每次允许刷新的 soft deadline 刷新都递增 generation。
- 堆不保存 token；快照始终归 Request 的 UpdateCoalescingState 所有。
- 每轮调度前先处理已经到达的输入事件，再弹出所有已到期条目。
- generation 不匹配的条目是过期 deadline，直接跳过。
- 有效条目在同一 EngineCore 主线程中 flush。

这样，deadline 附近若新事件先到达，它会使旧条目失效；若 deadline 先被处理，则旧窗口先 flush，新事件进入下一窗口。两者都满足有界等待和不可变 work 语义。

### dormant 请求

开启 controller 后，某请求可能有尚未 flush 的 latest Update，却没有 active 或已放行 work。它不应持续留在 running 集合做空 GPU 调度。

- 已有 active 或已放行 Append/Update work 不受影响，继续正常调度。
- 若请求已经 `flush_ready` 但被 Append gate 阻挡，或仅等待 `full_merge` 的 source 终止事件，则标为
  无 deadline 的 dormant。
- 若 admission 已允许且仅等待有界 deadline，则标为带 deadline 的 dormant；EngineCore 在最近
  deadline 到达时被唤醒，flush 后再把请求加入普通等待队列。

第一版应采用下文的 **deadline-aware 阻塞等待**，不能每毫秒空 `schedule()` 检查时间；
后者会引入 CPU 空转并污染实验。

### EngineCore 的 gated deadline-aware 阻塞等待

这是后续实现需要修改 `vllm/v1/engine/core.py` 的位置；当前代码尚未具备该能力。

#### 当前 EngineCore 为什么会无限等输入

当前 `EngineCoreProc.run_busy_loop()` 的主循环是：

~~~python
while self._handle_shutdown():
    self._process_input_queue()
    self._process_engine_step()
~~~

其中 `EngineCore.has_work()` 把 `scheduler.has_requests()` 视为有工作。若没有可运行
请求，`_process_input_queue()` 会执行 `input_queue.get(block=True)`，无限期阻塞，直到
IO 线程从 REST/ZMQ 收到新的 ADD、ABORT 或其他控制消息并写入 `input_queue`。

这对普通请求是正确的，但对**admission 已允许的有界 Update 窗口**不够：一个 dormant 请求可能拥有
`latest_pending_token_ids`，却没有 active work 或已放行 work。若没有新的 Update、
`PROMPT_FINISHED` 或 abort 到达，现有 EngineCore 不知道它的 soft/hard deadline，因而
永远不会自行 flush。相反，gate 阻挡的 `flush_ready` 与 `full_merge` 不应因 timer 反复唤醒，继续
等待输入恰好是正确行为。

反过来，不能让这种 dormant 请求继续被当作普通 scheduler work。否则
`scheduler.has_requests()` 始终为真，EngineCore 会不断进入 `step()` 和
`scheduler.schedule()`；没有 GPU work 时当前路径只短暂 sleep，形成低效忙转，而不是
真正等待 deadline。

#### Scheduler 与 EngineCore 的职责边界

Scheduler 在 controller 存在时持有它；deadline 最小堆和 dormant 集合归 controller 所有。Scheduler 仅转发两个由 EngineCore 主线程调用的接口；以下名称是建议的实现接口：

~~~python
def get_next_prompt_stream_deadline(self) -> float | None:
    """返回有效 coalescing deadline 的最早绝对 monotonic 时间。"""

def flush_due_update_coalescing(self, now: float) -> int:
    """放行所有到期窗口，并将因此可运行的请求重新加入等待队列。"""
~~~

- dormant 请求不应被当作可运行 GPU work。实现时建议新增
  `Scheduler.has_runnable_engine_work()`，并让 `EngineCore.has_work()`、`step()` 中
  “是否进入 `scheduler.schedule()`”的早退判断，以及“无模型执行时短暂 sleep”的判断
  都使用 runnable 语义；不能只改 `has_work()`，否则 timeout 返回后仍可能进入空转路径。
  `has_requests()` 若还承担请求生命周期统计，可保留其原语义。
- dormant 集合不混入普通 `waiting` / `running` 队列；请求对象仍可保留在按
  request id 索引的长期状态表中，以接收后续 Update、finish 或 abort。
- `get_next_prompt_stream_deadline()` 只读最小堆，跳过 generation 失效、请求已取消、策略已变为
  `full_merge` 或已无 pending Update 的陈旧条目。
- `flush_due_update_coalescing()` 是唯一将 `latest_pending_token_ids` 物化为不可变
  `pending_work_items` 的入口。它完成后，请求才重新成为普通 runnable request。

#### 主循环的目标顺序

第一版不创建定时器线程。只把 EngineCore 空闲时原本无限期的 queue wait 改为“最多等待
到最近 deadline”：

~~~python
while self._handle_shutdown():
    # 处理已经到达的 REST/ZMQ 输入；可能创建、刷新或取消 deadline。
    self._process_input_queue()

    if self.scheduler.has_prompt_stream_coalescer():
        # 同一主线程检查到期窗口；必要时把 dormant 请求转回 runnable 队列。
        self.scheduler.flush_due_update_coalescing(time.monotonic())

    # 只调度已经有实际 GPU work 的普通请求。
    self._process_engine_step()
~~~

`_process_input_queue()` 在 `not self.has_work()` 时的等待逻辑改为概念上的：

~~~python
if not self.scheduler.has_prompt_stream_coalescer():
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
Update 放行；若没有有效 admission deadline，`timeout=None`，仍是原有的无限等待。关闭时不会出现
这个 timeout 路径。因此“EngineCore 因 deadline 醒来”并不是新增一个线程或主动中断 GPU；它只是
启用 controller 后，原有 EngineCore 主线程从带 timeout 的阻塞 `Queue.get()` 返回。

#### 输入事件与 deadline 同时发生

在同一 EngineCore 主线程上采用固定顺序：先处理已进入 `input_queue` 的事件，再检查
deadline。

- 新 Update 先被处理时，它可以刷新 soft deadline 并递增 generation；堆内旧条目在
  随后的检查中失效。
- queue wait 先 timeout 时，主线程先 flush 旧窗口；稍后到达的新 Update 进入下一窗口。
- 无论哪一种先发生，hard deadline 不会被刷新，且已放行 work 永远不可变，因此不需要
  锁来保护 `Request`、`pending_work_items` 或 KV block table。Append / Update source-finished
  事件与最后一个 Append work 的 completion callback 也在这个主线程串行化：只有
  `required_append_epoch` 已提交，Update work 才可变为 runnable。

`process_input_queue_block=False` 的既有非阻塞路径应保持原样；deadline 检查仍在每轮
主循环发生，但它不应向该特殊路径额外引入阻塞。

## flush 后的 KV 执行与 Append epoch barrier

本节定义 gate 之后，冻结 Update work 如何与既有分段 KV 路径衔接。合并器只决定**是否和何时**
创建 work；它不直接持有 block id，也不直接调用 worker。

### 冻结 work 的语义

严格模式下，每个已放行的 Update work 至少冻结以下逻辑字段：

| 字段 | 含义 | 写入时刻 |
| --- | --- | --- |
| `target_update_token_ids` | 本 work 对应的完整 Update 快照 $U_p$。 | flush。 |
| `execution_base_token_ids` | flush 时读取的 $U_e$；它仅在该 work 内冻结为执行前驱。 | flush。 |
| `update_lcp_len` | $L(U_e,U_p)$，只比较 Update token。 | flush。 |
| `required_append_epoch` | 本 work 所需的 `PREFIX_APPEND` 物理版本；严格 gate 下为最终 $E_f$。 | flush。 |
| `prefix_append_logical_len` | 该 epoch 下可见 Prefix+Append 的逻辑 token 长度。 | flush 或由 epoch 查表。 |

`required_append_epoch` 是 **scheduler dependency**，不是让 worker 临时拼接任意 block id 的捷径。
Request 的 `PromptStreamState` 仍是 block table 的唯一权威：它根据已提交的 prefix / update
segment、各段逻辑长度和既有 padding 规则，构造 worker 所需的 slot mapping / block table。

### 从放行到提交的严格顺序

```text
Append source finished
    → final_append_epoch = E_f，重新检查既有 Update 窗口是否可以冻结 work
    → 最后一个 PREFIX_APPEND work 计算并提交 E_f
    → Scheduler 观察 committed_append_epoch >= required_append_epoch
    → Update work 才可成为 active
    → 按 source-local LCP 截断 UPDATE segment 并计算新后缀
    → UPDATE KV 以 (U_p, E_f) 提交
    → question barrier 解除
```

1. **flush 只创建不可变 work。** 它不截断 UPDATE KV、不释放 block，也不把 work 直接塞给 worker。
2. **Scheduler 先检查 Append dependency。** 若 `committed_append_epoch < required_append_epoch`，该
   Update work 不能成为 active。即使同一 Request 的 FIFO 已让 Append work 排在它前面，这个检查
   仍应保留为防御性条件。
3. **dependency 满足后才计算执行 LCP。** 对严格 gate，UPDATE KV 当前 epoch 与
   `required_append_epoch` 相同，才可按 work 内冻结的
   $L(\text{execution\_base\_token\_ids},\text{target\_update\_token\_ids})$ 延后截断并复用
   Update 前缀；epoch 不同则必须视为 LCP 为零并重建，不能错误复用 token LCP。
4. **worker 的 attention 视图**是当前已提交的 `PREFIX_APPEND` 加上 Update LCP KV；新计算 token
   只写入独立 `UPDATE` segment。Append block 不因 Update 被截断或覆写。
5. **完成提交**后才令 `update_committed_token_ids = U_p`，同时记录
   `update_committed_append_epoch = required_append_epoch`。work 对象可回收，但 committed snapshot
   与对应 UPDATE KV 继续保留给下一代 Update 比较。

逻辑 token 与物理 block 边界必须继续分开看：`PREFIX_APPEND` 末 block 未满而 UPDATE segment 从
独立 block 起始时，现有分段实现可能含有 segment 间 padding。该 padding / attention-mask /
position 近似属于第一项分段 KV 设计的既有边界；本合并器不得通过改变 `update_lcp_len` 或伪造
Append token 来处理它。它只保证任何可复用 UPDATE KV 都绑定正确的逻辑 prefix epoch。

若目标 $U_p$ 是其 work 内执行基准 $U_e$ 的严格前缀，$R(U_e,U_p)=0$，仍要创建零 token
状态转换 work：它负责截断多余 UPDATE KV、更新 `(U_c,E_A)` 和解除后续依赖，不能简单丢弃。

### 严格与近似执行语义的边界

- `append_gate=none` 是无 gate 的近似 / 消融路径：Append 与 Update work 按实际到达与调度顺序
  竞争，Update 使用当时已提交的 Append 前缀。
- `hold_until_append_or_update_finished` 是**合并后早放行**的近似路径：Append 未结束期间仍只暂存和
  覆盖 Update；一旦 Update source 结束，就放行该最终合并快照，并使用当时可见的 Append epoch。
- `hold_until_append_finished` 是严格路径：所有 Update work 都等待最终 $E_f$，因此 Update KV 可被
  解释为基于完整 Append 前缀的结果。

三者的区别是计算时看见多少 Append，而不是“前两者正确、后一者才正确”。前两者应在论文中
作为近似语义的性能 / 质量权衡单独报告；严格路径则作为完整 Append 上下文的对照。

## 预抢占、取消、重试与协议异常

### 预抢占 / 物理 KV 丢失

preemption 释放的是物理 KV，不是逻辑 source 状态。正常新 Update 到达绝不删除已放行 work；但
preemption 使其物理前驱不存在时，允许清空受影响的 active / pending 物化 work，并从 Request 的
权威逻辑状态确定性 replay。这是故障恢复，不是合并替换。

1. 释放 PREFIX_APPEND 与 UPDATE 的物理 block，递增 controller generation，废弃 deadline heap
   中的旧条目。
2. 清空 `active_work_item`、物化的 `pending_work_items`、`update_committed_token_ids`、
   `update_committed_append_epoch` 与 `last_admitted_token_ids`。
3. 保留 `PromptSourceState.token_ids`、source finished 状态、`append_epoch` /
   `final_append_epoch`，以及单一窗口中的 `latest_pending_token_ids`；它们代表已接受的逻辑输入。
4. 先按最新逻辑 Append 重建 PREFIX_APPEND。gate 是否允许 admission 仍由 source-finished 状态决定；
   preemption 不能绕过 gate 强制计算 Update。
5. 将当前最新 Update 快照作为**恢复窗口的第一代**：重置 window 的时间戳、deadline、事件数与
   `flush_ready`，但可以保留 EWMA 作为历史先验；随后按同一 policy 和 gate 重新判断是否 flush。
   这不是人为创建的新窗口 arrival，而是丢失物理 KV 后对同一逻辑状态的确定性恢复。

恢复 work 的 `execution_base_token_ids` 在 flush 时重新读取恢复后的 `$U_e$`，
`required_append_epoch` 指向重建后的物理 epoch，而不是旧 block 编号。

### 取消、重复、乱序与 finish 协议

- abort、engine error 或 Request 完成后，controller 删除 request id 的 dormant 标记；heap 遗留条目
  通过 request 不存在或 generation 不匹配自然失效。
- 乱序 sequence number 仍留在 source pending-event buffer；只有按序接受后才更新单一
  `UpdateCoalescingState` 窗口。
- 重复 chunk 或 token 完全相同的 Update snapshot 不创建 work，也不刷新 `quiet_debounce` deadline。
- 相同 version 但 token 内容不同仍为协议错误，合并器不能掩盖 source version 违规。
- V0 的 `PROMPT_FINISHED` 前置条件是两个 source 均已 sealed；若未来允许它隐式 close source，
  请求协议必须明确声明、原子地关闭哪些 source，并拒绝任何后续 event。不能用 finish 启动缺少
  知识 chunk 的 question。

## 正确性不变量

- **最终快照不丢失**：每个合法 Update 都先进入 $U_o$，并覆盖同一个尚未放行目标 $U_p$；
  gate 只影响能否把 $U_p$ 变成 work。合法结束后，最终 $U_o$ 必须可追溯到最终 Update work。
- **严格 gate 安全**：`hold_until_append_finished` 下，Append 未逻辑封闭时不创建 UPDATE work；最终
  Update work 的 `required_append_epoch` 必须等于 `final_append_epoch`。
- **逻辑封闭与物理提交分离**：Append source finished 可以创建依赖 $E_f$ 的 work；只有
  `committed_append_epoch >= E_f` 才允许它 active。
- **LCP + epoch 双重正确性**：Update LCP 只比较 Update token；同一执行语义下复用 UPDATE KV
  还要求 `update_committed_append_epoch == required_append_epoch`。严格路径要求该 epoch 为最终
  $E_f$；近似路径允许它是执行时 epoch $E_s$。
- **放行后不可变**：除 preemption / error recovery 外，新 Update 只能覆盖未放行的 $U_p$，不能
  删除、覆盖或缩短已放行 work。
- **KV 截断安全**：event 到达、窗口合并、deadline 到期和 gate 阻挡都不触碰 UPDATE KV；仅在对应
  work active、Append dependency 与 FIFO 前驱满足后，才执行延后截断。
- **有界等待的范围**：soft/hard deadline 与 TTFT slack 从首个 Update 开始度量。若 gate 阻挡
  admission，到期窗口进入 `flush_ready`，其实际物化可晚于 deadline；这段额外等待必须单独统计，
  不能伪称为 policy 的有界等待。
- **Question barrier 完整**：启动 question Prefill 前，两个 source 已按协议封闭，最终 Append epoch
  已提交，且 $U_p$、`flush_ready`、`pending_work_items`、`active_work_item` 中没有未完成的最终
  Update 状态转换。
- **单线程控制**：所有 source、gate、window、deadline、work 与 KV 状态仅由 P 节点 EngineCore 主
  线程变更；router / FastAPI 线程只投递输入。
- **默认路径不变**：config 为 `None` 时不创建 controller / heap / dormant 状态，也不改变普通
  scheduler 或现有 PromptStream 的即时 Update 行为。

## 统计、对照与实验

### 公平的 token 成本口径

“少算 token”必须说明相对于哪个语义正确的对照：

| 记号 | 对照 | 用途 |
| --- | --- | --- |
| $C_{\mathrm{ungated}}$ | 无 gate、每个 Update 立即计算。 | 历史近似路径的计算量参考，不作严格正确性基线。 |
| $C_{\mathrm{strict\_immediate}}$ | `hold_until_append_finished + immediate`。 | 严格语义下的主在线对照。 |
| $C_{\pi}$ | 同一 Append gate 与 Update policy $\pi$ 下的执行。 | 本文策略实际 Update 重算量。 |

对某次 flush：

$$
C_{\mathrm{actual}}=R(B,W),
\qquad
B=\text{work.execution\_base\_token\_ids},\quad
W=\text{work.target\_update\_token\_ids}.
$$

主结果中的策略 token 节省为：

$$
S^{\mathrm{strict}}_{\mathrm{token}}
= C_{\mathrm{strict\_immediate}}-C_{\pi},
\qquad
\mathrm{saving\_ratio}
=\frac{S^{\mathrm{strict}}_{\mathrm{token}}}
       {\max(C_{\mathrm{strict\_immediate}},1)}.
$$

这不会把 Append gate 带来的延后 admission，错误归功于某个 Update policy。若另行报告相对旧路径
的端到端节省，必须标记为 `vs_ungated_approximate`。

### 必要的运行时统计

- `update_window_events`、`replaced_pending_snapshots`、`window_wait_ms`：衡量单一 Update
  窗口吸收了多少 Update 波动及其等待。
- `append_gate_block_ms`、`flush_ready_count`、`admitted_update_versions`、
  `execution_reduction_ratio`：区分 gate 阻挡与 policy 合并效果。
- `recomputed_update_tokens`、`saved_update_tokens`、`token_saving_ratio`：按严格口径统计。
- `append_epoch_at_admission`、`required_append_epoch`、`update_committed_append_epoch` 与 dependency
  wait time：验证无跨 epoch 误复用。
- `flush_reason`：`immediate`、`soft_deadline`、`hard_deadline`、`ttft_slack`、`update_finished`、
  `prompt_finished`、`joint_finish`、`recovery_replay`。
- retained / released Update blocks、partial-block overwrite 次数和 segment padding 的逻辑长度：区分
  token 节省与 block churn。
- P 节点 GPU 时间、队列长度、GPU 利用率、TTFT p50/p95/p99、goodput；将 Update window waiting、
  append-gate blocking 和 scheduler queueing 分开报告。
- 仅近似早放行实验：`early_release_append_epoch`、后续 Append token 数、近似窗口长度，以及
  与严格模式相比的 TTFT / 回答质量差异。

不同可见 PREFIX_APPEND 长度下，相同 $R$ 的 GPU 时间并不相同。后续可拟合：

$$
\hat{T}_{\mathrm{prefill}}(P,R)=aR+bPR+cR^2+d,
$$

其中 $P$ 是 work 绑定 epoch 下可见的 Prefix+Append 逻辑长度，$R$ 是 Update 重算 token；于是

$$
\hat{G}^{\mathrm{GPU}}
=\hat{T}_{\mathrm{strict\_immediate}}-\hat{T}_{\pi}.
$$

### 实验臂与 Oracle

| 实验臂 | `append_gate` | Update policy | 作用 |
| --- | --- | --- | --- |
| `ungated_immediate` | `none` | immediate | 旧近似路径的计算 / 延迟参考。 |
| `strict_gate_immediate` | `hold_until_append_finished` | immediate | 主正确性在线对照。 |
| `joint_finish_merge` | `hold_until_append_finished` | `full_merge`，等两个 source sealed | 与“Update 完全合并”直接比较。 |
| `strict_fixed_window` | `hold_until_append_finished` | fixed，10 / 20 / 40 / 80 ms | 固定等待。 |
| `strict_quiet_debounce` | `hold_until_append_finished` | debounce + 同一 hard deadline | 吸收 burst。 |
| `strict_adaptive` | `hold_until_append_finished` | adaptive | EWMA、收益、负载与 TTFT slack。 |
| `approximate_early_release` | `hold_until_append_or_update_finished` | 任一 policy | Update source 结束即放行合并快照；报告近似前缀与严格模式的性能 / 质量权衡。 |

`Offline Oracle` 不能绕过 gate。给定 trace，它沿用每个窗口的真实到达时间和同一 policy 的
最大等待约束；但严格 gate 下，只能在 Append sealed、最终 epoch 已确定之后物化 work。若 $U_0$
是执行基准、$U_i$ 是窗口内快照、$D$ 为最大窗口等待，令 `DP[i]` 为最终提交 $U_i$ 时的最小累计
重算量：

$$
\mathrm{DP}[i]
=\min_{\substack{0\leq j<i\\
t_i-t_{j+1}\leq D}}
\left[\mathrm{DP}[j]+R(U_j,U_i)\right].
$$

Oracle 的每个候选 work 都绑定同一最终 $E_f$，不能免费使用 gate 关闭期间最终才出现的 Append
context。

主要自变量包括 Append / Update 的结束相对顺序、Update 间隔、burst 长度、快照长度、LCP 比例、
Append 长度、QPS、P 节点利用率和 $D_{\mathrm{hard}}$。重点图包括严格策略相对 `joint_finish_merge` 的 TTFT survival curve、相对
`strict_gate_immediate` 的 token 节省与额外等待、不同 QPS 下 p50/p95/p99 TTFT 与 goodput，以及
Update window / append-gate / scheduler waiting 的分解。

## 分阶段实现

1. **配置骨架与严格 gate**：解析 JSON config、保留 `None` 默认路径；实现
   `AppendUpdateGateState`、Append epoch、逻辑 sealed / 物理 committed 区分和
   `required_append_epoch` barrier。先实现 `strict_gate_immediate` 与 `joint_finish_merge`。
2. **固定 policy 与 EngineCore 唤醒**：实现 `fixed_window`、`quiet_debounce`、deadline generation
   和 dormant 标记。只有 controller 开启时加入 deadline-aware `Queue.get(timeout=...)`；gate 阻挡或
   `full_merge` 的窗口不保留到期 heap 条目，继续等待外部输入，禁止 1 ms busy-spin。
3. **恢复与可观测性**：实现 preemption replay、abort、乱序、source-finished、finish 的状态矩阵；
   补齐 epoch、waiting、token / block、TTFT 指标和单元 / 集成测试。
4. **Adaptive 与 Oracle**：加入 EWMA、冷启动先验、$w_i^\star$、资源价格与 TTFT slack；实现遵守
   strict gate 的 offline Oracle。
5. **近似早放行评估**：实现 `hold_until_append_or_update_finished` 的 gate-open 转换与
   `early_release_append_epoch` 统计；比较它、`none` 与严格路径的 TTFT、token 成本和回答质量。

最重要的严格路径正确性准则不是“少算了多少”，而是：在相同 source-finish / question 边界下，
最终提交的 `(PREFIX_APPEND final epoch, Update snapshot, question)` 必须与
`strict_gate_immediate` 完全一致；合并只能跳过尚未放行、且已被后续 Update snapshot 覆盖的
中间版本。近似路径则必须忠实保留“执行时可见 Append 前缀”的定义，并单独评估质量代价。
