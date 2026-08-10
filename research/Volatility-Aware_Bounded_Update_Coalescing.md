# Volatility-Aware Bounded Update Coalescing

> **研究定位**：混合来源流式 Prefill 的第二个创新点。
> **范围**：一个逻辑请求只有一个全量快照语义的 Update 源；暂不处理多 Update 源、跨源 KV 排布与跨源公平性。
> **状态**：完整设计，尚未接入 scheduler。现有系统已经具备分段 KV、冻结 Update work、FIFO 执行与延后 UPDATE KV 截断；本文在此基础上设计 Update 合并层。

## 核心结论

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

## 与分段 KV 设计的关系

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

本文只决定**何时让一个已到达的 Update 快照进入 scheduler**，不改变最终 prompt 语义、分段 KV 排布或 P/D 路径。

## 版本层次

同一个 Update 源在任意时刻可能同时有多种“当前版本”。它们不能混用。

| 记号 | 建议状态 | 含义 | 是否允许后续事件覆盖 |
| --- | --- | --- | --- |
| $U_o$ | PromptSourceState.token_ids | 最近接受的最新逻辑快照。 | 是 |
| $U_p$ | UpdateCoalescingState.latest_pending_token_ids | 当前合并窗口内最新、尚未放行的目标快照。 | 是 |
| $U_b$ | UpdateCoalescingState.execution_base_token_ids | 本窗口开始时冻结的执行基准；目标 work 将接在它之后执行。 | 否 |
| $U_e$ | UpdateCoalescingState.last_admitted_token_ids | 最近一次已放行的完整 Update 快照；可能尚未物化到 KV。 | 仅在下一次 flush 时前移 |
| $U_c$ | PromptStreamState.update_committed_token_ids | UPDATE KV 当前真实对应的完整快照。 | 仅在对应 work 完成时前移 |

$U_e$ 可以领先于 $U_c$。例如：

~~~text
UPDATE KV 已完成到 U1：          U_c = U1
U2 已经 flush 并等待/执行：       U_e = U2
U3 正在合并窗口中：               U_p = U3
~~~

若 U3 后续放行，它在执行链上必定接在 U2 之后，故实际 KV 复用必须比较 $U_2$ 与 $U_3$，而不是比较 $U_1$ 与 $U_3$。

这也是本文相对旧笔记的关键修正：**flush 的 LCP 基准不是一律使用 committed snapshot，而是使用窗口开始时冻结的 execution base。** 当系统允许已有 Update work 正在 active 或等待时，后者才是正确的前驱。

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

UpdateCoalescingState 放入长期 Request 的 PromptStreamState。它只保留当前窗口的必要状态，而不是保存全部历史版本。

~~~python
@dataclass
class UpdateCoalescingState:
    # 唯一可被后续事件覆盖、尚未进入 pending_work_items 的目标。
    latest_pending_token_ids: tuple[int, ...] | None = None

    # 本窗口开始时冻结的实际执行基准。
    execution_base_token_ids: tuple[int, ...] | None = None

    # 最近一次已放行的完整 Update 版本，即未来执行链的前沿。
    last_admitted_token_ids: tuple[int, ...] = ()

    # 使用 P 节点 monotonic clock 的窗口时间状态。
    first_pending_time: float | None = None
    last_update_time: float | None = None
    soft_deadline: float | None = None
    hard_deadline: float | None = None
    generation: int = 0

    # 在线统计。
    pending_event_count: int = 0
    hypothetical_eager_tokens: int = 0
    interval_ewma: float | None = None
    merge_gain_ewma: float | None = None
~~~

约束如下：

- 更新前的 PromptSourceState.token_ids 就是上一条原始事件快照。处理新事件时先冻结旧值，再覆盖 source.token_ids，因而不必重复长期保存 previous_event_snapshot。
- execution_base_token_ids 在一个窗口内不变；新事件只能替换 latest_pending_token_ids、更新统计和刷新 soft deadline。
- last_admitted_token_ids 在 flush 时立即前移，而不是等 GPU 完成后才前移。它表示未来执行顺序。
- update_committed_token_ids 只在 work 计算完成后更新，表示物理 UPDATE KV 的真实内容。
- flush 后清空本窗口的 pending、execution base、deadline、hypothetical eager cost 和 pending event count；EWMA 可以跨窗口保留。

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

## 两类比较：到达统计与 flush 执行

新 Update 到达时不应沿着“pending → active → committed”依次计算多个 LCP。正确设计只有两类比较：

| 比较 | 快照 | 用途 | 会否创建 work |
| --- | --- | --- | --- |
| 到达比较 | 上一原始事件 $P$ 与新快照 $N$ | 假想 eager 成本、覆盖收益、EWMA | 否 |
| flush 比较 | 固定执行基准 $U_b$ 与窗口最新快照 $U_p$ | 实际 LCP、KV 截断边界、冻结 work | 是 |

### 到达比较

假设窗口正在等待 $P$，新快照 $N$ 到达；$U_b$ 是窗口开始时冻结的 execution base。

若所有快照都立即执行，窗口内假想 eager 成本为：

$$
C_{\mathrm{eager}}
=
R(U_b,U_1)
+
\sum_{j=1}^{m-1}R(U_j,U_{j+1}).
$$

第一条窗口事件 $U_1$ 到达时：

$$
\mathrm{hypothetical\_eager\_tokens}=R(U_b,U_1).
$$

后续事件 $P\rightarrow N$ 到达时：

$$
\mathrm{hypothetical\_eager\_tokens}
\mathrel{+}=R(P,N).
$$

此刻已知 $P$ 被覆盖，可以观测跳过 $P$ 的局部收益：

$$
G
=
\max\left(
0,
R(U_b,P)+R(P,N)-R(U_b,N)
\right).
$$

$G$ 只用于更新预测；$P$ 从未进入 pending_work_items。

### Flush 比较

等待窗口结束时，唯一目标是 $U_p$。系统只计算：

$$
L_{\mathrm{execute}}=L(U_b,U_p),
$$

并创建：

~~~python
PromptStreamWorkItem(
    phase=PromptStreamExecutionPhase.UPDATE,
    update_token_ids=U_p,
    update_common_prefix_len=L_execute,
)
~~~

随后立即推进执行前沿：

~~~text
last_admitted_token_ids = U_p
latest_pending_token_ids = None
execution_base_token_ids = None
~~~

此后这个 work 不再受合并策略影响。正在等待的旧快照 $P$ 不应成为实际 KV 基准，因为它从未被放行。

## 在线放行决策

策略回答的问题是：当前最新快照到达后，应立即 flush，还是最多等待一小段时间以吸收可能马上到来的新快照？

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

## 有界等待

每个等待窗口维护两个 deadline：

| deadline | 定义 | 新事件到达时 | 作用 |
| --- | --- | --- | --- |
| soft deadline | 最近事件时刻加 $w_i^\star$ | 可以刷新 | 吸收短 burst |
| hard deadline | 窗口首事件时刻加 $D_{\mathrm{hard}}$ | 绝不能刷新 | 防止无限等待与饥饿 |

标准 P/D 路径的 TTFT 剩余 slack 为：

$$
S_i
=
D_{\mathrm{TTFT}}-t_{\mathrm{now}}
-\hat{T}_{\mathrm{prefill}}
-\hat{T}_{\mathrm{transfer}}
-\hat{T}_{\mathrm{D1}}.
$$

$\hat{T}_{\mathrm{D1}}$ 表示 D 节点产生第一个 token 的预测时间。TTFT-only 实验不含 transfer 和 D 首 token，应去掉相应项。

若 $d_{\mathrm{hard}}$ 是绝对 hard deadline，最大可等待时间为：

$$
W_i^{\max}
=
\max\left(
0,
\min\left(
d_{\mathrm{hard}}-t_{\mathrm{now}},
S_i
\right)
\right).
$$

最终 flush 时刻为：

$$
t_{\mathrm{flush}}
=
\min\left(
d_{\mathrm{soft}},
d_{\mathrm{hard}},
t_{\mathrm{now}}+S_i
\right).
$$

若 $S_i\leq0$，必须立刻 flush。新事件只能刷新 soft deadline，不能刷新 hard deadline。

### 具体例子：连续 burst、TTFT 约束与自动 flush

假设已经放行的执行前沿为 $U_e=U_0$。在 $t=0\,\mathrm{ms}$，第一个新
Update 快照 $U_1$ 到达；策略此时给出 $w^\star=30\,\mathrm{ms}$，并且：

- $D_{\mathrm{hard}}=100\,\mathrm{ms}$，所以 hard deadline 固定为
  $t=100\,\mathrm{ms}$。
- 根据该请求剩余的 TTFT 预算以及预估的 P/D 工作，任何本窗口的 flush 最迟必须
  在 $t=70\,\mathrm{ms}$ 完成。也就是说，在任意后续时刻，
  $t_{\mathrm{now}}+S_i=70\,\mathrm{ms}$。

随后出现一个短 burst：

| 时刻 | 到达快照 / 动作 | $U_b$ 与 $U_p$ | soft deadline | 最终计划的 flush 时刻 |
| --- | --- | --- | --- | --- |
| $0\,\mathrm{ms}$ | $U_1$ 到达，打开窗口 | $U_b=U_0$，$U_p=U_1$ | $30\,\mathrm{ms}$ | $\min(30,100,70)=30\,\mathrm{ms}$ |
| $18\,\mathrm{ms}$ | $U_2$ 到达，覆盖 pending 目标 | $U_b=U_0$，$U_p=U_2$ | 刷新为 $48\,\mathrm{ms}$ | $\min(48,100,70)=48\,\mathrm{ms}$ |
| $43\,\mathrm{ms}$ | $U_3$ 到达，覆盖 pending 目标 | $U_b=U_0$，$U_p=U_3$ | 刷新为 $73\,\mathrm{ms}$ | $\min(73,100,70)=70\,\mathrm{ms}$ |
| $70\,\mathrm{ms}$ | 没有更多 Update 到达，EngineCore 因 deadline 醒来 | flush $U_0\rightarrow U_3$ | 不再适用 | 立即放行不可变 work |

这个例子说明四件事：

- $U_1$ 和 $U_2$ 从未进入 `pending_work_items`，因此不会产生两次无用的 Update
  Prefill；最终只放行 $U_0\rightarrow U_3$。
- $U_2$、$U_3$ 只能刷新 soft deadline；它们不能把 hard deadline 从
  $100\,\mathrm{ms}$ 向后推。
- 本例中 TTFT slack 更紧，因此即使 soft deadline 已经被刷新到
  $73\,\mathrm{ms}$，仍必须在 $70\,\mathrm{ms}$ flush。若没有 TTFT 限制，则这
  一窗口最晚会在 $100\,\mathrm{ms}$ 被 hard deadline 强制结束。
- 即使 $43\,\mathrm{ms}$ 之后再也没有事件，EngineCore 也必须在
  $70\,\mathrm{ms}$ 主动醒来检查 deadline；只在新事件到达时判断，会让这个窗口
  永远悬挂。

若在 $55\,\mathrm{ms}$ 收到 `PROMPT_FINISHED`，则它是更强的 question barrier：
系统不等待 $70\,\mathrm{ms}$，而是在 $55\,\mathrm{ms}$ 强制 flush
$U_0\rightarrow U_3$，再等待这个 work 及此前已放行的 work 按 FIFO 计算完成，最后
才调度用户问题。

## 完整状态机

### 新 Update 事件到达

在 P 节点 EngineCore 主线程中处理一个合法且 token 内容变化的 Update 快照 $N$：

1. 覆盖 PromptSourceState.token_ids 前读取旧快照 $P$。
2. 更新 source 的 token、version、sequence number 和到达时间，使 source 始终表示 $U_o$。
3. 若没有合并窗口：
   - 冻结 $U_b=\mathrm{last\_admitted\_token\_ids}$。
   - 设置 $U_p=N$。
   - 初始化 first pending time、hard deadline、pending event count。
   - 设置 hypothetical eager cost 为 $R(U_b,N)$。
4. 若已有窗口：
   - 旧 $U_p=P$ 被新快照 $N$ 覆盖。
   - 累计 $R(P,N)$，计算 $G$，更新 merge gain EWMA。
   - 用本次到达间隔更新 interval EWMA 和 $\hat{\lambda}$。
   - 令 $U_p=N$，仅刷新 soft deadline。
5. 按阈值、$w_i^\star$、TTFT slack 和 hard deadline 决定立即 flush 或登记 deadline。

token 完全相同、仅 version 等元数据变化的事件不产生新 Prefill work，也不应无意义地重置一个合并窗口。

### Flush

soft deadline、hard deadline、TTFT slack 或强制事件触发时：

1. 读取当前 $U_b$ 和 $U_p$。
2. 计算 $L(U_b,U_p)$。
3. 以完整 $U_p$ 和该 LCP 创建一个不可变 Update work，append 到 pending_work_items 的末尾。
4. 立即设置 $U_e=U_p$，清空当前合并窗口，并使旧 deadline generation 失效。
5. 不改写 active_work_item，不删除任何 pending work，不在 flush 时截断 UPDATE KV。

UPDATE KV 截断仍延后到该 work 真正被 scheduler 选中时。此时其前驱 work 已按 FIFO 物化，系统才能安全按该 work 的 LCP 截断 UPDATE 尾部。

若 $U_p$ 是 $U_b$ 的严格前缀，$R(U_b,U_p)=0$，仍必须生成零 token 状态转换：它负责截断多余 UPDATE KV、更新 committed 快照并继续后续 work，不能简单丢弃。

### 强制 flush 与 question barrier

以下情形必须立即 flush：

- soft deadline 到期。
- hard deadline 到期。
- TTFT slack 耗尽。
- 唯一 Update source 收到 source finished。
- 收到 PROMPT_FINISHED。
- 后续可选的空闲优化判定立即执行更好；它只能提前 flush，不能推迟 deadline。

PROMPT_FINISHED 必须在同一次 EngineCore 状态转换中先 flush 合并层的 $U_p$，再写入 suffix/question。否则 Question barrier 可能只看到空的 pending_work_items，错误地在尚未放行的 Update 前开始最终 question Prefill。

### Append 与 Update 的顺序

Append 不经过 UpdateCoalescingState。等待中的 Update 尚未放行时，后到 Append 仍可进入 pending_work_items 并正常竞争。

因此这里的 FIFO 是**放行后的 work FIFO**，不是严格的原始事件到达 FIFO。被延迟的 Update 在被 scheduler 选中时冻结已提交的 PREFIX_APPEND 视图；尚未提交的 Append 不会进入它的 attention。

这与现有“Append 和 Update 按实际可调度 work 竞争”的近似语义一致。若未来要求原始事件严格顺序，需加不可调度 barrier；它会阻塞后到 Append，不纳入第一版。

## Scheduler、并发与唤醒

### 不需要锁

FastAPI/router 不应直接修改长期 Request。外部 REST 事件进入 EngineCore 输入队列后，由同一个 EngineCore 主线程顺序执行：

~~~text
Scheduler.add_request
    → _apply_prompt_stream_request
    → Request.apply_prompt_stream_event
    → Scheduler.flush_due_update_coalescing
    → Scheduler.schedule
~~~

PromptStreamState、UpdateCoalescingState、pending_work_items、active_work_item 和 KV block table 都由该线程修改。因此：

- 不需要把 pending_work_items 改成带锁队列。
- 不需要 asyncio.Lock 或 threading.Lock；它们不能跨 API 进程保护 EngineCore 状态。
- 定时器线程最多向 EngineCore 投递 wake-up 或 deadline hint，绝不能直接 flush Request 或修改快照。

### deadline 最小堆

Scheduler 应维护一个最小堆，条目只包含：

~~~text
(deadline, request_id, generation)
~~~

- 每个新窗口和每次 soft deadline 刷新都递增 generation。
- 堆不保存 token；快照始终归 Request 的 UpdateCoalescingState 所有。
- 每轮调度前先处理已经到达的输入事件，再弹出所有已到期条目。
- generation 不匹配的条目是过期 deadline，直接跳过。
- 有效条目在同一 EngineCore 主线程中 flush。

这样，deadline 附近若新事件先到达，它会使旧条目失效；若 deadline 先被处理，则旧窗口先 flush，新事件进入下一窗口。两者都满足有界等待和不可变 work 语义。

### dormant 请求

某请求可能有 latest pending Update，但没有 active 或已放行 work。它不应持续留在 running 集合做空 GPU 调度。

- 已有 active 或已放行 Append/Update work 不受影响，继续正常调度。
- 若请求仅等待合并 deadline，则应进入 scheduler 管理的 dormant coalescing wait 集合。
- EngineCore 在最近 deadline 到达时被唤醒；flush 后再把请求加入普通等待队列。

第一版应采用下文的 **deadline-aware 阻塞等待**，不能每毫秒空 `schedule()` 检查时间；
后者会引入 CPU 空转并污染实验。

### EngineCore 的 deadline-aware 阻塞等待

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

这对普通请求是正确的，但对合并窗口不够：一个 dormant 请求可能拥有
`latest_pending_token_ids`，却没有 active work 或已放行 work。若没有新的 Update、
`PROMPT_FINISHED` 或 abort 到达，现有 EngineCore 不知道它的 soft/hard deadline，因而
永远不会自行 flush。

反过来，不能让这种 dormant 请求继续被当作普通 scheduler work。否则
`scheduler.has_requests()` 始终为真，EngineCore 会不断进入 `step()` 和
`scheduler.schedule()`；没有 GPU work 时当前路径只短暂 sleep，形成低效忙转，而不是
真正等待 deadline。

#### Scheduler 与 EngineCore 的职责边界

Scheduler 维护 deadline 最小堆和 dormant 集合，并提供两个仅由 EngineCore 主线程调用
的接口；以下名称是建议的实现接口：

~~~python
def get_next_prompt_stream_deadline(self) -> float | None:
    """返回有效 coalescing deadline 的最早绝对 monotonic 时间。"""

def flush_due_update_coalescing(self, now: float) -> int:
    """放行所有到期窗口，并将因此可运行的请求重新加入等待队列。"""
~~~

- dormant 请求不应被当作可运行 GPU work。实现时建议新增
  `Scheduler.has_runnable_requests()`，并让 `EngineCore.has_work()`、`step()` 中
  “是否进入 `scheduler.schedule()`”的早退判断，以及“无模型执行时短暂 sleep”的判断
  都使用 runnable 语义；不能只改 `has_work()`，否则 timeout 返回后仍可能进入空转路径。
  `has_requests()` 若还承担请求生命周期统计，可保留其原语义。
- dormant 集合不混入普通 `waiting` / `running` 队列；请求对象仍可保留在按
  request id 索引的长期状态表中，以接收后续 Update、finish 或 abort。
- `get_next_prompt_stream_deadline()` 只读最小堆，跳过 generation 失效、请求已取消或
  已无 pending Update 的陈旧条目。
- `flush_due_update_coalescing()` 是唯一将 `latest_pending_token_ids` 物化为不可变
  `pending_work_items` 的入口。它完成后，请求才重新成为普通 runnable request。

#### 主循环的目标顺序

第一版不创建定时器线程。只把 EngineCore 空闲时原本无限期的 queue wait 改为“最多等待
到最近 deadline”：

~~~python
while self._handle_shutdown():
    # 处理已经到达的 REST/ZMQ 输入；可能创建、刷新或取消 deadline。
    self._process_input_queue()

    # 同一主线程检查到期窗口；必要时把 dormant 请求转回 runnable 队列。
    self.scheduler.flush_due_update_coalescing(time.monotonic())

    # 只调度已经有实际 GPU work 的普通请求。
    self._process_engine_step()
~~~

`_process_input_queue()` 在 `not self.has_work()` 时的等待逻辑改为概念上的：

~~~python
deadline = self.scheduler.get_next_prompt_stream_deadline()
timeout = None if deadline is None else max(0.0, deadline - time.monotonic())

try:
    request = self.input_queue.get(block=True, timeout=timeout)
except queue.Empty:
    # 不是错误；最近的 coalescing deadline 到期。
    return
else:
    self._handle_client_request(*request)
~~~

`queue.Empty` 返回主循环后，紧随其后的 `flush_due_update_coalescing()` 会把到期的
Update 放行。因此“EngineCore 因 deadline 醒来”并不是新增一个线程或主动中断 GPU；它
只是原有 EngineCore 主线程从带 timeout 的阻塞 `Queue.get()` 返回。

#### 输入事件与 deadline 同时发生

在同一 EngineCore 主线程上采用固定顺序：先处理已进入 `input_queue` 的事件，再检查
deadline。

- 新 Update 先被处理时，它可以刷新 soft deadline 并递增 generation；堆内旧条目在
  随后的检查中失效。
- queue wait 先 timeout 时，主线程先 flush 旧窗口；稍后到达的新 Update 进入下一窗口。
- 无论哪一种先发生，hard deadline 不会被刷新，且已放行 work 永远不可变，因此不需要
  锁来保护 `Request`、`pending_work_items` 或 KV block table。

`process_input_queue_block=False` 的既有非阻塞路径应保持原样；deadline 检查仍在每轮
主循环发生，但它不应向该特殊路径额外引入阻塞。

## flush 后的 KV 执行

放行后仍沿用现有分段 KV 路径：

1. Update work 在 pending_work_items 中冻结，不再被合并层触碰。
2. work 成为 active_work_item 时，冻结当时已经提交的 PREFIX_APPEND 物理 token 视图。
3. scheduler 按该 work 的 source-local LCP 延后截断 UPDATE KV。
4. worker 读取 PREFIX_APPEND + UPDATE block table，只向 UPDATE segment 写新 KV。
5. work 完成后，update_committed_token_ids 更新为该完整快照；work 对象从 active_work_item 移除并可回收，KV 保留给下一代 Update 复用。

~~~text
合并层 latest pending
    → flush 后的冻结 work
    → active work
    → UPDATE KV committed
    → work 对象删除，KV 与 committed snapshot 保留
~~~

## 预抢占、取消与协议异常

### 预抢占

preemption 会释放两个物理 KV segment，现有系统也会清空 active/pending work 并从最新逻辑 source 状态安全 replay。合并层还必须：

- 增加 generation，使旧 deadline 条目失效。
- 清空 latest pending 和 execution base，避免引用已丢弃 work。
- 将 last admitted 重置为空 Update 基准。
- 以当前 source.token_ids 的完整最新快照重建；建议强制放行，不再等待合并窗口。

EWMA 可以保留为统计历史，但当前窗口的 hypothetical eager cost、pending event count 与 saved token 统计必须重置。

### 取消、重复与乱序

- abort 或 error 后，堆中遗留条目通过请求不存在或 generation 不匹配自然失效。
- 乱序 sequence number 仍停留在 source 的 pending events，不进入合并窗口。
- 重复 chunk 或完全相同的 token 快照不创建新 work。
- 相同 version 却具有不同 token 内容仍为协议错误，不能由合并机制掩盖。

## 正确性不变量

- **最终快照不丢失**：source finished 与 PROMPT_FINISHED 必定 flush 当前 $U_p$。
- **放行后不可变**：新 Update 只能影响下一窗口，不能删除、覆盖或缩短已放行 work。
- **执行 LCP 正确**：每个 work 只以冻结的 $U_b$ 和最终 $U_p$ 计算 source-local LCP。
- **KV 截断安全**：事件到达和合并等待不触碰 UPDATE KV；仅在对应 work 被选中、前驱 work 已完成后截断。
- **Update 源内化**：template、Append、question 不参与 Update LCP 或 UPDATE KV 失效范围。
- **陈旧度有界**：从窗口首事件到 flush 不超过 hard deadline，也不超过 TTFT slack。
- **Question barrier 完整**：启动 question Prefill 前，合并层、pending_work_items 和 active_work_item 中均无未完成 Update。
- **单线程控制**：所有 Request 状态和 KV 状态仅由 EngineCore 主线程变更。

## 统计、对照与实验

每个窗口 flush 时记录：

$$
C_{\mathrm{actual}}=R(U_b,U_p),
$$

$$
S_{\mathrm{token}}
=
\mathrm{hypothetical\_eager\_tokens}-C_{\mathrm{actual}}.
$$

建议记录：

- recomputed_update_tokens、saved_update_tokens、token_saving_ratio。
- observed_update_versions、admitted_update_versions、execution_reduction_ratio。
- waiting_time、最大陈旧度、soft/hard/slack/finish 各类 flush 原因。
- retained_update_blocks、released_update_blocks、partial-block overwrite 次数。
- P 节点 GPU 时间、队列长度、GPU 利用率、TTFT p50/p95/p99 和 goodput。
- 最终被放行的 Update 快照与 Immediate 对照的 token 完全一致性。

不同可见 PREFIX_APPEND 长度下，相同 $R$ 的 GPU 时间并不相同。后续可拟合：

$$
\hat{T}_{\mathrm{prefill}}(P,R)
=
aR+bPR+cR^2+d,
$$

其中 $P$ 为可见前缀长度，$R$ 为 Update 重算 token。由此得到 GPU 时间收益：

$$
\hat{G}^{\mathrm{GPU}}
=
\hat{T}_{\mathrm{eager}}
-
\hat{T}_{\mathrm{coalesce}}.
$$

对照策略：

| 策略 | 行为 |
| --- | --- |
| Immediate | 每个内容变化的 Update 立即 flush。 |
| Fixed window | 固定等待 10、20、40、80 ms，只放行窗口末最新快照。 |
| Quiet debounce | 固定静默窗口，仍使用同一 hard deadline。 |
| Adaptive | 使用 EWMA、收益预测、负载价格与 TTFT slack。 |
| Offline Oracle | 已知完整 trace，在相同最大等待约束下最小化 Update 重算 token。 |

给定完整 trace 与最大等待 $D$，令 $U_0$ 为初始基准，DP[i] 为最终提交 $U_i$ 时最小累计重算量：

$$
\mathrm{DP}[i]
=
\min_{\substack{0\leq j<i\\
t_i-t_{j+1}\leq D}}
\left[
\mathrm{DP}[j]+R(U_j,U_i)
\right].
$$

主要自变量包括 Update 间隔、burst 长度、快照长度、LCP 比例、Append 长度、QPS、P 节点利用率和 $D_{\mathrm{hard}}$。重点图包括额外等待与 Token Saving Ratio 的关系、不同 QPS 的 TTFT survival curve，以及 Adaptive 与 Oracle 的差距。

## 分阶段实现

1. **固定有界窗口**
   - 实现 UpdateCoalescingState、latest pending、execution base、deadline generation 和强制 flush。
   - 使用固定 soft window 与 hard deadline。
   - 覆盖最终快照、strict-prefix、PROMPT_FINISHED、active Update、Append 交错与 preemption 测试。

2. **请求级自适应**
   - 加入 interval EWMA、merge gain EWMA、冷启动先验与 $w_i^\star$。
   - 保持所有策略最终调用同一个 flush 函数，不能在 pending_work_items 内实现合并。

3. **负载与时延感知**
   - 加入 P 节点资源价格、TTFT slack 和基于 $(P,R)$ 的 GPU 时间预测。
   - 加入 dormant 请求的 deadline-aware 唤醒，避免空转。

4. **论文级评估**
   - 比较 Immediate、固定窗口、debounce、Adaptive 和 Offline Oracle。
   - 同时报告最终快照正确性、重算 token、GPU 时间、block churn、TTFT 与 goodput。

最重要的正确性准则不是“少算了多少”，而是：在同一 finish 边界下，最终实际放行的 Update 快照必须与 Immediate 路径完全一致；合并只能跳过已经被后续快照覆盖、且尚未放行的中间版本。
