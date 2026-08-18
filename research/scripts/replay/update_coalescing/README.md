# Update 合并实验设计

## 实验目的

Update 合并的目标不是简单地让所有 Update 等得更久，而是在有限等待内减少
重复 Prefill，同时控制最终用户问题的 TTFT。实验需要回答以下问题：

- **是否真的减少计算**：相同 Update 事件序列下，实际物化的 Update work 和
  Prefill token 是否减少。
- **节省是否转化为性能收益**：在不同 QPS 下，TTFT、吞吐和尾延迟是否改善。
- **等待代价是否可控**：soft window、hard deadline 和 Append gate 是否引入
  不可接受的额外等待。
- **策略是否适应不同波动性**：固定窗口、静默窗口和 Adaptive 是否分别适合
  稳定更新、突发更新和高波动更新。
- **Adaptive 是否值得其复杂度**：相对调优后的固定策略，在线 EWMA 决策是否
  在跨 workload 场景下更稳健。

当前系统只允许一个真实 Append source 和一个 Update source，本实验也只覆盖该
边界，不考虑多 Update source。

## 是否要测试所有参数

**不需要，也不应该做全部参数的笛卡尔积。** 当前配置同时包含策略选择、
安全边界、在线预测和冷启动参数。如果把所有取值完全组合，实验数量会快速膨胀，
而且许多组合没有清晰的研究含义。

参数应分成四层：

| 层级 | 参数 | 测试方式 |
| --- | --- | --- |
| 核心机制 | `update_policy` | 五种策略都要进入基线或主实验。 |
| 上下文语义 | `append_gate` | 主实验固定严格 gate，再做一次独立消融。 |
| 有界等待 | `soft_window_ms`、`hard_deadline_ms`、`ttft_deadline_ms` | 先筛 soft window，再测试 hard / TTFT deadline，不做全组合。 |
| Adaptive 参数 | EWMA、先验、`delay_price`、`load_price_scale` | 只对少量代表值做敏感性分析。 |

推荐使用**逐层筛选**：先找出合理窗口范围，再比较策略，最后只对胜出策略做
QPS sweep 和 Adaptive 参数消融。

## 被测配置与基线

### 控制组

| 名称 | P 节点配置 | 用途 |
| --- | --- | --- |
| `disabled` | 不传 `--prompt-stream-update-coalescing-config` | 原始逐 Update FIFO 行为，是主要无合并基线。 |
| `immediate_controller` | `append_gate=none`、`update_policy=immediate` | 启用 controller 但不等待，用于测量 controller、dormant 检查和状态维护本身的开销。 |
| `full_merge` | `append_gate=hold_until_append_finished`、`update_policy=full_merge` | 只计算最终目标快照，作为最大合并程度和最少 Update work 的参考上界。 |

`disabled` 与 `immediate_controller` 应在相同 workload 下接近。如果二者存在明显
TTFT 或吞吐差异，应先定位新增控制路径开销，不能直接进入策略对比。

### 主要策略

主实验固定：

```json
{
  "append_gate": "hold_until_append_finished"
}
```

在此基础上比较：

- `immediate`：每个 Update 立即 admission。
- `fixed_window`：从窗口第一条 Update 开始等待固定时长，新事件不刷新 soft
  deadline。
- `quiet_debounce`：每条有效 Update 都刷新 quiet window，但不能刷新 hard
  deadline。
- `adaptive`：根据 Update 到达间隔、合并收益和负载计算候选等待时间。
- `full_merge`：等待 Update source 结束或最终 Prompt barrier，不设置时间
  deadline。

### Append gate 消融

Append gate 不与全部 policy 和参数做全组合。选择一组已调好的
`fixed_window` 和一组 `adaptive`，分别比较：

- `none`：Append 和 Update 按已有近似语义竞争。
- `hold_until_append_finished`：主方案，Append 封闭后才允许 Update work 物化。
- `hold_until_append_or_update_finished`：任一来源先结束即可放行，用于评估更早
  overlap 的收益和近似语义代价。

## Workload 维度

### Update 到达频率

Update 合并只会在一个窗口内收到多条 Update 时产生收益，因此必须覆盖不同
更新频率。建议初始使用：

| 类型 | `--update-chunk-interval` | 预期现象 |
| --- | --- | --- |
| 低频 | `0.5` 秒 | 大多数窗口只有一个 Update，合并收益应很小。 |
| 中频 | `0.1` 或 `0.2` 秒 | 用于观察固定窗口和 debounce 的正常取舍。 |
| 高频 | `0.01` 或 `0.05` 秒 | 多条 Update 形成 burst，最能体现合并效果。 |

主筛选阶段可以只选中频与高频；低频作为负向对照，验证策略不会无故增加大量
TTFT。

### Update 快照波动性

仅改变 interval 不足以评价 Adaptive。还需要根据相邻 Update 快照的 source-local
LCP 比例对样本分桶：

$$
\rho_i = \frac{\operatorname{LCP}(U_{i-1}, U_i)}{\max(1, |U_i|)}.
$$

- **高 LCP**：多数内容稳定，仅尾部变化。逐事件计算本身已经较便宜。
- **中 LCP**：存在稳定前缀，同时有明显替换内容。
- **低 LCP**：快照变化剧烈，重复 Prefill 成本高，合并潜在收益最大。

建议同时记录 Update 快照长度、事件数和 LCP 分布。最终结果既报告全数据集指标，
也按这些桶分别报告，避免数据集组成掩盖策略差异。

### Append 与 Update 的相对结束时间

严格 Append gate 的收益依赖两个来源的相对时间。至少覆盖：

- Append 明显先结束，后续 Update 继续到达。
- 两个来源接近同时结束。
- Update 先结束，Append 仍持续到达。

可以通过 Append/Update 的 initial delay、chunk interval 和事件数量构造这三类
时间线。不要只使用一个固定的 Append interval，否则 gate 结论可能只对一种时间
关系成立。

### 系统负载

沿用旧实验的 QPS 起点：

```text
QPS = 0.5, 1, 2, 4
```

但正式运行前应先测量当前 P 节点的大致饱和点。最终 QPS 需要同时包含：

- 低负载：排除排队影响，观察单请求策略代价。
- 中负载：系统开始出现排队，观察减少 Prefill 是否释放容量。
- 高负载：接近饱和但仍能完成绝大多数请求，观察 P95/P99。

如果 QPS=4 已经产生大量超时或错误，它不能作为有效性能点，应降低 QPS 或明确
标记为 overload 实验。

## 指标设计

### 主要结果指标

| 指标 | 含义 |
| --- | --- |
| `finish_to_first_token_ms` | `PROMPT_FINISHED` 发出到 P 返回首 token 的 TTFT，是论文主延迟指标。 |
| TTFT P50 / P90 / P95 / P99 | 同时报告中位数和尾延迟，不能只报告均值。 |
| 成功率、错误率、超时率 | 防止某个策略通过丢弃或卡住请求获得较低 TTFT。 |
| 完成请求吞吐 | 单位时间成功完成的请求数，用于高负载实验。 |

TTFT 主实验建议继续使用 `--ttft-only`，让所有方案都在 P 生成首 token 后结束，
不引入 D 节点 KV load 和 decode 噪声。另选少量样本跑完整 P→D 链路，验证功能
闭环，但不要把两种结果混合统计。

### 合并机制指标

仅有 TTFT 不能证明性能提升来自 Update 合并。每个请求至少需要记录：

| 指标 | 含义 |
| --- | --- |
| `num_update_events` | P 接受的有效 Update 快照数。 |
| `num_update_works_admitted` | 合并后真正创建的 Update work 数。 |
| `num_update_works_completed` | 实际完成 Prefill 的 Update work 数。 |
| `hypothetical_eager_tokens` | 若每个 Update 都立即计算时需要 Prefill 的 Update token。 |
| `actual_merged_tokens` | 当前策略实际需要 Prefill 的 Update token。 |
| `saved_update_tokens` | 两者之差。 |
| `token_saving_ratio` | 节省 token 占 eager token 的比例。 |
| `coalescing_ratio` | 被吸收的 Update 事件比例。 |
| `flush_reason` | immediate、soft deadline、hard deadline、TTFT、source finished 或 Prompt finished。 |
| `window_wait_ms` | Update 从进入窗口到 admission 的等待时间。 |
| `append_gate_block_ms` | policy 已希望 flush，但仍被 Append gate 阻挡的时间。 |

建议使用以下口径：

$$
\mathrm{coalescing\_ratio}
= 1 - \frac{N_{\mathrm{admitted}}}{\max(1, N_{\mathrm{update\ events}})},
$$

$$
\mathrm{token\_saving\_ratio}
= \frac{C_{\mathrm{eager}}-C_{\mathrm{actual}}}
{\max(1, C_{\mathrm{eager}})}.
$$

当前回放 CSV 主要记录 REST 时序，尚不足以完整得到上述内部指标。正式批量实验前，
需要把 Request/Scheduler 的合并统计输出到 Engine 指标、结构化日志或请求结果中，
再由 replay 保存。**不要仅用客户端发送事件数推断实际 Prefill work 数。**

### 系统资源指标

- P 节点实际 scheduled/prefilled token 数。
- GPU utilization 与显存占用。
- Scheduler waiting/running/dormant 请求数。
- 每个请求的 queue delay。
- deadline heap 唤醒次数，以及 dormant 停放/恢复次数。

这些指标用于解释高 QPS 下的 TTFT 改善是否来自减少计算，以及是否引入了额外
调度开销。

## 分阶段实验计划

### 阶段 A：正确性和开关开销

使用单请求或低 QPS，固定同一批短样本：

1. 比较 `disabled` 与 `immediate_controller`。
2. 检查最终 Update 快照没有丢失，QUESTION 只在最终 Update work 之后执行。
3. 覆盖零 token Update、纯截断、单 Update、SOURCE_FINISHED 先后顺序和
   `PROMPT_FINISHED` barrier。
4. 记录 controller 开启但不合并时的 TTFT 和调度开销。

你已经通过 `research/scripts/simple_test` 完成了小规模 REST 闭环，这可以作为阶段
A 的初步结果；正式实验仍应把关键边界变成可重复的自动化用例。

### 阶段 B：策略初筛

固定严格 Append gate、一个中等 QPS、两种 Update interval，比较：

```text
disabled
immediate_controller
fixed_window
quiet_debounce
adaptive
full_merge
```

固定窗口和 debounce 先使用相同的 `soft_window_ms=20`、
`hard_deadline_ms=80`。Adaptive 先使用代码默认 EWMA 参数。这个阶段的目标是淘汰
明显无收益或等待代价过高的配置，不是给出最终论文结果。

### 阶段 C：窗口与 deadline 筛选

对 `fixed_window` 和 `quiet_debounce` 分别测试：

```text
soft_window_ms = 5, 10, 20, 40, 80
hard_deadline_ms = 4 × soft_window_ms
ttft_deadline_ms = null
```

先确定合理 soft window，再只对胜出的 1 至 2 个窗口测试：

```text
hard_deadline_ms = 40, 80, 160
```

`ttft_deadline_ms` 从 Request start 计时，长流式请求可能在最终问题到达前已经消耗
大部分预算，因此它应作为独立 SLO 消融，不与窗口筛选同时变化。

### 阶段 D：Adaptive 参数敏感性

只在 Adaptive 已经显示出潜力后进行：

- `interval_ewma_alpha`、`gain_ewma_alpha`：先同步取
  `0.1 / 0.25 / 0.5`，再判断是否需要拆开。
- `cold_start_prior_weight`：取 `0 / 4 / 8`，重点观察 Update 事件很少的短请求。
- `arrival_rate_prior_per_s`、`merge_gain_prior_tokens`：比较 `null` 与根据训练集
  统计得到的一组先验，不凭空枚举大量值。
- `delay_price`：围绕默认值取 `0.5 / 1 / 2`。
- `load_price_scale`：先比较 `0` 与一个非零值；只有负载感知明显有效时再细调。

这些参数采用一次改变一组的方式，不做全部组合。数据集统计得到的先验必须只使用
训练/调优子集，最终评估子集不能反向参与先验选择。

### 阶段 E：负载与泛化实验

从前面阶段选出：

- `disabled`。
- 最佳 fixed 或 debounce。
- 最佳 Adaptive。
- `full_merge`。

再执行完整 QPS sweep、Update interval sweep、LCP 分桶和 Append/Update 结束顺序
实验。这一阶段才用于论文主表和主图。

### 阶段 F：Append gate 消融

使用阶段 E 的最佳 fixed/adaptive 参数，只改变 `append_gate`。报告：

- token saving。
- TTFT。
- gate block time。
- 最终输出一致性或质量变化。

这样可以把“Update 窗口策略”和“是否等待 Append 封闭”两个创新因素分开解释。

## 推荐的首轮最小矩阵

为了快速确认趋势，第一轮可以采用：

- 样本数：`256`，固定样本 ID 和随机种子。
- 重复次数：每个配置 `3` 次。
- QPS：先使用 `1`；确认稳定后再扩展为 `0.5 / 1 / 2 / 4`。
- Update interval：`0.2` 和 `0.01` 秒。
- Append interval：先固定 `0.5` 秒。
- Gate：`hold_until_append_finished`。
- 策略：`disabled`、`immediate`、`fixed 20/80`、
  `debounce 20/80`、默认 `adaptive`、`full_merge`。
- 生成：全部使用 `--ttft-only`。

该矩阵为 $6 \times 2 \times 3=36$ 次运行，能够先判断主要趋势。不要在这一步
加入全部 EWMA、prior、price、gate 和 QPS 组合。

## 公平性与重复性

- 使用完全相同的样本顺序、QPS arrival seed 和 source interval。
- 每个配置启动独立 P 节点实例，或者重启同一个 P 节点后再运行；配置是 Engine
  启动参数，不能在请求级切换。
- 每次正式采集前使用同一批不计入结果的 warmup 请求。
- 不同策略的运行顺序随机化或轮换，避免机器温度和集群背景负载始终偏向某个策略。
- 每个点至少重复 3 次，报告均值/中位数及置信区间；TTFT survival curve 应使用
  请求级样本，而不是只画各次运行的均值。
- 采用配对分析：同一个 sample id 在不同策略下比较 TTFT 和 token saving。
- 结果中保留失败请求，不得只删除超时样本后报告成功请求延迟。

## 结果图表

论文主结果建议至少包含：

- **TTFT survival curve**：相同 QPS 下比较主要策略。
- **P95/P99 TTFT 对 QPS**：观察接近饱和时的尾延迟。
- **Update Prefill token 对策略**：直接展示实际计算量下降。
- **token saving 与 TTFT 的 Pareto 图**：横轴为节省 token，纵轴为 TTFT 或
  TTFT 增量，展示计算与等待的权衡。
- **admission work 数 / Update event 数**：展示合并程度。
- **flush reason 堆叠图**：解释策略主要由 soft、hard、source finish 还是 TTFT
  deadline 触发。
- **按 LCP 比例分桶的收益图**：说明策略在哪类 Update 变化中最有效。

## 计划中的脚本结构

后续实现建议保持以下边界：

```text
research/scripts/replay/update_coalescing/
├── README.md
├── replay_update_coalescing.py
├── run_policy_sweep.sbatch
├── summarize_coalescing.py
├── plot_coalescing_tradeoff.py
└── results/
```

- `replay_update_coalescing.py` 只负责复现相同 mixed-source 事件时间线和记录请求
  结果；它不在客户端自行模拟 policy。真正的合并策略只能由 P 节点执行。
- `run_policy_sweep.sbatch` 负责按配置重启或连接对应 P 节点，并确保结果目录名
  包含 policy、gate、窗口、QPS、interval 和 seed。
- `summarize_coalescing.py` 负责合并请求级与 Engine 级指标。
- `plot_coalescing_tradeoff.py` 负责 TTFT、token saving 和 Pareto 图。

在内部合并指标尚未导出之前，可以先复用旧 `replay_prompt_stream.py` 检查 TTFT，
但这种结果只能说明“开关后延迟发生变化”，不能单独支撑“Update 合并减少了计算”
这一论文结论。
