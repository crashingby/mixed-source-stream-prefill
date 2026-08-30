# Prompt Stream 高 QPS 抢占恢复与统计一致性 TODO

## 文档目标

本文记录 Update coalescing 在高 QPS 回放下暴露的问题，并给出后续代码修改的
实现基准。当前阶段只完成问题定位和方案设计，不修改源码。

需要同时处理三类问题：

1. 多个长期请求同时保留分段 KV，确实会造成 KV Cache 压力、驱逐和 preemption。
2. preemption 后的 Request recovery 会折叠已经 admission 的 Update work，但
   现有生命周期计数没有描述这种折叠。
3. replay 把 stats 校验失败与 HTTP/SSE 失败混成 `result_status=error`，导致已经
   返回首 token 的请求被从 TTFT 统计中删除。

最终修改必须保证：

- Question 不会越过最终 Append/Update barrier。
- policy admission、物理执行和 recovery replay 使用不同计数口径。
- bounded window 不会因为 repeated preemption 重新计时。
- replay 分开报告传输成功、首 token 成功、stats 有效和 formal-valid。
- 补齐压力观测后，再优化 Scheduler 的 KV victim 策略。

## 已有实现基础

以下能力已经存在，本 TODO 不重新设计：

- Prompt Stream 使用 `PREFIX_APPEND` 和 `UPDATE` 两个物理 KV 域。
- Update work 在正常路径进入 `pending_work_items` 后，不再被后续 Update 合并。
- controller 支持 Append gate、fixed window、quiet debounce、adaptive 和 full merge。
- P 节点已通过 `prompt_stream_stats` 返回 accepted、admitted、completed、eager、
  admitted/scheduled tokens、saved、extra 和 invalidated。
- `finish` 与 `finish_ttft` 都能将 stats 经 proxy 返回客户端。

当前问题集中在高压 recovery 和实验统计语义，而不是基础网络透传链路。

## 现有结果给出的证据

### QPS 4 的失败构成

检查 `research/scripts/replay/update_coalescing/results` 中现有 QPS 4 正式结果，共有
25,920 个请求：

| 分类 | 请求数 | 占比 |
| --- | ---: | ---: |
| 当前 `result_status=ok` | 23,449 | 90.47% |
| 当前 `result_status=error` | 2,471 | 9.53% |
| 已获得首 token 和完整 stats，但被 stats 断言改记为 error | 2,467 | 9.52% |
| 真正的 HTTP 错误 | 4 | 0.0154% |

所以当前 90.47% 的 `success_rate` 不是首 token 成功率。按是否真正收到首 token
计算，这批结果的成功率是 99.9846%。

2,467 条 stats 校验失败分为：

| 失败关系 | 请求数 |
| --- | ---: |
| `completed < admitted` | 1,495 |
| `admitted > accepted` 且 `completed < admitted` | 875 |
| 仅 `admitted > accepted` | 97 |

另有 2,036 条记录同时满足：

```text
scheduled_update_prefill_tokens < admitted_update_prefill_tokens
```

这个错误被更早抛出的 lifecycle 异常遮住了。

### 不同策略的分布

QPS 4 下，每个 profile 聚合三个 Update interval，共 4,320 个请求：

| Profile | 当前 error 比例 | 主要现象 |
| --- | ---: | --- |
| disabled | 0.1% | 4 个真正的 `/prompt_stream/event` 上游 502 |
| full-merge | 0% | 存在物理 replay，但单 admission 使旧布尔计数碰巧闭合 |
| fixed-window | 7.3% | `completed < admitted` |
| quiet-debounce | 9.2% | `completed < admitted` |
| strict-immediate | 10.1% | `completed < admitted` |
| immediate-controller | 30.5% | 同时出现两类 lifecycle 错误 |

全部 972 条 `admitted > accepted` 都来自 `immediate-controller`。它使用
`append_gate=none`，Update admission 会与 Append epoch 交错，正好触发当前
recovery 的 epoch 重计数问题。

### 当前汇总会低估高压 TTFT

`summarize_coalescing.py` 只使用 `result_status=ok` 的行计算 TTFT。stats error
倾向于出现在压力更高、preemption 更多的请求中，删除这些行会系统性美化尾延迟。

| Profile 与 Update interval | 当前 ok-only P95 | 所有首 token请求 P95 |
| --- | ---: | ---: |
| immediate-controller，0.01 s | 1,347 ms | 2,722 ms |
| immediate-controller，0.2 s | 1,143 ms | 1,836 ms |
| fixed-window，0.2 s | 2,119 ms | 3,044 ms |

旧结果可以恢复真实 TTFT，但不能继续用于 recovery-safe 的 token lifecycle 结论。

## 高 QPS 如何触发 recovery

```text
更多逻辑请求按 offered QPS 到达
    -> replay 的 max_concurrency 限制同时活跃的 HTTP 请求
    -> 多个长期 Prompt Stream Request 同时持有 PA / UPDATE KV
    -> event 在 Request 内持续产生 Prefill work
    -> 计算 backlog 与 KV residency 同时增加
    -> allocate_slots() 返回 None
    -> 先驱逐 dormant / retained-waiting KV
    -> 仍不足时 preempt running Request
    -> Request.reset_prompt_stream_computed_state()
    -> 当前实现折叠历史 work 并重建最新状态
    -> lifecycle 计数与 replay 旧断言不一致
```

一个 event 不会变成新的 Scheduler `Request`。一个长期流式请求始终只有一个
`Request`，event 只增加或覆盖它的内部 work/window 状态。

QPS 控制逻辑请求到达速率，`max_concurrency` 控制客户端同时推进多少个请求。超出
并发上限的请求会在 semaphore 前等待，并计入 `queue_delay_ms`。现有回放常用
`max_concurrency=8`，P 节点常用 `MAX_NUM_SEQS=16`，因此不是 sequence 上限直接
拒绝请求。盲目提高 `MAX_NUM_SEQS` 可能让更多长期请求同时驻留 KV，反而恶化问题。

已有 P 日志在 QPS 4 阶段多次出现 99.4% 到 100% 的 GPU KV Cache 使用率，说明
真实资源压力存在。但当前 CSV 没有逐请求 preemption counter，所以只能说代码路径、
错误分布和日志高度一致；补齐可观测性后才能逐请求闭环证明。

## P0：Request recovery 的生命周期问题

### 多个 admission 被静默折叠

核心位置是 `vllm/v1/request.py` 的
`Request.reset_prompt_stream_computed_state()`。它当前会：

1. 收集 `active_work_item` 和 `pending_work_items`。
2. 用 `recovery_update_completion_pending: bool` 判断最新快照是否有未完成 admission。
3. 清空 active 和全部 pending work。
4. 将物理进度归零。
5. 重建“完整 Append -> 最新 Update -> Question”。

例如 preemption 前为：

```text
U1: completed
U2: active
U3: pending
```

latest-only recovery 可能变为：

```text
U1: 历史 completed 仍保留
U2: 被折叠，但没有 terminal 状态
U3: 最多通过一个 bool 继承 completion
```

最后得到 `admitted=3, completed=2`。这不必然说明 Question 使用了错误快照；
当前 recovery 会重建最新逻辑状态，decode 前也仍有 barrier。真正缺失的是 U2 已被
recovery 放弃，却没有 `cancelled_by_recovery` 终态。

以下两个 bool 无法表达多个 admission 的身份和状态：

```python
PromptStreamWorkItem.count_as_admitted_update_work
PromptStreamState.recovery_update_completion_pending
```

它们无法回答：

- recovery work 延续的是哪个 admission。
- 哪些 pending/active admissions 被折叠。
- 同一 admission 的多次 physical attempts 如何计数。
- recovery 中到达新 Update 时，旧 admission 如何终止。

而且 `_apply_prompt_stream_update()` 在 recovery work 完成前收到新 Update 时会清掉
`recovery_update_completion_pending`，最后一个旧 admission 也可能无声丢失。

### Recovery rebind 被误记成 policy admission

`flush_prompt_stream_update()` 当前将以下情况视为新 admission：

```text
target 与 stats_last_admitted_token_ids 不同
或者 target 相同但 Append epoch 不同
```

epoch 是执行依赖的一部分，本身没有问题。问题在于 recovery 重新走了普通 policy
flush：reset 清空物理 `last_admitted_*`，把 source 最新快照重新放入
`latest_pending_token_ids`，Scheduler 随后再次 flush。

`append_gate=none` 下，recovery 时 Append epoch 可能已经前移。同一个 Update target
因此再次增加 `update_works_admitted` 和 admitted tokens，却没有新 accepted event，
最终出现 `admitted > accepted`，并污染 saved/extra。

已有 CSV 甚至出现：

```text
accepted = 1
eager tokens = 1531
admitted tokens = 3062
extra tokens = 1531
```

这部分是 recovery/rebind 成本，不能记作新的 policy admission。

### Recovery 破坏 bounded window

reset 还会重置逻辑 coalescing window：

- `first_pending_time` 与 `last_update_time` 改成 recovery 时刻。
- soft/hard/TTFT deadline 被清空后重新计算。
- sticky `flush_ready` 被清除。
- pending count 与 hypothetical eager 被重建。

repeated preemption 会不断向后移动 hard deadline；已经 policy-ready、只被 gate 阻挡
的窗口也可能重新变成未 ready。这是实际调度语义错误，不能只通过放宽 replay
断言解决。

### 空 Update 与“没有 pending”混淆

recovery 使用类似下面的真假值判断：

```python
tuple(update_token_ids) if update_token_ids else None
```

它混淆了：

- `None`：没有 pending Update target。
- `()`：合法 Update 将快照缩短为空，需要零-forward truncation 并结算 admission。

后续必须使用显式状态或 sentinel 区分两者。

## P0：Replay 的状态和断言问题

### 旧断言不适用于 recovery

`research/scripts/replay/segmented_prefill/replay_common.py` 当前强制：

```text
admitted <= accepted
completed <= admitted
正式回放 completed == admitted
scheduled >= admitted
```

`completed <= admitted` 仍可作为基本保护；其余关系在当前 recovery 语义下不成立：

- 折叠旧 admission 时，`completed < admitted`。
- recovery rebind 被误计时，`admitted > accepted`。
- admission 未调度就被折叠时，`scheduled < admitted`；重复 replay 又可能使
  `scheduled > admitted`。

因此不能再用下面的差值推导 replay overhead：

```text
max(0, scheduled_update_prefill_tokens - admitted_update_prefill_tokens)
```

取消和 replay 会向相反方向影响差值，二者可能互相抵消。recovery tokens 必须按
execution origin 单独累计。

### Stats 异常覆盖了首 token成功

`replay_prompt_stream.py` 当前先成功消费 SSE 并写入 TTFT，再调用 stats 校验。校验
抛出的 `ReplayHttpError` 被外层网络异常一起捕获，使请求保持
`result_status=error`，还会尝试 abort 已经结束的请求。

`summarize_coalescing.py` 又只用 `result_status=ok` 的行计算 TTFT、吞吐和 counters，
所以同时造成 TTFT selection bias 和 stats coverage 偏差。

## P1：Scheduler 的真实压力点

### Dormant 与 retained-waiting KV

`scheduler.py` 允许无 runnable work 的 Prompt Stream Request 进入 dormant 并保留
segment KV。唤醒但尚未进入 running 的请求若仍持有 KV，会进入
`_prompt_stream_retained_waiting_req_ids`。

所以日志中的 `Running=0, Waiting=0` 不代表没有 Prompt Stream KV 驻留。当前日志
没有显示 dormant 请求数和持有 block 数。

### Allocation failure 路径

running 的 `allocate_slots()` 失败后：

1. 先驱逐 dormant 或 retained-waiting 请求。
2. 仍不足时 preempt running 请求。
3. 一旦本轮发生 running preemption，本轮不会继续 admission waiting 请求。

waiting admission 分配失败时也先驱逐 dormant；仍失败后直接 `break`，可能造成
head-of-line blocking。当前数据不足以证明它是主瓶颈，应先加指标再调整 FCFS。

### Victim 选择缺少成本模型

`_prompt_stream_retained_waiting_req_ids` 是 `set`，遍历顺序不稳定。dormant 路径也
没有综合以下信息：

- 持有 block 数和预计 replay tokens。
- 距 hard/TTFT deadline 的距离。
- 是否已收到 `PROMPT_FINISHED`。
- 最近活动时间和累计 preemption 次数。

当前可能把计算最多、最接近 finish 的请求先驱逐，放大重算和尾延迟。

### Adaptive load 会低估 dormant 压力

`_get_prompt_stream_prefill_queue_load()` 只使用 running、waiting、skipped-waiting
请求数，没有计入 dormant 持有的 blocks 和 Prefill backlog。它不是当前 fixed/
debounce 失败的直接根因，但会影响下一阶段 adaptive policy。

## 设计原则：拆成三层状态

### Source 层

```text
latest_source_token_ids
update_events_accepted
```

只由通过协议校验且内容真实变化的 Update 事件推进。preemption、epoch rebind 和
KV replay 都不能增加 accepted。

### Policy admission 层

```text
last_policy_admitted_token_ids
last_policy_admitted_append_epoch
next_update_admission_id
outstanding_admissions
```

跨 preemption 不回退。每个真正 policy flush 最多创建一个稳定 admission ID。
recovery 不得重新增加 policy admission 或 policy admitted tokens。

### Physical execution 层

```text
update_committed_token_ids
update_committed_append_epoch
physical_execution_base
execution_attempt_id
execution_origin
```

该层在 preemption 后归零，再由 recovery work 重建。同一个 policy admission 可以有
多个 physical attempts。

## Admission 数据结构

建议增加轻量 ledger：

```python
class UpdateAdmissionStatus(Enum):
    PENDING = auto()
    ACTIVE = auto()
    COMPLETED = auto()
    CANCELLED_BY_RECOVERY = auto()


@dataclass
class UpdateAdmissionRecord:
    admission_id: int
    target_token_ids: tuple[int, ...]
    required_append_epoch: int
    policy_prefill_tokens: int
    status: UpdateAdmissionStatus
```

只需为未终结 admission 保存 target；终结历史可压缩为 counters 和调试事件。

`PromptStreamWorkItem` 增加：

```python
update_admission_id: int | None
execution_attempt_id: int | None
execution_origin: Literal["policy", "recovery"]
recovery_reason: PromptStreamRecoveryReason | None
```

work 完成时按稳定 ID 结算，逐步移除两个含义不足的 bool。

## Recovery 语义选择

正常无 KV 丢失路径继续保持原强不变量：

> Update 一旦被 coalescer admission 并进入 FIFO，就不再被后续普通 Update 合并、
> 替换或取消。

资源 preemption 是额外语义边界，有两种实现。

### strict FIFO recovery

按原 admission 顺序 replay 所有尚未完成 work。

优点是完全保持 `admitted == completed`；缺点是 KV 已丢失后要重建 execution base
并重放过时快照，Append/Update 历史 epoch 也需要更多状态，高压下可能形成
preemption/replay 正反馈。

### latest-only recovery

保留当前恢复最终逻辑状态的方向，但显式记录折叠：

- 最新且仍需物化的 admission 保留原 ID 进入 recovery work。
- 其余 outstanding admissions 标记 `CANCELLED_BY_RECOVERY`。
- 已 completed 但 KV 丢失时，创建 `admission_id=None` 的 recovery-only work。
- 未 admission 的最新 window 继续服从原 gate/deadline，不能被 recovery 提前放行。

第一版建议采用 latest-only。正常 Update 路径仍不取消 admission；只有明确的 KV
resource recovery 可以取消，并必须计数。后续可增加 `recovery_mode=strict_fifo`
作为消融或 debug 模式。

## latest-only recovery 算法

### 1. 冻结逻辑状态

清空物理 work 前保存：

- source 最新完整 Update，允许为 `()`。
- 未 admission window 的 target、first/last time、绝对 deadlines、sticky-ready、
  EWMA 和 pending count。
- active/pending work 的 admission IDs、targets、状态与顺序。
- 当前 committed Update snapshot 与 Append epoch。

### 2. 结算 admissions

1. 若存在更新的未 admission window，取消已过时的 outstanding admissions；恢复 PA
   后继续等待原 policy，不提前 admission。
2. 若无未 admission window但有 outstanding admissions，保留最新需要物化的 ID，
   其余标记 cancelled。
3. 若所有 admissions 已 completed，只是 KV 丢失，创建 recovery-only work。

每个 admission 只能进入一次 `COMPLETED` 或 `CANCELLED_BY_RECOVERY`；重复 preemption
不能重复结算。

### 3. 重建物理链

```text
完整 PREFIX_APPEND recovery
    -> 必要的 UPDATE recovery
    -> 原 window 到期后产生的新 UPDATE work
    -> 显式 QUESTION work
```

recovery 使用新 attempt ID，但复用旧 admission ID 或使用 `None`。它只增加 physical
attempt 和 recovery token，不增加 policy admission。

### 4. 保留 window

recovery 不得修改：

- `latest_pending_token_ids`
- `first_pending_time`、`last_update_time`
- soft/hard/TTFT absolute deadline
- sticky `flush_ready` 与 ready reason
- EWMA 与观测数

若原 deadline 已经过期，Scheduler 下一次 reconcile 必须立即兑现。

### 5. 保持 Question barrier

Question 只有在以下条件满足后才能执行：

- sources 已按协议封闭。
- final Append epoch 已物化。
- 没有未 admission 的 ready window。
- 没有 outstanding policy admission。
- source 最新 Update 已物化。

decode 前验证：

```text
question_work_completed
implies
prefix_append_committed_logical_len == final_prefix_append_logical_len
and committed_append_epoch == final_append_epoch
and update_committed_token_ids == latest_update_source_token_ids
and no outstanding_update_admission
```

## Stats schema v2

### Policy counters

```text
update_events_accepted
update_policy_works_admitted
update_policy_works_completed
update_policy_works_cancelled_by_recovery
update_policy_works_outstanding
eager_update_prefill_tokens
policy_admitted_update_prefill_tokens
policy_saved_update_prefill_tokens
policy_extra_update_prefill_tokens
```

成功结束时：

```text
policy_admitted = policy_completed + policy_cancelled_by_recovery
policy_outstanding = 0
```

strict FIFO 下 cancelled 必须为 0。只要 admission 仅由 accepted Update window flush
创建，还可验证 `policy_admitted <= update_events_accepted`。recovery/epoch rebind 不算
policy admission。

### Physical counters

```text
update_execution_attempts_started
update_execution_attempts_completed
update_execution_attempts_preempted
update_recovery_works_created
update_tokens_scheduled_total
update_tokens_scheduled_for_recovery
prefix_append_tokens_scheduled_for_recovery
update_tokens_discarded_by_preemption
prefix_append_tokens_discarded_by_preemption
```

成功结束时：

```text
attempts_started = attempts_completed + attempts_preempted
```

`scheduled_update_prefill_tokens` 可以保留为兼容字段，但必须明确是所有 origin 的
总量，不能再与 policy admitted 做大小断言。

### 语义失效与资源浪费

当前 `update_tokens_invalidated` 只表示 snapshot/LCP 变化：

```text
update_tokens_invalidated_by_snapshot
```

preemption 丢失有效 KV 另记：

```text
update_tokens_discarded_by_preemption
prefix_append_tokens_discarded_by_preemption
```

论文应分别报告 policy saving、physical scheduled/recovery、snapshot invalidation 和
resource discard，不能混成一个 wasted token。

### Pressure counters

```text
prompt_stream_running_preemptions
prompt_stream_dormant_evictions
prompt_stream_retained_waiting_evictions
prompt_stream_allocation_failures_running
prompt_stream_allocation_failures_waiting
```

allocation failure 的结构化事件至少记录 requested/free/watermark blocks、running/
waiting/dormant/retained counts、dormant held blocks、victim blocks 和预计 replay
tokens。

`PromptStreamTokenStats` 增加 `metrics_schema_version=2`。过渡期可保留旧字段，但它们
必须映射到 policy 层。现有 `prompt_stream_stats` 是字典透传，主传输链无需为每个新
标量单独改协议。

## Replay 状态模型

`RequestMetric` 建议新增：

```text
transport_status
first_token_received
stats_validation_status
stats_validation_error
formal_result_status
request_error_stage
metrics_schema_version
```

处理规则：

1. 网络异常只更新 transport 与 error stage。
2. 收到首 token 后立即保留 TTFT，后续 stats 问题不能清除。
3. stats 校验返回结构化结果，不抛到网络异常层。
4. 已经完成 SSE 的请求不再 abort。
5. `--fail-fast` 区分 transport fail-fast 与 formal-data fail-fast。

Summary 至少输出：

```text
num_first_token_success / first_token_success_rate
num_transport_error / transport_success_rate
num_stats_valid / stats_valid_rate
num_formal_valid / formal_valid_rate
```

TTFT 使用所有 `first_token_received=True` 请求；policy/token 指标仅使用 stats valid
请求，并同时报告 coverage；formal trade-off 使用首 token、配置与 stats 全有效请求。

旧 CSV 可兼容恢复 TTFT：`finish_to_first_token_ms` 非空且 `completion_chunks>0` 即
推断首 token成功；旧 lifecycle error 标为 `legacy_stats_semantics_incompatible`，
不能用于 recovery-safe token 结论。

## P1：压力缓解方向

### 显式 victim policy

将 retained-waiting 的无序 set 与 dormant 简单顺序统一为可测试候选结构。victim
score 可综合：

```text
释放 blocks 越多，越倾向回收
replay token 成本越高，越不倾向回收
deadline 越近，越不倾向回收
已收到 PROMPT_FINISHED，越不倾向回收
短时间内已多次 preempt，越不倾向再次回收
```

第一版不必求最优公式，但顺序必须确定并可记录。

### Dormant KV block 预算

增加默认关闭的 block 高水位，例如
`prompt_stream_dormant_kv_max_fraction` 或
`prompt_stream_dormant_kv_max_blocks`。超过预算时主动回收低成本 victim，避免等到
`allocate_slots()` 临界失败后逐个被动驱逐。

### Waiting head-of-line

先统计队首因 blocks 不足连续停留的轮数、后续可 fit 请求数和等待时间。只有确认
HOL 占比明显后，再考虑转入 `skipped_waiting`，并必须加入 aging 防止大请求饥饿。

### Adaptive load

后续可观测：

```text
runnable_load = runnable_requests / max_num_running_reqs
kv_load = used_kv_blocks / total_kv_blocks
backlog_load = pending_prefill_tokens / token_budget
effective_load = max(runnable_load, kv_load, backlog_load)
```

第一阶段只记录，不立即改 adaptive 决策。

## 逐文件修改计划

### `vllm/v1/request.py`

- [ ] 增加 admission ID、attempt ID 和 lifecycle ledger。
- [ ] 分开 source、policy 和 physical chain。
- [ ] recovery 不得增加 policy admitted work/token。
- [ ] 保留原 window、absolute deadline、sticky-ready 和 EWMA。
- [ ] 区分空 Update `()` 与没有 pending `None`。
- [ ] 按 recovery mode 结算 completed/cancelled/outstanding。
- [ ] 扩展 schema v2 stats。
- [ ] decode 前验证 final Append、latest Update 和 Question barrier。

### `vllm/v1/core/sched/scheduler.py`

- [ ] reset 时传入 running/dormant/retained/cache-reset recovery reason。
- [ ] 记录 allocation failure 与 KV pressure snapshot。
- [ ] 接收 Request 的 recovery accounting delta。
- [ ] recovery 后重新注册原 absolute deadline，不重新计时。
- [ ] 后续实现确定性 victim policy 和可选 dormant block budget。
- [ ] 观测 HOL 后再决定是否跳过 waiting 队首。

### KV Cache 管理层

- [ ] 暴露每个 Prompt Stream segment 持有的 block 数。
- [ ] PA 与 UPDATE 的释放统计分开。
- [ ] 资源驱逐不计为 snapshot invalidation。

### Engine 输出、REST 与 simple test

- [ ] 在现有 `prompt_stream_stats` 中加入 schema v2 字段。
- [ ] 保持 P -> proxy -> replay 原样透传，proxy 不重算。
- [ ] 更新 `research/scripts/simple_test/prompt_stream_stats.py` 校验新守恒式。
- [ ] abort/error 通过 partial stats 或结构化日志保留压力信息，但不伪造成功 finish。

### Replay 与分析脚本

- [ ] `segmented_prefill/replay_common.py` 解析 policy/physical/recovery counters。
- [ ] 用新守恒式替换 `completed==admitted` 和 `scheduled>=admitted`。
- [ ] stats 校验不覆盖 transport 状态。
- [ ] `replay_prompt_stream.py` 拆分网络与 stats 异常边界。
- [ ] `summarize_coalescing.py` 分别汇总四类成功/有效率。
- [ ] TTFT 使用所有首 token成功请求。
- [ ] token 图显示 stats coverage，不使用 ok-only 高压样本。
- [ ] 更新 update-coalescing README 的旧验收描述。

## 测试设计

### Request 纯状态测试

- [ ] `U1 completed + U2 active + U3 pending` 后 preemption。
- [ ] 多个 pending admissions 在 latest-only 下只取消一次。
- [ ] strict FIFO 下全部按原 ID 完成且 cancelled 为 0。
- [ ] completed Update 丢 KV 后只增加 recovery attempt/token。
- [ ] active Update 部分计算后 preempt，正确统计 discard/replay。
- [ ] recovery 中到达新 Update，旧 admission 不会无声丢失。
- [ ] Append epoch 改变后恢复相同 target，不重复 accepted/admitted。
- [ ] pending window 遇到 preemption，deadline 和 sticky-ready 不变。
- [ ] deadline 已过期时 recovery，下一次 reconcile 立即 flush。
- [ ] repeated preemption 不重复结算。
- [ ] 非空 Update 缩短为空后 recovery，零-forward work 正确结算。
- [ ] disabled/enabled 两条路径满足各自 lifecycle 不变量。
- [ ] PROMPT_FINISHED 在 recovery 前后到达，Question barrier 始终成立。

现有 `test_preemption_replays_all_segments` 只验证物理视图归零，没有覆盖 lifecycle、
window、epoch 与 repeated recovery。

### Scheduler 压力测试

- [ ] 使用 fake/tiny KV pool 让 running `allocate_slots()` 返回 `None`。
- [ ] 验证 dormant、retained-waiting、running 的回收顺序和 reason。
- [ ] 验证 worker removal、queue membership 与 segment blocks 不泄漏。
- [ ] 验证确定性 victim 顺序。
- [ ] 验证 repeated failure 不重复取消 admission。
- [ ] Question 前验证 final Append epoch 与 latest Update 已 committed。

### Replay 解析测试

- [ ] 首 token成功但 stats invalid：TTFT 保留，transport 成功，stats 失败。
- [ ] start/event/finish HTTP 错误准确记录 stage、status 与 body。
- [ ] SSE timeout 和 stats missing 分开。
- [ ] schema v1 进入 legacy 兼容路径。
- [ ] schema v2 latest-only 满足 completed+cancelled 守恒。
- [ ] schema v2 strict FIFO 满足 completed==admitted。
- [ ] TTFT 样本数等于 first-token success，而非 formal-valid。

## 后续压力验证

核心状态与 CPU 测试通过后，再固定 sample manifest、arrival seed、并发和 source
interval，运行：

```text
QPS = 2.0, 2.5, 3.0, 3.5, 4.0
```

每个点同时报告 offered QPS、achieved first-token throughput、queue delay、四类成功/
有效率、preemption/eviction、policy saving、recovery scheduled/discarded、snapshot
invalidated 和 TTFT P50/P95/P99。

只有 stats coverage 足够且 recovery counters 守恒后，高 QPS token saving 与 TTFT
trade-off 才能作为论文结果。

## 实施顺序

1. [ ] 冻结 recovery mode 与 schema v2 术语。
2. [ ] 先拆 replay transport/stats 状态，恢复现有结果的真实 TTFT。
3. [ ] 在 Request 加 admission ledger，修复 lifecycle。
4. [ ] 保留原 coalescing window 与 absolute deadlines。
5. [ ] 增加 preemption、eviction 和 recovery token 观测。
6. [ ] 更新 validator、summary、图表与 README。
7. [ ] 完成无 GPU 的 Request/Scheduler 测试。
8. [ ] 小规模 P 节点验证 schema 和 Question barrier。
9. [ ] 重跑 QPS 压力矩阵。
10. [ ] 根据 pressure 数据实现 victim policy 与 dormant KV budget。

## 验收标准

- [ ] decode 前 final Append、latest Update 和 Question barrier 全部成立。
- [ ] recovery 不重新开始 hard/TTFT deadline，不清除 sticky-ready。
- [ ] empty Update 与 repeated preemption 不丢状态或重复计数。
- [ ] policy admission、physical attempt、recovery replay 三层不混用。
- [ ] latest-only 终态满足 `admitted=completed+cancelled`。
- [ ] strict FIFO 终态满足 `admitted=completed` 且 cancelled 为 0。
- [ ] snapshot invalidation 与 preemption discard 分开。
- [ ] 首 token成功的请求不会因 stats invalid 从 TTFT 中消失。
- [ ] 图表报告 TTFT 样本数和 stats coverage。
- [ ] QPS 4 可逐请求关联 preemption/eviction 与 recovery counters。
- [ ] 默认未启用扩展时不改变普通 Scheduler 主路径。

## 当前不能直接下的结论

- 不能把 QPS 4 的 9.53% `result_status=error` 称为服务失败率。
- 不能逐请求证明 2,467 条 stats error 全由 preemption 触发；当前缺 counter。
- 不能用旧结果的 ok-only TTFT 评价高压尾延迟。
- 不能用 `scheduled-admitted` 推导 recovery overhead。
- 不能把 4 个 event 502 直接归因于队列压爆，仍需对应 P/proxy 日志。
- 不能在 schema v2 修复前把 QPS 4 saved/extra 作为正式论文结论。

## 暂不纳入第一轮修改

- 不立即实现 adaptive 的新预测公式。
- 不立即改变 vLLM 全局 FCFS/priority policy。
- 不把正常 finish/abort 释放 KV 记为 preemption waste。
- 不把 block 空余容量或 PA padding 记为 invalidated。
- 不在 proxy 端重新 tokenize 或推导 Engine 计数。
- 不在缺少 pressure 数据前直接提高 `MAX_NUM_SEQS` 或并发。
