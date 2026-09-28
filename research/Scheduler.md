# 流式请求的调度、停放与唤醒

## 核心规则

一个 PromptStream 是一个长期 `Request`。Append、Update 和最终 Question 是该请求内部的
work，不会各自创建一个全局调度请求。

**所有 PromptStream 都使用同一套 dormant（停放）规则，与是否开启 Update 合并无关。**
首次提交的请求进入 `waiting`；有 work 时参与正常调度；当前 work 全部完成、只等待后续输入
或合并 deadline 时离开 `running`；产生新 work 后先回到 `waiting`，再竞争运行名额。

这解决的是空闲流式请求占用 `MAX_NUM_SEQS` 名额的问题。停放会保留已经计算的 KV，
**不会自动释放 GPU 显存**，也不代表请求已经完成。

## 全局调度队列与请求内 work

### Scheduler 管理 Request

| 容器 | 保存的请求 | 如何离开 |
| --- | --- | --- |
| `waiting` | 新请求、需要恢复计算的被抢占请求，以及从 dormant 唤醒的请求。 | 运行名额、token budget、KV 等条件允许时准入 `running`。 |
| `running` | 已准入、正在计算或有后续计算要调度的请求。 | 完成、取消、抢占，或耗尽当前 work 后停放。 |
| `skipped_waiting` | 暂时受阻的等待请求，例如等待远端 KV、grammar、原生 streaming input，或本轮暂时无法准入。 | 对应阻塞条件解除后重新尝试调度。 |
| `_prompt_stream_dormant_reasons` | 模板已完成且当前没有 work，只等外部输入或 Update 策略条件的 PromptStream。 | 新 work 产生时回到 `waiting`；取消或完成时清理；显存回收时进入恢复流程。 |

`dormant` 是 Scheduler 独立维护的集合，不是新的 `RequestStatus`。停放请求的 status 可以
是 `WAITING`，但它不在普通 `waiting` 或 `skipped_waiting` 队列内。判断实际位置必须看
容器归属，不能只看 status。

`Scheduler.requests[request_id]` 继续保存同一个请求对象，因此后续 REST event、finish 和
abort 仍能找到它。逻辑未完成数量包括 dormant 请求；即时可运行数量不包括它们。

### Request 管理内部 work

| 状态 | 含义 | 是否要求请求参与调度 |
| --- | --- | --- |
| 模板 bootstrap 未完成 | 固定模板的 KV 尚未准备好。 | 是，先完成模板。 |
| `active_work_item` | 当前已选择、尚未完成的 Append、Update 或 Question work。 | 是，包括分多轮计算的情况。 |
| `pending_work_items` | 已准入的 FIFO 工作链，还未轮到执行。 | 是，不能因暂时没有新事件而停放。 |
| `latest_pending_token_ids` | 合并窗口内尚未放行的 Update 目标。 | 单独存在时不要求 GPU 调度；达到策略条件才创建 work。 |
| 可以 decode | 最终问题等依赖已经完成，可以生成。 | 是。 |

这里两个 pending 的层级不同：合并窗口里的 pending 快照可以覆盖；
`pending_work_items` 里的冻结 work 已经承诺执行。普通 `waiting` 则是 Request 级别的
资源等待队列，不是这两个内部队列的替代品。

## 停放条件与在途计算

`Request.should_make_prompt_stream_dormant()` 判断逻辑条件：

1. 请求具有 `PromptStreamState`。
2. 固定模板 bootstrap 已完成。
3. `active_work_item` 为空。
4. `pending_work_items` 为空。
5. 当前还不能进入 decode。

Scheduler 还检查请求的生命周期和 `_prompt_stream_inflight_steps`：**已经提交给 worker 的
步骤必须先返回，才能停放。** 这是为了区分“本轮 token 已被调度”与“对应 KV 已经计算完成”。
即使 scheduler 已经前移进度计数，也不能在 GPU 结果尚未处理时删除 worker row、释放运行名额
或把仍在写入的 KV 当成可安全驱逐的缓存。

因此，“当前没有新事件”不是充分条件。已有 work 没算完、恢复 work 没算完，或者 GPU 还有
在途步骤时，请求继续参与调度；token budget 暂时耗尽也不等于应该进入 dormant。

另有一种零 forward 的情况：准备 execution view 时，某个 work 已能直接完成状态转换，
例如目标快照不需要新增计算。Scheduler 会在 prepare 后再次 reconcile；若工作链已耗尽，
可以直接停放，不必保留到下一轮空调度。

满足条件后，`Scheduler._make_prompt_stream_request_dormant()`：

- 将请求从 `running`、`waiting` 或 `skipped_waiting` 中移出，登记停放原因。
- 保留 Request 的来源内容、work 状态和分段 KV block 所有权。
- 登记 worker row removal，让后续维护步骤移除 worker persistent batch 中的请求行。

worker row removal 不等于释放 KV。它只移除暂时不需要参与 batch 的执行视图；维护步骤本身
仍被视为 EngineCore 可执行工作，不会因为所有请求都停放而永远得不到处理。

## 事件、执行反馈与 deadline 如何唤醒请求

`Scheduler._reconcile_prompt_stream_request()` 将“是否形成新 work”和“是否该停放”衔接起来。
开启合并器时，它还检查 Update flush、最终 Question 准入和 deadline；关闭时仍进行共同的
停放与唤醒处理。

| 到达的内容或时刻 | 产生的工作 | dormant 中的处理 |
| --- | --- | --- |
| 有效 Append delta | 新的 `PREFIX_APPEND` work。 | 唤醒进入 `waiting`。 |
| 关闭合并时的有效 Update | 按原即时路径创建 Update work。 | 唤醒进入 `waiting`。 |
| `immediate` Update | controller 立即放行的 Update work。 | 唤醒进入 `waiting`。 |
| 尚未达到策略条件的 Update | 只修改合并窗口，尚无 work。 | 继续停放；定时策略维护 deadline，adaptive 登记执行反馈候选。 |
| adaptive 背压解除且 token / slot 压力都低于阈值 | 冻结并放行最新 Update work。 | 经 waiting 重新准入，不需要新事件或定时器。 |
| 有界窗口 deadline 到达 | 冻结并放行 Update work。 | 唤醒进入 `waiting`。 |
| Update source-finished 或 `PROMPT_FINISHED` | 按既有规则放行最后窗口，并在依赖满足时准备 Question。 | 有可执行工作则唤醒。 |
| 重复事件、无变化快照或仅记录来源结束 | 可能不产生 work。 | 不因收到 HTTP 请求本身而唤醒。 |
| abort | 取消请求。 | 清理状态，结束生命周期。 |

`Scheduler._wake_prompt_stream_request()` 删除 dormant 标记后，通过原有等待队列入口入队。
恢复顺序始终是 **dormant → waiting → running**，不能直接绕过资源准入回到 `running`。
如果资源充足，这两次转换可能发生在同一轮调度里。

唤醒后，已有 KV 仍可复用。Scheduler 将仍持有分段 KV 的等待请求登记到
`_prompt_stream_retained_waiting_req_ids`，并要求 worker refresh；真正进入 running 后清理
retained 标记。等待期间如发生显存回收，请求会改走恢复流程。

### 一个没有开启合并的例子

| 时刻 | 请求内部状态 | Scheduler 位置 |
| --- | --- | --- |
| `start` 提交模板 | bootstrap 尚未完成。 | `waiting`，准入后 `running`。 |
| 模板结果返回，后续事件还未到达 | bootstrap 完成，无 active/pending work。 | `dormant`。 |
| Append A 到达 | 新建 A 的 work。 | `waiting`，资源允许后 `running`。 |
| A 的部分 token 完成 | active work 仍未完成。 | `running`，继续计算。 |
| A 全部完成，但 Update 尚未到达 | 无 work、尚未结束。 | `dormant`。 |
| Update U 到达 | 即时创建 U 的 work。 | `waiting`，然后 `running`。 |
| U 计算期间 `PROMPT_FINISHED` 到达 | U 继续执行，Question 遵守原有最终依赖。 | 保持可运行，不因收到 finish 就提前 decode。 |
| 最终问题完成并开始生成 | 已允许 decode。 | `running`，直到终止。 |

所有 PromptStream 都沿用显式 QUESTION work 的 FIFO 路径；Question 必须等待此前知识
工作完成并满足最终依赖，才能执行和生成。统一停放不会让 Question 越过这条 barrier。

## 抢占、显存回收与恢复

停放和抢占的资源含义不同：

| 行为 | KV 是否保留 | 后续计算 |
| --- | --- | --- |
| 正常停放 dormant | 保留 `PREFIX_APPEND` 和 `UPDATE` blocks。 | 唤醒后继续使用现有 KV。 |
| running preemption | 被回收的 KV 释放。 | 根据权威来源内容重建执行状态，进入 waiting 等待恢复。 |
| dormant eviction | 停放请求的 KV 释放。 | 重建恢复工作，再参与 waiting/running 调度。 |
| retained-waiting eviction | 尚未重新准入但仍持有的 KV 被释放。 | 改为恢复计算，不能继续声称此前物理进度仍有效。 |

恢复工作也属于 work，因此即使没有新事件，被抢占请求仍可能需要重新计算模板、Append 和
Update。是否再次停放以重建后的执行状态为准；没有任何恢复工作时才允许重新停放。

合并窗口中的快照、原始 deadline 和窗口 token 统计 按原恢复规则保留；恢复 attempt 与策略 admission
分别统计，不能把重算误记为新的合并收益。已有 victim 选择和分段恢复方法沿用当前实现。

统一 dormant 不会解决“很多长期请求同时占用 KV”的显存瓶颈。它释放 running 名额后，新的
请求可能更容易准入，因此仍需观察总 KV 使用率、驱逐次数和恢复开销，不能把运行名额利用率
改善直接等同于显存占用下降。

## EngineCore 空闲、计数和关闭

EngineCore 区分“逻辑上还有请求”和“现在有事可做”：

- `get_num_unfinished_requests()` / `has_unfinished_requests()` 保留 dormant 请求，保证
  生命周期与 connector accounting 正确。
- `has_runnable_engine_work()` 不因纯 dormant Request 而持续执行空调度，但仍计算
  waiting/running、worker removal、已完成请求清理和 connector pending push。在途计数
  阻止请求过早离开 running；EngineCore 的 `has_work()` 还保留对 `batch_queue` 的检查，
  保证已提交 batch 的输出能够收尾。
- 关闭合并时，没有 Update deadline；纯等输入的 EngineCore 可以阻塞等待新的 ADD、ABORT
  或其他控制消息。
- 开启有界合并时，controller 的最小堆提供最近 deadline；输入队列最多阻塞到该时刻，随后
  同一主线程放行 work。`full_merge` 没有时间 deadline，等待来源或请求结束事件。
- `adaptive` 不设置时间 deadline。Scheduler 在事件、回包与调度机会变化时重新判断，
  EngineCore 在阻塞输入前也检查候选，避免纯 dormant 时永远等不到下一条事件。
  已放行 Update 的 pending/active 状态与 `num_update_inflight_steps` 共同构成执行背压。
- DP 协调同样区分逻辑未完成与当前 runnable work；纯 dormant 请求不会单独维持计算波次。
  合并配置原有的 DP 限制仍保留，这次没有新增完整异步或 DP 服务验证结论。
- shutdown 不再接受新的来源事件时，纯等待输入的 dormant 请求会被取消并清理，避免服务
  退出永远等待不会再到达的事件。取消后的 worker / connector 清理仍需完成。

日志中的 `Running` / `Waiting` 不等于逻辑未完成总数：当前 `make_stats()` 的
`num_waiting_reqs` 仍统计普通 waiting，skipped 另有字段。因此看到 `Running=0`、
`Waiting=0` 时，也可能存在仍持有 KV 的 dormant 请求；应结合
`get_num_unfinished_requests()` / `get_request_counts()` 理解请求数量。

controller 管理定时策略的 deadline，以及 adaptive 的候选集合和在途 token 需求，不拥有 dormant 集合。
`Scheduler._prompt_stream_dormant_reasons` 与 `_prompt_stream_inflight_steps` 为所有 PromptStream
服务。控制状态由 EngineCore/Scheduler 的既有线程顺序修改，不增加定时器线程或队列锁。

## 配置、实验与实现边界

不需要新增启动参数。`MAX_NUM_SEQS` 仍对应 `max_num_seqs`：它限制同时准入 running 的
Request 数量，不是整个服务的未完成请求总数，也不是每个请求的事件数或 KV block 数。

`disabled` 与 `immediate-controller` 现在具有一致的空闲停放规则，但不应据此宣称两者的
计算路径完全相同。现有 controller 的 Append epoch / Update LCP 检查、统计和 Question
准入方法仍保留；本次没有重新定义这些规则。

旧实验若使用“disabled 空闲仍占 running，controller 可停放”的版本，与新版本的队列行为
不完全相同。比较新策略时应使用同一代码版本重新采集结果；这份笔记不宣称已经完成新的
性能实验。

### 关键源码

| 位置 | 需要关注的接口或状态 |
| --- | --- |
| [Request 状态机](../vllm/v1/request.py) | `should_make_prompt_stream_dormant()`、active/pending work、bootstrap、decode barrier 与恢复状态。 |
| [Scheduler](../vllm/v1/core/sched/scheduler.py) | dormant/inflight/retained-waiting 集合，`_make_prompt_stream_request_dormant()`、`_wake_prompt_stream_request()`、`_reconcile_prompt_stream_request()`、驱逐与请求计数。 |
| [Scheduler 接口](../vllm/v1/core/sched/interface.py) | EngineCore 查询 runnable work、deadline 与 shutdown 清理的接口。 |
| [EngineCore](../vllm/v1/engine/core.py) | 空闲输入等待、GPU 步骤结果处理、deadline 唤醒与 shutdown。 |
| [完整设计总结](DONE.md) | 分段 KV、事件定义、Question barrier、恢复统计与实验结论。 |
| [Update 合并设计](Volatility-Aware_Bounded_Update_Coalescing.md) | 合并窗口、policy、deadline和不变的 admission 规则。 |

### 验证范围

针对性的状态机/调度测试应覆盖：关闭合并也能停放；bootstrap 和在途 GPU 步骤不能提前
停放；Append/Update/finish 产生 work 后只进入一次 waiting；重复事件和窗口覆盖不误唤醒；
抢占与 dormant KV 驱逐产生恢复工作；abort/完成删除停放状态；逻辑计数与 idle/shutdown
判定不丢请求；开启合并的 deadline 和 FIFO 行为继续成立。

本次 CPU 单元检查共 **61 项通过**，其中针对性测试 57 项：
[Request 测试](../tests/v1/test_prompt_stream_request.py) 38 项，
[Scheduler 测试](../tests/v1/test_prompt_stream_scheduler.py) 19 项。覆盖真实
`_update_after_schedule()` / `update_from_output()` 路径：两个在途步骤的计数从 2 到 1
时不能停放，降至 0 后才允许停放；另有 4 项已有普通 / Async grammar abort、
Masked Padding finish / abort 相关用例通过。

没有启动 vLLM 服务。单元检查只能验证这些状态转换，完整 P/D 服务、异步执行配置和高负载
资源行为仍应通过实际部署实验确认。
