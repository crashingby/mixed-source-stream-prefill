# 流式 Prompt Prefill 当前实现

本文记录当前代码已经实现的行为。描述范围从 `EngineCoreRequest` 转成长期
`Request` 开始，一直到 scheduler、worker KV block table、NIXL P/D transfer
以及实验 proxy。

当前实现已经删除普通连续 prompt 的 KV fallback。只要一个请求进入
prompt stream 模式，它就必须走 segment-aware 路径。

## 阶段性联调结果

`research/scripts` 下三个最小前端脚本曾在真实 P/D 分离链路上完成 smoke
test：

| 测试脚本 | 覆盖场景 | 当前结果 | 说明 |
| --- | --- | --- | --- |
| `append_frontend.py` | append-only 文本事件 | 通过 | 验证稳定 chunk 单调进入 `PREFIX_APPEND` 域，并能 finish 后请求 D decode。 |
| `update_frontend.py` | 单 update source 文本快照 | 通过 | 验证 update 完整快照、source-local LCP、`UPDATE` 域 prefill 和 finish decode 链路。 |
| `mixed_frontend.py` | append / update 交错事件 | 通过 | 验证 `append -> update -> append -> append -> update -> finish` 这类混合事件序列可以跑通。 |

这三个测试均走实验 proxy：

```text
frontend script
  -> research/scripts/proxy_server.py
  -> P /v1/prompt_stream/start
  -> P /v1/prompt_stream/event
  -> P /v1/prompt_stream/finish
  -> D /v1/completions
```

目前可以确认的工程闭环是：

- RESTFUL start / event / finish 能把文本事件送入 P 节点，并由 P 节点完成 tokenization。
- Append chunk 能进入 `PREFIX_APPEND` 物理 KV 域。
- Update snapshot 能进入 `UPDATE` 物理 KV 域，并使用 update 源内 LCP。
- finish 后最终用户问题能接在 update 后面继续 prefill。
- proxy 能拿到 P 返回的物理 prompt token ids 和 `kv_transfer_params`，再请求 D 节点 decode。
- append-only、update-only 和混合模式都已经能返回非错误 completion。

这些结果来自本轮 exact manifest、长度保护和 connector 生命周期修复之前，
只能说明基础 REST/P/D 链路曾经跑通，不能替代对当前代码的重新联调。本轮修改
已通过定向单元测试，但尚未在实时双 GPU 集群上重新验证。

即使重新联调通过，也仍然只是 smoke test，不等价于完整正确性证明。当前仍
保留这些近似和限制：

- `PREFIX_APPEND` 末尾 padding 会参与 attention。
- Append-after-Update 不强制重算已有 update KV，属于当前论文原型的近似语义。
- 当前只覆盖单 update source，尚未测试多 update source 或 source_finished 自动收尾。
- 当前测试主要验证链路和 block 排布没有直接崩溃，还没有系统化比较启用/禁用 NIXL 的输出一致性。

## 设计结论

一个逻辑请求只拥有两个物理 KV 域：

```text
request_id
├── PREFIX_APPEND: prefix + 所有 append + block 对齐 padding
└── UPDATE: 当前 update 快照 + 最终用户问题 + 后续 decode token
```

KV manager 内部使用两个 key：

| 物理域 | KV manager key | 生命周期 |
| --- | --- | --- |
| `PREFIX_APPEND` | 原始 `request_id` | prefix 固定，append 只在尾部增长。 |
| `UPDATE` | `request_id::prompt_stream::update` | update 后缀可失效；问题和 decode 接在其后。 |

Question 不再单独使用第三个 key。这样只有 `PREFIX_APPEND -> UPDATE` 一个
物理边界，Update 与 Question 之间可以沿用普通连续 block 分配。

## 当前 KV Cache block 管理总览

当前实现的核心选择是：**逻辑上有 prefix、append、update、question 四段，
物理 KV block 只分成两个域**。这样做是为了让 Append 可以持续尾部增长，同时让
Update 有独立的失效边界。

```text
逻辑 prompt:
prefix + append + update + question

物理 KV:
PREFIX_APPEND blocks = prefix + append + padding
UPDATE blocks        = update + question + decode
```

两个物理域仍然归属于同一个长期请求，但在 KV manager 内部使用不同 key 管理：

- **`PREFIX_APPEND` 使用原始 `request_id`**：prefix 固定，append 只允许尾部追加。
  这个域的 block 不会因为 Update 改变而释放。
- **`UPDATE` 使用派生 key `request_id::prompt_stream::update`**：Update 快照变化时，
  只在这个域里按 source-local LCP 截断；Question 和后续 decode token 也接在这个域后面。

申请 block 时使用 **segment-local 坐标**。例如 Append 阶段只看
`PREFIX_APPEND` 当前长度和已计算进度；Update 阶段只看 Update 域当前已经保留了
多少 token 的 KV。worker 真正执行时，再由 scheduler 拼出本轮需要安装的完整
block table。

### Append 阶段的 block 管理

Append 阶段只写 `PREFIX_APPEND` 域：

```text
worker token view  = PA_physical
worker block table = PREFIX_APPEND blocks
write segment      = PREFIX_APPEND
```

其中：

```text
PA_physical = prefix + append + padding
```

`padding` 是为了让 `PREFIX_APPEND` 的物理长度按 block size 对齐。这样 finish
时把 `PREFIX_APPEND blocks + UPDATE blocks` 直接交给 D 节点时，D 侧普通连续
block table 的位置计算不会把 Update 的第一个 token 落到 PA 的半空 block 里。

如果 Append 又来了新 chunk：

- 旧的 PA 逻辑 token 对应的 KV 保留。
- 旧 PA 尾部 padding 会被新的 append token 覆盖。
- 如果当前 PA blocks 容量不够，KV manager 继续给原始 `request_id` 申请新 block。
- 已经存在的 Update blocks 不释放、不重算，也不会出现在 Append 阶段的 worker view。

因此 Append-after-Update 的当前语义是：**Append 只推进 PA 域，Update 域保持原样**。
这符合本项目当前的近似设计，但不是严格完整 prompt 重算。

### Update 阶段的 block 管理

Update 阶段写 `UPDATE` 域，但读取两个域：

```text
worker token view  = PA_physical + U_new
worker block table = PREFIX_APPEND blocks + UPDATE blocks
write segment      = UPDATE
```

Update 的复用边界只由 Update source 内部决定：

```text
L = LCP(U_old, U_new)
C_U = min(旧 Update 已计算进度, L)
```

事件到达后：

- `PREFIX_APPEND` 的 blocks 和进度不变。
- `UPDATE` 域的已提交进度回退到 `C_U`。
- KV manager 只截断 `request_id::prompt_stream::update` 在 `L` 之后的 blocks。
- 下一轮 prefill 从 `U_new[C_U:]` 开始，前缀 KV 是 `PA_physical KV + U_new[:C_U] KV`。

这里的 `L` 是语义上允许保留的最长前缀，`C_U` 是实际上已经写入 KV 的安全复用进度。
如果旧 Update 还没算完，即使 LCP 更长，也不能跳过尚未真实写入的 KV。

### Question 和 decode 的 block 管理

`PROMPT_FINISHED` 到达后，Question 不创建第三个物理域，而是追加到 `UPDATE`
域后面：

```text
worker token view  = PA_physical + U + Q
worker block table = PREFIX_APPEND blocks + UPDATE blocks
write segment      = UPDATE
```

Question prefill 完成后，P 节点会采样出唯一的首 token。这个 token 属于后续
decode 序列，但它只是当前 forward 产生的输出，还没有作为输入经过下一轮
forward，因此 P 侧并没有它的 KV。当前实现只允许一种精确范围：

```text
remote_num_tokens = physical_prompt_len
D prompt len = physical_prompt_len + 1
```

D 先为 `physical_prompt_len` 个外部 token 申请和读取 blocks；KV load 完成后，
再在 D 本地 forward P 采样的首 token，并从它的 logits 继续生成第二个输出 token。

### finish 时交给 D 的 block 顺序

P 节点 finish 时导出的远端 block ids 固定按下面顺序拼接：

```text
remote_block_ids = PREFIX_APPEND block ids + UPDATE block ids
```

此时 `UPDATE blocks` 已经包含：

```text
当前 Update 快照 + Question
```

scheduler 会在调用 connector 前校验：

```text
len(remote_block_ids[group]) == ceil(physical_prompt_len / block_size)
```

如果两个 segment 拼起来不能形成 D 侧可解释的连续物理 block table，会直接失败，
不会把错位 block ids 交给 NIXL。

P 还会随 `kv_transfer_params` 导出权威 manifest：

```text
prompt_stream_exact_transfer = true
prompt_stream_physical_prompt_len
prompt_stream_remote_block_counts
prompt_stream_remote_block_sizes
prompt_stream_prefill_output_tokens
```

proxy、D scheduler 和 NIXL worker 会逐层校验同一份 manifest。worker 发现
local / remote / manifest 三方计数不一致时，只把当前请求标记为 KV load
失败并打印 request id，不再通过 `assert` 结束整个 EngineCore。exact 模式的
完整不变量是：

```text
remote_num_tokens = physical_prompt_len
D_prompt_len = physical_prompt_len + prefill_output_tokens
prefill_output_tokens in {0, 1}
block_count[group] = ceil(physical_prompt_len / block_size[group])
```

校验不只比较总 block 数，还会严格检查字段类型、KV group 数、每组 block
count、P/D block size、远端 block id，以及 physical / remote / D prompt
三种 token 长度。任何一项异常都按单请求 fail closed，不让错误 manifest
进入 NIXL read。

## 大规模回放暴露的问题与修复

### D 为什么会出现 prefix cache

D 收到的是普通 completion `Request`，并不带 P 侧的
`PromptStreamState`。原来的通用 NIXL 路径会先调用
`get_computed_blocks()` 查询 D 本地 APC，再调用
`get_num_new_matched_tokens()` 查询 P 的外部 KV；加载成功后还会把收到的
blocks 写回普通 APC。

这会造成两个问题：

- Update 的 KV 是本项目分段 attention 近似结果，不能和普通全量 prefill
  共用 token hash namespace。
- D 本地 prefix hit 会改变本次需要加载的 local block 数，而 P 的分段
  block table 是固定的，二者不能继续套用通用的尾部裁剪推断。

当前通过 `prompt_stream_exact_transfer` 识别 D 侧请求，并做了完整隔离：

- **Local Prefix cache** 是 D 自己的自动前缀缓存（APC）命中。exact
  prompt-stream 请求既不读取也不写入这个缓存，因此这类请求的 local hit
  固定为 0，也不会用分段近似 KV 污染普通 token hash namespace。
- prompt-stream 请求不受 NIXL `kv_recompute_threshold` 影响；短请求也会
  真正读取 P 的 KV。
- 加载后的近似 KV 不写入普通 APC，Request 结束时按普通私有 blocks 释放。
- **External prefix cache** 是通过 NIXL 从 P 加载的外部 KV 命中。该指标仍会
  增长，而且一次 exact transfer 可以接近 100%；它不表示 D 本地 APC 命中，
  也不和禁用 Local Prefix cache 矛盾。

### block 数不匹配的边界根因

失败样本曾出现：

```text
physical_prompt_len = 3664 = 229 * 16
P remote blocks     = 229
D prompt len        = 3665
```

旧代码把 P 采样的首 token 计入 `remote_num_tokens`，D 因而按 3665 个外部
token 申请 230 个 blocks，再和 P 的 229 个 blocks 进入
`_apply_prefix_caching()`，最终触发断言。

现在 `remote_num_tokens` 被强制覆盖为物理 prompt 长度 3664。D 的第一次
异步 load 只申请 229 个 blocks；首 token 留到 load 完成后本地 forward，
届时才按需要申请第 230 个 tail block。

### 超长请求和 unrecognized request

此前 REST event ack 只表示事件已经入队。超长事件在 scheduler 中结束长期
Request 后，随后到达的同 ID 事件会走 `add_request(existing is None)`，
错误地创建一个新 Request。旧失败生命周期的 `reqs_not_processed` 与新
生命周期的 `reqs_in_batch/reqs_to_send` 随后发生冲突，D 通知 P 时就会出现：

```text
Potentially invalid KV blocks for unrecognized request
```

当前使用四层保护：

- 数据构建阶段通过每个网页 Append chunk、每个 ANNS 文档和每个 Update
  快照的 token 预算控制样本长度，回放阶段不再提前查看完整请求长度。
- P serving 为每个活跃会话维护一个无 KV 的 Request 长度镜像；事件进入
  EngineCore 前检查实际阶段的最大运行序列上界。这个上界包含 segment
  padding 和为 P 首 token 预留的 `+1`，超限时使用 EngineCore 内部 request
  id 真正 abort 长期请求。
- Scheduler 为已经因超长结束的 request id 保存 terminal tombstone，拒绝
  core-only 后续事件复活。
- NIXL lifecycle 以当前待发送请求为准，丢弃同 id 的 stale
  `not_processed` 标记，避免旧失败生命周期覆盖新状态。

这里的 `+1` 是给 P 采样并交给 D 本地 forward 的首 token 预留序列位置。

## 逻辑状态和执行状态已经分离

`PromptStreamState` 保存长期逻辑事实：

- 固定 `prefix_token_ids`。
- 每个 Append / Update source 的当前 token、版本、序号和去重状态。
- 所有 Append source 共享一条按事件提交顺序增长的请求级 token 日志；交错
  到达的 source 不会在重建 prompt 时被重新分组。
- 最终用户问题 `suffix_token_ids`。
- 请求是否收到 `PROMPT_FINISHED`。
- `PREFIX_APPEND`、Update 和 Question 各自真正完成 prefill 的进度。

`PromptStreamExecutionView` 保存 worker 本轮必须共同使用的执行事实：

- 当前阶段是 `PREFIX_APPEND`、`UPDATE` 还是 `QUESTION`。
- 本轮写入哪个物理 KV 域。
- worker 需要安装哪些 segment block table。
- worker 本轮看到的完整物理 token ids。
- 本轮开始前的 `num_computed_tokens`。
- `UPDATE` 域在全局物理序列中的起点。

这两个对象不能合并。逻辑 prompt 始终是：

```text
prefix + append + update + question
```

物理执行 prompt 则可能包含对齐 padding，并且 Append 事件期间不会把已有
Update token 放进本轮 worker view。

## 为什么要有阶段状态机

REST `/event` 是 ack 接口。前端可以在 GPU 尚未完成一次 prefill 时连续发送：

```text
start -> append 0 -> append 1 -> update -> finish
```

因此不能用“最后收到的事件”直接覆盖一个 active view。当前 scheduler 每轮
都根据目标长度和已提交进度选择第一个未完成阶段：

```text
PREFIX_APPEND -> UPDATE -> QUESTION -> DECODE
```

这保证即使 `finish` 最先进入 EngineCore 队列，Question 也不会把尚未计算的
Append 或 Update 当作现成 KV 前缀。

中间阶段结束时，worker 为简化 chunked prefill 可能仍计算 logits。scheduler
会丢弃这些临时 sampled tokens；只有 Question 也完成后，采样结果才作为真正
decode 输出。

## PREFIX_APPEND 的 block 对齐

两组独立 block id 不能无条件直接拼接。设 block size 为 16，
`PREFIX_APPEND` 有 19 个 token。如果直接使用：

```text
[PA block 0, PA block 1, UPDATE block 0]
```

全局位置 19 仍会映射到 block table 下标 1，而不是 `UPDATE block 0`。

当前第一版使用显式物理 padding 解决这个问题：

```text
PA_physical = prefix + append + padding
len(PA_physical) % block_size == 0
```

padding token id 在 Request 初始化时确定：

- prefix 非空时，复用 prefix 最后一个 token id。
- prefix 为空时，使用 token id 0。

padding 会真正完成 prefill，也会参与 attention。因此它是当前近似计算的一部分，
不是严格等价的 attention mask。后续若要消除这项误差，需要实现以下方案之一：

- attention mask 能跳过物理 padding。
- finish 时把两段 KV compact/copy 成严格连续 blocks。
- attention backend 原生支持 segment offset。

## Append 事件如何计算

设：

```text
P = 固定 prefix
A_old = 之前已经到达的 append tokens
A_new = 本次新增 chunk
U = 当前 update 快照
C_PA = min(旧 PA 已计算进度, len(P + A_old))
```

`A_old` 使用全局事件提交顺序，而不是先按 source 分组再拼接。否则
`A -> B -> A` 会把最后一个 A chunk 插回 B 之前，破坏“只在尾部追加”和旧
KV 可保留的前提。

Append 的逻辑目标变成：

```text
P + A_old + A_new
```

本轮 worker 执行视图是：

```text
token ids          = P + A_old + A_new + 新 padding
num_computed       = C_PA
worker block table = PREFIX_APPEND blocks
write segment      = PREFIX_APPEND
```

需要 prefill 的 token 是：

```text
(P + A_old + A_new + 新 padding)[C_PA:]
```

本次可见的 KV 前缀是：

```text
(P + A_old)[:C_PA] 对应的 PA KV
```

如果旧 PA 已经全部提交，`C_PA = len(P + A_old)`，上述公式才简化成“只计算
`A_new + 新 padding`，并读取完整 `P KV + A_old KV`”。如果多个事件在 GPU
完成前快速到达，scheduler 会先补齐尚未提交的旧 PA token，不能把它们误认为
已经存在的 KV。

如果上一轮 PA 末尾已有旧 padding，本次从旧逻辑尾部开始覆盖它。KV manager
只在 PA 容量不足时继续申请新 blocks。

已经存在的 Update blocks 不释放、不重算，也不放进本次 Append worker view。
这正是当前论文原型要求的近似语义。

## Update 事件如何计算

当前只支持一个 Update source，但同一请求可以连续收到多代 Update 快照。设第
`i` 个到达快照为：

```text
U_(i-1) = 上一条到达的 Update 快照
U_i     = 当前到达的完整 Update 快照
L_i     = LCP(U_(i-1), U_i)
```

`L_i` 只在 Update source 内计算，不包含 prefix/template，也不包含 Append。

**事件到达时不再覆盖旧 work。** `Request._apply_prompt_stream_update()` 会：

1. 把长期 source 的最新逻辑状态更新为 `U_i`；这个状态用于最终 Question 和
   P/D transfer 的最终布局。
2. 追加一个冻结的 `UPDATE` work，内部保存 `update_token_ids=U_i` 与
   `update_common_prefix_len=L_i`。
3. 保持已有 `active_work_item` 和 `pending_work_items` 原样；正在计算的
   `U_(i-1)` 必须先完成，之前排队的更早快照也必须按 FIFO 继续计算。
4. 不修改 `update_num_computed_tokens`，也不立即截断 `UPDATE` blocks。

因此，`PromptStreamState` 同时保存两种不同事实：

- `source.token_ids`：最新到达、最终应该交给 Question / D 节点的 Update 快照。
- `update_committed_token_ids` 与 `update_num_computed_tokens`：当前实际写入
  `UPDATE` KV 域的那一代快照及其真实进度；它可能暂时落后于最新 source。

当 scheduler 真正从 FIFO 队列选中 `U_i` 时，设 `C_old` 是上一代物化快照已经
写入 UPDATE 域的 token 数，则本 work 的可复用进度为：

```text
C_i = min(C_old, L_i)
```

此时 scheduler 才在 `Request.prepare_prompt_stream_execution()` 之后、KV slots
分配之前，取出 `take_prompt_stream_update_kv_truncation()` 返回的 `C_i`，只截断
派生 `UPDATE` key 的尾部 blocks。`PREFIX_APPEND` blocks 和进度始终保持不变。

本轮 worker 执行视图是：

```text
token ids          = PA_physical + U_i
num_computed       = len(PA_physical) + C_i
worker block table = PREFIX_APPEND blocks + UPDATE blocks
write segment      = UPDATE
```

本次真正需要 prefill 的 tokens 是：

```text
U_i[C_i:]
```

它使用的 KV 前缀是：

```text
PA_physical KV + U_i[:C_i] KV
```

如果 `U_i` 是 `U_(i-1)` 的严格前缀，`C_i == len(U_i)`，就没有新 token 要
forward。状态机仍会完成这次仅截断的切换、更新 `update_committed_token_ids`，
并继续取下一个 work 或 Question，不能留下一个零 token 的 active work。

普通全局 `truncate_blocks_from_position(request_id, global_lcp)` fallback 已经
完全删除。即使 Update 当前还没有 blocks，截断也只会对派生 Update key 做
no-op，不会触碰 PA。

## PROMPT_FINISHED 如何计算用户问题

`PROMPT_FINISHED` 携带最终用户问题 `Q`。它只是请求级输入 barrier，不会跳过
前面尚未完成的阶段。当前接口要求 `Q` 至少包含一个 token；空 Question 不会
产生 final forward，容易在之前的临时 sampled token 已丢弃后形成时序相关卡死，
因此直接 fail closed。

只有 PA 和 Update 都提交后，Question 才进入 worker：

```text
token ids          = PA_physical + U + Q
num_computed       = len(PA_physical) + len(U)
worker block table = PREFIX_APPEND blocks + UPDATE blocks
write segment      = UPDATE
```

因此 Question KV 直接追加在 Update blocks 后面，不建立第三个物理域。

Question 完成后，decode barrier 才打开。判断条件不再只看一个全局计数，而是
同时检查：

- `PREFIX_APPEND` 物理 token 全部提交。
- 当前 Update 快照全部提交。
- Question 全部提交。
- 当前物理 prompt 已全部完成 prefill。

## PROMPT_FINISHED 的同步顺序

当前源码里的同步不是靠 REST handler 阻塞，也不是靠 `SOURCE_FINISHED` 事件把
所有来源统一 join，而是靠长期 `Request` 内部的 work item 队列和 decode
barrier 一起完成。

事件进入 scheduler 时分两种情况：

- 如果 `request_id` 已经存在，`Scheduler.add_request()` 会调用
  `_apply_prompt_stream_request(existing, incoming)`，把临时 `incoming`
  事件应用到长期 `existing Request` 上。临时 `Request` 不保存长期状态。
- 如果是新的 prompt stream 请求，则先对这个请求执行
  `request.apply_prompt_stream_event()`，再进入 `waiting` 队列。

`SOURCE_FINISHED` 的作用很轻：`Request._mark_prompt_source_finished()` 只把对应
source 的 `finished=True` 写入长期状态。它不会新增 work item，也不会等待这个
source 已经提交的最后一个 chunk prefill 完成。因此 `SOURCE_FINISHED` 更像
输入侧的“这个 source 不会再来了”，不是 GPU 计算侧的完成通知。

`PROMPT_FINISHED` 到达后，`Request._mark_prompt_stream_finished()` 会：

- 校验最终用户问题 `Q` 非空。
- 把 `Q` 写入 `prompt_stream.suffix_token_ids`。
- 设置 `prompt_stream.finished=True`。
- 将 `question_num_computed_tokens` 重置为 0。
- 刷新逻辑 prompt revision，但不直接进入 decode。

真正的顺序控制发生在 `Request._select_prompt_stream_execution_view()`：

- 如果当前有 `active_work_item`，继续完成它。
- 如果没有 active，但 `pending_work_items` 里还有 Append / Update，则先取队首
  work item 计算。
- 只有当 `active_work_item is None` 且 `pending_work_items` 为空，并且
  `prompt_stream.finished=True` 时，Question 才会被组装成
  `PA_physical + U + Q` 进入 `QUESTION` phase。

也就是说，如果某个 Append / Update chunk 已经被 P 节点 ack 并写入长期
`Request`，即使它的 prefill 还在 running 或 waiting，后来的
`PROMPT_FINISHED` 也不会越过它。finish 只会把 Question 挂到状态里，等前面
已经进入状态机的 source work item 完成后再计算 Question。

decode barrier 由 `Request.can_decode_from_prompt_stream()` 控制。它要求：

- `prompt_stream.finished=True`。
- 没有 `active_work_item`。
- `pending_work_items` 为空。
- `prefix_append_committed_logical_len` 已追上最终 PA 逻辑长度。
- `prefix_append_num_computed_tokens` 已追上 PA 物理长度，包括 padding。
- `update_num_computed_tokens` 已追上当前 Update 长度。
- `question_num_computed_tokens` 已追上最终 Question 长度。
- `num_computed_tokens >= num_prompt_tokens`。

`Scheduler._cap_prompt_stream_num_new_tokens()` 还会在
`PROMPT_FINISHED` 之前把本轮调度 token 数裁剪到剩余 prompt 范围内，避免请求
在输入未封口时跨入 decode。worker 为中间阶段顺手采样出的 token 会在
`Request.should_discard_prompt_stream_sampled_tokens()` 为 True 时被 scheduler
丢弃。

当前边界也要明确：scheduler 只能等待**已经到达并成功 ack 到长期 Request 的
事件**。如果某个 source 的最后一个 chunk 因网络或前端时序原因还没有发到 P，
而 `PROMPT_FINISHED` 已经先到，scheduler 无法知道那个未来 chunk 的存在。这个
顺序必须由 proxy / frontend 保证：所有要纳入回答的 source event 收到 P ack
后，再发送 `PROMPT_FINISHED`。

## active_work_item 和 pending_work_items 如何维护

`active_work_item` 和 `pending_work_items` 都保存在长期
`PromptStreamState` 中。它们不是 vLLM 全局 scheduler 的三个请求队列，而是
**单个逻辑 Request 内部的流式 prompt 小队列**。

两个字段的职责不同：

- `pending_work_items` 表示已经到达、已经应用到长期 `Request`，但还没有被这个
  Request 内部状态机选中计算的 Append / Update 工作。
- `active_work_item` 表示当前已经被选中、正在 chunked prefill 的那一项工作。
  一个 work item 进入 active 后，中途不会因为 Question 到达而被普通插队。

Append 事件到达时，`Request._apply_prompt_stream_append()` 会先更新 source 状态
和请求级 `append_token_ids`。如果事件里有 token，就追加一个
`PREFIX_APPEND` work item 到 `pending_work_items`：

```text
pending_work_items.append(PREFIX_APPEND work)
```

这个 work item 记录目标 `prefix_append_logical_len`。它不会立即覆写
`PREFIX_APPEND` 尾部 padding，只有后续真正被 scheduler 选中时才开始写 KV。

Update 事件到达时，`Request._apply_prompt_stream_update()` 会先计算相邻快照的
source-local LCP，但**不会**回退当前 `update_num_computed_tokens`，也不会调用
任何“丢弃旧 Update work”的函数。它会把下面两个值冻结到一个新的队尾
`UPDATE` work：

```text
work_item.update_token_ids          = 本次 U_i 完整快照
work_item.update_common_prefix_len  = LCP(U_(i-1), U_i)
```

已有队列和当前执行状态都保留：

- `pending_work_items` 中所有旧 Update work 都继续保留，按原到达顺序计算。
- `active_work_item` 如果正是旧 Update，也继续使用自己的冻结快照完成，不会被
  新事件置空。
- Append work 也保留；Append 与 Update 仍在同一个请求内 FIFO 竞争。
- 新 Update 到达时不截断 blocks。这样旧 active Update 的 KV 前缀不会被提前
  释放，worker 也不会被迫从一个不匹配的新视图继续执行。

`source.token_ids` 会立即变为最新 `U_i`，但 `update_committed_token_ids` 仍指向
当前实际物化在 UPDATE KV 域的旧快照。这两个状态在存在排队 Update 时可以不同，
是“保留所有历史 work”所必需的区分。

每轮 scheduler 准备执行视图时，会调用
`Request.prepare_prompt_stream_execution()`，内部再调用
`_select_prompt_stream_execution_view()` 和 `_start_next_prompt_stream_work_item()`。
这里维护从 pending 到 active 的迁移：

- 如果固定模板还没有 bootstrap 完，先创建或继续一个 `PREFIX_APPEND` active
  work，用来计算模板 KV。
- 如果已经有 `active_work_item`，直接继续它，不从 pending 里取新任务。
- 如果没有 active 且 pending 非空，从 `pending_work_items.popleft()` 取一个
  work item，放入 `active_work_item`。

取出 Append work 时，会把它的起始计算位置设成
`prefix_append_committed_logical_len`。这表示新 Append 从已经稳定提交的 PA
逻辑尾部开始写，并覆盖旧 padding。

取出 Update work 时，它的 Update 快照已经在**事件到达时**冻结；这里仅冻结它
当前能看到的 PA 快照，并根据当前 UPDATE 域的真实物化进度计算可复用位置：

```text
work_item.prefix_append_logical_len = prefix_append_committed_logical_len
work_item.prefix_append_token_ids   = PA_physical(committed_logical_len)
C_i = min(update_num_computed_tokens, work_item.update_common_prefix_len)
work_item.num_computed_tokens       = C_i
```

随后 `Request` 记录 `C_i` 为待处理的 UPDATE 域截断边界。
`Scheduler._prepare_prompt_stream_execution()` 在这项 work 已经被选中后、
`allocate_slots()` 之前调用 `take_prompt_stream_update_kv_truncation()`，只释放
UPDATE segment 在 `C_i` 之后的尾部 blocks。

这个冻结动作有两个作用：如果 Update 先被选中，后面才来的 Append 不会被这个
Update work 读取；如果 Append 先完成并提交，后续 Update 才会看到新的 PA KV。
同时，多个 Update work 也不会再都错误读取最新的 source 快照。

forward 完成后，`Scheduler._update_after_schedule()` 会调用
`Request.advance_prompt_stream_computed_tokens(num_scheduled_tokens)`。这里按当前
phase 推进进度：

- `PREFIX_APPEND` phase 推进 `prefix_append_num_computed_tokens`。当本 work 的
  物理 token，包括 padding，都算完后，更新
  `prefix_append_committed_logical_len`，并把 `active_work_item=None`。
- `UPDATE` phase 推进当前冻结快照的 `update_num_computed_tokens`。当该快照算完后，
  把它写入 `update_committed_token_ids`，再把 `active_work_item=None`，从而让下一代
  Update work 基于这份真实物化 KV 进行 LCP 复用。
- `QUESTION` phase 推进 `question_num_computed_tokens`。Question 不是
  `pending_work_items` 中的 work item，它只有在 source work 都清空后才由
  `prompt_stream.finished=True` 触发。

因此，如果 Question 到达时还有 chunk 正在 prefill，状态会是：

```text
active_work_item = 某个 Append 或 Update
pending_work_items = 后面尚未被选中的 Append / Update
prompt_stream.finished = true
suffix_token_ids = Q
```

下一轮 scheduler 仍然先继续 active，再消费 pending。只有当：

```text
active_work_item is None
pending_work_items 为空
```

并且 Question 尚未算完时，才会切换到 `QUESTION` phase。

preemption 时会走 `Request.reset_prompt_stream_computed_state()`。它会清空
`active_work_item` 和 `pending_work_items`，把物理计算进度归零，但保留长期
source 状态。随后按当前逻辑快照重建安全顺序：

```text
如果已有 Append token，加入一个 PREFIX_APPEND work
如果已有 Update token，加入一个 UPDATE work
```

这个 replay 顺序牺牲了原始交错信息，但保证被释放的 KV blocks 不会被错误复用。
它是正常事件路径“保留所有已入队 Update work”的唯一例外：preemption 已经释放了
物理 KV，当前恢复策略只能从最新逻辑快照安全重建。

## Append 在 Update 之后到达

如果 Update 已经计算完，又到达 Append chunk，行为如下：

1. worker 暂时切回只包含 `PREFIX_APPEND` 的执行视图。
2. 从旧 PA 的逻辑尾部覆盖 padding，计算新 Append 与新 padding。
3. Update blocks 原样保留。
4. 后续 Update 变化或 Question 到达时，再安装 `PA + UPDATE` 完整 block table。

这意味着旧 Update KV 不会因为 Append 增长而重算。它仍编码计算当时看到的
PA 上下文和位置，最终却被拼到最新 PA 后面。这是本项目明确选择的近似注意力
语义，不是标准完整 prompt prefill 的严格等价结果。

## KVCacheManager 已实现的行为

prompt stream 使用专门的 segment allocator：

- 分配时使用当前 write segment 的派生 key。
- token 数和已计算进度都使用 segment-local 坐标。
- allocator 返回本轮新申请的 blocks。
- scheduler 在 worker phase 切换时另外读取完整 execution blocks。
- Update 截断只操作 Update key。
- abort / finish 会释放两个 key。
- preemption 会释放两个域，并把物理进度清零；长期 source 状态保留，之后按
  `PA -> Update -> Question` 完整 replay。

prompt stream segment blocks 当前不会进入普通 prefix cache。原因是 Update KV
不仅由 Update token 决定，还依赖事件发生时的 PA 上下文；沿用普通 prompt
hash 会产生错误共享。

当前 allocator 明确限制：

- 只支持 block size 一致的 full-attention / MLA KV groups。
- 不支持 sliding-window、Mamba、chunked-local attention 或 cross-attention。
- P 节点 start 会拒绝任何 speculative decoding 配置；segment allocator 也不
  接受 lookahead slots。
- 不接受本地 prefix-cache hit、外部 KV load 或 encoder KV。
- P 节点只允许同步调度和 pipeline parallel size 1；async scheduling / PP 会在
  `AsyncLLM.add_prompt_stream_request()` 入口直接拒绝，避免事件修改在途 batch。
- 当前只支持 NIXL 直接 device KV transfer。host-buffer 模式对同一 partial
  block 的增量覆盖还缺少重新保存语义，因此也会在 start 入口直接拒绝。

不满足这些条件会 fail closed，而不是退回普通连续 block table。

## Scheduler 到 worker 的刷新方式

phase 切换时必须同时更新三项内容：

```text
token ids + num_computed_tokens + 完整 block table
```

旧实现把 prompt mutation 塞进 `CachedRequestData.resumed_req_ids`，但 force
refresh 时只读取原始 `request_id` 的 PA blocks，而且 cached worker 的 token
视图可能没有更新。

当前实现改为：

- phase 切换或 token view 变化时，把请求作为 `NewRequestData` 发给 worker。
- legacy GPU runner 复用 `_update_streaming_request()`，移除旧 batch row 后完整
  更新 prompt、计算位置和 block table。
- V2 GPU runner 复用已有 remove/re-add 路径，并以 `overwrite=True` 安装整表。
- 同一 phase 的后续 chunk 只走 cached diff，不重复重装完整状态。

同时修复了一个 token view bug：重新绑定 `_all_token_ids` 后会同步重建
`ConstantList`，scheduler 不会再把落后一个事件的 token ids 发给 worker。

## P/D transfer 的最终布局

P 节点结束时导出的 block ids 固定为：

```text
PREFIX_APPEND block ids + UPDATE block ids
```

这里的 Update blocks 已经包含：

```text
当前 Update 快照 + Question + 已经真正 forward 的后续 token KV
```

connector 调用前会逐 KV group 校验：

```text
actual block count == ceil(physical_prompt_len / block_size)
```

如果拼接结果不是普通 D 请求能够解释的连续物理布局，会直接抛出错误，不再把
错误 block ids 交给 NIXL。

P 在 final Question forward 后采样出的首 token 尚未作为输入经过下一轮
forward，因此不在远端 KV manifest 中。exact prompt-stream transfer 只允许：

```text
remote_num_tokens = len(PA_physical + Update + Question)
D prompt tokens   = PA_physical + Update + Question
                  + 0/1 个 P sampled token
```

D 为 sampled token 可能额外申请一个本地 block，尤其是物理 prompt 恰好 block
对齐时。NIXL D 侧严格按 manifest 把 local load block ids 裁剪到
`ceil(physical_prompt_len / block_size)`，只从 P 拉取物理 prompt；可选的首
token 由 D 本地 forward。当前实现不允许用 `remote_num_tokens` 覆盖这个 token。

## proxy 如何和物理布局对齐

P 侧 `kv_transfer_params` 生成时会附带当前物理 prompt token ids。serving 层
将它从 connector 参数中取出，作为 finish 响应的 `prompt_token_ids` 返回。

proxy 收到后会用这份序列完整替换自己的逻辑 token 镜像，再追加 P 侧返回的
output token ids。D 节点最终收到：

```text
PA_physical + Update + Question + P 侧已返回的 output tokens
```

发往 D 之前会强制验证：

```text
remote_num_tokens = physical_prompt_len
D_prompt_len = physical_prompt_len + len(P sampled tokens)
len(P sampled tokens) in {0, 1}
```

此外，`kv_transfer_params.prompt_stream_physical_prompt_len` 必须和 P 返回的
`prompt_token_ids` 长度一致；manifest 的类型、group、count 和 block size
也必须一致。不一致时 proxy 返回 502，不会继续请求 D。

P 当前只允许 `n=1`，所以 sampled token 最多一个。D 返回非流式 completion 后，
proxy 会把这个 P 首 token 的文本和 token id 合并回最终响应，并修正 usage；若首
token 已触发 stop，或原请求只要求一个 token，则不再启动 D。

## RESTFUL 链路

当前实验链路保持：

```text
frontend
  -> research/scripts/proxy_server.py
  -> P /v1/prompt_stream/start
  -> P /v1/prompt_stream/event
  -> P /v1/prompt_stream/finish
  -> D /v1/completions
```

P 节点接口仍支持 text / token ids 双模式；进入 `PromptStreamEvent` 前统一完成
tokenization。start 保存一个长期 output collector，event 只返回 ack，finish
等待最终输出和 `kv_transfer_params`。当前 proxy MVP 只接受 completions API，
但已经同时支持 `stream=false` 和 `stream=true`：流式路径先发送 P 首 token，
再转发 D 的后续 SSE。chat prompt 的 renderer 镜像尚未实现，因此会在 start
入口直接返回 400，而不是发送错位 KV。

P 和 proxy 在 finish 成功后都会删除各自的长期 session，避免已完成 request id
被错误复用。

## 已添加的单元测试和脚本

`tests/v1/test_prompt_stream_request.py` 

当前覆盖：

- 固定模板先完成 bootstrap prefill；随后 Append 与 Update 作为同一个请求内的
  FIFO work item 竞争，不再由 `PREFIX_APPEND` 的硬编码优先级决定。
- Update 先获调度、Append 后到时，Update 冻结当时已经提交的
  `PREFIX_APPEND` 物理 token/KV 快照，不能读取该 Append chunk。
- 连续快速到达的 Append、Update、Finish 仍按阶段顺序执行。
- `PREFIX_APPEND` 的 padding 和物理 token view。
- 多个 Append source 交错到达时保持全局尾部追加顺序。
- 空 Question 被拒绝。
- Append-after-Update 不回退 Update 已提交进度；即使新 PA 跨 block 边界，
  Update block ids 也保持不变，worker 仍按新的 `PA ids + Update ids` 安装整表。
- Update 使用 source-local LCP。
- Update 截断不改变 PA blocks。
- 两个 segment 使用独立 key 分配，并严格按 `PA ids + Update ids` 合并。
- Question 复用 Update 的 partial block，不创建第三个 key。
- preemption 释放两个 key 后从长期 source 状态 replay。
- D prompt 多出本地 sampled token 尾块时，NIXL load block ids 正确裁剪。
- `_all_token_ids` 与 `ConstantList` 始终一致。

当前环境没有安装 `pytest` 和 `ruff`，所以没有运行正式 pytest / ruff 命令。
已经完成的本地代码级验证是：


直接测试运行时只有无 GPU 环境下的 NVML / Triton 提示，没有断言失败。

新增 `research/scripts` 下的端到端 smoke test 脚本：

- `append_frontend.py`：append-only 文本模式。
- `update_frontend.py`：单 update source 文本快照模式。
- `mixed_frontend.py`：单 append source + 单 update source 的混合文本模式。

这三个脚本都已经在实验 P/D 分离链路上跑通过。

## 当前明确未完成的内容

- 已完成 append-only、update-only、mixed 的真实 P/D smoke test，但还没有形成可重复的自动化集成测试。
- padding 会参与 attention，尚未实现严格 mask 或 KV compaction。
- Append-after-Update 保留旧 Update KV 是论文近似，位置和上下文不是完整重算结果。
- 当前只支持一个 Update source。
- 乱序事件会暂存，但缺口补齐后的自动消费尚未实现。
- 如果 P 首 token 已满足停止条件、proxy 无需启动 D，或 D 请求在 KV 拉取前失败，
  P 已进入 delay-free 状态的 NIXL blocks 当前只能等待 lease 到期释放；这会产生
  短期显存占用和 expired-KV 日志，但不改变成功请求的 token/block 排布。
- prompt embeds、多模态位置编码、Mamba、sliding window 和 speculative decode
  尚未接入 segment allocator。
- 异步多 batch 场景还没有用 prompt revision 对在途旧结果做完整 fencing；当前
  代码已拒绝 async scheduling 和 PP>1，而不是静默运行不安全路径。
- 物理 prompt 加上 P 必采样的首 token 超过 `max_model_len` 时，scheduler 会把
  该请求标记为 `FINISHED_ERROR` 并释放两个 segment；start 和后续 event 都走
  请求级失败，不再让长度异常退出整个调度循环。由于 event 是异步 ack，具体
  超长原因还没有通过独立错误通道回传给已经返回的 `/event`。
- 旧的 `total_tokens_recomputed` 已删除。当前通过最终响应中的
  `prompt_stream_stats` 分别报告 `admitted_update_prefill_tokens`、
  `scheduled_update_prefill_tokens` 和 `update_tokens_invalidated`，避免把逻辑
  admission、Scheduler 重复调度和语义失效混为同一指标。

下一步最有价值的验证是把三类 smoke test 固化成可重复脚本或 pytest 集成测试，
并系统记录 P 日志中的：

```text
physical prompt len
remote_num_tokens
merged block counts
```

与 proxy 日志中的 D prompt 长度关系。之后再比较启用和禁用 NIXL 的输出差异，
判断剩余误差主要来自 padding 近似、Append-after-Update 近似，还是 NIXL 实际复制
的远端/本地 block id 对。
