# 流式 Prompt Prefill 当前实现说明

这份文档记录当前分支已经实现的内容。重点不是历史讨论过程，而是说明现在代码里已经形成的边界、状态模型和调度路径。

当前实现已经把流式 prompt 的核心逻辑基本隔离到 `EngineCoreRequest -> Request` 之后：

- `EngineCoreRequest` 只负责携带一次入站事件。
- `Request` 保存长期状态，并把多次事件合成为当前有效 prompt。
- `Scheduler` 负责把事件应用到长期 `Request`，并根据变化释放失效 KV、刷新 worker 侧状态。
- 前端如何构造事件、如何把 chunk 传进 engine，后续再设计。

## 核心设计目标

这个实现面向混合来源的流式 prefill，主要支持两类来源：

- **Append source**：来源内容只追加，不修改过去已经提交的 token。
- **Update source**：来源内容以快照方式更新，新版本可以替换上一版本的部分或全部 token。

整体 prompt 的逻辑布局固定为：

```text
prefix/template + append segment + update segment + suffix/user question
```

在 P/D 分离语境下，P 节点最终传给 D 节点的 KV cache 也应该按照同样顺序理解：

```text
通用模板 KV + Append 部分 KV + Update 部分 KV + 用户问题 KV
```

这里需要特别注意：Update 部分的 prefill 是论文设计中的近似计算，不要求等价于完整 prompt 一次性 prefill。当前目标是先把状态管理、调度边界和 KV 释放路径搭起来。

## 事件模型

流式 prompt 的传输层事件定义在 `vllm/v1/engine/prompt_stream.py`。

### `PromptStreamEventType`

当前事件类型包括：

- `APPEND`：某个来源追加一段 token。
- `UPDATE`：某个来源提交一个新版本的完整 token 快照。
- `SOURCE_FINISHED`：某个来源结束，不再发送新内容。
- `PROMPT_FINISHED`：整个 prompt stream 封口，并携带最终用户问题。

其中 `PROMPT_FINISHED = "prompt_finished"` 是请求级 barrier。它不是某个 source 的结束事件，而是表示最终用户问题已经到达，P 节点可以完成最后一段 prefill。

### `PromptSourceType`

当前来源语义只有两种：

- `APPEND`：该 source 的 token 只能追加。
- `UPDATE`：该 source 的 token 表示当前版本快照。

source 创建后不能改变类型。如果同一个 `source_id` 先作为 Append 出现、后面又作为 Update 出现，`Request` 会报错。

### `PromptStreamEvent`

`PromptStreamEvent` 是一次入站事件，主要字段语义如下：

| 字段 | 含义 |
| --- | --- |
| `request_id` | 事件所属请求。必须和长期 `Request.request_id` 一致。 |
| `source_id` | 同一个请求内的来源实例，例如检索器、网页源、ANN source。 |
| `event_type` | 本次事件是 append、update、source finished 还是 prompt finished。 |
| `source_type` | source 采用 Append 还是 Update 语义；`PROMPT_FINISHED` 可以不依赖它。 |
| `chunk_id` | 幂等键，用于避免重复应用同一个 chunk。 |
| `version` | 来源内容版本；Update 用它过滤旧版本。 |
| `sequence_no` | 来源内部的到达顺序，用于检测乱序或缺口。 |
| `token_ids` | Append 的新增 token、Update 的完整快照，或 `PROMPT_FINISHED` 的最终用户问题。 |
| `arrival_time` | 前端观察到内容可用的时间，用于后续调度和延迟分析。 |

## EngineCoreRequest 的边界

`EngineCoreRequest` 现在只增加了一个字段：

```python
prompt_stream_event: PromptStreamEvent | None = None
```

这个字段的定位很轻：

- 普通请求保持为 `None`。
- 流式 prompt 的后续事件通过这个字段进入 engine core。
- `EngineCoreRequest` 不保存长期流式状态。
- `EngineCoreRequest` 不负责判断 Append / Update 怎么改 prompt。

真正的状态管理从 `Request.from_engine_core_request()` 之后开始。

## Request 的长期状态

`Request` 现在是流式 prompt 的长期状态持有者。它不是单次事件，而是 scheduler 里持续存在的请求对象。

新增的关键字段包括：

- `prompt_stream_event`：本次从 `EngineCoreRequest` 转来的入站事件。对于长期 Request，事件被应用后会清空。
- `prompt_stream`：长期流式 prompt 状态，类型是 `PromptStreamState | None`。
- `prompt_stream_needs_worker_refresh`：当 prompt token 或 block 状态发生变化时，要求 scheduler 刷新 worker 侧 cached request 数据。

### `PromptStreamState`

`PromptStreamState` 保存一个请求内所有流式 source 的当前状态：

| 字段 | 含义 |
| --- | --- |
| `prefix_token_ids` | 流式 source 之前的固定 prompt，例如 system prompt 或通用模板。 |
| `sources` | `source_id -> PromptSourceState` 的映射。 |
| `source_order` | source 首次出现顺序；重建 prompt 时按类型分段后仍保持各段内部稳定顺序。 |
| `suffix_token_ids` | `PROMPT_FINISHED` 携带的最终用户问题。 |
| `finished` | 请求级 prompt stream 是否已经封口。 |
| `revision` | prompt 每次真实变化后递增。 |
| `total_tokens_invalidated` | 累计失效 token 数。 |
| `total_tokens_recomputed` | 预留统计字段，目前还没有完整维护。 |

### `PromptSourceState`

每个 source 的状态保存在 `PromptSourceState` 中：

- `source_id`：来源 ID。
- `source_type`：Append 或 Update。
- `token_ids`：该来源当前有效 token。
- `version`：当前有效版本。
- `next_sequence_no`：下一条期望到达的序号。
- `seen_chunk_ids`：已经应用过的 chunk，用于去重。
- `pending_events`：乱序到达但暂时不能应用的事件。
- `finished`：该 source 是否结束。
- `last_arrival_time`：最近一次事件到达时间。

当前 `pending_events` 只是暂存乱序事件，尚未实现自动补洞消费。

### `PromptStreamLayout`

`PromptStreamLayout` 用于显式描述当前 prompt 在 KV cache 中的逻辑分段：

```python
@dataclass(frozen=True)
class PromptStreamLayout:
    prefix_len: int
    append_len: int
    update_len: int
    suffix_len: int
```

`Request.get_prompt_stream_layout()` 返回四段长度。后续接 P/D transfer 时，可以用它把 KV cache 的逻辑边界传给 connector 或 debug 工具。

## Request 如何应用事件

Scheduler 面向流式 prompt 的统一入口是：

```python
Request.apply_prompt_stream_event(event)
```

它根据 `event.event_type` 分发到不同处理函数：

| 事件 | 处理函数 | 主要行为 |
| --- | --- | --- |
| `APPEND` | `_apply_prompt_stream_append()` | 追加该 source 的 token。 |
| `UPDATE` | `_apply_prompt_stream_update()` | 用新快照替换该 update source 的 token。 |
| `SOURCE_FINISHED` | `_mark_prompt_source_finished()` | 标记单个 source 结束。 |
| `PROMPT_FINISHED` | `_mark_prompt_stream_finished()` | 写入最终用户问题，并封口整个 prompt stream。 |

所有会修改 prompt 的事件都会先经过 `_validate_prompt_stream_can_mutate()` 校验。

当前校验规则是：

- 已经生成 output token 后，不能再修改 prompt。
- 当前只支持 token-id prompt，不支持 prompt embeds 或 mixed token/embed 输入。
- `APPEND`、`UPDATE`、`PROMPT_FINISHED` 必须携带 `token_ids`。

## Append 语义

Append 事件表示某个 source 新增了一段 token。

当前行为是：

- 如果 prompt stream 已经 `finished`，拒绝追加。
- 如果 source 已经 `finished`，拒绝追加。
- 如果 `chunk_id` 已经出现过，返回 no-op，避免重复应用。
- 如果 `sequence_no` 不是期望值，把事件放入 `pending_events`，暂不应用。
- 否则把 `event.token_ids` 追加到该 source 的 `token_ids` 后面。

Append 应用后会调用 `_sync_prompt_from_stream_state()`，把新的流式状态同步回 `Request.prompt_token_ids` 和 `_all_token_ids`。

## Update 语义

Update 事件表示某个 source 当前版本的完整快照。

当前行为是：

- MVP 阶段最多只允许一个 Update source。
- 如果 prompt stream 已经 `finished`，拒绝更新。
- 如果 source 已经 `finished`，拒绝更新。
- 如果 `chunk_id` 已经出现过，返回 no-op。
- 如果 `sequence_no` 不是期望值，把事件放入 `pending_events`，暂不应用。
- 如果 `version` 小于当前版本，认为是旧事件，返回 no-op。
- 如果 `version` 相同但 token 内容不同，认为是冲突事件，直接报错。
- 否则用 `event.token_ids` 替换该 update source 的 `token_ids`。

Update 应用前会先比较旧 update 快照和新 update 快照，得到 **update 源内 LCP**。
然后把这个局部 LCP 加上 `prefix_len + append_len`，转换成 scheduler 可以使用的全局 KV 保留边界。
最后再触发 `_sync_prompt_from_stream_state()`。

## PROMPT_FINISHED 语义

`PROMPT_FINISHED` 是整个 prompt stream 的封口事件。

它的 `token_ids` 被解释为最终用户问题，写入：

```python
PromptStreamState.suffix_token_ids
```

当前行为是：

- 如果还没有 `PromptStreamState`，先初始化。
- 如果 prompt stream 尚未 finished，写入 suffix，设置 `finished=True`，然后同步 prompt。
- 如果 prompt stream 已经 finished 且 suffix 完全一致，返回 no-op。
- 如果 prompt stream 已经 finished 但 suffix 不一致，报错，避免同一个请求被两个不同最终问题封口。

这个事件到达后，scheduler 不会立刻跳过最终用户问题。它只是解除“不能 decode”的 barrier；最终用户问题仍然需要被 prefill 完，之后才会自然进入首 token decode。

## Prompt 重建和 revision

每次 Append、Update 或 PROMPT_FINISHED 改变有效 token 后，Request 会调用：

```python
_sync_prompt_from_stream_state(old_prompt_token_ids)
```

同步流程是：

1. 用 `_rebuild_prompt_from_stream_state()` 重建全局 prompt。
2. 计算或接收 scheduler 可安全保留的全局 KV 前缀长度 `common_prefix_len`。
   - Append 和 `PROMPT_FINISHED` 使用旧全局 prompt 与新全局 prompt 的 LCP。
   - Update 使用 update 源内 LCP，再加上 `prefix_len + append_len` 偏移。
3. 如果 token 序列没有变化，返回 `PromptStreamMutation.noop()`。
4. 如果发生变化，更新：
   - `prompt_token_ids`
   - `_all_token_ids`
   - `num_prompt_tokens`
   - `num_computed_tokens`
   - `prompt_stream.revision`
   - `prompt_stream.total_tokens_invalidated`
   - `prompt_stream_needs_worker_refresh`
5. 如果公共前缀短于旧 prompt，清空 `block_hashes` 和 `_prompt_embeds_per_block_hashes`，然后重新计算 block hashes。

重建 prompt 的固定顺序是：

```text
prefix_token_ids
+ 所有 Append source token_ids
+ 所有 Update source token_ids
+ suffix_token_ids
```

这就是当前 Request 侧和后续 D 节点 KV 排布需要对齐的逻辑顺序。

## PromptStreamMutation

`PromptStreamMutation` 是 Request 返回给 scheduler 的变化摘要。

它包含：

| 字段 | 含义 |
| --- | --- |
| `applied` | 事件是否真实改变了 prompt。 |
| `old_num_prompt_tokens` | 修改前 prompt 长度。 |
| `new_num_prompt_tokens` | 修改后 prompt 长度。 |
| `common_prefix_len` | scheduler 可安全保留的全局 KV 前缀长度。Update 场景下它等于 `prefix_len + append_len + update_source_lcp`。 |
| `num_tokens_to_invalidate` | 已经 prefill 过但现在失效的 token 数。 |
| `prompt_revision` | 修改后的 prompt revision。 |

Scheduler 不需要理解 Append / Update 的内部细节，只需要根据这个 mutation 判断是否释放 KV、是否刷新 worker 侧缓存。

## Scheduler 如何接收流式事件

`Scheduler.add_request()` 现在区分两种情况。

### 已存在长期 Request

如果同一个 `request_id` 已经在 `self.requests` 里，并且新来的临时 `Request` 带有 `prompt_stream_event`：

```python
self._apply_prompt_stream_request(existing, request)
return
```

这里的关键点是：

- `incoming Request` 只是事件载体。
- `existing Request` 才保存长期状态。
- 事件应用完成后，`incoming.prompt_stream_event` 会被清空。

这就是当前实现中 “EngineCoreRequest 之后隔离” 的主要边界。

### 第一次进入 scheduler 的 Request

如果 `request_id` 还不存在，但这个 Request 自带 `prompt_stream_event`：

- scheduler 会先在这个 Request 上应用该事件。
- 然后清空 `request.prompt_stream_event`。
- 再把这个 Request 放入 waiting queue，并登记到 `self.requests`。

这使得第一条事件也能初始化长期 `PromptStreamState`。

## Scheduler 的 decode barrier

流式 prompt 请求在 `PROMPT_FINISHED` 前不能进入 decode。

对应逻辑在：

```python
Scheduler._cap_prompt_stream_num_new_tokens(request, num_new_tokens)
```

行为是：

- 普通请求不受影响。
- 已经 `PROMPT_FINISHED` 且 prompt token 已经 prefill 完的请求不裁剪。
- 未封口的流式 prompt 请求，只允许调度剩余 prompt token。
- 如果 prompt token 已经 prefill 到当前末尾，但还没收到 `PROMPT_FINISHED`，本轮调度 token 数会变成 0，从而等待最终用户问题。

对应的 Request 判断函数是：

```python
Request.is_prompt_stream_finished()
```

它要求同时满足：

- `prompt_stream.finished == True`
- `num_computed_tokens >= num_prompt_tokens`

因此 `PROMPT_FINISHED` 到达后，suffix 仍然会被正常 prefill；只有 suffix 也完成后，才允许进入 decode。

## KV cache 失效路径

当 Request 应用事件后返回的 `PromptStreamMutation.num_tokens_to_invalidate > 0`，scheduler 会调用：

```python
self.kv_cache_manager.truncate_blocks_from_position(
    existing,
    mutation.common_prefix_len,
)
```

当前新增了三层接口：

```text
KVCacheManager.truncate_blocks_from_position(request, keep_tokens)
KVCacheCoordinator.truncate_blocks_from_position(request_id, keep_tokens)
SingleTypeKVCacheManager.truncate_blocks_from_position(request_id, keep_tokens)
```

含义是：保留 `keep_tokens` 对应的安全前缀，释放之后的 KV blocks。

需要注意：KV cache 是 block 粒度管理的，所以 `SingleTypeKVCacheManager` 会根据 block size 向上取整保留 block。这意味着当前是 block 级截断，不是精确 token 级截断。

## Worker 侧 cached request 刷新

Prompt 改变后，Request 会设置：

```python
prompt_stream_needs_worker_refresh = True
```

`Scheduler._make_cached_request_data()` 会检查这个标志：

- 把该 request 加入 `resumed_req_ids`。
- 重新发送完整 `all_token_ids`。
- 如果需要，重新获取当前 block ids。
- 发送完后清掉 `prompt_stream_needs_worker_refresh`。

这样 worker 侧不会继续使用旧 prompt token 或旧 block table。

## 当前实现的近似语义

当前代码在 token 序列层面使用统一布局：

```text
prefix + append + update + suffix
```

Update 快照变化时，当前实现不再直接用全局 prompt LCP 推导失效边界，而是先计算：

```text
update_source_lcp = LCP(old_update_source_tokens, new_update_source_tokens)
```

然后转换成全局 KV 保留边界：

```text
common_prefix_len = prefix_len + append_len + update_source_lcp
```

这样 scheduler 仍然只需要处理一个全局 token 位置，但这个位置的来源语义是 **update 源内 LCP**。

当前仍然没有专门的 attention mask、分段 block table 或显式 KV segment metadata 来表达 Append / Update 的物理分域。因此现在可以认为：**Update 的失效边界已经按源内 LCP 计算，但物理 KV 分域仍是 MVP 近似实现**。

## 当前已经完成的边界

已经完成的部分包括：

- `EngineCoreRequest` 增加 `prompt_stream_event`，作为一次事件入口。
- `Request.from_engine_core_request()` 把事件转入 Request。
- `Request` 保存长期 `PromptStreamState`。
- `Request` 支持 Append、Update、Source Finished 和 Prompt Finished 四类事件。
- `PROMPT_FINISHED` 会把最终用户问题写入 suffix。
- `Request` 会把流式状态重建为统一 prompt token 序列。
- `PromptStreamMutation` 把 prompt 变化摘要返回给 scheduler；Update 场景下其 `common_prefix_len` 来自 update 源内 LCP 加全局偏移。
- scheduler 会把重复 `request_id` 的流式事件应用到长期 Request。
- scheduler 在 prompt stream 封口前阻止提前 decode。
- scheduler 会根据公共前缀释放失效 KV blocks。
- worker cached request 会在 prompt 改变后刷新。
- `Request.get_prompt_stream_layout()` 已经提供四段逻辑长度，供后续 P/D transfer 接入。

## 当前还没有完成的部分

后续仍需要继续设计或实现：

- **前端事件入口**：还没有定义外部 API 如何把 chunk、source、version 和最终用户问题转成 `PromptStreamEvent`。
- **乱序事件消费**：`pending_events` 目前只暂存乱序事件，还不会在缺口补齐后自动应用。
- **多 Update source**：当前明确限制最多一个 Update source。
- **精确 KV 分域**：目前是 block 级 LCP 截断，还没有真正表达 Append KV 和 Update KV 的物理分区。
- **P/D transfer 扩展**：还没有把 `PromptStreamLayout` 写入 `kv_transfer_params` 或 connector 协议。
- **测试用例**：还需要补 Append、Update、PROMPT_FINISHED、KV 截断和 decode barrier 的单元测试 / 集成测试。
- **统计字段**：`total_tokens_recomputed` 目前是预留字段，还没有系统维护。
- **prompt embeds 支持**：当前流式 prompt mutation 只支持 token-id prompt。

## 推荐的下一步

比较自然的下一步是先补最小测试，而不是马上改前端：

- 构造一个 Request，连续应用 Append 事件，确认 prompt 按 append segment 增长。
- 构造一个 Update source，提交两个版本，确认 update 源内 LCP、全局 KV 保留边界和 `num_tokens_to_invalidate` 正确。
- 发送 `PROMPT_FINISHED`，确认最终用户问题进入 suffix。
- 在 scheduler 层模拟重复 `request_id` 的事件进入，确认长期 Request 被更新，临时 Request 不保留状态。
- 检查 `PROMPT_FINISHED` 前不会进入 decode，`PROMPT_FINISHED` 后 suffix prefill 完才允许 decode。

这些测试能先保护当前边界，然后再继续设计前端 chunk API 和 P/D transfer metadata。
