# 流式 Prefill 已完成设计记录

## 当前已经新增的结构

目前我们已经把流式 prefill 的信息从普通请求字段里拆出来，形成了一个比较清楚的两层结构：

- `EngineCoreRequest.prompt_stream_event`：负责把一次流式 prefill 事件从前端传进 engine core。
- `Request.prompt_stream_event`：负责把这次入站事件继续转交给 scheduler。
- `Request.prompt_stream`：负责保存一个请求在 scheduler 生命周期里的长期流式 prompt 状态。

这一步的核心意图是：**不要把 append、update、source finish、request finish 这些语义散落成多个布尔字段**，而是统一封装成事件和状态。

## 新增类的职责

### `PromptStreamEventType`

`PromptStreamEventType` 表示这次事件到底是什么操作。

目前设计了四类：

- `APPEND`：某个来源追加了一段 token。
- `UPDATE`：某个来源给出了一个新版本的完整 token 快照。
- `SOURCE_FINISHED`：某个来源已经结束，不会再发送新内容。
- `REQUEST_FINISHED`：整个请求的流式 prompt 输入已经结束。

这个枚举后面会在 `Request.apply_prompt_stream_event()` 里真正用到，用来分发到 append、update、source finish 和 request finish 的处理逻辑。

### `PromptSourceType`

`PromptSourceType` 表示某个来源采用哪种语义。

- `APPEND`：只允许尾部追加，不允许改写已有 token。
- `UPDATE`：允许用新版本快照替换该来源当前内容。

这个类型描述的是 **source 的长期语义**，不是单次事件本身。比如一个检索来源如果是 update 类型，那么它可能连续发送多个 `UPDATE` 事件，每个事件带一个更高版本的完整快照。

### `PromptStreamEvent`

`PromptStreamEvent` 是传输层事件，回答的是：**这一次来了什么？**

它包含：

- `request_id`：同一个流式 prompt 请求的所有事件必须属于同一个 engine 内部请求 ID。
- `source_id`：区分不同来源，例如 `web_search_0`、`dense_retriever_0`。
- `event_type`：说明这次事件是 append、update、source finished 还是 request finished。
- `source_type`：说明这个来源采用 append 语义还是 update 语义。
- `chunk_id`：用于去重和幂等，避免重试导致同一段内容重复应用。
- `version`：用于 update 语义下判断新旧版本。
- `sequence_no`：用于检查同一来源内部是否乱序或缺失。
- `token_ids`：append 时表示新增 token，update 时表示该来源当前版本的完整 token 快照。
- `arrival_time`：记录内容到达时间，后面可以用于 debounce、延迟统计和调度策略。

注意：`PromptStreamEvent` 不保存累计 prompt，它只描述一次输入事件。

### `PromptSourceState`

`PromptSourceState` 是 `Request` 内部的单来源状态，回答的是：**这个 source 目前累积成什么样？**

它保存：

- `source_id`：来源 ID。
- `source_type`：来源语义，创建后应保持稳定。
- `token_ids`：该来源当前有效 token。
- `version`：该来源当前有效版本。
- `next_sequence_no`：下一条期望收到的序号。
- `seen_chunk_ids`：已经应用过的 chunk，用于去重。
- `finished`：该来源是否结束。
- `last_arrival_time`：该来源最近一次事件到达时间。

### `PromptStreamState`

`PromptStreamState` 是 `Request` 内部的完整流式 prompt 状态，回答的是：**这个请求当前的多源 prompt 状态是什么？**

它保存：

- `sources`：从 `source_id` 到 `PromptSourceState` 的映射。
- `source_order`：多来源拼接 prompt 的稳定顺序。
- `finished`：请求级流式 prompt 是否结束。
- `revision`：prompt 每次真实变化后递增，用于防止 scheduler 或 model runner 使用旧状态 decode。
- `total_tokens_invalidated`：累计失效 token 数。
- `total_tokens_recomputed`：累计重算 token 数。

这个状态是 scheduler 后面真正会依赖的对象。

### `PromptStreamMutation`

`PromptStreamMutation` 是 `Request.apply_prompt_stream_event()` 返回给 scheduler 的结果，回答的是：**这次事件对 prompt 和 KV cache 有什么影响？**

它保存：

- `applied`：事件是否真的改变了 prompt。
- `old_num_prompt_tokens`：应用事件前的 prompt 长度。
- `new_num_prompt_tokens`：应用事件后的 prompt 长度。
- `common_prefix_len`：新旧 prompt 的最长公共前缀。
- `num_tokens_to_invalidate`：已经 prefill 过但现在需要失效的 token 数。
- `prompt_revision`：应用事件后的 prompt 版本。

这个类的重点是把 `Request` 和 `Scheduler` 的职责隔开：

- `Request` 负责判断 prompt 怎么变。
- `Scheduler` 负责根据 mutation 去处理 KV cache、重新调度和 decode barrier。

## 数据流

当前设计的数据流是：

```text
前端 / client
  -> EngineCoreRequest(prompt_stream_event=event)
  -> EngineCore.preprocess_add_request()
  -> Request.from_engine_core_request()
  -> Request(prompt_stream_event=event)
  -> Scheduler.add_request()
  -> existing_request.apply_prompt_stream_event(event)
  -> PromptStreamMutation
  -> Scheduler / KVCacheManager 处理 KV 失效和重新调度
```

这里最关键的是：`EngineCoreRequest` 不保存长期状态，`Request` 才保存长期状态。

## 和 Stream2LLM 的区别

Stream2LLM 的实现更直接：

- 在 `Request` 上加 `is_streaming_prompt`。
- 在 `Request` 上加 `is_streaming_prompt_finished`。
- append 时直接扩展 `prompt_token_ids` 和 `_all_token_ids`。
- update 时计算最长公共前缀，然后回退 `num_computed_tokens`。
- 用 `cache_invalidated_this_step` 通知 scheduler 这一轮需要重置 worker 侧 block table。

我们现在的设计更偏向多源场景，因此没有直接照搬它的字段，而是拆成：

- `PromptStreamEvent`：单次事件。
- `PromptSourceState`：单个来源的长期状态。
- `PromptStreamState`：整个请求的多源长期状态。
- `PromptStreamMutation`：给 scheduler 的变更摘要。

这样后续才能支持 append source 和 update source 混合存在。



### `PromptStreamState` 不能只靠 `prompt_stream_event` 推断

`prompt_stream_event` 是一次性事件，`prompt_stream` 是长期状态。

后面第一次收到流式事件时，需要显式初始化：

```python
self.prompt_stream = PromptStreamState(
    sources={},
    source_order=[],
    finished=False,
    revision=0,
    total_tokens_invalidated=0,
    total_tokens_recomputed=0,
)
```

否则 `is_prompt_streaming()` 只看 `self.prompt_stream is not None` 时，会认为这个请求还不是流式 prefill 请求。

## 下一步实现计划

- 实现 `Request.apply_prompt_stream_event()`，先只做事件类型分发。
- 实现 `_ensure_prompt_stream_state()`，第一次收到流式事件时初始化 `PromptStreamState`。
- 先实现 append 语义：只允许尾部追加、重复 chunk 跳过、source finished 后拒绝追加。
- 再实现 update 语义：把 `token_ids` 当作该 source 的完整快照，计算全局 prompt 的最长公共前缀。
- Scheduler 最小接入已完成：遇到 duplicate request id 且带 `prompt_stream_event` 时，不走现有 `StreamingUpdate`，而是调用 `existing_request.apply_prompt_stream_event()`。

## Request 侧接口框架更新

这一轮主要补的是 `vllm/v1/request.py` 里的流式 prompt 语义层接口。目标不是让 scheduler 现在就完整跑通，而是先让 `Request` 能独立维护流式 prompt 状态，并给 scheduler 暴露少量稳定接口。

### 当前可以认为已经准备好的 scheduler-facing 接口

`Request` 目前给 scheduler 看的接口主要有四个：

```python
request.prompt_stream_event
request.apply_prompt_stream_event(event)
request.is_prompt_stream_finished()
request.can_decode_from_prompt_stream()
request.get_prompt_stream_revision()
```

它们的职责分别是：

- `prompt_stream_event`：保存这一次从 `EngineCoreRequest` 转过来的流式 prompt 事件。scheduler 后续在 duplicate `request_id` 分支里会读取这个字段。
- `apply_prompt_stream_event(event)`：scheduler 的主入口。scheduler 不需要知道 append、update、source finish 的细节，只把事件交给 `Request`，然后拿到 `PromptStreamMutation`。
- `is_prompt_stream_finished()`：判断请求级流式 prompt 是否已经结束。普通非流式请求返回 `True`。
- `can_decode_from_prompt_stream()`：判断当前请求是否越过 decode barrier。普通请求返回 `True`；流式 prompt 请求必须同时满足 prompt 已结束和 prompt 已 prefill 完。
- `get_prompt_stream_revision()`：返回当前 prompt 版本。后续 scheduler 可以把它放进调度快照，避免旧版本 prompt 的执行结果被误用于新版本 prompt。

所以现在可以说：**Request 侧给 scheduler 的接口骨架已经有了，并且 scheduler 已经做了最小调用**。但这还不是端到端完成，因为部分 KV 失效、revision 快照过滤和测试还没有补齐。

### `PromptStreamMutation.noop()`

新增了：

```python
PromptStreamMutation.noop(num_prompt_tokens, prompt_revision)
```

这个方法用于构造“没有真实改变 prompt”的返回值，例如：

- 重复 `chunk_id`。
- 旧版本 update。
- 乱序事件暂存到 `pending_events`。
- 单个 source finished。
- request finished。

它的意义是让 scheduler 不需要特殊判断这些情况，只看：

```python
mutation.applied
mutation.num_tokens_to_invalidate
mutation.prompt_revision
```

如果 `applied=False` 且 `num_tokens_to_invalidate=0`，scheduler 通常不需要做 KV invalidation。

### `PromptStreamState` 的前缀和后缀

这一轮给 `PromptStreamState` 补了两个字段：

```python
prefix_token_ids: list[int]
suffix_token_ids: list[int]
```

原因是流式 source 往往不是完整 prompt 的全部内容。真实请求通常像这样：

```text
固定前缀 + 多个流式 source 内容 + 固定后缀
```

例如：

```text
system prompt / 用户问题
+ 检索结果 source 0
+ 检索结果 source 1
+ answer prefix
```

因此 `_ensure_prompt_stream_state()` 第一次初始化时会把当前已有的 `prompt_token_ids` 保存进 `prefix_token_ids`。后面重建全局 prompt 时使用：

```text
prefix_token_ids + 按 source_order 拼接的 source token_ids + suffix_token_ids
```

这个设计避免第一次收到 stream event 时把原始 prompt 覆盖掉。

### `PromptSourceState.pending_events`

`PromptSourceState` 新增了：

```python
pending_events: dict[int, PromptStreamEvent]
```

这个字段用于暂存乱序到达的事件。当前 MVP 逻辑是：

- 如果 `event.sequence_no != source.next_sequence_no`，先把事件放进 `pending_events`。
- 本次返回 `PromptStreamMutation.noop()`。
- 暂时不自动消费 pending 队列。

也就是说，现在只是把乱序事件保护起来，避免错误应用；后续如果要支持乱序恢复，需要补“缺口补齐后连续消费 pending events”的逻辑。

### `apply_prompt_stream_event()` 的分发逻辑

当前 `apply_prompt_stream_event()` 做了两层事情。

第一层是校验：

```python
if event.request_id != self.request_id:
    raise ValueError(...)
```

第二层是按事件类型分发：

```python
APPEND -> _apply_prompt_stream_append(event)
UPDATE -> _apply_prompt_stream_update(event)
SOURCE_FINISHED -> _mark_prompt_source_finished(event)
REQUEST_FINISHED -> _mark_prompt_stream_finished(event)
```

这意味着 scheduler 后续只需要调用一个统一入口，不需要自己区分不同事件内部怎么改 prompt。

### Append 事件的当前语义

`_apply_prompt_stream_append()` 当前实现的是最小 append 语义：

- 不允许在请求级 prompt stream finished 后继续 append。
- 不允许对已经 finished 的 source append。
- 如果 `chunk_id` 已经出现过，返回 `noop()`。
- 如果 `sequence_no` 不是期望值，把事件放进 `pending_events`，返回 `noop()`。
- 如果事件合法，则把 `event.token_ids` 追加到该 source 的 `token_ids`。
- 然后调用 `_sync_prompt_from_stream_state()` 重建全局 prompt，并生成 `PromptStreamMutation`。

Append 对 scheduler 的效果是：

```text
如果只是尾部追加，common_prefix_len 通常等于旧 prompt 长度。
num_computed_tokens 不会被回退。
scheduler 后续只需要继续 prefill 新增部分。
```

### Update 事件的当前语义

`_apply_prompt_stream_update()` 当前实现的是 source 级完整快照 update：

- 不允许在请求级 prompt stream finished 后继续 update。
- 不允许对已经 finished 的 source update。
- 如果 `chunk_id` 已经出现过，返回 `noop()`。
- 如果 `sequence_no` 不是期望值，把事件放进 `pending_events`，返回 `noop()`。
- 如果 `event.version < source.version`，视为旧版本，返回 `noop()`。
- 如果同一个版本出现不同内容，抛出 `ValueError`。
- 如果事件合法，则用 `event.token_ids` 替换该 source 当前 token 快照。
- 然后调用 `_sync_prompt_from_stream_state()` 重建全局 prompt。

Update 对 scheduler 的效果是：

```text
Request 会计算新旧全局 prompt 的最长公共前缀。
Request 会把 num_computed_tokens 回退到公共前缀以内。
Request 会返回 num_tokens_to_invalidate。
scheduler 后续根据 mutation 释放失效 KV，并重新 prefill 后缀。
```

### Source finished 和 request finished

当前有两个 finished 级别：

- `_mark_prompt_source_finished()`：只标记某个 source 结束，不改变全局 prompt。
- `_mark_prompt_stream_finished()`：标记整个请求级流式 prompt 结束。

这两个函数目前都返回 `PromptStreamMutation.noop()`，因为它们不改变 token 内容。

对 scheduler 来说，真正影响 decode barrier 的是请求级：

```python
request.is_prompt_stream_finished()
request.can_decode_from_prompt_stream()
```

单个 source finished 只是 `Request` 内部状态，scheduler 不应该直接关心。

### `_sync_prompt_from_stream_state()` 做了什么

`_sync_prompt_from_stream_state()` 是当前最关键的内部函数。它把 `PromptStreamState` 同步回 vLLM 原有调度字段。

当前流程是：

1. 用 `_rebuild_prompt_from_stream_state()` 重建新的全局 prompt。
2. 计算旧 prompt 和新 prompt 的最长公共前缀 `common_prefix_len`。
3. 如果 prompt 没变，返回 `noop()`。
4. 计算 `num_tokens_to_invalidate = max(0, old_num_computed_tokens - common_prefix_len)`。
5. 更新：
   - `self.prompt_token_ids`
   - `self._all_token_ids`
   - `self.num_prompt_tokens`
   - `self.num_computed_tokens`
   - `self.prompt_stream.revision`
   - `self.prompt_stream.total_tokens_invalidated`
6. 如果公共前缀小于旧 prompt 长度，清空 `block_hashes` 和 `_prompt_embeds_per_block_hashes`。
7. 调用 `update_block_hashes()` 重新生成 prefix cache 相关哈希。
8. 返回 `PromptStreamMutation`。

这个函数是 Request 和 scheduler 的分界线：

- Request 负责把 prompt 状态改正确。
- Scheduler 负责根据 mutation 去处理真实 KV block 释放和重新调度。

### 当前实现的限制

这部分很重要，后续 review 代码时需要特别看：

- 当前只支持 token-id prompt，不支持 `prompt_embeds` 或 mixed token/embed 输入。`_validate_prompt_stream_can_mutate()` 里已经显式拒绝。
- 当前没有自动消费 `pending_events`，乱序事件只会暂存。
- 当前没有实现 debounce，update 事件会立即应用。
- 当前 `source_order` 是按 source 首次出现顺序确定的；多源排序策略后续还需要设计。
- 当前 `suffix_token_ids` 只是预留字段，还没有入口设置。
- 当前 `total_tokens_recomputed` 还没有更新逻辑，因为 recompute 发生在 scheduler/model runner 侧。
- Scheduler 已经有了最小接入：duplicate `request_id` 的 `prompt_stream_event` 会应用到 existing `Request`；但 `SchedulerOutput` revision 快照、部分 KV 失效和测试还没完成。

### 后续 scheduler 接入状态

这一项现在已经有了最小实现：

- `Scheduler.add_request()` 已经识别 duplicate `request_id` 且 `prompt_stream_event is not None` 的情况。
- 事件会被应用到 scheduler 中的 existing `Request`。
- `schedule()` 已经加入流式 prompt decode barrier。
- `_make_cached_request_data()` 已经能在 prompt 改变后刷新 worker cached state。

仍未完成的是：部分 KV 失效、revision 快照过滤、pending events 自动消费和测试。

## Scheduler 接入后的真实长期状态边界

这一轮把前面讨论的关键边界落到了代码里：**incoming Request 只是事件载体，scheduler 里已有的 existing Request 才保存长期流式 prompt 状态**。

### 新的对象关系

当前数据流应该这样理解：

```text
EngineCoreRequest(prompt_stream_event=event)
  -> Request.from_engine_core_request()
  -> incoming Request
      - prompt_stream_event != None
      - prompt_stream 通常为 None
  -> Scheduler.add_request(incoming)
  -> existing = scheduler.requests[request_id]
  -> existing.apply_prompt_stream_event(event)
  -> existing.prompt_stream 保存长期状态
```

也就是说：

- `incoming.prompt_stream_event` 是这一次 append/update/finish 事件。
- `incoming.prompt_stream` 不应该保存长期状态。
- `existing.prompt_stream` 才是长期状态账本。
- `scheduler.requests[request_id]` 是长期请求对象的唯一入口。

这和 Stream2LLM 的设计一致：Stream2LLM 也是用新来的 `EngineCoreRequest` 携带 chunk/update，然后修改 `scheduler.requests[request_id]` 里的 existing `Request`。

### `Scheduler.add_request()` 的新逻辑

现在 `Scheduler.add_request()` 在发现 duplicate `request_id` 时会优先检查：

```python
if request.prompt_stream_event is not None:
    self._apply_prompt_stream_request(existing, request)
    return
```

这条分支排在原有 `resumable` / `StreamingUpdate` 逻辑之前，原因是流式 prefill 和 vLLM 现有 streaming session 语义不同：

- 流式 prefill 是同一个 prompt 在 decode 前持续变化。
- `resumable` 是一次 generation session 结束后继续输入下一段。

所以流式 prefill 事件不应该走 `StreamingUpdate.from_request()`。

### `_apply_prompt_stream_request()` 做了什么

新增的 `_apply_prompt_stream_request(existing, incoming)` 做三件事：

1. 从 incoming Request 取出 `prompt_stream_event`。
2. 调用长期 Request：

   ```python
   mutation = existing.apply_prompt_stream_event(event)
   ```

3. 如果 `mutation.num_tokens_to_invalidate > 0`，说明已有 KV 后缀不能继续复用，于是采用保守策略释放 KV。

当前代码里明确区分：

```text
incoming Request：只负责携带事件
existing Request：负责保存和更新长期 PromptStreamState
```

这是本轮最重要的逻辑修正。

### 当前 KV 失效策略已改为分域截断版

Stream2LLM 里有：

```python
invalidate_cache_from_position(existing_request, common_prefix_len)
```

它能按 common prefix 保留一部分 KV，只释放后缀。

新版 vLLM 原本没有 Stream2LLM 那种公开的按 token 位置失效接口，所以这一轮新增了 `truncate_blocks_from_position()` 链路：

```python
KVCacheManager.truncate_blocks_from_position(request, keep_tokens)
KVCacheCoordinator.truncate_blocks_from_position(request_id, keep_tokens)
SingleTypeKVCacheManager.truncate_blocks_from_position(request_id, keep_tokens)
```

含义是：

- `keep_tokens` 是安全复用边界。
- KV manager 会按 block size 向上取整，保留覆盖该边界的 blocks。
- 边界之后的尾部 blocks 会被释放。
- scheduler 不再把 `num_computed_tokens` 强行置 0，而是保留 `Request.apply_prompt_stream_event()` 已经回退后的进度。

这个版本比全释放更符合分域语义：Append segment 的 KV 可以保留，Update segment 只释放与上一版本最长公共前缀之后的 KV。

### worker cached state 刷新标记

`Request` 新增了：

```python
self.prompt_stream_needs_worker_refresh = False
```

当 `Request._sync_prompt_from_stream_state()` 发现 prompt 真实改变后，会设置：

```python
self.prompt_stream_needs_worker_refresh = True
```

原因是 scheduler 自己的 `Request` 变了还不够，worker 侧也缓存了 request 的 token/block 状态。如果 prompt 变了但 worker 没刷新，就可能出现：

```text
scheduler 认为 prompt 已 append/update
worker 仍然使用旧 token 序列或旧 block table
```

所以 `_make_cached_request_data()` 现在会检查这个标记：

```python
force_worker_refresh = req.prompt_stream_needs_worker_refresh
```

如果为 True：

- 把该 request 加入 `resumed_req_ids`。
- 把完整 `all_token_ids` 重新发给 worker。
- 用 `kv_cache_manager.get_blocks(req_id)` 发送当前完整 block table。
- 然后清掉 `prompt_stream_needs_worker_refresh`。

这个逻辑的意图是让 worker 把这个 request 当作类似 resumed/preempted request 重新装载，而不是继续 append 到旧 cached state 上。

### decode barrier 的 scheduler 侧实现

新增了：

```python
Scheduler._cap_prompt_stream_num_new_tokens(request, num_new_tokens)
```

它的语义是：

- 普通请求或者已经 `REQUEST_FINISHED` 的流式 prompt 请求，不裁剪。
- 如果流式 prompt 还没结束，只允许调度剩余 prompt token。
- 如果 prompt token 已经 prefill 完，但 request 级流式输入还没结束，本轮调度 token 数变成 0。

也就是说，流式 prompt 未结束时：

```text
可以继续 prefill 已经到达的 prompt token
不能跨入 decode
```

这个函数已经接入两个地方：

- running request 的调度路径。
- waiting request 的调度路径。

waiting request 如果因为 decode barrier 得到 `num_new_tokens == 0`，会被移到本轮 skipped waiting，等待后续 stream event 让 prompt 继续增长或结束。

### 当前已经形成的调度模型

现在 scheduler 看流式 prompt 的方式变成：

```text
1. 新事件到达：Scheduler.add_request(incoming)
2. 找到长期请求：existing = self.requests[request_id]
3. 修改长期状态：existing.apply_prompt_stream_event(event)
4. 如有 KV 失效：释放 blocks，并把 existing.num_computed_tokens 置 0
5. 如 prompt 改变：标记 worker refresh
6. schedule() 使用原有 catch-up 逻辑计算 num_new_tokens
7. decode barrier 把 num_new_tokens 限制在 prompt 范围内
8. worker 收到 refreshed cached data 后使用新 token/block 状态
```

### 现在仍然没有完成的部分

这一步只是把长期状态和 scheduler 接口接起来了，还没有完全完成论文系统需要的全部内容：

- 还没有实现新版 vLLM 的部分 KV 失效；现在是保守全释放。
- 还没有实现 `pending_events` 的自动消费。
- 还没有实现 debounce。
- 还没有实现显式的初始流式请求标记。如果一个请求一开始没有 `prompt_stream_event`，scheduler 仍然会把它当普通请求；后续可能需要 `is_prompt_streaming_request` 或类似字段。
- `source_order` 仍然是 source 首次出现顺序，后续需要根据论文设定确定稳定排序策略。
- `suffix_token_ids` 仍然只是预留。
- `prompt_stream_revision` 已经能从 Request 获取，但还没有真正写入 `SchedulerOutput` 快照，也没有在 `update_from_output()` 里做旧版本结果过滤。
- 还没有针对流式 prompt 写单元测试。

### 目前可以先 review 的重点

你现在 review 代码时，建议优先看这几件事是否符合你的本意：

- `PromptStreamState` 是否应该挂在 existing `Request` 上，而不是 incoming `Request` 上。
- 初始 prompt 是否应该作为 `prefix_token_ids` 保存。
- 多 source 拼接顺序是否可以先用首次出现顺序。
- update 事件是否应该被解释为 source 级完整快照。
- 当前保守全释放 KV 是否可以作为 MVP，还是必须尽早做部分失效。
- decode barrier 是否应该以 `REQUEST_FINISHED` 为准，而不是所有 source finished 自动推断。

## Append / Update 分域 KV 逻辑修正

这一轮根据新的论文语义修正了整体逻辑：Append 和 Update 的 KV cache 不再被当作一个全局 prompt 的统一 LCP 来理解，而是按来源类型分域。

### 固定计算布局

`Request._rebuild_prompt_from_stream_state()` 现在按固定布局重建 token：

```text
prefix_token_ids
+ 所有 Append source token_ids
+ 所有 Update source token_ids
+ suffix_token_ids
```

这替代了之前的“按 source 首次出现顺序拼接”。原因是你的语义要求：

- Append KV 是稳定累积的基础。
- Update 的新版本 prefill 要基于已有 append KV 和 update 前缀 KV。
- Update 快照变化不应该影响 Append 对应的 KV cache。

### Append 事件现在如何影响 KV

Append 事件会扩展 append segment。由于 update segment 位于 append segment 之后，如果之前已经计算过 update token，那么 append 增长后 update token 的上下文发生变化，需要从新的 append 边界后重新计算。

因此 append 事件的效果是：

```text
保留：prefix + 旧 append segment
新增计算：new append chunk
后续 update segment：如果已算过，需要在调度中重新计算
```

当前代码通过重建布局和计算 `common_prefix_len` 得到这个安全边界。

### Update 事件现在如何影响 KV

Update 事件被解释为 update source 的新版本完整快照。处理时：

1. 更新该 source 的 `token_ids`。
2. 重建全局布局：`prefix + append + update + suffix`。
3. 计算旧布局和新布局的 LCP。
4. 由于 append segment 位于 update segment 之前，LCP 会自然包含：
   - fixed prefix。
   - 所有 append token。
   - update snapshot 的最长公共前缀。
5. scheduler 使用该 LCP 作为 `keep_tokens` 截断 KV blocks。

这对应你指定的近似计算：

```text
update suffix prefill context = [已有所有 append KV] + [update 前缀 KV]
```

### 新增 KV 截断接口

为了不再保守释放全部 KV，这一轮新增了按位置截断接口：

```python
KVCacheManager.truncate_blocks_from_position(request, keep_tokens)
```

底层会继续调用 coordinator 和 single-type manager：

```python
KVCacheCoordinator.truncate_blocks_from_position(request_id, keep_tokens)
SingleTypeKVCacheManager.truncate_blocks_from_position(request_id, keep_tokens)
```

当前是 block 粒度截断：

```python
num_blocks_to_keep = cdiv(keep_tokens, block_size)
```

所以如果 `keep_tokens` 落在某个 block 中间，会保留整个 block。这个选择符合 vLLM 现有 KV block 管理粒度，但后续需要验证是否会过度保留 update 不匹配 token 的 KV。

### Scheduler 的变化

`Scheduler._apply_prompt_stream_request()` 现在不再做全释放：

```python
self.kv_cache_manager.truncate_blocks_from_position(
    existing,
    mutation.common_prefix_len,
)
```

并且不再强制：

```python
existing.num_computed_tokens = 0
```

因为 `Request.apply_prompt_stream_event()` 已经将 `num_computed_tokens` 回退到了安全 LCP 内。scheduler 只负责释放 LCP 之后的 KV blocks，并通知 worker 刷新 cached request state。

### 当前仍需确认的问题

- 多个 Update source 同时存在时，现在的 LCP 是对聚合后的 update segment 计算；是否需要每个 update source 独立 KV 域，还需要进一步确定。
- Append 新 chunk 到达后，当前布局会让后续 update segment 失效并重新计算；这符合“update 基于已有所有 append KV”的语义，但会带来额外 update recompute。
- block 粒度截断可能保留超出 token 级 LCP 的 partial block，后续需要测试注意力输出是否会受影响。
- 当前还没有专门的 attention mask 或 block table 分段表达 `[append KV] + [update prefix KV]`，仍然借助单序列布局近似实现。

