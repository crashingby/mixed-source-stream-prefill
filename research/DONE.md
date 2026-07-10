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
- 最后改 scheduler：遇到 duplicate request id 且带 `prompt_stream_event` 时，不走现有 `StreamingUpdate`，而是调用 `existing_request.apply_prompt_stream_event()`。
