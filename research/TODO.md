# Prompt Stream Token 统计与响应透传 TODO

## 目标

为每个成功完成的 Prompt Stream 请求返回一份可审计的 Prefill token 统计，
同时服务两个研究问题：

- **分段流式 Prefill**：统计已经计算、但因 Update 快照变化而失效的 KV token，
  用于比较 segmented stream 与 merged-update baseline 的无效计算量。
- **Update 合并**：统计相对逐 Update 立即计算基线少创建了多少 Prefill token
  工作，用于验证合并策略是否真正减少计算。

统计必须以 Engine 内部的 token、work 和 KV 状态为准。proxy 与 replay 只负责
透传和保存，不能根据文本长度或客户端事件数反推。

## 考察结论

### 当前已经存在的状态

`vllm/v1/request.py` 已经有以下基础：

- `PromptStreamState.total_tokens_invalidated`：已经在新 Update work 激活时，
  按旧的已计算 Update 进度减去可复用 LCP 进行累计。
- `PromptStreamState.total_tokens_recomputed`：字段存在，但目前没有形成完整维护
  逻辑。
- `UpdateCoalescingState.hypothetical_eager_tokens`：当前只统计一个尚未 flush
  窗口的 eager token 成本；每次 flush 后会清零，不能直接表示整个请求累计值。
- `PromptStreamWorkItem.update_common_prefix_len`：可以得到一次真正 admission 的
  Update work 需要计算多少 token。

当前失效统计发生在 `_start_next_prompt_stream_work_item()`：

```text
invalidated = old_update_num_computed_tokens - reusable_update_lcp_tokens
```

这个位置是正确的，因为只有旧 Update work 已完成、下一代 Update 真正准备执行时，
旧 UPDATE KV 后缀才会失效。Update 事件刚到达或仍在 coalescing window 中时不能
提前增加失效 token。

### 当前返回链路

现有成功完成链路是：

```text
Request / Scheduler
    -> EngineCoreOutput
    -> RequestOutput
    -> PromptStreamServing._wait_for_finish_output()
    -> PromptStreamFinishResponse
    -> proxy /v1/prompt_stream/finish 或 /finish_ttft
    -> replay SSE parser
```

当前 `PromptStreamFinishResponse` 只包含 token、文本和 `kv_transfer_params`，没有
Prompt Stream 统计。proxy 构造的 completions JSON/SSE 也没有保存 Engine 内部的
失效或合并数据。

### 是否应放入网络响应

**应该。** 每请求实验需要把 Engine 侧最终统计关联到对应 sample id，只依赖服务端
聚合日志会增加跨进程匹配和日志丢失风险。

但该字段不应放进 OpenAI 标准 `usage`：

- `usage.prompt_tokens` 表示最终输入 token 数，不表示 GPU 实际执行量或反事实节省量。
- invalidated/saved token 是本研究系统的执行指标，不是计费语义。
- 修改 `usage` 会让普通 OpenAI 客户端误解请求长度。

建议新增独立顶层扩展字段：

```json
{
  "prompt_stream_stats": {
    "coalescing_enabled": true,
    "update_events_accepted": 12,
    "update_works_admitted": 4,
    "update_works_completed": 4,
    "eager_update_prefill_tokens": 2048,
    "admitted_update_prefill_tokens": 640,
    "executed_update_prefill_tokens": 640,
    "saved_update_prefill_tokens": 1408,
    "extra_update_prefill_tokens": 0,
    "update_tokens_invalidated": 96
  }
}
```

该扩展只出现在 Prompt Stream 路径，不修改普通 `/v1/completions` 的响应契约。

## 统计口径

### Update 失效 token

字段建议命名为 `update_tokens_invalidated`，明确当前统计只对应 UPDATE KV 域。

对每个真正开始执行的 Update work：

```text
invalidated_i = max(0, old_computed_update_tokens - reused_lcp_tokens)
update_tokens_invalidated = sum(invalidated_i)
```

语义如下：

- 只统计**已经计算并写入 UPDATE KV，但因逻辑 Update 变化而不能继续复用**的
  token。
- 连续多代 Update 使同一逻辑位置反复失效时允许重复计数，因为这些确实是多次
  已执行但被丢弃的工作。
- pending Update 被 coalescer 覆盖时没有产生 KV，因此不属于 invalidation，而属于
  saved token。
- 普通完成、abort 时释放全部 KV 不算 invalidation。
- 内存压力 preemption、cache reset 导致的 KV 释放不算语义 invalidation；如需研究
  系统驱逐，应单独增加 `evicted_prefill_tokens`。
- Prefix+Append 尾部为了分块而加入的 padding 不算 invalidated token。

现有 `total_tokens_invalidated` 已接近这个定义，后续应重命名或在导出时映射为更
明确的 `update_tokens_invalidated`。

### Update 合并少算 token

必须先明确反事实基线：**同一条合法 Update 事件时间线中，每个 Update 都立即形成
work，并使用相邻原始 Update 快照的 source-local LCP。**

对一个 coalescing window：

```text
eager_window_tokens
    = 窗口内每个 Update 相对上一条原始 Update 的待 Prefill token 之和

admitted_window_tokens
    = 最终 admission target 相对上一条已 admission target 的待 Prefill token
```

请求生命周期累计：

```text
eager_update_prefill_tokens = sum(eager_window_tokens)
admitted_update_prefill_tokens = sum(admitted_window_tokens)
net_update_prefill_tokens_avoided
    = eager_update_prefill_tokens - admitted_update_prefill_tokens
saved_update_prefill_tokens = max(0, net_update_prefill_tokens_avoided)
extra_update_prefill_tokens = max(0, -net_update_prefill_tokens_avoided)
```

同时返回 eager、admitted、saved 和 extra，而不是只返回一个 saved 数字。这样在
Append epoch 改变、严格 gate 迫使 Update 从零重新计算等情况下，不会用截断到零的
saved 字段掩盖实际额外计算。

`saved_update_prefill_tokens` 只在启用 coalescing config 时有意义：

- config 为 `None` 时建议返回 `null`，表示未建立合并反事实统计。
- `immediate` controller 应返回接近 0，可用于验证计数逻辑和 controller 开销。
- fixed、debounce、adaptive 和 full_merge 返回请求级累计值。

### Admission 与实际执行应分开

`admitted_update_prefill_tokens` 是算法决定创建的逻辑 work 成本，不一定等于 GPU
最终执行量。preemption 或 replay 可能让同一段 token 再次被调度。

因此建议额外统计：

```text
executed_update_prefill_tokens
```

它应在 Update phase 的 scheduled tokens 被确认推进时累计，并包含因 preemption
发生的重复执行。论文可以同时报告：

- algorithmic saving：eager 对 admitted。
- realized saving：eager 对 executed。

第一版如果暂时不能得到严格的 worker-completed token，可以先记录 Scheduler 实际
scheduled 的 Update token，但字段名必须明确为 `scheduled_update_prefill_tokens`，
不能伪称 completed。

## Request 侧修改方向

- [ ] 新增可序列化的 `PromptStreamTokenStats`，或在 Request 内保存同等字段。
- [ ] 将 `PromptStreamState.total_tokens_invalidated` 明确为 Update 语义失效累计；
  保持当前在 Update work 激活时增加的时机。
- [ ] 新增请求级 `total_eager_update_prefill_tokens`，不能复用 flush 后会清零的
  `hypothetical_eager_tokens`。
- [ ] 新增请求级 `total_admitted_update_prefill_tokens`。在
  `flush_prompt_stream_update()` 冻结 Update work 时，使用最终 target 与 execution
  base 的 LCP 增加该值。
- [ ] 默认无 coalescer 的即时 Update work 也增加 admitted/completed/invalidated 等
  通用计数，但 saved 字段保持 `None`。
- [ ] 新增 `update_events_accepted`、`update_works_admitted`、
  `update_works_completed`，用于校验 token 统计是否与 work 生命周期一致。
- [ ] 明确重复事件、完全相同快照和协议拒绝事件不增加 accepted 计数。
- [ ] 在请求成功结束前提供只读 `build_prompt_stream_stats()`，返回不可变快照；
  不使用 take-and-clear，避免 finish 日志和网络响应竞争消费。
- [ ] 整理未完整维护的 `total_tokens_recomputed`：要么用明确口径实现，要么删除，
  避免它和 admitted/executed/invalidated 三种计数混用。

## Scheduler 与 Engine 输出修改方向

- [ ] Scheduler 在 P 节点生成最终首 token、释放 Request 之前调用
  `build_prompt_stream_stats()`。必须先快照再 `_free_request()`。
- [ ] `EngineCoreOutput` 增加可选 `prompt_stream_stats` 字段，普通请求为 `None`。
- [ ] `OutputProcessor` 将该字段传入 `RequestOutput`。
- [ ] `RequestOutput` 增加同名可选字段，并在 streamed output 聚合时保留最新的非空
  stats。
- [ ] stats 只随完成 Prompt Stream Prefill 的那条 P 输出发送一次，避免每轮 chunked
  Prefill 都复制请求级统计。
- [ ] abort/error 没有正常 finish 响应时，仍通过结构化日志或 Engine metrics 保存
  部分统计，但不伪造成功网络响应。

不建议把 stats 临时塞入 `kv_transfer_params`：该对象属于 P/D KV connector，最终会
传给 D 节点。实验统计与 KV manifest 无关，使用独立 EngineCoreOutput 字段可以避免
connector 忽略未知字段、序列化兼容和 D 侧误读问题。

## P 节点 REST 响应修改方向

- [ ] `PromptStreamPrefillOutput` 增加 `prompt_stream_stats`。
- [ ] `_wait_for_finish_output()` 从 `RequestOutput.prompt_stream_stats` 读取并校验。
- [ ] `PromptStreamFinishResponse` 增加：

```python
prompt_stream_stats: dict[str, int | bool | None]
```

- [ ] `/v1/prompt_stream/start` 与 `/event` 仍只返回 ACK，不附带累计统计。中途返回
  会产生多个互相覆盖的口径，且增加响应开销。
- [ ] `/v1/prompt_stream/finish` 在成功时返回最终请求级 stats。

## proxy 响应修改方向

proxy 收到 P 节点 finish JSON 后，应原样保存 `prompt_stream_stats`，不能重新计算。

- [ ] 非流式 `/v1/prompt_stream/finish`：在最终 completions JSON 顶层加入
  `prompt_stream_stats`。无论请求是否继续访问 D，该字段都来自 P。
- [ ] 流式 `/v1/prompt_stream/finish`：在 P 首 token 对应的第一个 SSE JSON chunk
  顶层加入 `prompt_stream_stats`。后续 D chunk 不重复携带。
- [ ] `/v1/prompt_stream/finish_ttft`：必须在唯一首 token SSE chunk 中加入 stats。
  replay 的 `--ttft-only` 会在首 token 后立即返回，如果 stats 放在 `[DONE]` 前的
  额外尾帧，客户端将无法读取。
- [ ] `_build_prefill_only_response()`、`_merge_prefill_output()` 和
  `_build_prefill_stream_chunk()` 使用同一个 helper 附加 stats，避免三个路径字段不一致。
- [ ] proxy 不把 `prompt_stream_stats` 放进发给 D 的 `kv_transfer_params` 或 decode
  request。

建议 SSE 首帧形态：

```json
{
  "id": "cmpl-<request_id>",
  "object": "text_completion",
  "choices": [
    {
      "index": 0,
      "text": "...",
      "token_ids": [123],
      "finish_reason": null
    }
  ],
  "prompt_stream_stats": {
    "saved_update_prefill_tokens": 1408,
    "update_tokens_invalidated": 96
  }
}
```

## replay 结果修改方向

- [ ] 扩展 `SseMeasurement`，即使未开启 `--save-responses`，也从首帧提取
  `prompt_stream_stats`。当前 `payloads` 只在保存完整响应时使用，不能依赖它。
- [ ] 扩展 `RequestMetric` 和 `request_metrics.csv`，至少保存：
  `saved_update_prefill_tokens`、`update_tokens_invalidated`、
  `eager_update_prefill_tokens`、`admitted_update_prefill_tokens`、
  `executed_update_prefill_tokens`、`update_events_accepted` 和
  `update_works_completed`。
- [ ] segmented-prefill 实验重点比较 segmented stream 与 merged-update baseline
  的 `update_tokens_invalidated` 和实际 Update executed tokens。
- [ ] update-coalescing 实验重点比较 saved/admitted/executed、TTFT 和 work 数。
- [ ] summary.json 增加每个字段的 sum、mean、P50/P95；token 总量主结果应优先
  报告全体请求求和，而不是请求平均值。
- [ ] 结果校验：成功 Prompt Stream 请求缺少 stats 时标记为实验数据错误，不能
  静默填 0。

## 最小正确性用例

- [ ] Append-only：saved 为 `null`，invalidated 为 0。
- [ ] 单 Update：coalescing enabled 时 eager 等于 admitted，saved 为 0。
- [ ] 两个相同 Update：第二个不创建 work，不增加 invalidated；是否计入 accepted
  需要与当前幂等语义保持一致，推荐不计。
- [ ] Update 尾部追加：LCP 复用旧 KV，invalidated 为 0。
- [ ] Update 中部替换：invalidated 等于旧已计算快照中 LCP 之后的 token 数。
- [ ] Update 严格缩短：产生零-forward truncation，仍应累计被截断的 invalidated
  token。
- [ ] coalescing window 内三次 Update 只 admission 最后一版：eager 为三次即时
  delta 之和，admitted 为 base 到最终 target 的 delta，saved 为二者差。
- [ ] strict Append gate：Update 在 gate 内被覆盖时只增加 eager/saved，不提前增加
  invalidated。
- [ ] preemption 后恢复：executed 可以大于 admitted，但 saved 的算法口径不因
  replay 被重复增加。
- [ ] `finish_ttft`：首个 SSE payload 同时包含首 token 与完整 stats。
- [ ] 完整 P 到 D decode：最终非流式 JSON 或首个流式 chunk 保留同一份 P stats，
  D 侧不修改。

## 实施顺序

1. [ ] 先完成 Request 生命周期计数和纯状态测试。
2. [ ] 再透传 `EngineCoreOutput -> RequestOutput -> PromptStreamFinishResponse`。
3. [ ] 修改 proxy 的 JSON、完整 SSE 和 TTFT-only SSE 三条返回路径。
4. [ ] 修改 replay CSV 与 summary，确保 stats 缺失时 fail closed。
5. [ ] 用 `research/scripts/simple_test` 检查单请求返回结构。
6. [ ] 用 segmented-prefill 的三模式小样本验证 invalidated 统计差异。
7. [ ] 用 update-coalescing 的最小策略矩阵验证 saved token 与 TTFT 关系。

## 暂不纳入第一版

- 不把正常 finish/abort 释放的全部 KV 记为 invalidated。
- 不把 block 内未使用容量或 PREFIX_APPEND padding 记为 invalidated。
- 不把普通 OpenAI completions 的 prefix cache hit 混入 Prompt Stream saved token。
- 不尝试在 proxy 端根据文本重新 tokenize 后估算 Engine token。
- 不在 start/event ACK 中持续返回中间统计。
