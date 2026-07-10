# 多源流式 Prefill TODO

## 项目目标

在同一个 RAG 请求中同时接收两类异步上下文，并尽可能将检索延迟与 LLM
prefill 重叠：

- **Append 源**：新 chunk 到达后稳定追加，已到达内容通常不再变化。
- **Update 源**：维护版本化候选集合，已有 chunk 可能被替换、重排或删除。
- **核心目标**：在控制质量损失的前提下，减少失效与重复计算的 prefill token，
  降低 TTFT，并提高高并发场景下的吞吐量。

当前先把 Stream2LLM 作为参考实现，不预设它可以直接复用。阅读源码后，需要
补充其输入协议、调度流程、KV Cache 生命周期以及与当前 vLLM 版本的差异。

## 第一阶段：明确问题与正确性边界

- [x] **定义 chunk 数据模型**：采用请求级事件和来源级状态分离的设计，事件
  携带来源、版本、顺序和 token，`Request` 负责维护各来源的当前状态。
- [x] **定义 Append 语义**：MVP 只允许来源内部尾部追加，不允许撤回或重排；
  重复事件幂等丢弃，乱序事件按 `sequence_no` 暂存后顺序应用。
- [x] **定义 Update 语义**：MVP 使用来源级完整快照，不实现增量 patch；插入、
  替换、删除和重排都表现为一个更高版本的新快照。
- [x] **定义完成条件**：每个来源独立结束；所有来源结束后关闭 prompt，只有
  prompt 已关闭且所有 prompt token 已计算完成时才允许 decode。
- [ ] **定义一致性要求**：确定系统最终必须等价于完整上下文 prefill，还是允许
  使用改变注意力关系的近似结果。
- [ ] **写出注意力可见性矩阵**：明确 Append 和 Update token 在不同到达时刻
  可以关注哪些历史 token。
- [ ] **构造最小事件序列**：覆盖 Append-only、Update-only、Append 后
  Update、Update 后 Append、两类事件交错和高频 Update。

### Chunk 数据模型

#### Stream2LLM 的现状

Stream2LLM 没有独立的 chunk 数据类，而是复用 `EngineCoreRequest`：

- `request_id` 标识同一个流式请求。
- `prompt_token_ids` 在 Append 时表示新增 token，在 Update 时表示新的完整
  prompt，字段语义依赖 `is_prompt_update`。
- `is_streaming_prompt`、`is_streaming_prompt_finished` 和
  `is_prompt_update` 使用三个布尔值表达事件类型和完成状态。
- `arrival_time` 记录请求构造时间，但没有 `source_id`、版本号、来源内序号和
  chunk 幂等标识。

这套接口适合“一个请求只有一种流式输入模式”的实验，但不适合混合多源：后端
无法区分事件来自哪个来源，也无法判断重复、过期和乱序事件。

#### 建议新增的事件结构

建议新增独立的 `PromptStreamEvent`，不要继续给 `EngineCoreRequest` 堆叠布尔值。
初始请求仍携带采样参数和固定 prompt；后续只传轻量事件。

```python
class PromptSourceType(str, enum.Enum):
    APPEND = "append"
    UPDATE = "update"


class PromptStreamEventType(str, enum.Enum):
    DATA = "data"
    SOURCE_FINISHED = "source_finished"
    REQUEST_FINISHED = "request_finished"


class PromptStreamEvent(msgspec.Struct):
    request_id: str
    source_id: str
    source_type: PromptSourceType
    event_type: PromptStreamEventType
    chunk_id: str
    version: int
    sequence_no: int
    token_ids: list[int]
    arrival_time: float
```

字段语义如下：

| 字段 | 语义与约束 |
|---|---|
| `request_id` | Engine 内部稳定请求 ID；同一混合流请求的所有事件必须一致 |
| `source_id` | 同一请求内的来源实例，例如 `web_search_0`、`anns_0` |
| `source_type` | 来源采用 Append 还是 Update 语义；创建后不可改变 |
| `event_type` | 区分数据、单来源结束和请求输入流结束，避免多个布尔组合出非法状态 |
| `chunk_id` | 事件幂等键；同一来源内必须唯一，重试发送不得重复应用 |
| `version` | 来源内容版本；Append 可等于 `sequence_no`，Update 必须严格递增 |
| `sequence_no` | 来源内传输顺序，从 0 或 1 单调递增，用于检测缺口和乱序 |
| `token_ids` | Append 为新增 token；Update 为该来源当前版本的完整 token 快照 |
| `arrival_time` | 前端观察到内容可用的时间，使用单调时钟计算排队与 debounce 延迟 |

还应补充以下工程字段，但不放进首个 MVP 的必需集合：

- `trace_headers`：跨前端、EngineCore 和 scheduler 关联 trace。
- `metadata`：保存检索分数、文档 ID 等实验信息，不进入模型 token 语义。
- `deadline`：为 debounce、来源超时和调度策略提供截止时间。

#### 建议新增的来源状态

事件是传输对象，`Request` 还需要保存每个来源的可变状态：

```python
@dataclass
class PromptSourceState:
    source_id: str
    source_type: PromptSourceType
    token_ids: list[int]
    version: int
    next_sequence_no: int
    seen_chunk_ids: set[str]
    pending_events: dict[int, PromptStreamEvent]
    finished: bool
    last_arrival_time: float
```

当前 vLLM 的 `Request` 建议增加：

- `is_prompt_streaming`：与现有 `resumable` 分开，明确表示 decode 前流式 prompt。
- `prompt_stream_finished`：请求级 prompt 是否关闭。
- `prompt_sources`：`source_id -> PromptSourceState`。
- `prompt_source_order`：来源在最终 prompt 中的确定性布局顺序。
- `fixed_prefix_token_ids` 和 `fixed_suffix_token_ids`：保存模板前缀与最终问题等
  非来源 token，避免它们混入某个来源的版本语义。
- `last_prompt_event_time`：最近一次有效事件时间，用于调度和 debounce。
- `total_invalidated_tokens`、`total_recomputed_tokens`：研究指标。
- `prompt_revision`：每次全局 prompt 发生有效变化时递增，供 scheduler 保存
  调度快照并检测执行期间发生的更新。

#### 在当前 vLLM 中的建议落点

| 文件 | 建议增加的内容 |
|---|---|
| `vllm/v1/engine/__init__.py` | 定义可序列化的 `PromptStreamEvent` 与枚举，或定义专用 `EngineCoreRequestType` |
| `vllm/v1/engine/core_client.py` | 增加同步/异步 `update_prompt_stream` 发送接口 |
| `vllm/v1/engine/core.py` | 解码事件并转交 scheduler，不在这里直接修改 KV 状态 |
| `vllm/v1/request.py` | 定义 `PromptSourceState`，保存多源状态并提供纯状态变更方法 |
| `vllm/v1/core/sched/scheduler.py` | 串行应用事件，更新 prompt、KV 和调度状态 |

新版 vLLM 已经存在 `StreamingUpdate`、`streaming_queue` 和重复 `request_id`
处理，可以参考它的 per-request 排队方式。但现有实现会等一次生成停止后再把新输入
接到 prompt，不能直接承担 Stream2LLM 的 decode 前连续 prefill。

### Append 语义

#### MVP 规则

- **只允许尾部追加**：事件中的 `token_ids` 追加到对应来源的
  `PromptSourceState.token_ids` 末尾。
- **不允许撤回、替换或重排**：需要修改已有内容时，该来源必须声明为 Update，
  不能给 Append 增加隐式回滚语义。
- **来源内部严格有序**：只应用 `sequence_no == next_sequence_no` 的事件。
  更大的序号进入有界 `pending_events`；更小的序号视为重复或过期。
- **幂等去重**：已经存在于 `seen_chunk_ids` 的事件直接忽略，不能重复追加 token。
- **结束后不可追加**：`finished=True` 后收到 DATA 事件应返回明确错误，而不是静默
  修改 prompt。
- **decode 后不可追加**：MVP 保持 Stream2LLM 的约束，一旦产生 output token，
  后续 prompt 事件全部拒绝。

#### 状态更新接口

建议在 `Request` 上提供只负责状态的接口：

```python
def apply_append_event(self, event: PromptStreamEvent) -> PromptMutation:
    ...
```

返回的 `PromptMutation` 至少包含：

- `old_prompt_len` 和 `new_prompt_len`。
- `common_prefix_len`。
- `invalidated_computed_tokens`，Append 正常情况下应为 0。
- `prompt_revision`。
- 是否真正应用、因重复跳过或因乱序暂存。

真正的 KV 分配和释放仍由 scheduler/KV manager 完成，不应由 `Request` 直接操作。

#### 与 Stream2LLM 的差异

Stream2LLM 的 `append_streaming_prompt_tokens()` 直接扩展全局
`prompt_token_ids`，适用于单 Append 源。多源版本需要先更新来源局部状态，再按
确定性布局生成全局 prompt。否则一个来源的事件到达顺序会隐式改变其他来源在
prompt 中的位置，导致无法解释的 KV 失效。

### Update 语义

#### MVP 选择：来源级完整快照

每个 Update DATA 事件携带该来源当前版本的**完整 token 快照**。不在首版支持
`insert/delete/replace/move` 等增量 patch 操作：

- 插入表示新快照多出一段 token。
- 替换表示新快照对应区间发生变化。
- 删除表示新快照缺少旧 token。
- 重排表示新快照中的 token 顺序发生变化。

采用完整快照后，以上操作统一退化为“比较旧快照与新快照”，更适合作为论文的
精确重算基线，也与 Stream2LLM 的 ANNS Update 语义一致。

#### 版本与顺序规则

- `version` 必须严格大于当前版本；相同版本且内容相同视为幂等重试。
- 相同版本但 token 不同属于协议错误，必须拒绝并记录。
- 低版本事件属于过期更新，直接丢弃并增加 stale event 指标。
- `sequence_no` 负责传输顺序，`version` 负责内容新旧，两者不能混为一个字段。
- 高频更新可以先进入 scheduler 侧 debounce 队列，只应用窗口结束时的最高版本。

#### 状态更新接口

建议新增：

```python
def apply_update_snapshot(self, event: PromptStreamEvent) -> PromptMutation:
    ...
```

处理流程为：

1. 校验请求仍处于 prompt streaming 阶段，来源类型为 Update。
2. 校验 `chunk_id`、`sequence_no` 和 `version`。
3. 用新快照替换该来源的 `token_ids`。
4. 按 `prompt_source_order` 重建全局 prompt。
5. 计算旧全局 prompt 与新全局 prompt 的 token 级 LCP。
6. 将 `num_computed_tokens` 回退到安全复用边界。
7. 重建受影响的 `block_hashes`，返回需要失效的 KV 范围。
8. scheduler 调用新版多 KV group 失效接口并安排后续 re-prefill。

注意：LCP 必须针对**最终全局 prompt**计算，不能只比较 Update 来源自己的局部
快照。来源在全局 prompt 中的起始位置和前面的 Append 长度都会改变实际 KV
复用边界。

### 完成与取消语义

#### 单来源结束

- `SOURCE_FINISHED` 只关闭指定 `source_id`，不代表整个 prompt 完成。
- 结束事件也必须携带合法的 `sequence_no`，确保它排在该来源最后一个 DATA
  事件之后。
- 来源结束后，其 token 状态保持不变并继续参与全局 prompt，不释放 KV。
- 重复结束事件可以幂等忽略；结束后再收到 DATA 属于协议错误。

#### 所有来源结束

满足以下条件时设置 `prompt_stream_finished=True`：

- 所有注册来源都已经 `finished=True`。
- 每个来源的 `pending_events` 为空，不存在尚未补齐的序号缺口。
- 固定 suffix（例如问题、结束模板）已经加入全局 prompt。
- 没有等待 debounce 提交的 Update 版本。

`REQUEST_FINISHED` 可以作为前端显式屏障，但不能绕过上述检查。若仍有未结束来源，
应返回错误或按预先定义的超时策略关闭，不能静默丢事件。

#### 允许开始 decode

只有同时满足以下条件，scheduler 才能接收 sampled token：

```python
request.prompt_stream_finished
and request.num_computed_tokens >= request.num_prompt_tokens
and scheduled_prompt_revision == request.prompt_revision
```

其中 `scheduled_prompt_revision` 必须在 `schedule()` 时写入
`SchedulerOutput`。这沿用 Stream2LLM 保存 finished 快照的思路，并进一步防止
model execute 期间收到新事件后，旧版本 sampled token 被误认为有效输出。

#### 请求取消与异常

- 客户端取消继续走当前 `abort_requests()` / `finish_requests()` 路径。
- 取消时清空所有来源的 `pending_events` 和 debounce 队列，并释放请求持有的
  全部 KV blocks。
- 来源失败是否结束整个请求必须显式配置：MVP 建议 **fail closed**，任一必需
  来源失败就终止请求；可选来源可以标记失败后按空快照完成。
- 序号缺口、来源超时和队列溢出不能无限等待；需要记录错误原因并终止或按配置
  降级。

### 四项语义对应的最小源码改动

| 修改区域 | MVP 需要添加的内容 | 可参考的现有实现 |
|---|---|---|
| 传输协议 | `PromptStreamEvent`、事件枚举、client 发送与 core 解码入口 | Stream2LLM 的 `STREAM_ADD`；新版 vLLM 的 `ADD` 序列化链路 |
| `Request` 状态 | `prompt_sources`、revision、完成状态、Append/Update 纯状态方法 | Stream2LLM 的两个 token 更新方法；新版 `StreamingUpdate` |
| scheduler 入口 | 根据稳定 `request_id` 串行处理事件并维护 per-request event queue | 新版 `Scheduler.add_request()` 的重复请求和 `streaming_queue` 处理 |
| KV 失效 | 根据全局 LCP 截断所有 KV group，并同步 block hashes | Stream2LLM 的 `invalidate_cache_from_position()`，但需按新版 coordinator 重写 |
| 调度输出 | 传递 `scheduled_prompt_revision` 和 prompt finished 快照 | Stream2LLM 的 `req_prompt_stream_finished` |
| 输出处理 | revision 匹配且 prompt 完成后才接受 sampled token | Stream2LLM `update_from_output()` 的 partial prompt 抑制逻辑 |
| 取消清理 | 清空事件队列、来源状态和 KV，复用当前 finished/abort 生命周期 | 当前 vLLM 的 `finish_requests()` 与 `_free_request()` |

## 第二阶段：阅读参考实现与定位 vLLM 改动点

- [x] **梳理 Stream2LLM**：已经确认流式 chunk 从实验驱动层经内部 client
  进入 `EngineCore`，由同一 `request_id` 对应的 `Request` 保存增量状态，
  scheduler 在 prompt 未结束时执行 prefill 但丢弃采样结果，结束后才进入 decode。
- [ ] **对齐版本差异**：确认 Stream2LLM 基于哪个 vLLM 版本，区分可直接移植、
  需要适配和已被新版 vLLM 覆盖的部分。
- [ ] **梳理当前 vLLM 请求链路**：从 API 请求、输入预处理、调度器、
  `EngineCore`、模型执行到 KV Cache 管理，标出请求状态的传递位置。
- [ ] **定位增量输入入口**：确认运行中请求能否追加 token；如果不能，确定需要
  扩展的请求消息、队列和状态机。
- [ ] **定位调度入口**：确认新 chunk 到达时由谁触发 prefill，以及它与 decode、
  preemption（抢占）和 continuous batching（连续批处理）的优先级关系。
- [ ] **定位 KV Cache 控制点**：确认 block 分配、共享、失效、释放和重计算分别
  由哪些模块负责。
- [ ] **形成源码改动清单**：调研后将本节替换为准确的文件、类、函数和调用关系，
  不凭模块名称猜测实现位置。

### Stream2LLM 已实现内容

本次核对以 Stream2LLM artifact 提交 `5e2763d` 和本地复现实验提交
`b88394e` 为参考。后者主要修改实验配置、回放脚本和结果文件，没有改动
`stream2llm/vllm/v1` 下的核心实现。

| 层级 | Stream2LLM 实现位置 | 已实现内容 |
|---|---|---|
| 实验驱动 | `experiments/crawler/replay_query_crawls.py` | 回放网页 trace，将内容按到达时间作为 Append chunk 发送 |
| 实验驱动 | `experiments/anns/replay_anns_queries.py` | 回放 ANNS trace，每次用当前 `PipelinePool` 重建完整 prompt 并发送 Update |
| 请求构造 | `experiments/crawler/stream_client.py` | 提供初始、Append、Update 三类 `EngineCoreRequest` 构造函数 |
| 进程通信 | `stream2llm/vllm/v1/engine/core_client.py` | 为每个请求建立输入/输出队列，通过 ZMQ 连续发送 `STREAM_ADD` |
| 请求协议 | `stream2llm/vllm/v1/engine/__init__.py` | 增加 streaming、finished、update 标志和失效 token 输出指标 |
| 请求状态 | `stream2llm/vllm/v1/request.py` | 维护流式状态、最近 chunk 时间、累计失效量，并实现 Append 与 LCP Update |
| 请求入口 | `stream2llm/vllm/v1/engine/core.py` | 根据 `request_id` 识别后续输入，并在 Append 与 Update 路径间分流 |
| KV 管理 | `stream2llm/vllm/v1/core/kv_cache_manager.py` | Update 后释放 LCP 之后的 GPU/CPU KV blocks，并通知 worker 重建 block table |
| 调度控制 | `stream2llm/vllm/v1/core/scheduler.py` | 允许 partial prompt 提前 prefill，保存 finished 快照，prompt 未结束时禁止输出 |
| 调度策略 | `stream2llm/vllm/v1/core/scheduler.py` | 实现 `fcfs`、`mcps`、`lcas` 排序以及 recompute/swap 代价选择 |
| 调度输出 | `stream2llm/vllm/v1/core/scheduler_output.py` | 将流式状态、finished 状态和 block-table 重置信息传给 worker |
| 执行层 | `stream2llm/vllm/v1/worker/gpu_model_runner.py` | 复用已有增量 token、block table 和恢复请求路径，执行新到达 token |

Stream2LLM 的控制流可以概括为：

1. 初始请求、Append 和 Update 都复用同一个 `request_id`。
2. Append 只发送新增 token，保留 `num_computed_tokens` 和已有 KV blocks。
3. Update 发送新的完整 prompt，在 token 级计算最长公共前缀 LCP。
4. `num_computed_tokens` 回退到 LCP，KV manager 释放其后的 blocks。
5. scheduler 继续调度欠缺的 token；worker 使用已有 block table 读取历史 KV。
6. prompt 未 finished 时，本轮采样结果被丢弃；finished 后才返回 decode token。

### 当前 vLLM 的版本差异

- **请求入口已经改变**：当前 `EngineCore.add_request()` 接收的是内部 `Request`，
  不再直接接收并转换 `EngineCoreRequest`。流式更新的分流点需要重新选择，不能
  原样复制 Stream2LLM 的 `EngineCore.add_request()`。
- **消息类型发生冲突**：Stream2LLM 使用 `EngineCoreRequestType.STREAM_ADD =
  b"\\x04"`，当前 vLLM 的 `b"\\x04"` 已经表示 `EXECUTOR_FAILED`。不能复制该
  枚举值，需要扩展新的消息类型或复用当前 client 的其他安全输入通道。
- **调度器目录已经迁移**：当前实现位于
  `vllm/v1/core/sched/scheduler.py`，并拆出了 `interface.py`、`output.py` 和
  `request_queue.py`，不能替换旧版单文件 scheduler。
- **调度输出改成批量结构**：当前 `CachedRequestData` 以多请求数组和
  `resumed_req_ids` 传输状态，不再是每个请求一个带布尔字段的对象。
- **KV Cache 已支持多 group**：当前 `KVCacheManager` 通过 coordinator 管理
  多种 KV cache group。旧版直接裁剪 `req_to_gpu_blocks`/`req_to_cpu_blocks`
  的代码不能移植，失效操作必须对所有 group 保持一致。
- **prefix cache 状态更复杂**：当前 `Request` 持有 `block_hashes`，KV manager
  会缓存 blocks。修改 prompt 后还需要同步 block hash、cache 引用和统计信息，
  不能只修改 token list 与 `num_computed_tokens`。
- **执行层基本机制仍然存在**：当前 worker 仍用 `new_block_ids`、
  `num_computed_tokens` 和 `resumed_req_ids` 区分追加与恢复，并维护持久化
  block table。这部分可以复用现有机制，不应先修改 attention kernel。
- **当前 `resumable` 不是 prompt streaming**：它用于生成流的暂停/续传，不能
  直接替代 Stream2LLM 的“decode 前持续增加或更新 prompt”。

### 移植价值判断

#### 可以直接参考或近似直接移植

- [ ] **复用实验事件语义**：保留“同一请求 id + 初始/Append/Update/Finish
  事件”的建模方式，但补充 `source_id`、`version` 和 `sequence_no`。
- [ ] **复用 LCP 算法作为精确基线**：Update 收到完整 prompt 后计算 token 级
  LCP，并将已计算位置回退到 LCP。算法本身简单稳定，但状态更新需要适配新版。
- [ ] **复用 Append 状态不变量**：Append 只能发生在 decode 前，只增加 prompt
  token，不修改已计算前缀，也不能在 finished 后继续追加。
- [ ] **复用实验 trace 驱动思路**：Crawler 和 ANNS replay driver 可以作为单源
  对照组；新增 mixed-source driver，在同一请求中交错回放两类事件。
- [ ] **复用指标定义**：保留 TTFT、E2E、失效 token、抢占事件和吞吐量采集，
  但重新定义论文中的用户侧 TTFT 起点。

#### 思路可复用，但必须按当前 vLLM 重写

- [ ] **重写请求协议**：在当前 `EngineCoreRequest` 或独立 update 消息中表达
  Append/Update/Finish；不要复用旧 `STREAM_ADD` 的枚举值。
- [ ] **重写 client 输入流**：参考 Stream2LLM 的 per-request queue，但适配当前
  `EngineCoreClient`、序列化结构、DP wave 和多 client output routing。
- [ ] **扩展当前 `Request`**：增加多源事件状态、finished 状态、版本号、累计
  invalidation 等字段，并在修改 prompt 后重建必要的 block hashes。
- [ ] **重写 KV 后缀失效接口**：基于当前 coordinator 和多 KV group 实现
  `invalidate_from_position`，正确处理 partial block、prefix cache 引用和 block
  释放顺序。
- [ ] **扩展当前 scheduler**：保存“本轮调度时 prompt 是否结束”的快照，避免
  chunk 在 model execute 期间到达后错误接收 sampled token。
- [ ] **复用新版恢复通道**：KV 截断后将请求放入 `resumed_req_ids`，使 worker
  用新 block ids 重建状态；必须满足当前实现中 resumed request 不在 persistent
  batch 的断言，不能只设置一个标志。
- [ ] **扩展输出统计**：优先接入当前 `PrefillStats` 或 metrics 系统，不把研究
  指标永久塞进稳定的 `EngineCoreOutput` 公共接口。

#### 暂不直接移植

- [ ] **暂不移植旧 model runner 与 attention 代码**：当前 vLLM 已经具备增量
  token、paged KV、block table 和 resumed request 执行路径。先证明控制面无法
  表达需求，再考虑修改执行层。
- [ ] **暂不移植 recompute/swap predictor**：它解决显存抢占选择，不是混合源
  prefill 的首要功能；而且旧实现用 `num_new_tokens` 近似 swap block 数。
- [ ] **暂不移植 FCFS/MCPS/LCAS 全局排序**：先用当前默认 scheduler 跑通正确性
  基线，再将多源事件新鲜度、失效代价和 debounce 纳入新的调度策略。
- [ ] **不照搬 LCAS 当前实现**：源码按 `last_chunk_arrival_time` 升序排列，与
  “most recent first”的注释相反，需要先确定论文期望语义。
- [ ] **不照搬 profile 拟合方法**：artifact 使用二次多项式，而论文描述为分段
  线性；若后续需要代价模型，应重新采样和验证误差。

### 推荐的最小改动边界

第一版按照以下顺序推进，尽量让执行层保持不变：

1. **实验层**：构造 mixed-source 事件回放器，先不接真实搜索和 ANNS 服务。
2. **协议层**：让同一请求能够安全发送 Append、Update 和 Finish 事件。
3. **Request 层**：维护版本、prompt、LCP、`num_computed_tokens` 和统计信息。
4. **KV 管理层**：实现当前多 group 架构下的精确后缀失效。
5. **Scheduler 层**：允许 partial prompt prefill，并在 finished 前抑制输出。
6. **Worker 层**：优先复用当前 `resumed_req_ids` 和 block table 重建路径。
7. **优化层**：正确性通过后，再加入 debounce、混合源调度和近似分区 KV。

## 第三阶段：建立无优化基线

- [ ] **实现离线完整上下文基线**：等待所有来源结束，统一过滤、去重、rerank，
  再执行一次完整 prefill。
- [ ] **实现朴素流式基线**：每次事件到达后重新拼接当前上下文，并对失效后缀
  重新执行 prefill。
- [ ] **支持三种固定布局**：Append 在前、Update 在前、按事件到达时间排列。
- [ ] **记录 KV 失效范围**：输出最长公共前缀、失效 token 数和重算 token 数。
- [ ] **验证基线正确性**：朴素重算的最终 token、logits 和完整上下文结果应在
  允许误差内一致。

## 第四阶段：实现混合流式请求

- [ ] **扩展请求协议**：让同一个请求接收多个来源的 chunk 事件，而不是把每次
  更新伪装成新请求。
- [ ] **增加请求级状态机**：管理来源状态、当前版本、已处理 token、待处理事件、
  prefill 进度和 decode 启动条件。
- [ ] **实现 Append 路径**：只处理新到达的稳定后缀，并复用此前 Append token
  的 KV Cache。
- [ ] **实现 Update 路径**：比较新旧候选版本，识别稳定前缀、变化区间和必须
  重算的后缀。
- [ ] **处理过期事件**：丢弃乱序到达的旧版本，避免旧 Update 覆盖新状态。
- [ ] **处理请求终止**：覆盖完成、取消、超时和来源异常，确保 KV block 与事件
  缓冲区被释放。

## 第五阶段：设计 KV Cache 与注意力方案

- [ ] **实现精确重算方案**：以最终结果等价为目标，只复用语义上仍然有效的
  最长前缀 KV Cache。
- [ ] **实现分区 KV 原型**：分别维护 Append 与 Update 的逻辑分区，验证两类
  缓存能否独立增长和失效。
- [ ] **明确分区方案的注意力缺口**：例如先计算的 Update token 没有关注后来
  到达的 Append token，不能将其误称为精确 KV 复用。
- [ ] **决定近似修正策略**：比较周期性全量刷新、局部重算、屏障式合并或显式
  attention mask 等方案。
- [ ] **定义刷新条件**：根据累计缺失 token、Update 版本数、时间窗口或质量风险
  触发精确重算。
- [ ] **验证 block 生命周期**：覆盖分支更新、抢占、swap、prefix cache 命中和
  请求结束，避免引用错误或 block 泄漏。

## 第六阶段：Update Debounce 与调度策略

- [ ] **实现 debounce 时间窗**：短时间内多个 Update 只对最新版本执行一次
  prefill。
- [ ] **实现事件合并**：在调度前合并同一来源的连续更新，并统计被跳过的版本。
- [ ] **支持多种触发条件**：固定时间窗、累计变化 token 数、候选集合稳定度和
  GPU 空闲机会。
- [ ] **处理饥饿问题**：为持续更新的来源设置最大等待时间。
- [ ] **定义调度优先级**：比较 Append prefill、Update prefill、刷新任务和
  decode 在不同负载下的优先顺序。
- [ ] **加入背压机制**：更新速度超过处理速度时限制队列长度，只保留必要版本。

## 第七阶段：后处理与质量对照

- [ ] **完整后处理组**：等待所有来源完成后执行全局过滤、去重和 rerank。
- [ ] **无后处理组**：按来源返回结果直接进入流式 prefill。
- [ ] **轻量在线后处理组**：仅执行可增量完成的去重、阈值过滤或来源内排序。
- [ ] **定义最终上下文顺序**：固定来源优先级、相关性排序或到达顺序，避免实验
  中隐式改变输入语义。
- [ ] **定义质量指标**：根据任务选择 EM、F1、Recall、答案正确率、引用准确率
  或 judge score。

## 第八阶段：可观测性与实验

- [ ] **增加事件级 trace**：记录来源、版本、到达时间、排队时间、调度时间、
  prefill 区间和失效区间。
- [ ] **统计 `Invalidated KV tokens`**：衡量输入变化造成的缓存失效量。
- [ ] **统计 `Recomputed prefill tokens`**：衡量实际重复计算量。
- [ ] **统计 `Wasted GPU time`**：估计最终被丢弃计算占用的 GPU 时间。
- [ ] **测量 TTFT 与 p95 TTFT**：报告不同检索延迟和 Update 频率下的结果。
- [ ] **测量高 QPS 吞吐量**：观察混合 prefill、decode 干扰和调度公平性。
- [ ] **测量质量变化**：比较完整后处理、朴素流式、精确复用和近似分区方案。
- [ ] **执行消融实验**：分别关闭 debounce、分区 KV、周期刷新和在线后处理。
- [ ] **建立工作负载矩阵**：覆盖来源数量、Append/Update 比例、chunk 大小、
  Update 频率、检索延迟和并发请求数。

## 测试清单

- [ ] **单元测试**：覆盖版本比较、事件合并、失效区间计算和状态转换。
- [ ] **正确性测试**：将精确方案的最终 logits 与完整上下文基线对齐。
- [ ] **乱序与重复测试**：覆盖重复事件、旧版本晚到和来源结束后继续更新。
- [ ] **资源测试**：覆盖取消、超时、抢占和异常路径，检查 KV block 是否释放。
- [ ] **并发测试**：覆盖多个混合流请求同时运行以及与普通请求共存。
- [ ] **性能回归测试**：确保未启用该功能时不会明显影响 vLLM 原有请求。

## 当前最小可行版本

- [ ] 仅支持一个 Append 源和一个 Update 源。
- [ ] Append 只允许尾部追加；Update 使用单调递增版本号和全量候选快照。
- [ ] 暂不在流式阶段执行全局 rerank，只保留离线完整上下文对照组。
- [ ] 先实现**精确重算基线**，确认事件协议、状态机和指标采集正确。
- [ ] 再加入固定 debounce 时间窗，验证减少无效 prefill 的收益。
- [ ] 最后实现分区 KV 近似原型，并单独报告质量损失。

## 待确认问题

- [ ] Stream2LLM 源码位于哪里，基于哪个 vLLM commit 或 release？
- [ ] 首要目标是精确等价，还是允许通过近似注意力换取更低 TTFT？
- [ ] Update 候选集合是否保持固定顺序，替换 chunk 后是否需要全局 rerank？
- [ ] decode 应在什么条件下启动，启动后是否仍允许接收上下文更新？
- [ ] 首轮实验使用真实搜索/ANNS 系统，还是先用可控延迟的事件回放器？
