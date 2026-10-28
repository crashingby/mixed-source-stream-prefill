# 稳定 Prefix KV 的提前传输设计、实现与实验

当前 P/D 交接由 D 生成全部回答；P-only 使用 `prefill_ready` 控制事件测量 P 侧 TTFT。协议、兼容字段与验证命令见 [P/D 交接说明](scripts/server/PD-Handoff.md)。

## 设计状态与结论

本文记录混合来源流式 Prefill 的稳定 KV 提前传输方案、当前源码实现和截至 **2026-10-05** 的实验结论。区间级 NIXL WRITE、D 只接收会话、最终 block table 收养、专用 proxy 与配对 replay 已实现。除真实双卡 smoke 外，已完成原有性能集的两轮对照、稳定知识占比与 QPS 的受控矩阵，以及一次短区间 nsys 排查。

**历史实验说明：** 下述正式实验及已有数值使用旧版 P 首 token 交付协议；当前执行协议见上方链接。正式实验在同一主机的两张 A30 上进行，共提交 5120 次非预热测量请求，5119 次完整成功且统计有效。顺序负载下，受控场景的 P→D 交接 P95 改善约 7.4%–51.5%；但部分并发场景的 P 首 token 和客户端续写等待反而变慢。因此当前结论是**可以把稳定 KV 的传输移出最终交接，并缩短特定条件下的交接尾延迟**，不是全负载、无代价的端到端加速。

核心目标是在知识源持续产生内容时，把已经稳定的 `PREFIX_APPEND` KV 提前送到 D，隐藏最终 P/D 交接中的部分传输时间。不提前计算 D 的输出，不改变 Update 的近似语义，也不优化抢占恢复。

结论如下：

- **不需要每个 Append 都由 proxy 向 D 提交一次生成请求。** Proxy 提前建立一次 P/D 接收会话，之后 Append 仍只提交给 P；计算完成后的 block 就绪消息与增量传输由 connector 处理。
- **D 不通过假 decode 接收 KV。** Worker 复用 `kv_connector_no_forward()`，用零模型 token 的步骤推进传输；接收会话不进入模型 `running`，最终数据齐备后才收养到生成 Request。
- **普通 Push 与增量 Push 分开处理。** 原有 `NixlPushConnector` 的终态路径仍保留；开启 `stable_kv_push` 后，新增 controller 与 worker 区间协议处理运行期间的稳定块。
- **仅提前发送真正完成的完整真实 token 块。** `PREFIX_APPEND` 最后包含 padding 的块可能被后续 Append 改写，留到 finish 再传。Update 与 Question 同样留到最终交接。
- **现有 TTFT-only 实验不能证明这项优化有效。** Proxy 先返回 P 完成通知，再等待 D；提前传输主要影响 D 首个回答 token 的等待时间和完整响应时间。

分段计算与近似语义见 [DONE.md](DONE.md)，空洞掩码与位置修正见 [Masked-Padding.md](Masked-Padding.md)，运行名额与 dormant 规则见 [Scheduler.md](Scheduler.md)。本方案只叠加传输流水线，不重新设计这些计算路径。

服务启动见 [server README](scripts/server/README.md)，指标口径与可复制命令见 [stable_kv_push README](scripts/replay/stable_kv_push/README.md)，逐轮实验记录见 [EXPERIMENTS.md](scripts/replay/stable_kv_push/EXPERIMENTS.md)。验证固定为 g2c9 上一个 P、一个 D，g2c14 上 proxy 与 replay；模型为 Qwen2.5-3B，Masked Padding 开启。输入包括原有 `mix_source_perf_qwen` 和三个独立生成的稳定知识占比场景，没有改写原有性能集或质量集。

## 原有 P/D 分离流程与修改理由

### 普通 Proxy 在 finish 后才调用 D

当前 [proxy_server.py](scripts/server/proxy_server.py) 的 `_start_prompt_stream()` 会选择并绑定 P、D 两个实例，但只向 P 发 `/prompt_stream/start`。绑定 D 的 HTTP client 不等于已经在 D EngineCore 中创建请求。

| 时刻 | 当前 proxy 行为 | 当前 D 状态 |
| --- | --- | --- |
| `start` | 向 P 创建长期 Request，设置 `max_tokens=1` 和完整 PD 的 transfer 参数。 | 尚无本请求，也未分配目标 blocks。 |
| Append / Update | `/event` 只转发 P，维护 P token ids 的镜像。 | 仍无本请求。 |
| `finish` | 向 P 提交 Question，等待 P 返回首 token、完整物理 prompt 和 KV manifest。 | 仍未接到生成请求。 |
| P finish 返回后 | 构造 D 的普通 completions 请求，带完整物理 prompt 和 transfer 参数。 | 开始正常请求准入、分配 blocks 和加载 KV。 |
| 远端加载完成 | 继续转发 D 输出。 | 解除远端 KV 等待，开始本地 forward / decode。 |

源码入口是 `_start_prompt_stream()`、`add_prompt_stream_event()`、`finish_prompt_stream()` 和 `_stream_pd_completion()`。其中 SSE 路径明确**先 yield P 完成通知，再向 D 发请求**。

[toy_proxy_server.py](../tests/v1/kv_connector/nixl_integration/toy_proxy_server.py) 也采用先向 P 发 `max_tokens=1` 请求、再把 transfer 参数交给 D 的基本交接方式；它没有为流式 Append 提前创建接收端的机制。当前研究 proxy 另外补充了物理 prompt、padding 与严格 manifest 校验，不能为了复用 toy proxy 而丢掉这些信息。

### Pull 的三个调用点

原有 `start_p_node.sbatch` / `start_d_node.sbatch` 使用 `NixlConnector`。在 [connector.py](../vllm/distributed/kv_transfer/kv_connector/v1/nixl/connector.py) 中，它是 `NixlPullConnector` 的别名，并非根据请求自动切换 Push。

| 调用点 | 当前职责 | 为什么不能单独实现提前传输 |
| --- | --- | --- |
| `get_num_new_matched_tokens()` | 判断 D 可以从 P 加载多少已经计算的 token；流式 manifest 使用 `remote_num_tokens` 限定范围。 | D 没有请求或最终 manifest 时，这条普通准入路径不会自行出现。 |
| `update_state_after_alloc()` | D 分配本地 blocks 后，建立本地目标与 P 远端 blocks 的对应关系。 | 默认以本次普通请求的完整外部 KV 为单位，不是一个持续增长的接收会话。 |
| `start_load_kv()` | Worker 根据已生成的 metadata 发起 NIXL READ。 | 发起 READ 不会自动处理最终问题屏障、目标扩容及多次接收状态。 |

对应实现位于 [pull_scheduler.py](../vllm/distributed/kv_transfer/kv_connector/v1/nixl/pull_scheduler.py) 和 [pull_worker.py](../vllm/distributed/kv_transfer/kv_connector/v1/nixl/pull_worker.py)。P 的 `request_finished()` 暴露完整远端 block IDs 并延迟释放源 blocks，D 才能在 finish 后来读取它们。

当前 D 会进入 `WAITING_FOR_REMOTE_KVS`。在 [Scheduler](../vllm/v1/core/sched/scheduler.py) 的 `_try_promote_blocked_waiting_request()` 中，收到请求级 `finished_recving` 后会返回普通 waiting / running 路径。因此，**现有远端等待屏障只表示本次加载结束，不表示流式知识输入已经结束**。把第一批提前 KV 塞入现有路径，会有过早解除屏障的风险。

### 原有 Push 等待 P 的终态 blocks

显式选择 `NixlPushConnector`、未开启 `stable_kv_push` 时，数据方向为 P 发起 NIXL `WRITE`，时序是：

1. D 在 `update_state_after_alloc()` 中分配目标 blocks，生成 Push registration。
2. P 在 `request_finished()` 中保存最终 blocks，写入 `_newly_finished_push_blocks`。
3. `build_connector_meta()` 把 registration 与 finished blocks 传给 worker。
4. P 的 `nixl-push-writer` 线程匹配两份信息，调用 `_do_start_push_kv()`，再执行 `_xfer_blocks_for_req()` 的 `WRITE`。
5. 完成通知通过 `get_finished()` 转成整请求的 `finished_sending` / `finished_recving`。

源码位于 [push_scheduler.py](../vllm/distributed/kv_transfer/kv_connector/v1/nixl/push_scheduler.py) 与 [push_worker.py](../vllm/distributed/kv_transfer/kv_connector/v1/nixl/push_worker.py)。Registration 与 P finished blocks 可以任意先到，writer 会等待匹配；但 P finished blocks 当前只有终态才发布。

原有路径存在三个不能直接用于增量传输的边界；新实现为此增加了独立的区间协议：

- Worker 的 transfer handle、registration 和完成通知以请求 ID 为主键，通知包含请求 ID 与 TP 信息，尚不能区分同一个请求的多批区间。
- `_xfer_blocks_for_req()` 当前会把 P/D 不同长度的 block 列表裁短。增量协议不能沿用这种行为，必须验证精确的范围、组数和块数；否则“传输成功”可能只代表错误的子集完成。
- 当前 `finished_sending` 会推进整请求释放，`finished_recving` 会解除整请求远端等待。它们不能用于普通提前区间，否则会提前释放 P blocks 或让 D 开始生成。

## 增量 Push 的职责分工

理论上可以让 D 收到区间消息后分批 READ，即增量 Pull。但它同样需要接收会话、动态分配、区间完成和最终 decode 屏障，不能通过重复调用普通 completions 自动解决。

实现采用 **`NixlPushConnector` 扩展的区间级 WRITE 协议**。
[StableKVPushWorker._submit_write()](../vllm/distributed/kv_transfer/kv_connector/v1/nixl/stable_push.py)
精确匹配源、目标 descriptors 后调用 `make_prepped_xfer("WRITE", ...)`，由 P 主动写入 D。同一个 WRITE 的 NIXL notification 向 D 报告 `received`；P 通过本地 handle 的完成状态报告 `sent`，两个完成方向分别汇合到各端 controller。

控制与数据各自承担的职责如下：

| 部分 | 职责 | 不承担的职责 |
| --- | --- | --- |
| 前端 | 提交 start、Append / Update、source finished 与最终 Question。 | 不提交目标 block IDs，不判断 GPU 是否完成。 |
| Proxy | 选择并绑定 P/D；建立接收会话；finalize；汇合输出与取消。 | 不搬运 KV，也不把每个 Append 改造成 D 的生成请求。 |
| P Scheduler | 判定完整稳定范围真正完成，固定源 block IDs 与代次。 | 不凭 HTTP ACK 宣布 KV 就绪。 |
| D Scheduler / KV manager | 逐步分配接收 blocks，处理容量与最终收养。 | 不为接收 KV 执行模板、Question 或假采样。 |
| NIXL worker | 交换目标注册消息，执行 WRITE，报告区间完成 / 失败。 | 不在后台线程直接修改 Scheduler 的请求和队列。 |

“每次 Append 都产生传输”也不是要求：一个短 chunk 可能尚未产生新完整块，多次 Append 可合成一批；一个长 chunk 的 prefill 分多轮完成时，也可以逐步产生多个可发送范围。

## 什么 KV 可以提前发送

### 完整真实块构成稳定前缀

令 $B$ 为逻辑 KV block 的 token 容量，$C(t)$ 为 P 在时刻 $t$ **真正完成 forward 的连续 Prefix+Append 真实 token 数**，不含 padding。可提前发送的连续完整块数量为：

$$
N(t)=\left\lfloor\frac{C(t)}{B}\right\rfloor.
$$

稳定范围是物理 token 区间 $[0,N(t)B)$。这里的稳定只针对同一存储代次、相同模板和只增不改的 Append；不涵盖重算后换块、模型变更或内存释放。

后续 Append 在块内覆盖的是之前的 padding，而不是已经存在的真实 token。因果 attention 下，新增尾部 token 不会改变前面真实 token 的 KV。因此，已经完成的完整真实块可以只传一次。

### 尾块与 Update 不提前发送

假设 $B=16$：

| 已完成真实 Prefix+Append 长度 | 当前 PA 物理长度 | 可提前发送 | 暂不发送 |
| --- | --- | --- | --- |
| 19 | 32 | 第 0 块，16 个真实 token。 | 第 1 块含 3 个真实 token 和 13 个 padding。 |
| 新 Append 后达到 37 | 48 | 第 0、1 块；第 1 块现在已全是真实 token。 | 第 2 块含 5 个真实 token 和 11 个 padding。 |
| 最终仍为 37 | 48 | 已成功接收的前两块无需重传。 | finish 时传第 2 块，再传 UPDATE 域。 |

开启 Masked Padding 后，padding 仍有物理槽位和 KV；只是后续真实 query 不读取它们，并修正语义位置。因此 mask 不会把动态尾块变成可以提前发送的稳定块。

Update 的内容与 LCP 会变化，已算 KV 还可能绑定较早的 Append 上下文。第一版不提前传 Update。最终直接复制 P 已有的 Update / Question KV，**不借传输机会重新计算成另一套 attention 结果**。

### 不能直接使用当前 computed 水位

这是从源码发现的关键实现风险。[Scheduler](../vllm/v1/core/sched/scheduler.py) 的 `_update_after_schedule()` 在生成 `SchedulerOutput` 后就调用 [Request](../vllm/v1/request.py) 的 `advance_prompt_stream_computed_tokens()`。它会推进 `prefix_append_num_computed_tokens`，在 work 达到边界时更新 `prefix_append_committed_logical_len`，甚至清空 active work。

这些状态用于调度流水线，**不能单独作为 NIXL 可以读源内存的 GPU 完成证明**。`/event` 的 ACK 更早，只证明事件被接受和送入控制路径。

实现落在 [StableKVPushController](../vllm/v1/core/stable_kv_push.py)：

- `capture_scheduled()` 在 scheduler 提前推进 computed 计数之前，保存该 `SchedulerOutput` 内 PA 阶段的请求、真实 token 结束位置和 `num_preemptions`。
- `complete_scheduled()` 在对应同步 worker 步骤回包后，确认请求仍有效且未被抢占，再将完整真实块交给 `_offer_prefix()`。回包不依赖已经切换的 active work。
- `_offer()` 固定本批源 block IDs，并为其增加独立传输引用。`offered_blocks` 表示已经发布到哪一个序列块，并不表示网络已完成到那里。
- P bind 晚于模板完成时，只在没有 `_prompt_stream_inflight_steps` 的情况下读取已完成的 committed logical 长度。
- [Push Worker](../vllm/distributed/kv_transfer/kv_connector/v1/nixl/push_worker.py) 将 WRITE 命令交给后台 writer 前执行 `torch.cuda.synchronize()`，保证此前计算流写入对传输可见。首版使用这个保守同步边界，没有实现更细粒度 CUDA event 依赖。

因此首版要求 `--no-async-scheduling`、TP/PP/DP 均为 1。它没有声称已经支持多个异步在途批次的连续完成前沿。后续若放宽这个约束，需要重新设计批次完成汇合，而不能直接移除启动检查。同步在这里承担正确性职责；后文的 nsys 结果没有支持将它认定为本轮退化的主要原因。

## D 使用独立的只接收会话

### 不先创建一个普通生成 Request

新增 `StableKVSession`，由 D Scheduler 的 `StableKVPushController.sessions` 持有。它属于一个业务请求，但在最终接收就绪前不是普通模型生成 Request。

理由是当前普通 Request 的 token、采样、output queue、KV allocation 和远端加载都以生成生命周期为中心。此时 D 尚不知道完整 prompt，不应该伪造 prompt 或设置 `max_tokens=0` 绕过生成。

实际会话状态与对应字段如下：

| 状态 | 用途 |
| --- | --- |
| `session_id` | Proxy 生成的唯一传输身份，与 P 实际内部 `request_id` 分别保存。 |
| `request_id`、`peer`、`producer` | 确定请求、对端 engine / host / port 和本端角色，生命周期中不换 D。 |
| `generation` | 所有区间与通知携带的代次；首版会话从 0 开始，抢占后失败，不原地重建下一代。 |
| `reservation_id` | KV manager 的独立 ownership key：`stable-kv:<session_id>:<generation>`。 |
| `ranges` | `range_id → StableKVRange`，保存精确源/目标 IDs、半开序列块区间、`granted` 与 `completed`。 |
| `final_tokens`、`final_params` | finish 后冻结的完整物理 prompt、最终 transfer manifest 和 mask。 |
| `adopted_request_id` | 记录块表已经交给哪个 D 生成请求，阻止旧会话再次释放它。 |
| `state`、`last_activity`、`local_drained`、`peer_drained` | 接收、最终就绪、收养、失败及安全释放状态。 |

一个 block 在 D 分配出来，不等于已有有效 KV。最终 `num_computed_tokens` 只能来自经过完成通知和 manifest 校验的有效连续范围。

### 接收不占模型运行名额，但占显存

Receive-only 会话不进入普通 `running`，也不计入模型 `MAX_NUM_SEQS`；它没有模型 token budget。它占用的 D blocks 必须真实计入同一个 KV 显存池，并受会话数、总保留块数与准入阈值限制。

Start 的 reserve 只创建会话，不为每个请求预分配 `MAX_MODEL_LEN` 对应的全部 blocks。D 的 `_grant_pending()` 根据 offer 逐步扩容；所有尚未收养会话的块占用受 `stable_kv_max_reserved_fraction` 限制，默认不超过总块池的 0.5。`reserve_stable_kv_blocks()` 同时检查块池空闲量和 watermark。**D 暂时不能分配时保留待处理 offer，之后重试，不发布不存在的目标地址。**

提前接收使 D 在尚不能生成的阶段持有 KV，这不是免费优化。第一版不研究如何优先驱逐这些会话，但必须明确容量不足、取消与超时的终止行为；不能无限保留或无限等待。

### 零模型步骤驱动 connector

[GPUModelRunner.execute_model()](../vllm/v1/worker/gpu_model_runner.py) 在 `total_num_scheduled_tokens == 0` 且存在 KV connector 时，会进入 [kv_connector_no_forward()](../vllm/v1/worker/kv_connector_model_runner_mixin.py)。它仍执行 metadata 绑定、`start_load_kv()`、完成与错误回收，而不执行本请求的模型 forward / sampling。

首版已把开放会话接入 `has_runnable_engine_work()` 和 `has_requests()`。EngineCore 在步前、scheduler 回包后处理 controller 消息；writer 只向线程安全事件队列写入，不直接分配或释放 block pool。

**当前采用有界轮询，没有实现专门的后台消息唤醒 FD。** 只要还有尚未收养/清理的开放会话，EngineCore 可运行零 token connector 步骤；没有模型执行时 sleep `stable_kv_poll_interval_ms`，默认 5 ms。这样能及时搬运 worker 收到的 offer/grant/completion，同时不运行 dummy 模型。

这与最初设想的“完全空闲时睡眠，消息驱动唤醒”有差别：开放空会话也会产生有限频率的 metadata-only 调度，控制握手可能跨几个 5 ms 周期。它是首版明确的 CPU / 交接时延成本，实验需包含在 `finalize_wait_ms` 和 P→D 时延中。未来可用真正的事件唤醒替换，但两组对照应保持同一实现。

现有 external launcher + DP 零 token 分支有集体协调 dummy run 特例；首版 DP=1，不使用该变体。

## 建立、增量传输与最终生成的时序

### 建立一次接收关系

外部前端继续使用 start / event / finish，不必发送完整未来 prompt。专用 [stable_kv_proxy.py](scripts/server/stable_kv_proxy.py) 调用以下已实现内部接口：

| 内部接口 | 调用方向 | 行为 |
| --- | --- | --- |
| `POST /v1/stable_kv/reserve` | Proxy → D | 创建接收会话，返回会话身份和 D peer 信息；仅 ACK，不创建生成 output queue。 |
| `POST /v1/stable_kv/describe` | Proxy → P/D | 查询配置、NIXL peer 身份、TP 和 block size。 |
| `POST /v1/stable_kv/bind` | Proxy → P | 将 P 长期 Request 与 D 接收会话绑定；仅 ACK。 |
| `POST /v1/stable_kv/finalize` | Proxy → D | 固定最终物理 prompt 与 manifest，返回当前 ready 状态；仅 ACK，不生成。 |
| `POST /v1/stable_kv/status` | Proxy → P/D | 查询 ready、错误、区间完成数和传输统计。 |
| `POST /v1/stable_kv/abort` | Proxy → P/D | 幂等取消发送、接收或生成会话，停止写入后释放。 |
| `POST /v1/completions` | Proxy → D | 只有 finalize ready 后才调用，创建真正的输出 collector 并收养块表。 |

这个过程不读取未来事件，也不需要整请求 preflight。专用 proxy 的 start 按以下顺序完成：

1. 绑定本轮唯一 P/D，通过 `describe` 获取 P peer 信息。
2. 向 P 提交现有 start，得到实际 P 内部 request ID；proxy 生成独立 UUID `session_id`，此时不要求已有 KV manifest。这个 P start 不挂旧的普通 PD transfer 参数，随后由 bind 接管增量传输生命周期。
3. 向 D reserve，使用实际 P request ID 和同一个 session ID，得到 D 接收目标身份。
4. 向 P bind target，将新会话通过 ack 型 EngineClient / EngineCore 控制接口绑定到长期 Request。
5. 三步均成功后才向前端返回 start 成功；失败时回滚已创建的两侧资源。

模板可能在绑定完成前已经算完；bind 的无在途检查处理这个窗口。路由还提供 `POST /v1/stable_kv/control` 统一入口，使用 JSON 的 `op` 选择上述操作；专用 proxy 使用具名路由。`EngineClient.stable_kv_control()` 经 AsyncLLM 和 EngineCore utility 进入主线程，HTTP handler 不直接操作 Scheduler。

### 由计算完成触发区间发送

| 时间线 | P 行为 | D 行为 |
| --- | --- | --- |
| start 绑定完成 | 计算模板，记录完成水位。 | 仅持有接收会话。 |
| Append 到达 | 正常入 pending / active，执行 PA prefill。 | 不收到新的普通生成请求。 |
| 新完整真实块完成 | 固定 `[start_block, end_block)`，发布就绪范围。 | 接收控制消息，主线程分配对应目标块。 |
| 目标块注册返回 | 匹配同代次源 / 目标范围，提交 NIXL WRITE。 | 等待这一批数据写入。 |
| 区间传输完成 | 解除本次传输的额外 pin；仍保留长期请求的 PA KV。 | 记录有效区间，不解除最终生成屏障。 |
| Update 到达 | 使用现有 PA + UPDATE 计算与 LCP 语义。 | 不计算、不提前接收 Update。 |
| Question 到达 | 按现有 FIFO 与依赖屏障完成最终 prefill，采样一个 token；终态 hook 发布剩余 blocks。 | 可接收最终区间，仍等待 manifest 屏障。 |
| finalize | 对在途最终 blocks 保留独立 pin，等待安全完成。 | 固定 manifest，等待所有范围齐备并检查映射，然后标记 ready。 |
| 普通 D completions | D 输入保持 P 的原始物理 prompt。 | 创建生成 Request、收养块表，进入正常 waiting/running 后执行。 |

P 完成新稳定范围后发送 `offer`；D worker 的通知经输出 metadata 到达 EngineCore，由 `_grant_pending()` 分配并发送 `grant`；P 根据 grant 提交精确区间 WRITE。该协议实现于 [stable_push.py](../vllm/distributed/kv_transfer/kv_connector/v1/nixl/stable_push.py)，不复用原有整请求 `PUSH_REG` 的终态含义。Proxy 不参与逐 chunk 的容量握手。

区间完成须确认 WRITE 已完成并对 D 计算可见，覆盖全部相关层与 KV groups。以后支持 TP 时，还须汇合所有必要 rank 的完成信号；不能收到一个 rank 的通知就宣布整批完成。

### Finalize 不产生第三套 KV

P 的最终 block table 仍由 [KVCacheManager.get_prompt_stream_blocks()](../vllm/v1/core/kv_cache_manager.py) 按顺序导出：`PREFIX_APPEND blocks + UPDATE blocks`。最终并不要求 P/D 的物理 block ID 相同，只要求同一个序列块序号的内容相同。

例如 D 已收到前面的 PA 块，就继续使用这些目标 blocks；P 的 PA 尾块映射到下一个 D block，UPDATE 域接在 PA 全部物理块之后。对每个 KV group 保存显式映射：

$$
\text{序列块序号 }j\quad\longmapsto\quad
(\text{P 源块 ID},\ \text{D 目标块 ID},\ \text{generation}).
$$

不能仅把“提前传了两块”作为跳过最终 manifest 前两项的依据。`_check_ready()` 按序验证没有缺口、每批 completed、源 IDs 与最终 manifest 的同位置切片一致。未完成区间继续等待；映射不一致则失败，不在首版中静默重建后继续生成。

KV manager 已实现两步：

- **Reserve**：`KVCacheManager.reserve_stable_kv_blocks()` 使用接收会话的 ownership key 分配，不伪造生成 Request，也不把新分配槽位直接当成有效 KV。
- **Adopt**：`adopt_stable_kv_blocks()` 经 `KVCacheCoordinator.adopt_reserved_blocks()` 把分组表从 reservation key 转交到真实 D request ID，GPU 地址不变，不复制 tensor，不先 free 再重新 allocate。未来 decode 需要的额外槽位再走正常分配。

Adopt 完成后，D 正常请求才进入 `waiting`，受原有运行名额和 token budget 控制。会话标记为 `adopted` 并保存真实 request ID，直到 proxy 完成收尾；已经转交的 blocks 只由正常 Request 释放，reservation 清理不能再释放一次。

Finalize 到达时，已经发布但尚未完成的 early 区间不会被重新归类或重传。P 只从 `offered_blocks` 之后发布最终尾部；D 汇合原有在途 WRITE 和最终 WRITE 的完成通知，再统一检查 ready。区间通知可以有多次，但只有一条最终生成准入屏障；不是等 early 全部结束后才允许提交尾部 WRITE。

## 最终屏障、P 完成通知与 Masked Padding

### 进入生成需要三个条件

D 可以开始生成必须同时满足：

1. 已收到最终 manifest 与完整生成参数，知道知识源和 Question 的最终交接范围。
2. manifest 要求的所有 KV 区间都完成接收，没有缺口、旧代次或在途写入。
3. D 普通 Request 已成功创建、收养 blocks 并通过正常调度准入。

提前区间完成通过 `StableKVWorkerMetadata.events` 的 `sent` / `received` 报告，不进入整请求 `finished_sending` / `finished_recving`。D 创建 Request 时先验证 session 为 `ready`，再收养，而不再触发普通全量远端 load。

P 的 Question 仍按现有规则排队，等待已准入知识 work 完成。D 的最终屏障不会替代 P 的 Question 屏障，两者保护的是不同阶段。

### P 完成通知与 D 独立生成

设最终 P 物理 prompt 长度为 $L$，远端 KV 覆盖 $[0,L)$。P 内部采样的 token 不进入 D 输入或客户端回答，proxy 只发送 `prefill_ready` 控制事件。

D 收养完整物理 prompt 的 KV 后，将 computed 长度设为 $L-1$，重算最后一个 prompt token，再生成第一个回答 token。D 保留完整输出预算，即使 P 内部采样得到 stop，也仍由 D 独立决定生成终止。

### 最终 manifest 仍是权威布局

最终交接必须继续携带：完整物理 prompt token ids、有效 KV token 数、分组 block IDs / counts / sizes、逻辑与物理长度、padding 元数据和零值的交接输出计数。

开启 Masked Padding 时，沿用 `prompt_stream_masked_padding={version, start, end}`。D 根据最终区间屏蔽中间 padding，并修正后续语义位置；早期无需知道最后会有多少 padding，因为接收阶段根本不执行 attention。

第一版要求 P/D 相同逻辑和 kernel block size、兼容布局，对每组最终有效块数严格验证为覆盖 $L$ 所需的数量。D 可以另外有 decode 容量，但不得把它混入已经有效接收的 KV 范围。`_finalize()` 检查 exact-transfer、物理长度、交接输出计数为零、组数、块数、block size 和 session 身份；`adopt()` 再验证 D prompt 恰好为已冻结的物理 prompt，并保持相同 mask。

外部流式 KV 的 adopt 不叠加一次 D 本地 prefix-cache 命中：接收到的 block table 已经是本请求的权威布局。首版保持该接收 / 收养路径的 blocks 不参与可复用缓存发布，避免 masked / 近似语义与正常 prompt hash 混用；普通请求的 prefix cache 不受影响。

## 分批协议与资源安全

### 区间完成不是请求完成

每批使用 `(session_id, generation, range_id)` 标识。`range_id` 由代次和半开块序号区间组成，`StableKVRange` 保存所有 groups 的精确源 / 目标 IDs。Block size 与 peer 兼容性在绑定和 WRITE 前检查。

控制消息包括 `offer`、`grant`、`transfer_stats`、`abort` 和 `drained`；区间完成为 `sent` / `received`。新的 notification 使用独立协议前缀和 msgpack，与旧的请求级通知分开。Worker 保存冻结后的映射指纹：同一 range 的完全相同重试幂等，换地址或换区间的重试报错。源/目标组数、每组块数和 descriptor 数必须严格相等，禁止裁短。

P 的区间写完只表示这一批额外传输引用可以解除，**不表示长期 P Request 结束**。D 的区间接收完成只将对应 `StableKVRange.completed` 置为真，不表示允许采样。最终生成 Request 的生命周期与区间资源回收分别推进。

### 源和目标都需要在途引用保护

同一业务请求依然拥有 PA / UPDATE blocks；NIXL 在途传输另外持有短期 pin / 引用。取消、抢占和 prefix-cache reset 不能在 WRITE 仍访问内存时把这些 blocks 交给其他请求。

即使不优化抢占，也必须处理以下正确性情况：

- **P 一旦抢占，本次 stable push 会话显式失败。** `invalidate_request()` 标记错误，后续 controller poll 发起 abort；客户端必须新建请求重试。首版没有实现“generation++ 后恢复提前传输”，也没有跨代次自动复用。
- D 接收 blocks 被取消或需要释放时，先停止新增 WRITE，等待 / 取消已有 handle 并确认不再写入，再释放目标引用。
- 每批在 `_offer()` 用 `block_pool.touch()` 增加独立源引用，WRITE 完成并安全释放句柄后才解除。P 原请求在 finish 可以正常释放自身引用，尚在途的最终块仍由这些额外引用保护。
- 重复消息只更新同一传输批次一次。请求 ID 重用也不能复活已经关闭的 session；保留短期 tombstone 拒绝迟到消息。
- Finalize 先于某个提前 WRITE 完成时等待其完成，不能同时对相同目标范围发一份竞争写入。

会话使用独立的无活动超时，默认 600 秒。控制/传输活动更新 `last_activity`，取消后双方交换 `drained`，确认没有在途 WRITE 再归还 D reservation 和残留源 pin。正常收养后的 D blocks 由生成 Request 生命周期负责。

唯一可简化的情况是 D 从未授予任何目标地址：例如 reserve 已成功、P bind 却失败。此时不存在远端 WRITE 的可能，D 在本地 drain 完成后即可清理，不必永远等待尚未建立的 P 会话回复。

**超时不是强制复用地址的授权。** 对端进程失联、无法取得 drained 时，会话可能保守保留引用；首版没有安全探测远端进程死亡后强制回收的机制。这种故障须显式报告，必要时重启受影响服务，不能为了回收显存而让旧 WRITE 覆盖新请求。

### Proxy 与输出生命周期一起清理

普通 proxy 的 `/finish` 在 P 返回后移除绑定；专用 `stable_kv_proxy.py` 会保留绑定到完整输出或异常收尾，并向 P/D 都发送 abort 控制来结束传输会话。

Reserve 和 finalize 都没有生成 output queue。**Finalize 只提交布局并等待 ready；随后一次普通 D completions 才创建并消费生成 collector。** 这是相对于最初设计中“finalize 直接生成”的接口细化，收养仍发生在这次真实 Request 的 scheduler 准入之前。

前端 disconnect、显式 abort、P/D 错误和握手失败都进入两侧清理路径；HTTP 返回清理 ACK 不代表在途 WRITE 已瞬间消失，真正释放由 drained 屏障控制。原有 P-only `/start_ttft` / `/finish_ttft` 保持在普通 proxy；专用早传 proxy 不提供 TTFT-only 路径。

## 首版范围与源码修改落点

首版启动检查要求 TP=1、PP=1、DP=1、decoder-only、同步调度、无 speculation 和 LoRA；各 KV group 必须为同尺寸 `FullAttentionSpec`。实际 launcher 使用 CUDA KV、同模型、相同 dtype / layout 与 block size。跨节点网络仍需 NIXL peer 可达与兼容检查通过；本轮验证固定同一节点两张卡，不据此宣称跨节点性能。

专用 proxy 当前只支持 completions 语义，并要求 `n=1`、`echo=false`。对外路由为 `/v1/prompt_stream/start`、`event`、`finish`、`abort` 和 `/health`；它不是普通研究 proxy 全部 OpenAI 路由的替代品，也没有 chat completions 或 TTFT-only 路径。

Masked Padding 的 0 / 1 配置均要明确传递；Update coalescing 可以独立开关，因为早传只关注 PA 完成水位，不依赖 Update 的 flush policy。

| 源码位置 | 已实现职责 |
| --- | --- |
| [stable_kv_proxy.py](scripts/server/stable_kv_proxy.py) | 独立 proxy；reserve / bind / finalize、P 完成通知、双侧取消和 D 续写计时。 |
| [Stable KV router](../vllm/entrypoints/serve/stable_kv/api_router.py)、[serving](../vllm/entrypoints/serve/stable_kv/serving.py)、[protocol](../vllm/entrypoints/serve/stable_kv/protocol.py) | 内部 ACK 型会话路由和配置、输入校验。 |
| [EngineClient](../vllm/engine/protocol.py)、[AsyncLLM](../vllm/v1/engine/async_llm.py)、[core_client.py](../vllm/v1/engine/core_client.py) | `stable_kv_control` 传递到 EngineCore utility。 |
| [StableKVPushController](../vllm/v1/core/stable_kv_push.py) | PA 完成区间、offer/grant、接收会话、最终 manifest、pin、统计及取消状态机。 |
| [Scheduler](../vllm/v1/core/sched/scheduler.py)、[EngineCore](../vllm/v1/engine/core.py) | 调度提交/回包 hook、正常请求收养、P 抢占失效、5 ms 零模型轮询。 |
| [KVCacheManager](../vllm/v1/core/kv_cache_manager.py)、[KVCacheCoordinator](../vllm/v1/core/kv_cache_coordinator.py) | 独立 reservation、容量检查、原 GPU 地址的 block table 所有权转交。 |
| [NIXL metadata](../vllm/distributed/kv_transfer/kv_connector/v1/nixl/metadata.py)、[Push Scheduler](../vllm/distributed/kv_transfer/kv_connector/v1/nixl/push_scheduler.py) | 增量 command / event 的 scheduler-worker 双向传递，独立于请求级完成通知。 |
| [StableKVPushWorker](../vllm/distributed/kv_transfer/kv_connector/v1/nixl/stable_push.py)、[Push Worker](../vllm/distributed/kv_transfer/kv_connector/v1/nixl/push_worker.py) | 后台多批 WRITE、严格 descriptor 配对、CUDA 可见性、telemetry 与 drain。 |
| [Model runner mixin](../vllm/v1/worker/kv_connector_model_runner_mixin.py) | 复用已有 no-forward 执行，使接收不产生模型输出。 |

### 启用配置与实验开关

P 启动时传入：

```json
{
  "kv_connector": "NixlPushConnector",
  "kv_role": "kv_producer",
  "kv_load_failure_policy": "fail",
  "kv_connector_extra_config": {
    "stable_kv_push": true,
    "stable_kv_poll_interval_ms": 5,
    "stable_kv_max_sessions": 128,
    "stable_kv_max_reserved_fraction": 0.5,
    "stable_kv_session_timeout_s": 600
  }
}
```

D 使用相同配置，仅将 `kv_role` 换为 `kv_consumer`。除布尔能力开关外的四个参数都有上述默认值，launch 脚本只传入必要的 `stable_kv_push=true`。

| 参数 | 约束和用途 |
| --- | --- |
| `stable_kv_poll_interval_ms` | 正有限数；零模型步骤的轮询间隔，影响控制延迟和 CPU 开销。 |
| `stable_kv_max_sessions` | 正整数；同一 EngineCore 的活跃会话数限制。 |
| `stable_kv_max_reserved_fraction` | 在 `(0,1]` 内；D 未收养 reservation 的总块数比例上限。 |
| `stable_kv_session_timeout_s` | 正有限数；无活动会话超时与关闭身份保留窗口。 |

模型启动同时使用 `--no-async-scheduling --prompt-stream-masked-padding --generation-config vllm`。
P 显式指定 `--attention-backend CHUNK_SPARSE_ATTN`，D 显式指定
`--attention-backend TRITON_ATTN`。D 的 `unified_attention` 完整读取历史 KV，
只屏蔽 padding，不做块稀疏；两组对照保持相同的逐角色后端配置。
不设置 `stable_kv_push` 时没有新增 controller，会继续使用原有 connector 路径。

实验开关是外部 start JSON 的 `early_stable_kv_push`：

- `true`：PA 整块真正完成后发布 offer；finish 只发布剩余尾部。
- `false`：同样在 start 建立 reserve/bind，但暂不发布 PA offer；finish 发布全部范围。

两种模式使用同一对 P/D 和同一套 Push 协议，避免把“是否创建会话”或 Pull/Push 后端差异误当作早传收益。专用 replay 分别通过 `--early-push` 与 `--no-early-push` 选择，不需要重启服务。

## 如何验证收益和正确性

### 区分 P 完成通知与 D 首个回答 token

当前 proxy 先通知 P 完成，再交付 D 的回答。提前 KV Push 没有直接缩短 P 最终 Question prefill 的计算路径，所以不能只看 P 侧 TTFT 判断传输优化。

专用 proxy 和 replay 已增加以下指标，在完整 PD 生成 workload 下比较：

| 指标 | 实际计量口径 |
| --- | --- |
| `p_finish_to_d_first_token_ms` | **主要指标**：proxy 读完 P finish JSON，到收到 D 的首个回答 token；包含 finalize、传输屏障、D 提交、排队及本地执行。 |
| `d_request_to_first_token_ms` | Proxy 发 D completions 到收到 D 首 token；当前提交点位于 finalize ready 之后，所以不含之前的剩余 KV 等待。 |
| `finalize_wait_ms` | Proxy 内部 finalize/ready 阶段，包含控制轮询、目标扩容和传输等待；不是纯传输时间。 |
| `finish_to_p_first_client_ms` / `finish_to_d_first_client_ms` | 客户端发 finish 至收到 P 完成通知 / D 首 token；前者是 P 侧计时，后者是本协议的用户等待。 |
| `early_transferred_bytes` / `final_transferred_bytes` | NIXL 完成 handle 的 `totalBytes`，按该区间 early/final 属性累加。 |
| `final_transfer_ms` | 最终类别各 handle 的 NIXL `xferDuration` 从微秒转为毫秒后求和。它不含 HTTP 或分配时间，多批重叠时也不等于墙钟等待。 |
| `early_byte_fraction` | early 字节除以总字节。它表示提前类别的份额，**不能保证所有这些字节在 P finish 前已完成**。 |
| `received_blocks_at_finalize` | D 收到最终 manifest 时已完成的序列块数；判断提前区间是否已经就位。 |
| `completed_bytes_before_p_finish` / `completed_blocks_before_p_finish` | P finish 时主线程已观测到 WRITE 完成的字节/块数；未回报的完成不计入，是保守的提前完成水位。 |
| `finish_to_done_ms`、`decode_tpot_ms`、`decode_tps` | 客户端完整响应时间，以及 D 首至末 token 区间的平均间隔/速率。TPS 按真实 token IDs 计数；完整响应还包含最终统计汇合与会话清理。 |
| 成功率与 `metrics_status` | HTTP/SSE 完整成功和统计合法性分别记录；不能隐藏失败后只比较少量快请求。 |

Proxy、worker、replay 各自在自身进程内计时，不相减跨机器单调时钟。输出 SSE
使用 `event=prefill_ready` 通知 P 完成，回答帧标记 `token_source=decode`。
最终 metadata 帧附带 `stable_kv_metrics`，replay 必须读到 `[DONE]`。

`p_finish_to_d_first_token_ms` 的起点是 proxy 读完 P finish JSON，不是客户端提交 finish 的时刻，也不是 GPU 完成的直接时间戳。它适合测量最终交接阶段，但不能替代客户端延迟。每条请求先测量完整间隔，再计算其分位数；**不能将 P 阶段 P95 与交接 P95 相加，称为客户端 P95**，不同分位数通常不对应同一个请求。

源码中，proxy 在 D 的 `[DONE]` 之后还等待 P 的传输计账完成，发送最终 metadata，并在 generator 的 `finally` 中清理两侧会话，随后才结束客户端 SSE。因此 `finish_to_done_ms` 是协议的完整响应时延，不是单纯的模型生成耗时。HTTP 200 也不足以代表成功：SSE 中可能出现错误或提前 EOF，必须同时检查完整输出与 `metrics_status`。

`D 预留块数 × 持有时间` 仍是建议补充的显存成本指标，当前回放器没有自动统计。
抢占在首版会表现为请求失败；不能把它描述为已经优化的恢复场景。

单个请求可能重叠的只是稳定 PA 部分；UPDATE / Question 和 PA 尾块仍在关键路径上。收益上限取决于 PA 的实际字节占比、提前窗口、网络带宽及 D 排队。即使提前复制成功，如果 D 准入等待占主要时间，续写改善也可能很小。

已提供的实验包含相同 Push 实现的早传开关对照，保持模型、prompt、事件时序、采样和 masked 配置一致。汇总按样本配对，核对数据 SHA256 和模板哈希，并报告全量成功率。若以后加入 Pull，只作为额外基线单独标注。

### 不新增计算近似的验证

本优化不引入新的 attention 近似：它复制 P 已计算的 KV，不在 D 重建另一套结果。对照首先检查最终 token view、padding 区间、有效 KV 长度与逐序列块映射，再比较生成结果；不能仅凭输出“看起来正常”判断传输正确。

“不新增近似”不等于所有并发 A/B 都必然输出相同 token。既有 Update work 可能使用当时已经完成的 Append 上下文；不同调度时序会改变这种近似的实际落点，模型批处理也可能有数值非确定性。本轮顺序负载的输出一致是回归证据，不能扩展成全部竞争调度或问答质量无损的证明。

后续测试应覆盖：短 PA 不足一块、跨块 Append、Update 与 Append 交错、部分块就绪、绑定晚到、finalize 先到、区间通知重复 / 乱序、取消与旧 generation 通知、无 D decode 的 stop / 单 token 预算、并发多个接收会话及显存不足。单 kernel 正确性不能替代这些跨进程生命周期验证。

在不改变既有近似的前提下，早传开关应复制相同 KV；生成 token 对照仍要考虑模型运行的数值非确定性，并用固定配置和内容检查区分数值差异与错块 / 漏块。

## 实现验证与早期 smoke

实现阶段的早期记录为 118 项 CPU 验证通过；2026-10-04 正式矩阵的配套记录为下面六个文件共 100 项通过，并记录 Ruff、shell 语法与 whitespace 检查通过。这是不同阶段的历史验证记录，不将两组人数相加，也不将旧数字作为当前工作树的重新验证结果。本次文档同步没有运行测试或启动服务。覆盖范围包括区间协议、controller/proxy、replay，以及已有流式 Scheduler 的停放、唤醒与合并调度回归：

- [test_stable_kv_push.py](../tests/v1/kv_connector/unit/test_stable_kv_push.py)：真实 worker 协议的输入、descriptor 配对、重试、telemetry 和 drain 边界，用模拟 NIXL wrapper 驱动。
- [test_stable_kv_controller.py](../tests/v1/test_stable_kv_controller.py)：完整真实块发布、完成范围、保留引用、最终 manifest、D 收养与错误分支。
- [test_stable_kv_proxy.py](../tests/v1/test_stable_kv_proxy.py)：reserve/bind、首 token 交接、D 输出标记、完整输出和清理。
- [test_stable_kv_replay.py](../tests/v1/test_stable_kv_replay.py)：P/D 首 token 区分、空文本 token、多行 SSE、最终指标、错误和提前 EOF。

还回归了 [test_prompt_stream_scheduler.py](../tests/v1/test_prompt_stream_scheduler.py)，确认未启用新能力时原有 dormant、执行背压与恢复规则保持可用。

历史验证使用的命令如下，不创建额外推理服务：

```shell
.venv/bin/python -m pytest \
  tests/v1/test_prompt_stream_scheduler.py \
  tests/v1/test_stable_kv_controller.py \
  tests/v1/test_stable_kv_proxy.py \
  tests/v1/test_stable_kv_replay.py \
  tests/v1/kv_connector/unit/test_stable_kv_push.py \
  tests/v1/kv_connector/unit/test_nixl_push_connector.py -q
```

### 真实双卡 smoke

2026-10-01 使用 g2c9 上单 P/D、Qwen2.5-3B、TP=1、Triton、Masked Padding 开启和显存比例 0.85；proxy / replay 位于 g2c14。数据为 `mix_source_perf_qwen` 前四个样本，两组都先预热一条，再串行回放；Append interval 为 0.5 秒，Update interval 为 0.2 秒，生成上限 64 tokens。

| 观察项 | finish 后统一 Push | 提前 Push |
| --- | ---: | ---: |
| 完整输出 / 有效统计 | 4/4 | 4/4 |
| P finish → D 首 token，均值 | 303.0 ms | 270.9 ms |
| 最终类别 WRITE 耗时，均值 | 172.4 ms | 147.7 ms |
| 最终类别传输量，均值 | 66.09 MiB | 55.41 MiB |
| D 在 finalize 时已收到的块数 | 全部为 0 | 每条为 10 或 19 |

四组配对的输出 token 序列全部一致，总传输字节数也逐样本一致；优化移动的是传输时机，不是少传 KV。每条提前类别共 19 块，但部分样本到 finalize 时只有 10 块已完成，因此不能把全部 early 字节都当成已经隐藏的传输。

另外以 QPS=2、客户端最大并发 2 回放四条提前传输请求，完整输出和统计均为 4/4 成功；这是跨会话交错的功能检查，不与串行对照混合汇总。
还执行了一条 start 后取消的真实请求，确认两侧会话均到达 `aborted`，没有停留在排空等待。验证结束后已停止本轮 P/D allocation 与 proxy，释放 GPU。

这里约 10.6% 的交接均值下降只是同机双卡、四条样本、固定顺序的一次 smoke 观察，不作为正式性能结论。首轮握手 / JIT 预热耗时没有混入上表；后续已完成下面的交替顺序、较大样本配对实验。

## 正式实验设计

### 部署与控制变量

2026-10-04 的正式实验比较同一增量 Push 实现的两个请求级开关：
`early_stable_kv_push=false` 为 finish-only 对照，`true` 为提前 Push。
两者均建立 receive-only 会话、消费完整 P/D 输出，不使用 TTFT-only；
**对照不是原有 NIXL Pull，也不是没有会话开销的普通 PD 请求。**

| 设置 | 正式实验取值 |
| --- | --- |
| P / D | g2c9 的两张 NVIDIA A30，同一个 Slurm allocation，每端一张卡 |
| Proxy / replay | g2c14，使用专用 stable KV proxy |
| 模型 | Qwen2.5-3B，服务名 `qwen2.5-3b` |
| 显存与长度 | P/D 显存比例 0.90；`MAX_MODEL_LEN=4096`；日志记录每端 KV 池容量 440,144 tokens |
| 计算路径 | TP=PP=DP=1，同步调度，eager，`TRITON_ATTN`，Masked Padding 开启 |
| 请求容量 | 服务 `MAX_NUM_SEQS=64`；回放 `MAX_CONCURRENCY=8` |
| 到达过程 | QPS=0 顺序执行；QPS=1/2 使用 Poisson，随机种子 2026 |
| 来源节奏 | Append interval=0.1 秒；Update interval=0.2 秒；两源 initial delay 与 source finish delay 均为 0 |
| 生成参数 | 输出上限 64 tokens，temperature=0，采样 seed=2026 |
| Update 合并 | 不启用，使用既有分段流式计算路径 |
| 重复与顺序 | 每个单元两轮，先 finish-only→early，再 early→finish-only；预热不进入统计 |

各数据与 QPS 单元串行运行，共用同一对 P/D，不将两个对照同时压到服务上。
正 QPS 是回放器设定的目标到达率；客户端并发上限可能使实际提交推迟，
不是服务吞吐保证。这里的“顺序负载”指 QPS=0、一次完成一个业务请求，
不是请求内只有一个来源。

没有人工扣除握手、轮询或 CPU 成本。原有数据第一轮 finish-only 开始时
出现一次 `reduce_segments` JIT 提示，该轮仍保留全部正式样本；反向第二轮
用于检查顺序和预热因素。受控正式阶段日志未发现新增 JIT、抢占或运行期 OOM。
因此本轮部分整体退化不能直接解释为抢占。

### 原有性能集与稳定占比场景

原有 `mix_source_perf_qwen` 取前 128 条，仅测 QPS=0，两轮完整 A/B。
另外通过 [build_stable_kv_push_dataset.py](scripts/data_processing/build_stable_kv_push_dataset.py)
生成三个独立性能场景，每个场景 128 条，最终知识总长固定为 2048 tokens：

| Prefix 中的 Append 长度 | 最终 Update 长度 | 稳定知识占比 | Append chunks | Update 快照数 |
| ---: | ---: | ---: | ---: | ---: |
| 256 | 1792 | 12.5% | 2 | 5 |
| 1024 | 1024 | 50% | 2 | 5 |
| 1792 | 256 | 87.5% | 2 | 5 |

稳定知识占比的分母是 Append 与最终 Update 的知识 token 总数，不包含模板、
Question、padding 或输出。实际传输量包含这些物理布局因素，二者不能混用。
长度检查包含 chat template、Question、物理 padding 和 64-token 输出预算。
每条请求使用独有模板标识；同一编号跨场景保持问题、模板标识与随机种子一致。
这些合成输入只用于性能，没有标准答案，不承担质量评估。

相邻 Update 完整快照的 source-local LCP 固定为其长度的 75%。
设最终 Update 长度为 $U$，无抢占、无合并时，五代 Update 的名义新计算量为：

$$
U+4\left(U-0.75U\right)=2U.
$$

因此三个场景的知识累计 prefill 量分别为 3840、3072、2304 tokens，
尚未计入模板、Question 和 padding 的计算。**改变稳定占比也改变了 Update
重算工作量**：不同占比的原始时延不能作为提前传输的纯因果对照；每个场景
内部的 early / finish-only 才使用相同输入、来源节奏和计算语义。

### 样本量、失败与统计方法

| 正式测量 | 请求提交数 | 完整成功且统计有效 | 共同成功配对数 |
| --- | ---: | ---: | ---: |
| 原有性能集：128 条 × 两模式 × 两轮 | 512 | 512 | 256 |
| 受控矩阵：三占比 × 三 QPS × 两模式 × 两轮 × 128 条 | 4608 | 4607 | 2303 |
| 合计 | 5120 | 5119 | 2559 |

上述数量是重复测量的提交次数，不是 5120 个不同问题。受控矩阵 early 为
2304/2304 成功，finish-only 为 2303/2304。唯一失败位于 50% 占比、QPS=1、
第一轮 finish-only 的 `anns-query-000028`：start 返回 HTTP 502，
上游为 D `/stable_kv/reserve`，错误内容为空。原因尚未确认，不能据此认定
是显存、超时或具体 NIXL 故障。该单元第一轮有 127 对，其余单元每轮有 128 对。
未配对的那条 early 成功响应仍计入全量成功人数，失败也仍在分母中。

汇总遵守以下口径：

- 延迟对照使用两边完整成功、统计有效且有实际 D 输出的共同样本；同时报告全量成功率。
- 先在每轮共同样本上分别计算 P90/P95/P99，再取两轮分位数的中位数。
  没有池化两轮请求时延，也没有把“每请求加速比的 P95”当成这里的降幅。
- 下文降幅为 `(finish-only 汇总值 − early 汇总值) / finish-only 汇总值`，
  正值表示减少，负值表示退化。字节与完成比例先算每轮均值，再取两轮中位数。
- 两轮最小/最大范围不是置信区间；没有做显著性检验。
  每轮约 128 对时，P99 由极少数最慢请求决定，仅作为探索性尾部指标。

## 正式实验结果

### 原有性能集的顺序对照

两轮都完整成功，各有 128 对。单位为 ms，箭头表示 finish-only → early：

| 重复 | P→D 交接 P90 | 交接 P95 | 交接 P99 | 最终 WRITE duration 累计值 P95 | 客户端 P 首 token P95 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 第一轮 | 305.8 → 273.8 | 309.9 → 274.5 | 317.3 → 276.7 | 184.8 → 145.8 | 159.5 → 169.2 |
| 第二轮，反向顺序 | 311.6 → 273.1 | 314.2 → 274.3 | 318.1 → 279.4 | 186.8 → 146.1 | 164.3 → 164.5 |

交接 P95 两轮分别下降约 11.4%、12.7%，最终 WRITE duration P95 下降约
21.1%、21.8%。第一轮 P finish 前实际完成字节平均占总量 14.8%。
P 首 token 的 P95 第一轮增加约 9.7 ms，第二轮基本相同：可重复观察到的是
交接缩短，而不是 P 本身 TTFT 的改善。

### 稳定占比与 QPS 下的交接尾延迟

以下均为两轮分位数的中位数，单位 ms，箭头方向同上。

| 稳定知识占比 | QPS | 交接 P90 | 交接 P95 | 交接 P99 | P95 降幅 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 12.5% | 0 | 325.1 → 300.6 | 326.4 → 302.1 | 329.3 → 303.9 | 7.43% |
| 12.5% | 1 | 536.2 → 495.6 | 612.9 → 564.4 | 755.7 → 695.5 | 7.91% |
| 12.5% | 2 | 642.8 → 647.6 | 696.9 → 697.6 | 745.0 → 810.1 | −0.11% |
| 50% | 0 | 326.0 → 229.7 | 327.2 → 231.6 | 329.8 → 233.0 | 29.22% |
| 50% | 1 | 523.1 → 410.3 | 574.7 → 462.8 | 840.0 → 535.2 | 19.47% |
| 50% | 2 | 691.0 → 508.9 | 766.8 → 541.7 | 835.1 → 613.0 | 29.35% |
| 87.5% | 0 | 326.6 → 158.1 | 328.2 → 159.1 | 329.5 → 160.8 | 51.52% |
| 87.5% | 1 | 506.5 → 369.6 | 548.7 → 399.4 | 702.2 → 482.7 | 27.22% |
| 87.5% | 2 | 601.6 → 469.8 | 645.2 → 511.1 | 697.4 → 665.3 | 20.79% |

顺序负载下，三个场景交接 P95 的改善随稳定占比从约 7.4% 增至 51.5%。
并发负载下不保证单调收益：12.5%、QPS=2 的两轮交接 P95 一轮改善 5.75%，
另一轮退化 6.23%，汇总基本持平，P99 变差。不能从该单元推出稳定的尾部加速。

### 客户端首 token、D 续写与完整响应

这些指标直接测客户端时钟，不通过阶段分位数相加推导。单位为 ms。

| 稳定知识占比 | QPS | finish→D 首续写 P95 | 续写等待降幅 | finish→P 首 token P95 | 完整响应 P95 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 12.5% | 0 | 415.2 → 391.7 | 5.68% | 93.7 → 94.2 | 2080.6 → 2058.4 |
| 12.5% | 1 | 1019.5 → 1105.5 | −8.44% | 500.2 → 510.2 | 2876.8 → 2863.3 |
| 12.5% | 2 | 1310.2 → 1312.2 | −0.16% | 781.7 → 805.1 | 3323.0 → 3265.7 |
| 50% | 0 | 426.1 → 329.5 | 22.67% | 103.0 → 103.2 | 2111.6 → 2013.0 |
| 50% | 1 | 881.8 → 848.4 | 3.79% | 335.2 → 414.3 | 2683.5 → 2620.5 |
| 50% | 2 | 1209.9 → 1109.5 | 8.30% | 551.8 → 677.4 | 3035.4 → 3002.5 |
| 87.5% | 0 | 433.0 → 263.7 | 39.10% | 109.1 → 109.6 | 2126.9 → 1958.0 |
| 87.5% | 1 | 774.8 → 831.5 | −7.32% | 301.8 → 425.2 | 2569.5 → 2619.3 |
| 87.5% | 2 | 988.6 → 1156.2 | −16.95% | 492.9 → 756.6 | 2881.8 → 3078.6 |

交接改善与整体退化可以同时发生。例如 87.5%、QPS=2 的交接 P95 下降
20.79%，但客户端 P 首 token P95 增加约 263.7 ms，D 续写等待 P95 增加
16.95%，完整响应 P95 增加 6.83%。这些都是独立实测分位数，不能用
263.7 ms 减去另一个阶段的 P95 降幅来精确解释端到端 P95。

提前传输本来就不缩短 Question 的模型计算，而是把复制与控制工作前移；
当这部分工作影响同时进行的 prefill，P finish 可能更晚。50%、QPS=1/2
仍有客户端续写收益，但小于交接阶段收益。87.5%、QPS=1/2 则由 P 侧代价
抵消了交接改善。当前结果支持负载相关的 tradeoff，不支持“稳定部分越多，
所有延迟就一定越低”。

### 实际提前完成量与最终剩余传输

受控请求每条总传输量为 **74.8125 MiB**，两模式逐请求相同。
以下最终字节与实际完成比例使用逐轮均值的中位数，duration 使用逐轮 P95
的中位数。箭头仍为 finish-only → early。

| 稳定知识占比 | QPS | 最终类别 WRITE，MiB | P finish 前实际完成字节比例 | 最终 WRITE duration 累计值 P95，ms |
| --- | ---: | ---: | ---: | ---: |
| 12.5% | 0 | 74.81 → 64.61 | 0% → 13.64% | 195.2 → 169.2 |
| 12.5% | 1 | 74.81 → 64.61 | 0% → 13.64% | 292.1 → 257.4 |
| 12.5% | 2 | 74.81 → 64.61 | 0% → 13.59% | 291.6 → 295.8 |
| 50% | 0 | 74.81 → 37.61 | 0% → 49.73% | 195.9 → 99.6 |
| 50% | 1 | 74.81 → 37.61 | 0% → 49.73% | 308.3 → 170.8 |
| 50% | 2 | 74.81 → 37.61 | 0% → 49.73% | 326.8 → 208.1 |
| 87.5% | 0 | 74.81 → 10.61 | 0% → 85.82% | 196.0 → 28.1 |
| 87.5% | 1 | 74.81 → 10.61 | 0% → 85.82% | 295.2 → 113.9 |
| 87.5% | 2 | 74.81 → 10.61 | 0% → 85.82% | 312.7 → 185.2 |

减少约 13.64%、49.73%、85.82% 的是**最终类别的字节数**，不是总字节数。
实际提前完成量与稳定知识占比不同，因为模板、Question 和整块布局改变分母，
且含 padding 的 PA 尾块不能提前发送。85.82% 的字节提前完成也不意味着
85.82% 的端到端时延被隐藏。

12.5%、QPS=2 时，P 主线程 finish 前观测到约 18.08 块完成，D 在 finalize
时已收到约 18.14 块，体现完成回报与观测时点不同；50% 和 87.5% 的平均
水位约为 66.14、114.14 块。已发布但仍在途的 early 字节仍需最终屏障等待，
不能只看 `early_byte_fraction` 就宣布已隐藏相应传输。

### 生成一致性与结论范围

原有顺序实验的 256 对，以及受控 QPS=0 的 768 对，完整输出 token 序列
全部一致。结合相同传输总量和严格 manifest，这是这些路径没有引入错块、
漏块等可观测回归的证据；不是逐元素 KV 数值比对，也不是标准答案正确率。
本轮没有重新做 SQuAD EM/F1，不将性能集结果替代
[近似计算质量评估](Approximate-Calculation-Evaluation.md)。

当前证据限定于一个模型、同主机 A30 双卡、TP=1、一种来源节奏和两个重复。
没有跨节点、TP>1、大显存硬件或更多负载的结论，也没有已实现抢占恢复。
没有抢占的这组结果可以展示传输与计算的竞争，但不能评价抢占优化。

## nsys 排查与传输后端解释

### 诊断条件与证据层次

2026-10-04 对 87.5% 稳定占比、QPS=2 做了额外诊断，配置与正式实验
相同。仅采集 P 进程树，使用 nsys 2023.4.4，开启 CUDA/NVTX，关闭 CPU
sampling；两路径各预热 8 条，随后四段各 8 条，顺序为 finish-only、early、
early、finish-only，共 32 次诊断请求全部完整成功。

这 32 次不进入前文正式性能表。工具本身会增加开销，短窗口只能用于定位，
不能替代 128 条样本的 P95。更早一段含 OS runtime、64 请求的采集没有取得
可用完整报告，不纳入下面证据。详细记录见
[nsys README](scripts/replay/stable_kv_push/nsys/README.md)。

### 显式设备同步不是已证实瓶颈

`torch.cuda.synchronize()` 对应的 DeviceSynchronize API 统计如下。
这是 CPU API 耗时，不是 GPU kernel 时长：

| Capture | 模式 | 调用数 | 累计耗时，ms | 最长一次，ms |
| ---: | --- | ---: | ---: | ---: |
| 1 | finish-only | 8 | 0.115 | 0.0209 |
| 2 | early | 17 | 0.179 | 0.0201 |
| 3 | early | 19 | 0.188 | 0.0173 |
| 4 | finish-only | 7 | 0.089 | 0.0243 |

提前组增加同步机会，但每段累计不足 0.2 ms，不能用它解释正式实验中
几百毫秒的 P 首 token 退化。此前“全设备同步导致主要退化”的猜测没有得到
这次 trace 支持；目前不删除这个保证源 KV 可见的正确性屏障。

这不等于全部同步都很便宜：模型主线程 `cudaEventSynchronize` 两段
finish-only 累计约 789/809 ms，early 约 598/587 ms，包含模型完成等待，
提前组在这项统计中没有增加。不同 API 的职责与计量不能混为一谈。

### 观察到细粒度 D2H 与模型提交重叠

四段各有 **76,608 次 D2H，每次 8192 bytes**，总量 598.5 MiB，正好对应
8 条请求 × 74.8125 MiB，即每请求 9576 次小拷贝。两模式总量相同，
不是 early 重复传输了全量 KV；early 将其中约 85.8% 前移到 P finish 之前。

GPU 上 D2H 累计执行时间每段约 140–143 ms。主要模型 kernel 在 stream 7，
D2H 使用其它 copy streams，多后台线程提交 `cuMemcpyAsync` 并等待相应
stream 完成。因此**实际已经使用不同 CUDA streams**，不是简单的“计算与
复制放在同一个 stream”问题。分流也不意味着底层资源完全隔离。

模型主线程中耗时至少 5 ms 的 launch API 如下：

| Capture | 模式 | 慢提交次数 | 与 copy API 时间重叠次数 | 最长一次，ms |
| ---: | --- | ---: | ---: | ---: |
| 1 | finish-only | 7 | 7 | 301.0 |
| 2 | early | 17 | 17 | 240.6 |
| 3 | early | 22 | 22 | 203.6 |
| 4 | finish-only | 8 | 8 | 269.0 |

例如 Capture 2 的最长提交持续 240.6 ms，期间同时存在 8640 个后台
copy API 调用。这里慢的是 **CPU 端 CUDA 提交调用**，不表示对应 kernel
在 GPU 上计算了 240 ms。不同线程累计的 API 时间也不能相加当成墙钟时延。
GPU kernel 累计执行时间四段约为 1685、1537、1537、1664 ms，early 未在
这项总量中变长；批处理时序与执行次数不同，仍不能据此宣布算子更快。

finish-only 也会与其它请求的 prefill 重叠传输；early 只是让传输更早、更分散。
这解释了为什么交接可以缩短，却存在影响 P 计算提交的风险。**已确认的是
小拷贝、慢提交及时间重叠；推测的是细粒度搬运造成提交路径竞争。**
尚未用控制变量实验确认具体 driver 锁、CPU 调度或 UCX/NIXL 内部阻塞，
也没有测出 PCIe/HBM 带宽饱和。论文不能写成这些底层原因已经定位。

### D2H 不是 PCIe 的必然要求

源码 [base_worker.py](../vllm/distributed/kv_transfer/kv_connector/v1/nixl/base_worker.py)
的 NIXL 默认后端为 UCX，本轮启动日志也确认使用 UCX，
`kv_buffer_device=cuda`、`use_host_buffer=False`。这表示 KV 池位于 GPU，
不保证底层传输没有 host staging；P 的 trace 已证明本次实际发生 D2H。
提前传输代码向 NIXL 提交 GPU 地址与 WRITE descriptors，没有在 Python
层主动将整个 KV tensor 转成 CPU tensor。

使用 UCX 不等于指定了某一种物理传输。NIXL 选择后端，UCX 再根据可用
能力与配置选择 transport/protocol；本轮没有抓取完整 endpoint transport
日志，**尚不知道为什么采用该 D2H 路径，也未确认是否能切换 CUDA IPC/P2P。**

2026-10-05 另在本机 g2c14 做过只读硬件检查：两张 A30 的拓扑为跨 NUMA
的 `SYS`，没有显示 NVLink；`nvidia-smi topo -p2p r/w` 双向均为 OK，
`cudaDeviceCanAccessPeer` 双向返回可访问。因此“没有 NVLink，所以一定
只能 D2H”不是成立的推论。但这是 **g2c14 的能力检查，不是实验机 g2c9 的
实测传输证据**，更不代表本轮 UCX 已启用 P2P 或启用后一定能消除竞争。
不能把本机可访问性写成实验链路已采用 GPU 直传。

## 源码核对与论文表述

### 当前文档与实现的对应关系

本次按 scheduler/controller、KV manager、worker、REST router 和专用 proxy
核对了关键路径。主协议与现有代码一致，下面这些边界应保留在实现说明中：

| 容易误写的内容 | 当前实现 |
| --- | --- |
| Append ACK 后就可复制 KV | 使用调度 capture 与同步 worker 回包确认完成，再固定完整真实块；ACK 不是 GPU 就绪证明。 |
| D 收到第一批就进入 decode | receive-only 会话保留显存但不进入模型 running；全部最终范围 ready 后，真实 D Request 才收养并准入。 |
| finalize 一步创建生成任务 | finalize 是 ACK 型布局冻结与屏障；proxy 随后用一次普通 D completions 创建 collector。 |
| finalize 重拉未完成 early 块 | 不重传已发布区间；等待在途 early 与新 final 区间统一完成。 |
| 收养时复制或重新申请全部块 | 转交 reservation 的 CPU 所有权表，GPU 地址不变；D 从 computed=L-1 开始重算最后一个 prompt token。 |
| 已实现纯事件驱动和异步多 TP | 当前是默认 5 ms 的 no-forward 轮询，只支持同步 TP/PP/DP=1。 |
| 普通 Pull 是本轮基线 | 两组共用增量 Push 协议与 start 会话，只改变稳定块是否提前发布。 |
| 传输总量减少，或者 P 侧 TTFT 必然改善 | 总字节相同；主要测量是 P finish→D 首个回答 token。P 侧完成延迟和用户侧 TTFT 分别报告。 |
| 同步或 PCIe 带宽已被确定为根因 | DeviceSynchronize 很小；小 D2H 与 CPU 慢提交相关，精确根因和实际 UCX transport 尚未确认。 |

本次修正了仅有四条 smoke、正式实验仍待进行的旧状态，补充了接口范围、
在途区间汇合、完整响应计时及历史验证口径。源码核对是描述一致性检查，
不是对所有竞态、故障或性能实现的穷尽验证。

### 可以支持的研究论点

本工作的机制特点是利用混合来源流式 Prefill 的稳定 PA 域，按真实完成的
整块增量发布；D 通过独立 receive-only ownership 接收，最终将原块表收养到
生成 Request，保留 Masked Padding 和精确 manifest 屏障。
它将知识流到达阶段的部分稳定 KV 传输与 P 计算重叠，而不提前传可变 Update。

可据现有数据写为：

> 在同主机 A30 双卡、Qwen2.5-3B 的配对完整 PD 实验中，稳定 KV 提前传输
> 保持总传输字节数不变，将部分复制移出最终交接。顺序负载的受控场景中，
> 随稳定知识占比由 12.5% 增至 87.5%，P→D 交接 P95 降低约 7.4%–51.5%，
> 客户端等待 D 首次续写的 P95 降低约 5.7%–39.1%。并发负载下，搬运与
> 计算的竞争可能抵消交接收益；短区间 profiling 观察到细粒度 D2H 与 CPU
> 模型提交的重叠，但尚未确定底层竞争根因。

论文中应把阶段收益、客户端总体验和实现代价分开呈现，而不是隐去退化
单元。可以将低竞争负载下的交接缩短作为主结果，并用并发负载说明适用范围；
不能只保留最有利的 QPS，也不能用减小 final 字节数替代实测延迟。

以下表述暂不受证据支持：全负载加速、零计算干扰、总通信量减少、
所有生成质量无损、跨节点已验证、已解决显存抢占、已确认带宽饱和或 driver 锁。
“计算与传输重叠”本身也不能未经相关工作核对就宣称首次提出；本文给出的
是当前系统的机制与实验依据，论文创新性仍需独立进行相关工作比较。

### 尚未实施的优化与额外评估

以下内容可以留作后续工作，不写成已完成的能力或收益：

- 核实 g2c9 的 peer 能力、UCX 实际 endpoint transport 和 host staging 原因，
  再评估适用的 GPU 直传路径；不假定换 transport 一定有收益。
- 合并发布批次、减少细粒度 descriptors/小拷贝或限制传输提交强度，
  评估能否降低计算提交竞争；保持稳定范围与完成屏障不变。
- 将控制轮询改为事件唤醒，或用 CUDA event 细化源 KV 可见性；当前 nsys
  没有证明删除全设备同步会改善正式结果。
- 增加 D 的 `预留块数 × 持有时间` 及内存压力评估，设计需要时的容量准入、
  多 rank 完成汇合、抢占后新代次恢复和失联后的安全回收。
- 增加重复、硬件/网络配置与来源时序，必要时做统计置信区间或质量复验；
  不将两轮范围当成这些扩展已经完成。

本文的设计说明、正式数据和诊断观察可作为当前论文草稿的输入；尚未实施
的优化不影响如实总结现有机制，但不能用预期优化后的结果替代已测数值。
具体启动与统计命令保留在 [回放 README](scripts/replay/stable_kv_push/README.md)，
逐轮记录与诊断分别见 [EXPERIMENTS.md](scripts/replay/stable_kv_push/EXPERIMENTS.md)
和 [nsys README](scripts/replay/stable_kv_push/nsys/README.md)。
