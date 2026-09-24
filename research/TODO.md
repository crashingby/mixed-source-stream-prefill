# 待办

## 定位 TP=2 时 NIXL producer 的 KV 注册阻塞

### 现象与边界

使用 Llama-3.1-8B-Instruct、`TENSOR_PARALLEL_SIZE=2`、
`ATTENTION_BACKEND=TRITON_ATTN` 启动 P 节点时，Slurm job `4263` 没有完成
server startup。两个 rank 都已完成模型加载、NCCL 初始化和 KV 容量规划，但随后
持续输出：

```text
No available shared memory broadcast block found in 60 seconds.
```

这不是请求执行阶段的问题，也不是 Triton JIT 或 OOM：P 尚未开始监听请求，日志
显示每张卡已完成模型加载（约 7.54 GiB），并已计算出约 12.26 GiB KV 容量。

### 已知执行位置

启动阶段的控制流为：

```text
EngineCore
  -> executor.initialize_from_config(...)
  -> worker.initialize_from_config(...)
  -> model runner 初始化 KV cache
  -> NixlConnector.register_kv_caches(...)
```

两个 worker 的最后一条正常日志均为：

```text
Registering KV_Caches. use_mla: False, kv_buffer_device: cuda,
use_host_buffer: False
```

该日志位于
`vllm/distributed/kv_transfer/kv_connector/v1/nixl/base_worker.py` 的
`register_kv_caches()` 内部。它之后依次会构造 region、调用
`get_reg_descs()`、`register_memory()`、创建 local transfer handler，并生成
agent metadata。当前日志级别不足以区分具体停在哪一步。

`shm_broadcast.py` 的警告是**后果而非根因**：EngineCore 已把
`initialize_from_config` RPC 发送给两个 TP worker，但至少一个 worker 尚未从
NIXL KV 注册路径返回；对应的 shared-memory broadcast block 因而无法释放。

### 可能原因

- 两个 TP rank 同时为各自 GPU 上的 HND KV cache 注册 NIXL/UCX GPU memory 时阻塞。
- 某张 GPU 的 CUDA IPC、UCX 或 GPUDirect 注册路径异常；单卡成功不能排除该问题。
- 本项目的 prompt-stream/NIXL producer 改动仍存在隐含单 TP rank 假设。

当前不能仅凭日志断言是 `register_memory()` 的唯一问题；必须补充 rank 级插桩后
再下结论。

### 排查顺序

1. 先让 P 以 TP=2 **且禁用 KV transfer** 启动。`finish_ttft` 不需要 D 或 KV
   transfer；若此配置成功，说明模型 TP 与 Triton 本身可用，问题限定在 NIXL 路径。
2. 在 `register_kv_caches()` 的以下边界记录 `tp_rank`、耗时和完成日志：
   `get_reg_descs()`、`register_memory()`、`register_local_xfer_handler()`、
   `get_agent_metadata()`。必要时为每个阶段加入可控超时或采集 worker stack。
3. 对比 TP=1 + NIXL、TP=2 + 无 NIXL、TP=2 + NIXL 三组启动日志；不要把
   FlashAttention/Triton、模型大小和 TP size 同时改变。
4. 在完成 P 侧定位后，再测试完整 P/D 都为 TP=2 的 NIXL transfer。P=TP2、D=TP1
   不应被当作完整 PD transfer 的有效配置。

### 当前缓解措施

`research/scripts/server/start_p_node.sbatch` 已支持 `ENABLE_KV_TRANSFER`，默认值为
`1`，因此不改变现有完整 P/D 行为。设为 `0` 时会完全省略
`--kv-transfer-config`；TTFT-only 的 TP=2 实验应使用此模式。完整 P/D 实验才启用
NIXL，并在上述根因定位完成后使用。
