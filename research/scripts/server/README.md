# Prompt Stream Server 脚本

这个目录负责启动实验链路中的服务端组件：

- `proxy_server.py`：实验代理服务器，RAG 前端和 replay 脚本只访问它。
- `start_p_node.sbatch`：在 Slurm 上启动 P 节点，即 NIXL producer。
- `start_d_node.sbatch`：在 Slurm 上启动 D 节点，即 NIXL consumer。

## proxy 服务器

`proxy_server.py` 接收 prompt stream RESTFUL 请求，并把请求分发给 P/D 节点。它负责维护一次 prompt stream 会话的 P 节点、D 节点、内部 `request_id` 和 decode prompt 镜像。

主要接口包括：

- `/v1/prompt_stream/start`：创建长期 prompt stream 请求，并转发给 P 节点。
- `/v1/prompt_stream/event`：转发 append、update 或 source_finished 事件。
- `/v1/prompt_stream/finish`：让 P 节点完成最终问题 prefill，再携带 `kv_transfer_params` 请求 D 节点 decode。
- `/v1/prompt_stream/finish_ttft`：只测 P 节点首 token TTFT，不启动 D 节点。
- `/v1/prompt_stream/abort`：中止仍在 proxy/P 节点侧保存的 prompt stream 会话。
- `/v1/completions`：普通 OpenAI completions 代理，用于 normal baseline。

启动模板：

```shell
.venv/bin/python research/scripts/server/proxy_server.py \
  --host 0.0.0.0 \
  --port <PROXY_PORT> \
  --prefiller-hosts <P_HOST> \
  --prefiller-ports <P_PORT> \
  --decoder-hosts <D_HOST> \
  --decoder-ports <D_PORT>
```

如果 P、D、proxy 不在同一个节点上，`<P_HOST>` 和 `<D_HOST>` 不能写 `127.0.0.1`。`127.0.0.1` 永远表示 proxy 当前所在机器自己。

调试 D 侧 KV transfer 时可以禁用远程 KV：

```shell
.venv/bin/python research/scripts/server/proxy_server.py \
  --host 0.0.0.0 \
  --port <PROXY_PORT> \
  --prefiller-hosts <P_HOST> \
  --prefiller-ports <P_PORT> \
  --decoder-hosts <D_HOST> \
  --decoder-ports <D_PORT> \
  --disable-kv-transfer
```

这个模式会让 D 节点本地重新 prefill 整个 prompt。正常验证 P -> D KV transfer 时不要加 `--disable-kv-transfer`。

## Slurm 启动 P/D 节点

建议在 vLLM 仓库根目录提交 job。脚本默认使用 `SLURM_SUBMIT_DIR` 作为 `REPO_ROOT`，并把日志写入 `research/scripts/logs/`。

先创建日志目录：

```shell
mkdir -p research/scripts/logs
```

提交 P 节点：

```shell
MODEL_PATH=<MODEL_PATH> \
SERVED_MODEL_NAME=<SERVED_MODEL_NAME> \
GPU_MEMORY_UTILIZATION=<GPU_UTIL> \
MAX_MODEL_LEN=<MAX_MODEL_LEN> \
MAX_NUM_SEQS=<MAX_NUM_SEQS> \
TENSOR_PARALLEL_SIZE=1 \
sbatch --nodelist=<P_HOST> research/scripts/server/start_p_node.sbatch
```

### P 节点的单节点 TP

`start_p_node.sbatch` 默认仍请求一张 GPU，并使用
`TENSOR_PARALLEL_SIZE=1`。需要 TP=2 时，在同一个节点申请两张 GPU，并
设置相同的 TP size：

提交命令中的 `--gres=gpu:2` 会覆盖脚本默认的 `#SBATCH --gres=gpu:1`。

```shell
TENSOR_PARALLEL_SIZE=2 \
MODEL_PATH=<MODEL_PATH> \
SERVED_MODEL_NAME=<SERVED_MODEL_NAME> \
GPU_MEMORY_UTILIZATION=<GPU_UTIL> \
MAX_MODEL_LEN=<MAX_MODEL_LEN> \
MAX_NUM_SEQS=<MAX_NUM_SEQS> \
sbatch --nodelist=<P_HOST> --gres=gpu:2 \
  research/scripts/server/start_p_node.sbatch
```

`--nodes=1` 已固定在脚本中，因此这只会使用同一主机上的两张卡，不会
形成多节点 TP。不要手动设置 `CUDA_VISIBLE_DEVICES`；Slurm 会把申请到的
两张卡暴露给 vLLM。`TENSOR_PARALLEL_SIZE` 必须不大于实际申请到的 GPU 数。

本轮 `finish_ttft` 实验只需要 P 使用 TP=2。若之后运行完整 P -> D KV
transfer 与 decode，应把 D 也改为同样的 TP=2 配置；当前
`start_d_node.sbatch` 仍是单卡脚本。


提交 D 节点：

```shell
MODEL_PATH=<MODEL_PATH> \
SERVED_MODEL_NAME=<SERVED_MODEL_NAME> \
GPU_MEMORY_UTILIZATION=<GPU_UTIL> \
MAX_MODEL_LEN=<MAX_MODEL_LEN> \
MAX_NUM_SEQS=<MAX_NUM_SEQS> \
sbatch --nodelist=<D_HOST> research/scripts/server/start_d_node.sbatch
```

常用环境变量：

| 变量 | 默认值 | 含义 |
| --- | --- | --- |
| `REPO_ROOT` | `SLURM_SUBMIT_DIR` | vLLM 仓库根目录 |
| `PYTHON_BIN` | `${REPO_ROOT}/.venv/bin/python` | Python 解释器 |
| `MODEL_PATH` | `${HOME}/Models/Qwen2.5-3B/` | 模型路径 |
| `SERVED_MODEL_NAME` | `qwen2.5-3b` | OpenAI API 中的模型名 |
| `HOST` | `0.0.0.0` | vLLM server 监听地址 |
| `PORT` | P 为 `8100`，D 为 `8200` | OpenAI server 端口 |
| `VLLM_NIXL_SIDE_CHANNEL_HOST` | 当前 Slurm 节点 `hostname` | NIXL 对外发布地址 |
| `NIXL_SIDE_CHANNEL_PORT` | P 为 `5600`，D 为 `5601` | NIXL side-channel 端口 |
| `GPU_MEMORY_UTILIZATION` | `0.65` | vLLM GPU 显存使用比例 |
| `MAX_MODEL_LEN` | `4096` | 最大上下文长度 |
| `TENSOR_PARALLEL_SIZE` | `1` | 仅 P 节点：同一 Slurm 节点内的 vLLM TP rank 数；值为 `2` 时提交命令还必须申请两张 GPU。 |
| `MAX_NUM_SEQS` | `16` | 最大并发序列数 |
| `ATTENTION_BACKEND` | 空，自动选择 | 显式指定 vLLM attention backend，例如 `TRITON_ATTN` 或 `FLASH_ATTN`。P/D 对照实验应使用相同值。 |
| `PROMPT_STREAM_MASKED_PADDING` | `0` | `1` 时传入 `--prompt-stream-masked-padding`，屏蔽分段 KV 中间 padding 并修正 RoPE；未指定后端时自动选择 `TRITON_ATTN`。完整 PD 模式的 P/D 均需开启。 |
| `GENERATION_CONFIG` | 空；masked 模式为 `vllm` | 传给 `--generation-config`。masked 模式默认覆盖 Qwen 的模型 generation config，避免其 `repetition_penalty=1.05` 将物理 padding 计入词频。 |
| `ENABLE_KV_TRANSFER` | `1` | 仅 P 节点：`1` 时使用 NIXL `kv_producer`；`0` 时完全省略 `--kv-transfer-config`。TP=2 的 TTFT-only 实验应设为 `0`。 |
| `PROMPT_STREAM_UPDATE_COALESCING_CONFIG` | 空，功能关闭 | 仅 P 节点使用的 Update 合并 JSON；非空时透传为 `--prompt-stream-update-coalescing-config`。 |

使用中间 padding mask 时，在原有提交命令中增加
`PROMPT_STREAM_MASKED_PADDING=1`。例如仅测 P 首 token：

```shell
PROMPT_STREAM_MASKED_PADDING=1 \
ATTENTION_BACKEND=TRITON_ATTN \
ENABLE_KV_TRANSFER=0 \
MODEL_PATH="${HOME}/Models/Qwen2.5-3B" \
SERVED_MODEL_NAME=qwen2.5-3b \
GPU_MEMORY_UTILIZATION=0.6 \
MAX_MODEL_LEN=4096 \
MAX_NUM_SEQS=16 \
sbatch --nodelist=g2c9 research/scripts/server/start_p_node.sbatch
```

完整 PD 生成保留 `ENABLE_KV_TRANSFER=1`，D 的原有提交命令也加上
`PROMPT_STREAM_MASKED_PADDING=1 ATTENTION_BACKEND=TRITON_ATTN`。proxy 和 P/D
进程都需重启以加载新代码；replay 的接口和 chunk 格式不变。不要在完整 PD
路径使用 proxy 的 `--disable-kv-transfer`，该诊断路径没有承诺复现带 mask
的分段语义，当前会拒绝这种组合。TTFT-only 不受此限制。

启用该开关时，P/D 脚本会默认追加 `--generation-config vllm`。这不是为了
改变模型输出偏好，而是避免 Qwen 模型目录里的默认 `repetition_penalty=1.05`
读取物理 padding token；若手动设置 `GENERATION_CONFIG`，必须保证
repetition/presence/frequency penalty 均为默认值，否则 masked 路径会明确拒绝。

不开此变量仍使用旧的 padding 近似。该功能独立于 Update coalescing；第一次
比较建议保持原有事件间隔和 coalescing 配置，只改变这个开关。已验证单算子及
无服务状态机，完整模型/P-D/replay 仍需运行验证。更多说明见
[Masked-Padding.md](../../Masked-Padding.md)。

启用 Update 固定窗口合并的 P 节点提交示例：

```shell
PROMPT_STREAM_UPDATE_COALESCING_CONFIG='{"update_policy":"fixed_window","soft_window_ms":20,"hard_deadline_ms":80}' \
MODEL_PATH=<MODEL_PATH> \
SERVED_MODEL_NAME=<SERVED_MODEL_NAME> \
sbatch --nodelist=<P_HOST> research/scripts/server/start_p_node.sbatch
```

不设置 `PROMPT_STREAM_UPDATE_COALESCING_CONFIG` 时保持逐 Update 事件立即创建
work 的原有行为。D 节点不要设置这个变量。当前 P 脚本支持同节点 TP，
但仍只支持 DP=1、PP=1 和单 in-flight batch；本脚本已有
`--no-async-scheduling`，不需要再添加异步调度参数。

只测 TTFT 时，P 生成首 token 后就结束，既不向 D 传 KV，也不需要初始化 NIXL。
特别是当前 TP=2 的 NIXL KV 注册仍在排查，应显式禁用：

```shell
ENABLE_KV_TRANSFER=0 \
TENSOR_PARALLEL_SIZE=2 \
sbatch --nodelist=<P_HOST> --gres=gpu:2 \
  research/scripts/server/start_p_node.sbatch
```

不设置或设为 `ENABLE_KV_TRANSFER=1` 时维持原有完整 P/D 配置。

跨节点 P/D 时，`VLLM_NIXL_SIDE_CHANNEL_HOST` 必须是对端可以访问的地址。脚本默认用 job 实际运行节点的 `hostname`，通常比 `localhost` 更安全。若集群 DNS 不可用，可以显式设置为日志中 `HOST_IP` 的内网 IP。

查看 job 和日志：

```shell
squeue -u "$USER"
scontrol show job <JOB_ID>
cat research/scripts/logs/vllm-p-<JOB_ID>.out
cat research/scripts/logs/vllm-d-<JOB_ID>.out
```

日志开头会打印：

```text
SLURM_JOB_ID=<JOB_ID>
SLURM_JOB_NODELIST=<GPU_NODE>
HOSTNAME=<GPU_NODE>
HOST_IP=<NODE_IP_LIST>
ROLE=prefill-producer 或 decode-consumer
OPENAI_PORT=<P_PORT 或 D_PORT>
NIXL_SIDE_CHANNEL_HOST=<NIXL_HOST>
NIXL_SIDE_CHANNEL_PORT=<NIXL_PORT>
CUDA_VISIBLE_DEVICES=<VISIBLE_GPU_ID>
SLURM_JOB_GPUS=<PHYSICAL_GPU_ID>
```

`HOSTNAME` 或 `HOST_IP` 就是 proxy 启动时填写的 `<P_HOST>` / `<D_HOST>`。

## 指定或排除节点

Slurm 只知道 GPU 是否被 Slurm job 分配，不一定知道显存是否真的空。如果节点上有手动启动的 GPU 进程或残留进程，`sinfo` 仍可能显示节点 `idle`，但 vLLM 会因为 free memory 不足启动失败。

指定节点：

```shell
sbatch --nodelist=<GPU_NODE> research/scripts/server/start_p_node.sbatch
```

排除节点：

```shell
sbatch --exclude=<GPU_NODE> research/scripts/server/start_d_node.sbatch
```

探测某个节点上 Slurm 分给你的 GPU 显存：

```shell
srun -p gpu --nodelist=<GPU_NODE> --gres=gpu:1 nvidia-smi
```

## 当前环境示例

这一节是当前实验环境的可复制命令。对外发布 README 时可以删除。

示例：P 跑在 `g2c13`，D 跑在 `g2c12`。

```shell
mkdir -p research/scripts/logs
```

```shell
VLLM_NIXL_SIDE_CHANNEL_HOST=g2c9 \
MODEL_PATH=/workplace/home/huangxinying/Models/Qwen2.5-3B/ \
SERVED_MODEL_NAME=qwen2.5-3b \
GPU_MEMORY_UTILIZATION=0.6 \
MAX_MODEL_LEN=4096 \
MAX_NUM_SEQS=16 \
sbatch --nodelist=g2c9 research/scripts/server/start_p_node.sbatch
```

```shell
VLLM_NIXL_SIDE_CHANNEL_HOST=g2c9 \
PROMPT_STREAM_UPDATE_COALESCING_CONFIG='{"update_policy":"fixed_window","soft_window_ms":20,"hard_deadline_ms":80}' \
MODEL_PATH=/workplace/home/huangxinying/Models/Qwen2.5-3B/ \
SERVED_MODEL_NAME=qwen2.5-3b \
GPU_MEMORY_UTILIZATION=0.6 \
MAX_MODEL_LEN=4096 \
MAX_NUM_SEQS=16 \
sbatch --nodelist=g2c9 research/scripts/server/start_p_node.sbatch
```

```shell
VLLM_NIXL_SIDE_CHANNEL_HOST=g2c14 \
MODEL_PATH=/workplace/home/huangxinying/Models/Qwen2.5-3B/ \
SERVED_MODEL_NAME=qwen2.5-3b \
GPU_MEMORY_UTILIZATION=0.6 \
MAX_MODEL_LEN=4096 \
MAX_NUM_SEQS=16 \
sbatch --nodelist=g2c14 research/scripts/server/start_d_node.sbatch
```

proxy：

```shell
.venv/bin/python research/scripts/server/proxy_server.py \
  --host 0.0.0.0 \
  --port 8000 \
  --prefiller-hosts g2c9 \
  --prefiller-ports 8100 \
  --decoder-hosts g2c9 \
  --decoder-ports 8200
```
