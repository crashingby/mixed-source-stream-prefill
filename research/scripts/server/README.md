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
sbatch --nodelist=<P_HOST> research/scripts/server/start_p_node.sbatch
```

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
| `MAX_NUM_SEQS` | `16` | 最大并发序列数 |
| `PROMPT_STREAM_UPDATE_COALESCING_CONFIG` | 空，功能关闭 | 仅 P 节点使用的 Update 合并 JSON；非空时透传为 `--prompt-stream-update-coalescing-config`。 |

启用严格 Append gate 与固定窗口的 P 节点提交示例：

```shell
PROMPT_STREAM_UPDATE_COALESCING_CONFIG='{"append_gate":"hold_until_append_finished","update_policy":"fixed_window","soft_window_ms":20,"hard_deadline_ms":80}' \
MODEL_PATH=<MODEL_PATH> \
SERVED_MODEL_NAME=<SERVED_MODEL_NAME> \
sbatch --nodelist=<P_HOST> research/scripts/server/start_p_node.sbatch
```

不设置 `PROMPT_STREAM_UPDATE_COALESCING_CONFIG` 时保持逐 Update 事件立即创建
work 的原有行为。D 节点不要设置这个变量。第一版只支持 P 节点上的
NIXL pull-mode `kv_producer`、DP=1、PP=1 和单 in-flight batch；本脚本已有
`--no-async-scheduling`，不需要再添加异步调度参数。

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
PROMPT_STREAM_UPDATE_COALESCING_CONFIG='{"append_gate":"hold_until_append_finished","update_policy":"fixed_window","soft_window_ms":20,"hard_deadline_ms":80}' \
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
  --decoder-hosts g2c14 \
  --decoder-ports 8200
```
