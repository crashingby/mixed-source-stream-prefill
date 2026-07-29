# Prompt Stream 实验脚本

这个目录放的是流式 prefill 的最小联调脚本。当前目标不是做完整 RAG 系统，而是先把
“前端 -> proxy -> P 节点 -> D 节点”的 RESTFUL 链路跑通，并确认 P/D 分离下的
NIXL KV transfer 可以工作。

## 脚本说明

### `proxy_server.py`

`proxy_server.py` 是 PD 分离场景下的实验代理服务器。RAG 前端只访问这个 proxy，proxy 负责：

- 接收 `/v1/prompt_stream/start`，为请求选择一个 P 节点和一个 D 节点。
- 把 start 请求转发给 P 节点的 `/v1/prompt_stream/start`。
- 保存 P 节点返回的内部 `request_id`，后续 event/finish 都使用这个 id。
- 接收 `/v1/prompt_stream/event`，把 append/update/source_finished 事件转发给同一个 P 节点。
- 接收 `/v1/prompt_stream/finish`，先让 P 节点完成最后用户问题的 prefill，
  校验物理 token/KV 覆盖范围，再把 `kv_transfer_params` 放进普通 OpenAI
  decode 请求转发给 D 节点。
- 接收 `/v1/prompt_stream/finish_ttft`，只返回 P 节点采样出的首 token，
  立刻结束 SSE，不启动 D 节点；这个路径专门用于 TTFT-only 实验。
- 把 P 节点采样出的首 token 接到 D prompt，并在 D 返回后合并回最终响应。
- 对 P 返回的 exact block manifest 做校验；D 只拉取物理 prompt KV，P 的
  首 token 在 D 本地 forward。

本轮修改提升了 NIXL connector 兼容版本。更新源码后必须同时重启 P、D 和
proxy，不能让新旧进程混跑。

### `append_frontend.py`

`append_frontend.py` 是一个最小前端/RAG mock。它默认只发送 append 文本事件：

1. 向 proxy 发送 `/v1/prompt_stream/start`，其中 `prompt` 是固定模板/prefix。
2. 连续向 proxy 发送多个 `/v1/prompt_stream/event`，每个事件都是 `event_type=append`。
3. 向 proxy 发送 `/v1/prompt_stream/finish`，其中 `event_type=prompt_finished`，文本是最后的用户问题。
4. 打印 proxy 返回的最终 decode 响应。

这个脚本故意不做 tokenizer。它发送原始文本，由 P 节点的 prompt stream RESTFUL serving 层负责把 `text` 转成 `token_ids`。

### `update_frontend.py`

`update_frontend.py` 是一个最小 update 语义前端/RAG mock。它默认只发送一个 update source 的文本快照：

1. 向 proxy 发送 `/v1/prompt_stream/start`，其中 `prompt` 是固定模板/prefix。
2. 连续向 proxy 发送多个 `/v1/prompt_stream/event`，每个事件都是 `event_type=update`。
3. 每个 update 事件的 `text` 都表示该 source 的**完整当前快照**，不是增量 patch。
4. 向 proxy 发送 `/v1/prompt_stream/finish`，其中 `event_type=prompt_finished`，文本是最后的用户问题。
5. 打印 proxy 返回的最终 decode 响应。

这个脚本用于测试 update 源内 LCP、旧 update 后缀失效、以及 finish 后 update KV 与 prefix/append/question 的组合逻辑。

### `mixed_frontend.py`

`mixed_frontend.py` 是一个 append/update 混合语义前端/RAG mock。它默认发送一组交错事件：

```text
append -> update -> append -> append -> update -> finish
```

这个脚本用于测试更接近真实 RAG 的事件顺序：稳定上下文持续 append，同时一个可变检索结果以 update 快照形式被刷新。每个 append 事件是新增 chunk；每个 update 事件仍然是同一个 update source 的**完整当前快照**，不是增量 diff。

脚本也支持用多个 `--event` 自定义事件顺序，例如 `--event append:...` 和 `--event update:...`。

### `start_p_node.sbatch` / `start_d_node.sbatch`

这两个脚本用于在 Slurm 集群上提交 P/D vLLM server：

- `start_p_node.sbatch`：启动 P 节点，即 NIXL producer。
- `start_d_node.sbatch`：启动 D 节点，即 NIXL consumer。

脚本默认使用 `SLURM_SUBMIT_DIR` 作为仓库根目录，因此建议在 vLLM 仓库根目录下提交。

## 占位符说明

下面的通用命令使用占位符。实际运行时请替换为你的环境值：

| 占位符 | 含义 | 示例 |
| --- | --- | --- |
| `<REPO_ROOT>` | 当前 vLLM 仓库根目录 | `/path/to/vllm` |
| `<MODEL_PATH>` | 本地模型目录或 Hugging Face 模型名 | `/path/to/Qwen2.5-3B` |
| `<SERVED_MODEL_NAME>` | OpenAI API 请求体里的 `model` 名称 | `qwen2.5-3b` |
| `<P_HOST>` | P 节点 hostname 或 IP | `gpu-node-p` |
| `<D_HOST>` | D 节点 hostname 或 IP | `gpu-node-d` |
| `<PROXY_HOST>` | proxy 所在机器的 hostname 或 IP | `127.0.0.1` |
| `<P_PORT>` | P 节点 OpenAI server 端口 | `8100` |
| `<D_PORT>` | D 节点 OpenAI server 端口 | `8200` |
| `<PROXY_PORT>` | proxy 端口 | `8000` |
| `<P_NIXL_PORT>` | P 节点 NIXL side-channel 端口 | `5600` |
| `<D_NIXL_PORT>` | D 节点 NIXL side-channel 端口 | `5601` |
| `<GPU_UTIL>` | vLLM 可占用 GPU 显存比例 | `0.6` |
| `<MAX_MODEL_LEN>` | vLLM 最大上下文长度 | `4096` |
| `<MAX_NUM_SEQS>` | vLLM 最大并发序列数 | `16` |
| `<GPU_NODE>` | Slurm GPU 节点名 | `g2c12` |

注意：`append_frontend.py --model` 要填写 `<SERVED_MODEL_NAME>`，不是 `<MODEL_PATH>`。
`update_frontend.py --model` 和 `mixed_frontend.py --model` 也一样要填写 `<SERVED_MODEL_NAME>`。
`<MODEL_PATH>` 只出现在启动 vLLM server 或提交 sbatch job 的命令中。

## 参数关系

### `max_tokens`

`append_frontend.py --max-tokens`、`update_frontend.py --max-tokens` 和 `mixed_frontend.py --max-tokens` 表示用户希望最终 completion 最多生成多少 token。
例如：

```shell
.venv/bin/python research/scripts/append_frontend.py \
  --proxy-url http://<PROXY_HOST>:<PROXY_PORT> \
  --model <SERVED_MODEL_NAME> \
  --max-tokens 64 \
  --temperature 0
```

proxy 内部会把发给 P 节点的请求强制改成：

```text
P max_tokens = 1
```

原因是 P 节点只负责 prefill 到最终问题，并采样首 token。之后 proxy 会把 D 节点的
`max_tokens` 减去 P 已经生成的 token 数。因此当用户设置：

```text
max_tokens = 64
```

实际链路通常是：

```text
P 节点生成 1 个 token
D 节点最多继续生成 63 个 token
最终 completion_tokens 约为 64
```

如果最终响应里 `finish_reason=length`，通常就是达到了这个总生成上限。

### `gpu_memory_utilization` 和 `max_num_seqs`

`<GPU_UTIL>` 控制 vLLM 预留多少 GPU 显存给 KV cache 和运行时状态。24 GiB A30 上，
`0.85` 往往要求约 20 GiB 空闲显存；如果节点上已有残留进程，启动会失败。

`<MAX_NUM_SEQS>` 控制最大并发序列数。联调阶段建议先用 `16`，如果 vLLM warmup sampler
仍然 OOM，可以降到 `8`。

## 单机手动启动模板

如果 P/D/proxy 都在同一台机器上，可以先用手动命令联调。

### 启动 P 节点

```shell
CUDA_VISIBLE_DEVICES=0 \
UCX_NET_DEVICES=all \
VLLM_NIXL_SIDE_CHANNEL_PORT=<P_NIXL_PORT> \
.venv/bin/python -m vllm.entrypoints.openai.api_server \
  --model <MODEL_PATH> \
  --served-model-name <SERVED_MODEL_NAME> \
  --trust-remote-code \
  --dtype auto \
  --gpu-memory-utilization <GPU_UTIL> \
  --max-model-len <MAX_MODEL_LEN> \
  --max-num-seqs <MAX_NUM_SEQS> \
  --host 0.0.0.0 \
  --port <P_PORT> \
  --enforce-eager \
  --no-async-scheduling \
  --kv-transfer-config '{"kv_connector":"NixlConnector","kv_role":"kv_producer","kv_load_failure_policy":"fail"}'
```

### 启动 D 节点

```shell
CUDA_VISIBLE_DEVICES=1 \
UCX_NET_DEVICES=all \
VLLM_NIXL_SIDE_CHANNEL_PORT=<D_NIXL_PORT> \
.venv/bin/python -m vllm.entrypoints.openai.api_server \
  --model <MODEL_PATH> \
  --served-model-name <SERVED_MODEL_NAME> \
  --trust-remote-code \
  --dtype auto \
  --gpu-memory-utilization <GPU_UTIL> \
  --max-model-len <MAX_MODEL_LEN> \
  --max-num-seqs <MAX_NUM_SEQS> \
  --host 0.0.0.0 \
  --port <D_PORT> \
  --enforce-eager \
  --no-async-scheduling \
  --kv-transfer-config '{"kv_connector":"NixlConnector","kv_role":"kv_consumer","kv_load_failure_policy":"fail"}'
```

### 启动 proxy

```shell
.venv/bin/python research/scripts/proxy_server.py \
  --host 0.0.0.0 \
  --port <PROXY_PORT> \
  --prefiller-hosts <P_HOST> \
  --prefiller-ports <P_PORT> \
  --decoder-hosts <D_HOST> \
  --decoder-ports <D_PORT>
```

如果 P、D、proxy 不在同一个节点上，`<P_HOST>` 和 `<D_HOST>` 不能写 `127.0.0.1`。
`127.0.0.1` 永远表示 proxy 当前所在机器自己。

## Slurm 启动模板

### 提交 P/D job

先创建日志目录：

```shell
mkdir -p research/scripts/logs
```

使用默认参数提交：

```shell
sbatch research/scripts/start_p_node.sbatch
sbatch research/scripts/start_d_node.sbatch
```

用环境变量覆盖常用参数：

```shell
MODEL_PATH=<MODEL_PATH> \
SERVED_MODEL_NAME=<SERVED_MODEL_NAME> \
GPU_MEMORY_UTILIZATION=<GPU_UTIL> \
MAX_MODEL_LEN=<MAX_MODEL_LEN> \
MAX_NUM_SEQS=<MAX_NUM_SEQS> \
sbatch research/scripts/start_p_node.sbatch
```

D 节点同理：

```shell
MODEL_PATH=<MODEL_PATH> \
SERVED_MODEL_NAME=<SERVED_MODEL_NAME> \
GPU_MEMORY_UTILIZATION=<GPU_UTIL> \
MAX_MODEL_LEN=<MAX_MODEL_LEN> \
MAX_NUM_SEQS=<MAX_NUM_SEQS> \
sbatch research/scripts/start_d_node.sbatch
```

如果从仓库根目录以外的位置提交，需要显式传：

```shell
REPO_ROOT=<REPO_ROOT> sbatch research/scripts/start_p_node.sbatch
```

### 指定或排除节点

Slurm 只知道 GPU 资源是否被 Slurm job 分配，不一定知道显存是否真的空。如果节点上有
手动启动的 GPU 进程或残留进程，`sinfo` 仍可能显示节点 `idle`，但 vLLM 会因为 free memory
不足启动失败。

指定节点：

```shell
sbatch --nodelist=<GPU_NODE> research/scripts/start_p_node.sbatch
sbatch --nodelist=<GPU_NODE> research/scripts/start_d_node.sbatch
```

排除节点：

```shell
sbatch --exclude=<GPU_NODE> research/scripts/start_d_node.sbatch
```

探测某个节点上 Slurm 分给你的 GPU 显存：

```shell
srun -p gpu --nodelist=<GPU_NODE> --gres=gpu:1 nvidia-smi
```

### 查看 job 和日志

```shell
squeue -u "$USER"
scontrol show job <JOB_ID>
```

脚本会在启动 vLLM 前打印节点和 GPU 信息：

```shell
cat research/scripts/logs/vllm-p-<JOB_ID>.out
cat research/scripts/logs/vllm-d-<JOB_ID>.out
```

日志开头会包含：

```text
SLURM_JOB_ID=<JOB_ID>
SLURM_JOB_NODELIST=<GPU_NODE>
HOSTNAME=<GPU_NODE>
HOST_IP=<NODE_IP_LIST>
ROLE=prefill-producer
OPENAI_PORT=<P_PORT>
NIXL_SIDE_CHANNEL_PORT=<P_NIXL_PORT>
CUDA_VISIBLE_DEVICES=<VISIBLE_GPU_ID>
SLURM_JOB_GPUS=<PHYSICAL_GPU_ID>
```

`HOSTNAME` 或 `HOST_IP` 就是 proxy 的 `<P_HOST>` / `<D_HOST>`。

## 运行前端测试脚本

### 运行 append-only 示例

```shell
.venv/bin/python research/scripts/append_frontend.py \
  --proxy-url http://<PROXY_HOST>:<PROXY_PORT> \
  --model <SERVED_MODEL_NAME> \
  --max-tokens 64 \
  --temperature 0
```

也可以自己指定 append chunk：

```shell
.venv/bin/python research/scripts/append_frontend.py \
  --proxy-url http://<PROXY_HOST>:<PROXY_PORT> \
  --model <SERVED_MODEL_NAME> \
  --chunk "检索片段 1：vLLM 的 EngineCore 保存请求状态。" \
  --chunk "检索片段 2：proxy 会先把事件发给 P 节点。" \
  --final-question "\n用户问题：proxy 在 decode 前做了什么？\n回答："
```

### 运行 update-only 示例

```shell
.venv/bin/python research/scripts/update_frontend.py \
  --proxy-url http://<PROXY_HOST>:<PROXY_PORT> \
  --model <SERVED_MODEL_NAME> \
  --max-tokens 64 \
  --temperature 0
```

也可以自己指定 update 快照。注意每个 `--snapshot` 都是同一个 source 的**完整当前快照**，不是增量 diff：

```shell
.venv/bin/python research/scripts/update_frontend.py \
  --proxy-url http://<PROXY_HOST>:<PROXY_PORT> \
  --model <SERVED_MODEL_NAME> \
  --snapshot "检索快照 v0：proxy 收到流式事件。" \
  --snapshot "检索快照 v1：proxy 收到流式 prefill 事件，然后请求 D 节点 decode。" \
  --final-question "\n用户问题：最新快照里 proxy 在 decode 前做了什么？\n回答："
```

### 运行 mixed append/update 示例

```shell
.venv/bin/python research/scripts/mixed_frontend.py \
  --proxy-url http://<PROXY_HOST>:<PROXY_PORT> \
  --model <SERVED_MODEL_NAME> \
  --max-tokens 64 \
  --temperature 0
```

默认事件顺序是：

```text
append -> update -> append -> append -> update -> finish
```

也可以用多个 `--event` 自定义混合事件顺序。格式是 `<append|update>:text`，命令行顺序就是发送顺序：

```shell
.venv/bin/python research/scripts/mixed_frontend.py \
  --proxy-url http://<PROXY_HOST>:<PROXY_PORT> \
  --model <SERVED_MODEL_NAME> \
  --event "append:稳定片段 1：EngineCore 保存长期请求状态。" \
  --event "append:稳定片段 2：Append chunk 单调加入 prefix_append KV 域。" \
  --event "update:可变快照 v0：proxy 收到流式事件。" \
  --event "update:可变快照 v1：proxy 收到流式事件，并在 decode 前请求 P 节点 prefill。" \
  --final-question "\n用户问题：稳定片段和最新快照说明 proxy 在 decode 前做什么？\n回答："
```

## 禁用 KV transfer 的对照实验

当 D 节点输出异常时，可以先让 proxy 加 `--disable-kv-transfer` 做 baseline：

```shell
.venv/bin/python research/scripts/proxy_server.py \
  --host 0.0.0.0 \
  --port <PROXY_PORT> \
  --prefiller-hosts <P_HOST> \
  --prefiller-ports <P_PORT> \
  --decoder-hosts <D_HOST> \
  --decoder-ports <D_PORT> \
  --disable-kv-transfer
```

这个模式用于区分两类问题：

- 如果禁用 KV transfer 后输出正常，说明 prompt/token ids 本身没问题，异常更可能来自 P -> D KV 内容、block 顺序或 block 对齐。
- 如果禁用 KV transfer 后输出仍然异常，说明问题可能在 prompt 构造、token ids 或普通 completions 请求本身。

正常验证 P -> D KV transfer 时不要加 `--disable-kv-transfer`。

## 当前环境示例

这一节是当前实验环境的可复制命令。对外发布 README 时可以删除。

### 当前模型和通用参数

```shell
export MODEL_PATH=/workplace/home/huangxinying/Models/Qwen2.5-3B/
export SERVED_MODEL_NAME=qwen2.5-3b
export GPU_MEMORY_UTILIZATION=0.6
export MAX_MODEL_LEN=4096
export MAX_NUM_SEQS=16
```

### 当前推荐的 Slurm 提交流程

示例：P 跑在 `g2c13`，D 跑在 `g2c12`。

这两个 job 位于**不同物理节点**时，必须分别设置
`VLLM_NIXL_SIDE_CHANNEL_HOST`。该变量是节点对外发布给 NIXL 对端的握手地址；
P finish 返回的 `kv_transfer_params.remote_host` 就来自 P 的这个值。默认值是
`localhost`，只适用于 P/D 位于同一物理节点的情况，跨节点时会让 D 错连到自己。

这里直接使用可被集群节点互相解析的 Slurm 主机名。若集群 DNS 不可用，可替换为
对应节点日志中 `HOST_IP` 的可达内网 IP（本环境优先使用 `10.4.1.x` 地址）。

```shell
mkdir -p research/scripts/logs
```

```shell
VLLM_NIXL_SIDE_CHANNEL_HOST=g2c13 \
MODEL_PATH=/workplace/home/huangxinying/Models/Qwen2.5-3B/ \
SERVED_MODEL_NAME=qwen2.5-3b \
GPU_MEMORY_UTILIZATION=0.6 \
MAX_MODEL_LEN=4096 \
MAX_NUM_SEQS=16 \
sbatch --nodelist=g2c13 research/scripts/start_p_node.sbatch
```

```shell
VLLM_NIXL_SIDE_CHANNEL_HOST=g2c12 \
MODEL_PATH=/workplace/home/huangxinying/Models/Qwen2.5-3B/ \
SERVED_MODEL_NAME=qwen2.5-3b \
GPU_MEMORY_UTILIZATION=0.6 \
MAX_MODEL_LEN=4096 \
MAX_NUM_SEQS=16 \
sbatch --nodelist=g2c12 research/scripts/start_d_node.sbatch
```

两个 job 日志启动后应分别确认：

```text
HOSTNAME=g2c13
NIXL_SIDE_CHANNEL_HOST=g2c13
NIXL_SIDE_CHANNEL_PORT=5600

HOSTNAME=g2c12
NIXL_SIDE_CHANNEL_HOST=g2c12
NIXL_SIDE_CHANNEL_PORT=5601
```

当前 sbatch 脚本会打印并传递 `NIXL_SIDE_CHANNEL_HOST` 与
`NIXL_SIDE_CHANNEL_PORT`。若提交命令没有显式设置 host，脚本会以该 Slurm job 的
`hostname` 作为默认值；显式设置仍可用于改用可达内网 IP。P finish 返回给 proxy
的 `kv_transfer_params.remote_host` 必须是 P 节点地址，不能是 `localhost` 或
`127.0.0.1`。

如果某个节点显存或工具链有问题，可以换成其它空闲节点，或排除问题节点：

```shell
sbatch --exclude=g2c15 research/scripts/start_d_node.sbatch
```

### 当前 proxy 启动示例

假设 P 日志显示 `HOSTNAME=g2c13`，D 日志显示 `HOSTNAME=g2c12`：

```shell
.venv/bin/python research/scripts/proxy_server.py \
  --host 0.0.0.0 \
  --port 8000 \
  --prefiller-hosts g2c13 \
  --prefiller-ports 8100 \
  --decoder-hosts g2c12 \
  --decoder-ports 8200
```

如果 P/D 都在同一节点，例如都在 `g2c13`，proxy 两侧 host 都填写该节点即可；
此时 side-channel 使用 `localhost` 也能工作，但仍建议保留 sbatch 的自动 host 设置，
这样后续迁移为跨节点部署时不会遗漏配置。

```shell
.venv/bin/python research/scripts/proxy_server.py \
  --host 0.0.0.0 \
  --port 8000 \
  --prefiller-hosts g2c13 \
  --prefiller-ports 8100 \
  --decoder-hosts g2c13 \
  --decoder-ports 8200
```

### 当前 append-only 测试示例

```shell
.venv/bin/python research/scripts/append_frontend.py \
  --proxy-url http://127.0.0.1:8000 \
  --model qwen2.5-3b \
  --max-tokens 64 \
  --temperature 0
```

### 当前 update-only 测试示例

```shell
.venv/bin/python research/scripts/update_frontend.py \
  --proxy-url http://127.0.0.1:8000 \
  --model qwen2.5-3b \
  --max-tokens 64 \
  --temperature 0
```

### 当前 mixed append/update 测试示例

```shell
.venv/bin/python research/scripts/mixed_frontend.py \
  --proxy-url http://127.0.0.1:8000 \
  --model qwen2.5-3b \
  --max-tokens 64 \
  --temperature 0
```

## RESTFUL 请求形状

这些 frontend 脚本对 proxy 发送的核心请求如下。`mixed_frontend.py` 只是按同一个 `request_id` 交错发送 Append event 和 Update event。

### Start

```json
{
  "api": "completions",
  "model": "qwen2.5-3b",
  "prompt": "fixed prefix/template",
  "stream": false,
  "max_tokens": 64,
  "temperature": 0.0
}
```

proxy 会把这个请求转发给 P 节点，并返回：

```json
{
  "request_id": "p-node-internal-request-id",
  "status": "started",
  "token_ids": [101, 102, 103]
}
```

### Append event

```json
{
  "request_id": "p-node-internal-request-id",
  "event": {
    "source_id": "demo_append_source",
    "event_type": "append",
    "source_type": "append",
    "chunk_id": "demo_append_source-0",
    "version": 0,
    "sequence_no": 0,
    "text": "Document chunk text",
    "arrival_time": 1783760000.0
  }
}
```

### Update event

```json
{
  "request_id": "p-node-internal-request-id",
  "event": {
    "source_id": "demo_update_source",
    "event_type": "update",
    "source_type": "update",
    "chunk_id": "demo_update_source-snapshot-0",
    "version": 0,
    "sequence_no": 0,
    "text": "Document snapshot text",
    "arrival_time": 1783760000.0
  }
}
```

update 事件的 `text` 是该来源当前版本的完整快照。P 节点内部会只在 update 源内比较新旧 token ids 的 LCP，并让不匹配的 update 后缀重新 prefill。

### Finish

```json
{
  "request_id": "p-node-internal-request-id",
  "event": {
    "source_id": "final_question",
    "event_type": "prompt_finished",
    "source_type": null,
    "chunk_id": "final-question",
    "version": 0,
    "sequence_no": 0,
    "text": "\nUser question: ...\nAnswer:",
    "arrival_time": 1783760001.0
  }
}
```

## 当前限制

- `append_frontend.py` 覆盖 append-only 文本模式。
- `update_frontend.py` 覆盖单 update source 的文本快照模式。
- `mixed_frontend.py` 覆盖单 append source + 单 update source 的文本混合模式；多 update source、source_finished 和 token_ids 双模式之后可以继续补测试脚本。
- proxy 当前只接受 `api=completions`，支持 `stream=false` 和 `stream=true`。
  chat renderer 的物理 token 镜像尚未实现，因此 chat 请求会在 start 阶段
  返回 400。
- `prompt_finished` 必须携带至少一个最终问题 token。
- P 节点当前要求同步调度、pipeline parallel size 1、无 speculative decoding、
  NIXL direct-device transfer，并且 KV groups 必须是 block size
  一致的 full attention / MLA；host-buffer 模式暂不支持。
- proxy 强制要求 P 节点在 finish 后返回物理 `prompt_token_ids`、
  `kv_transfer_params` 和合法的 `remote_num_tokens`；缺失或长度不一致时会
  fail closed 返回 502，不会继续请求 D。
- 启用 exact prompt-stream KV transfer 时，proxy 强制
  `remote_num_tokens == physical_prompt_len`。D prompt 只允许在物理 prompt
  后附加 P 产生的 `0/1` 个首 token；该 token 不属于远端 KV。
- proxy 还会严格检查 manifest 的字段类型、KV group 数、每组 block count、
  block size、block id 和 token 长度。异常时返回 502；D scheduler/worker
  也会把异常限制在单个请求，不让错位 KV 进入 NIXL read。
- Slurm 节点显示 `idle` 不代表 GPU 显存一定空闲；以 sbatch 日志中的 `nvidia-smi`
  输出为准。
- 这个目录里的脚本是研究原型工具，不是生产服务入口。
