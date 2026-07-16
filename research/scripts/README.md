# Prompt Stream 实验脚本

这个目录放的是流式 prefill 的最小联调脚本。当前目标不是做完整 RAG 系统，而是先把“前端 → proxy → P 节点 → D 节点”的 RESTFUL 链路跑通。

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
- 把 P 节点采样出的首 token 接到 D prompt，并在 D 返回后合并回最终响应。

### `append_frontend.py`

`append_frontend.py` 是一个非常小的前端/RAG mock。它默认只发送 append 文本事件，流程是：

1. 向 proxy 发送 `/v1/prompt_stream/start`，其中 `prompt` 是固定模板/prefix。
2. 连续向 proxy 发送多个 `/v1/prompt_stream/event`，每个事件都是 `event_type=append`。
3. 向 proxy 发送 `/v1/prompt_stream/finish`，其中事件是 `event_type=prompt_finished`，文本是最后的用户问题。
4. 打印 proxy 返回的最终 decode 响应。

这个脚本故意不做 tokenizer。它发送原始文本，由 P 节点的 prompt stream RESTFUL serving 层负责把 `text` 转成 `token_ids`。

## 启动顺序

下面命令按单机双卡示例编写：

- P 节点使用 `CUDA_VISIBLE_DEVICES=0`，监听 `8100`，NIXL side-channel 端口为 `5600`。
- D 节点使用 `CUDA_VISIBLE_DEVICES=1`，监听 `8200`，NIXL side-channel 端口为 `5601`。
- proxy 监听 `8000`。

如果机器显存比较紧张，优先调低 `--gpu-memory-utilization`。例如 24 GiB 卡上
`0.85` 可能要求约 20 GiB 空闲显存；如果启动时报 free memory 不足，可以先改成
`0.65` 或 `0.6` 做联调。sbatch 脚本还默认设置 `MAX_NUM_SEQS=16`，
用于减少 vLLM warmup sampler 和调度状态占用；如果仍然 OOM，可以继续降到 `8`。

### 1. 启动 P 节点

P 节点是 NIXL producer，负责流式 prefill，并在 finish 后把可被 D 拉取的
KV block 信息放进 `kv_transfer_params`。

```shell
CUDA_VISIBLE_DEVICES=0 \
UCX_NET_DEVICES=all \
VLLM_NIXL_SIDE_CHANNEL_PORT=5600 \
python -m vllm.entrypoints.openai.api_server \
  --model ~/Models/Qwen2.5-3B/ \
  --served-model-name qwen2.5-3b \
  --trust-remote-code \
  --dtype auto \
  --gpu-memory-utilization 0.65 \
  --max-model-len 4096 \
  --host 0.0.0.0 \
  --port 8100 \
  --enforce-eager \
  --no-async-scheduling \
  --kv-transfer-config '{"kv_connector":"NixlConnector","kv_role":"kv_producer","kv_load_failure_policy":"fail"}'
```

注意点：

- `--no-async-scheduling` 当前必须显式加上。prompt stream MVP 暂时拒绝异步调度，
  避免事件修改与在途 batch 交错。
- `VLLM_NIXL_SIDE_CHANNEL_PORT` 需要和 D 节点不同，避免端口冲突。
- `--served-model-name qwen2.5-3b` 是后续 OpenAI 请求体里的 `model` 字段。

### 2. 启动 D 节点

D 节点是 NIXL consumer，继续暴露普通 OpenAI `/v1/completions` 接口。proxy
会把 P 返回的物理 prompt token ids、P 首 token 和 `kv_transfer_params` 一起转发给它。

```shell
CUDA_VISIBLE_DEVICES=1 \
UCX_NET_DEVICES=all \
VLLM_NIXL_SIDE_CHANNEL_PORT=5601 \
python -m vllm.entrypoints.openai.api_server \
  --model ~/Models/Qwen2.5-3B/ \
  --served-model-name qwen2.5-3b \
  --trust-remote-code \
  --dtype auto \
  --gpu-memory-utilization 0.65 \
  --max-model-len 4096 \
  --host 0.0.0.0 \
  --port 8200 \
  --enforce-eager \
  --no-async-scheduling \
  --kv-transfer-config '{"kv_connector":"NixlConnector","kv_role":"kv_consumer","kv_load_failure_policy":"fail"}'
```

注意点：

- P/D 的 `--model` 和 `--served-model-name` 应保持一致。
- D 节点必须能访问 P 节点通过 NIXL side-channel 暴露的远端 blocks。
- 如果只想验证 prompt/token 构造，不想启用 KV 拉取，可以保留 D 节点启动参数，
  但在 proxy 启动时加 `--disable-kv-transfer`。

### 用 `sbatch` 启动 P/D 节点

如果登录节点上的 GPU 被占用，可以用 Slurm 把 P/D 节点提交到空闲计算节点。
本目录已经提供两个脚本：

- `start_p_node.sbatch`：启动 P 节点，即 NIXL producer，默认 OpenAI 端口 `8100`，
  NIXL side-channel 端口 `5600`。
- `start_d_node.sbatch`：启动 D 节点，即 NIXL consumer，默认 OpenAI 端口 `8200`，
  NIXL side-channel 端口 `5601`。

提交前先创建日志目录：

```shell
mkdir -p research/scripts/logs
```

然后提交两个 job：

```shell
sbatch research/scripts/start_p_node.sbatch
sbatch research/scripts/start_d_node.sbatch
```

如果你的集群分区名不是 `gpu`，需要先修改两个 sbatch 文件顶部的：

```shell
#SBATCH --partition=gpu
```

也可以按集群要求调整 `--gres=gpu:1`、`--cpus-per-task`、`--mem` 和 `--time`。

脚本默认把 `sbatch` 的提交目录 `SLURM_SUBMIT_DIR` 当作仓库根目录。因此建议在
`/workplace/home/huangxinying/Projects/vllm` 下提交；如果从别的目录提交，需要显式传
`REPO_ROOT=/workplace/home/huangxinying/Projects/vllm`。

脚本支持用环境变量临时覆盖常用参数，不需要每次编辑文件。例如：

```shell
MODEL_PATH=/workplace/home/huangxinying/Models/Qwen2.5-3B/ \
SERVED_MODEL_NAME=qwen2.5-3b \
GPU_MEMORY_UTILIZATION=0.6 \
MAX_NUM_SEQS=16 \
sbatch research/scripts/start_p_node.sbatch
```

D 节点同理：

```shell
MODEL_PATH=/workplace/home/huangxinying/Models/Qwen2.5-3B/ \
SERVED_MODEL_NAME=qwen2.5-3b \
GPU_MEMORY_UTILIZATION=0.6 \
MAX_NUM_SEQS=16 \
sbatch research/scripts/start_d_node.sbatch
```

如果需要改端口，也可以覆盖：

```shell
PORT=8110 NIXL_SIDE_CHANNEL_PORT=5610 sbatch research/scripts/start_p_node.sbatch
PORT=8210 NIXL_SIDE_CHANNEL_PORT=5611 sbatch research/scripts/start_d_node.sbatch
```

Slurm 只知道 GPU 资源是否被 Slurm job 分配，不一定知道显存是否真的空。如果某个节点上有
手动启动的进程或残留进程，`sinfo` 仍可能显示节点 `idle`，但 vLLM 会因为 free memory
不足启动失败。遇到这种情况，可以显式排除问题节点：

```shell
sbatch --exclude=g2c15 research/scripts/start_d_node.sbatch
```

也可以直接指定一个你想测试的节点：

```shell
sbatch --nodelist=g2c12 research/scripts/start_p_node.sbatch
sbatch --nodelist=g2c13 research/scripts/start_d_node.sbatch
```

如果想先探测某个节点上 Slurm 分给你的 GPU 显存，可以跑一个短命令：

```shell
srun -p gpu --nodelist=g2c12 --gres=gpu:1 nvidia-smi
```

脚本自身也会在启动 vLLM 前打印 `nvidia-smi`，日志里可以直接看到被分配 GPU 的
`memory.total`、`memory.used` 和 `memory.free`。

提交后查看 job：

```shell
squeue -u "$USER"
scontrol show job <JOB_ID>
```

最重要的是看日志，因为脚本会在启动 vLLM 前打印实际计算节点信息：

```shell
cat research/scripts/logs/vllm-p-<JOB_ID>.out
cat research/scripts/logs/vllm-d-<JOB_ID>.out
```

日志开头会包含类似信息：

```text
SLURM_JOB_ID=123456
SLURM_JOB_NODELIST=gpu-node-12
HOSTNAME=gpu-node-12
HOST_IP=10.4.1.115
ROLE=prefill-producer
OPENAI_PORT=8100
NIXL_SIDE_CHANNEL_PORT=5600
```

这里的 `HOSTNAME` 或 `HOST_IP` 就是 proxy 需要连接的地址。假设 P 节点日志里是
`gpu-node-12`，D 节点日志里是 `gpu-node-13`，proxy 应该这样启动：

```shell
.venv/bin/python research/scripts/proxy_server.py \
  --host 0.0.0.0 \
  --port 8000 \
  --prefiller-hosts gpu-node-12 \
  --prefiller-ports 8100 \
  --decoder-hosts gpu-node-13 \
  --decoder-ports 8200
```

如果集群 DNS 不能解析节点名，就改用日志里的 `HOST_IP`：

```shell
.venv/bin/python research/scripts/proxy_server.py \
  --host 0.0.0.0 \
  --port 8000 \
  --prefiller-hosts 10.4.1.115 \
  --prefiller-ports 8100 \
  --decoder-hosts 10.4.1.116 \
  --decoder-ports 8200
```

注意：`127.0.0.1` 只表示当前机器自己。P、D、proxy 如果不在同一个计算节点上，
proxy 的 `--prefiller-hosts` 和 `--decoder-hosts` 不能写 `127.0.0.1`。

### 3. 启动 proxy

```shell
.venv/bin/python research/scripts/proxy_server.py \
  --host 127.0.0.1 \
  --port 8000 \
  --prefiller-hosts 127.0.0.1 \
  --prefiller-ports 8100 \
  --decoder-hosts 127.0.0.1 \
  --decoder-ports 8200
```

常用参数：

- `--host` / `--port`：proxy 自己监听的地址。
- `--prefiller-hosts` / `--prefiller-ports`：一个或多个 P 节点地址，数量必须一致。
- `--decoder-hosts` / `--decoder-ports`：一个或多个 D 节点地址，数量必须一致。
- `--disable-kv-transfer`：调试开关。proxy 仍然会调用 P 节点完成流式 prefill，但不会把 `kv_transfer_params` 转发给 D 节点，D 节点会用 proxy 拼出的 prompt token ids 自己完整 prefill。

如果传入多个 P/D 节点，proxy 会用 round-robin 选择节点。

#### 禁用 KV transfer 的对照实验

当 D 节点输出异常时，可以先用 `--disable-kv-transfer` 做 baseline：

```shell
.venv/bin/python research/scripts/proxy_server.py \
  --host 127.0.0.1 \
  --port 8000 \
  --prefiller-hosts 127.0.0.1 \
  --prefiller-ports 8100 \
  --decoder-hosts 127.0.0.1 \
  --decoder-ports 8200 \
  --disable-kv-transfer
```

这个模式用于区分两类问题：

- 如果禁用 KV transfer 后输出正常，说明 prompt/token ids 本身没问题，异常更可能来自 P→D KV 内容、block 顺序或 block 对齐。
- 如果禁用 KV transfer 后输出仍然异常，说明问题可能在 prompt 构造、token ids 或普通 completions 请求本身。

正常验证 P→D KV 传输时不要加这个参数。

注意：`append_frontend.py` 的 `--model` 参数是 OpenAI API 请求体里的
`model` 字段，应该填写 P/D 节点启动时的 `--served-model-name`，例如
`qwen2.5-3b`，而不是本地模型路径。`--model ~/Models/...` 只应该出现
在启动 vLLM server 的命令里。

### 4. 运行 append-only 前端示例

```shell
.venv/bin/python research/scripts/append_frontend.py \
  --proxy-url http://127.0.0.1:8000 \
  --model qwen2.5-3b \
  --max-tokens 64 \
  --temperature 0
```

也可以自己指定 append chunk：

```shell
.venv/bin/python research/scripts/append_frontend.py \
  --proxy-url http://127.0.0.1:8000 \
  --model qwen2.5-3b \
  --chunk "检索片段 1：vLLM 的 EngineCore 保存请求状态。" \
  --chunk "检索片段 2：proxy 会先把事件发给 P 节点。" \
  --final-question "\n用户问题：proxy 在 decode 前做了什么？\n回答："
```

## RESTFUL 请求形状

`append_frontend.py` 对 proxy 发送的核心请求如下。

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

- `append_frontend.py` 只覆盖 append-only 文本模式。
- update/source_finished/token_ids 双模式之后可以继续补测试脚本。
- proxy 当前只接受 `api=completions` 和 `stream=false`。chat renderer 的物理
  token 镜像与流式响应中的 P 首 token 合并尚未实现，因此这两种请求会在
  start 阶段返回 400。
- `prompt_finished` 必须携带至少一个最终问题 token。
- P 节点当前要求同步调度、pipeline parallel size 1、无 speculative decoding、
  NIXL direct-device transfer，并且 KV groups 必须是 block size
  一致的 full attention / MLA；host-buffer 模式暂不支持。
- proxy 强制要求 P 节点在 finish 后返回物理 `prompt_token_ids`、
  `kv_transfer_params` 和合法的 `remote_num_tokens`；缺失或长度不一致时会
  fail closed 返回 502，不会继续请求 D。
- 启用 KV transfer 时，proxy 会检查
  `physical_prompt_len <= remote_num_tokens <= D_prompt_len`。如果不满足会返回
  502，不会让 D 静默加载错位 KV。
- 这个目录里的脚本是研究原型工具，不是生产服务入口。
