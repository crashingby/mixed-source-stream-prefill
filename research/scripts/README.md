# Prompt Stream 实验脚本

这个目录放的是流式 prefill 的最小联调脚本。当前目标不是做完整 RAG 系统，而是先把“前端 → proxy → P 节点 → D 节点”的 RESTFUL 链路跑通。

## 脚本说明

### `proxy_server.py`

`proxy_server.py` 是 PD 分离场景下的实验代理服务器。RAG 前端只访问这个 proxy，proxy 负责：

- 接收 `/v1/prompt_stream/start`，为请求选择一个 P 节点和一个 D 节点。
- 把 start 请求转发给 P 节点的 `/v1/prompt_stream/start`。
- 保存 P 节点返回的内部 `request_id`，后续 event/finish 都使用这个 id。
- 接收 `/v1/prompt_stream/event`，把 append/update/source_finished 事件转发给同一个 P 节点。
- 接收 `/v1/prompt_stream/finish`，先让 P 节点完成最后用户问题的 prefill，再把返回的 `kv_transfer_params` 放进普通 OpenAI decode 请求，转发给 D 节点。

### `append_frontend.py`

`append_frontend.py` 是一个非常小的前端/RAG mock。它默认只发送 append 文本事件，流程是：

1. 向 proxy 发送 `/v1/prompt_stream/start`，其中 `prompt` 是固定模板/prefix。
2. 连续向 proxy 发送多个 `/v1/prompt_stream/event`，每个事件都是 `event_type=append`。
3. 向 proxy 发送 `/v1/prompt_stream/finish`，其中事件是 `event_type=prompt_finished`，文本是最后的用户问题。
4. 打印 proxy 返回的最终 decode 响应。

这个脚本故意不做 tokenizer。它发送原始文本，由 P 节点的 prompt stream RESTFUL serving 层负责把 `text` 转成 `token_ids`。

## 启动顺序

下面端口只是示例。真实运行时，P/D 节点还需要按你的 KV connector、NIXL、PD 分离配置启动。

### 1. 启动 P 节点

P 节点需要使用已经注册 prompt stream router 的 OpenAI server，并监听例如 `8100` 端口。示例形式如下：

```bash
# 这里只展示形状；KV connector 参数需要按实际机器补齐。
.venv/bin/python -m vllm.entrypoints.openai.api_server \
  --host 127.0.0.1 \
  --port 8100 \
  --model ~/Models/Qwen2.5-3B/ \
  --served-model-name qwen2.5-3b
```

### 2. 启动 D 节点

D 节点继续暴露普通 OpenAI decode 接口，例如 `/v1/completions` 或 `/v1/chat/completions`，并监听例如 `8200` 端口。

```bash
# 这里只展示形状；KV connector 参数需要按实际机器补齐。
.venv/bin/python -m vllm.entrypoints.openai.api_server \
  --host 127.0.0.1 \
  --port 8200 \
  --model ~/Models/Qwen2.5-3B/ \
  --served-model-name qwen2.5-3b
```

### 3. 启动 proxy

```bash
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

如果传入多个 P/D 节点，proxy 会用 round-robin 选择节点。

注意：`append_frontend.py` 的 `--model` 参数是 OpenAI API 请求体里的
`model` 字段，应该填写 P/D 节点启动时的 `--served-model-name`，例如
`qwen2.5-3b`，而不是本地模型路径。`--model ~/Models/...` 只应该出现
在启动 vLLM server 的命令里。

### 4. 运行 append-only 前端示例

```bash
.venv/bin/python research/scripts/append_frontend.py \
  --proxy-url http://127.0.0.1:8000 \
  --model qwen2.5-3b \
  --max-tokens 64 \
  --temperature 0
```

如果想观察流式 decode，可以加：

```bash
.venv/bin/python research/scripts/append_frontend.py \
  --proxy-url http://127.0.0.1:8000 \
  --model qwen2.5-3b \
  --stream
```

也可以自己指定 append chunk：

```bash
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
  "status": "started"
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
- proxy 现在假设 P 节点能在 finish 后返回 `kv_transfer_params`，并且 D 节点能用普通 OpenAI 接口消费这些参数。
- 这个目录里的脚本是研究原型工具，不是生产服务入口。
