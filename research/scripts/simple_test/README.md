# Prompt Stream 最小测试脚本

这个目录放的是三个最小前端/RAG mock，用来验证 “前端 -> proxy -> P 节点 -> D 节点”的 RESTFUL 链路。

这些脚本默认发送原始文本，不在前端侧 tokenizer。P 节点的 prompt stream serving 层会把 `text` 转成实际 `token_ids`，并在响应里打印出来，方便排查。

运行前需要先按照 `research/scripts/server/README.md` 启动 P 节点、D 节点和 proxy。

三个脚本都会从 finish 响应顶层解析 `prompt_stream_stats`；字段缺失、类型非法或
统计关系不一致时会直接失败。重点输出包括 `saved_update_prefill_tokens`、
`scheduled_update_prefill_tokens` 和 `update_tokens_invalidated`。

## 参数关系

三个脚本的常用参数基本一致：

| 参数 | 含义 |
| --- | --- |
| `--proxy-url` | proxy base URL，不带 `/v1`，例如 `http://127.0.0.1:8000` |
| `--model` | OpenAI API 中的模型名，应填写 `<SERVED_MODEL_NAME>` |
| `--request-id` | 可选外部请求 id；P 节点仍会返回后续事件使用的内部 id |
| `--prefix` | start 请求里的固定模板/prefix |
| `--final-question` | finish 阶段发送的最终用户问题 |
| `--max-tokens` | 最终 completion 总生成 token 上限 |
| `--temperature` | 最终 decode temperature |
| `--event-delay` | 两个流式事件之间的 sleep 秒数，手动观察日志时有用 |

`--max-tokens` 是用户希望最终 completion 最多生成多少 token。proxy 内部会把发给 P 节点的请求强制改成 `max_tokens=1`，因为 P 节点只负责 prefill 到最终问题并采样首 token；之后 D 节点最多继续生成剩余 token。

## append-only 测试

`append_frontend.py` 默认只发送 append 文本事件：

1. 发送 `/v1/prompt_stream/start`，其中 `prompt` 是固定模板。
2. 连续发送多个 `/v1/prompt_stream/event`，每个事件都是 `event_type=append`。
3. 发送 `/v1/prompt_stream/finish`，其中 `event_type=prompt_finished`。
4. 打印 proxy 返回的最终 completion。

运行默认示例：

```shell
.venv/bin/python research/scripts/simple_test/append_frontend.py \
  --proxy-url http://<PROXY_HOST>:<PROXY_PORT> \
  --model <SERVED_MODEL_NAME> \
  --max-tokens 64 \
  --temperature 0
```

自定义 append chunk：

```shell
.venv/bin/python research/scripts/simple_test/append_frontend.py \
  --proxy-url http://<PROXY_HOST>:<PROXY_PORT> \
  --model <SERVED_MODEL_NAME> \
  --chunk "检索片段 1：vLLM 的 EngineCore 保存请求状态。" \
  --chunk "检索片段 2：proxy 会先把事件发给 P 节点。" \
  --final-question "\n用户问题：proxy 在 decode 前做了什么？\n回答："
```

## update-only 测试

`update_frontend.py` 默认只发送一个 update source 的文本快照。每个 update 事件的 `text` 都表示该 source 的**完整当前快照**，不是增量 patch。

这个脚本用于测试 update 源内 LCP、旧 update 后缀失效，以及 finish 后 update KV 与 prefix/append/question 的组合逻辑。

运行默认示例：

```shell
.venv/bin/python research/scripts/simple_test/update_frontend.py \
  --proxy-url http://<PROXY_HOST>:<PROXY_PORT> \
  --model <SERVED_MODEL_NAME> \
  --max-tokens 64 \
  --temperature 0
```

自定义 update 快照：

```shell
.venv/bin/python research/scripts/simple_test/update_frontend.py \
  --proxy-url http://<PROXY_HOST>:<PROXY_PORT> \
  --model <SERVED_MODEL_NAME> \
  --snapshot "检索快照 v0：proxy 收到流式事件。" \
  --snapshot "检索快照 v1：proxy 收到流式 prefill 事件，然后请求 D 节点 decode。" \
  --final-question "\n用户问题：最新快照里 proxy 在 decode 前做了什么？\n回答："
```

## mixed append/update 测试

`mixed_frontend.py` 默认发送一组交错事件：

```text
append -> update -> append -> update -> append -> finish
```

它用于测试更接近真实 RAG 的事件顺序：稳定上下文持续 append，同时一个可变检索结果以 update 快照形式被刷新。每个 append 事件是新增 chunk；每个 update 事件仍然是同一个 update source 的**完整当前快照**，不是增量 diff。

运行默认示例：

```shell
.venv/bin/python research/scripts/simple_test/mixed_frontend.py \
  --proxy-url http://<PROXY_HOST>:<PROXY_PORT> \
  --model <SERVED_MODEL_NAME> \
  --max-tokens 64 \
  --temperature 0
```

自定义混合事件顺序。`--event` 格式是 `<append|update>:text`，命令行顺序就是发送顺序：

```shell
.venv/bin/python research/scripts/simple_test/mixed_frontend.py \
  --proxy-url http://<PROXY_HOST>:<PROXY_PORT> \
  --model <SERVED_MODEL_NAME> \
  --event "append:稳定片段 1：EngineCore 保存长期请求状态。" \
  --event "append:稳定片段 2：Append chunk 单调加入 prefix_append KV 域。" \
  --event "update:可变快照 v0：proxy 收到流式事件。" \
  --event "update:可变快照 v1：proxy 收到流式事件，并在 decode 前请求 P 节点 prefill。" \
  --final-question "\n用户问题：稳定片段和最新快照说明 proxy 在 decode 前做什么？\n回答："
```

## 当前环境示例

这一节是当前实验环境的可复制命令。对外发布 README 时可以删除。

```shell
.venv/bin/python research/scripts/simple_test/append_frontend.py \
  --proxy-url http://127.0.0.1:8000 \
  --model qwen2.5-3b \
  --max-tokens 64 \
  --temperature 0
```

```shell
.venv/bin/python research/scripts/simple_test/update_frontend.py \
  --proxy-url http://127.0.0.1:8000 \
  --model qwen2.5-3b \
  --max-tokens 64 \
  --temperature 0
```

```shell
.venv/bin/python research/scripts/simple_test/mixed_frontend.py \
  --proxy-url http://127.0.0.1:8000 \
  --model qwen2.5-3b \
  --max-tokens 64 \
  --temperature 0
```
