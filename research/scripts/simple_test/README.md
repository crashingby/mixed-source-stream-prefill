# Prompt Stream 最小测试

当前 P/D 交接由 D 生成全部回答；P-only 使用 `prefill_ready` 控制事件测量 P 侧 TTFT。协议、兼容字段与验证命令见 [P/D 交接说明](../server/PD-Handoff.md)。

三个入口分别测试 Append、Update 和交错来源。现在既可以**直连 P 节点**，也可以
通过 proxy 测试 P-only 或完整 P→D 流程。默认输入是原始文本；客户端只渲染模型
chat template，P 负责分词。

| 文件 | 测试内容 |
| --- | --- |
| `append_frontend.py` | 同一 Append 来源连续追加正文。 |
| `update_frontend.py` | 同一 Update 来源多次完整替换快照。 |
| `mixed_frontend.py` | Append 与 Update 交错到达，最后只保留最新 Update。 |
| `frontend_common.py` | 公共参数、模板、HTTP、来源结束屏障及响应校验。 |
| `prompt_stream_stats.py` | schema v1/v2 统计及 recovery-aware 守恒校验。 |
| `tests/` | 不依赖模型或 GPU 的协议回归测试。 |

## 当前 P-only 服务

如果 P 按 `ENABLE_KV_TRANSFER=0` 在 `g2c10:8100` 启动，可直接运行：

```shell
.venv/bin/python research/scripts/simple_test/mixed_frontend.py \
  --target p --server-url http://g2c10:8100 \
  --model qwen2.5-3b --expected-masked-padding enabled
```

测试另外两个入口：

```shell
.venv/bin/python research/scripts/simple_test/append_frontend.py \
  --target p --server-url http://g2c10:8100 \
  --expected-masked-padding enabled

.venv/bin/python research/scripts/simple_test/update_frontend.py \
  --target p --server-url http://g2c10:8100 \
  --expected-masked-padding enabled
```

直连 P 使用 `/v1/prompt_stream/start`、`/event`、`/finish`，不需要 D 或 proxy。
**P-only 只返回一个输出 token**，用于验证 Prefill 和输入布局；它不会生成完整答案，
脚本会强制 start 的 `max_tokens=1`。`finish_response_ms` 是客户端从发送 finish
到消费完响应的耗时，包含网络和响应解析。

## 请求与检查

一次测试按下面顺序执行：

1. start 发送模型模板前缀，包括 system message 和 user header。
2. 按 CLI 或默认顺序发送 Append 增量、Update 完整快照；每条等待 ACK。
3. 为每个实际使用的来源发送一次 `source_finished`，使用该来源下一条序号。
4. 所有来源结束 ACK 后，finish 发送问题及模板后缀，保留 assistant header。
5. 解析并检查最终统计、masked-padding 清单和可用的实际 prompt token 视图。

请求只携带 `prompt` / `event.text`，没有客户端输入 token IDs。已渲染的 start
设置 `add_special_tokens=false`；实际 P token IDs 从 ACK 读取。当 P 返回物理
`prompt_token_ids` 时，脚本去掉清单指定的 padding，检查它等于 start ACK、全部
Append ACK、最新 Update ACK 和 finish ACK 的串接。该校验不要求服务端逐块分词
与整段分词相同。

最后的 `test summary` 会输出 `status=ok`、完成来源数、mask 区间及
`prompt_tokens_checked`。部分 proxy 返回不包含完整物理视图，此时该字段为 false，
仍会检查统计和可用 mask。异常或 Ctrl+C 会对已创建的会话发送 abort。

## proxy 模式

原有 `--proxy-url` 保留为 `--server-url` 的别名。默认 `--target proxy`，
默认地址仍是 `http://127.0.0.1:8000`。URL 可以带 `/v1`；HTTP 客户端不继承下载
模型时设置的外部代理。

P-only proxy 流程使用专门的 TTFT start/finish 路由，并消费完整的首 token SSE：

```shell
.venv/bin/python research/scripts/simple_test/mixed_frontend.py \
  --target proxy --proxy-url http://<PROXY_HOST>:8000 --ttft-only
```

完整答案测试需要启用 KV transfer 的 P、同模型 D 和普通 proxy：

```shell
.venv/bin/python research/scripts/simple_test/mixed_frontend.py \
  --target proxy --proxy-url http://<PROXY_HOST>:8000 \
  --max-tokens 64 --temperature 0
```

服务启动见 [server README](../server/README.md)。完整 masked-padding PD 模式需
P/D 都使用 Triton、启用 mask，并使用 `GENERATION_CONFIG=vllm`。
稳定 KV 提前 Push 使用自己的服务与 replay，见
[稳定 KV 实验说明](../replay/stable_kv_push/README.md)。

## 自定义数据与参数

Append 的 `--chunk`、Update 的 `--snapshot` 可重复指定；mixed 的 `--event` 格式
为 `append:text` 或 `update:text`，CLI 顺序就是发送顺序。每个 snapshot 都替换
同来源上一代正文。

```shell
.venv/bin/python research/scripts/simple_test/mixed_frontend.py \
  --target p --server-url http://g2c10:8100 \
  --event $'append:Paris is a city in France.\n' \
  --event $'update:The venue is London.\n' \
  --event $'update:The venue is Paris.\n' \
  --final-question $'\nQuestion: What is the latest venue?\nAnswer:'
```

| 参数 | 默认值与作用 |
| --- | --- |
| `--target` | `proxy`；`p` 表示直连 P。 |
| `--server-url` / `--proxy-url` | `http://127.0.0.1:8000`；实际调用的服务地址。 |
| `--ttft-only` | proxy 仅访问 P，不创建 D 请求。 |
| `--model` | `qwen2.5-3b`，填写 served model name。 |
| `--tokenizer-model` | `~/Models/Qwen2.5-3B`，提供与服务一致的模型模板；客户端不分词。 |
| `--prefix` | 该入口默认说明，作为 system message。 |
| `--final-question` | 默认示例问题，追加模板后缀后发送。 |
| `--raw-prompt` | 恢复旧原始 prefix/question 拼接方式，跳过模型模板。 |
| `--max-tokens` | `64`；仅完整 PD 使用，P-only 强制为 1。 |
| `--temperature` / `--seed` | `0` / `2026`；模型采样参数。 |
| `--event-delay` | `0`；每条数据 ACK 后等待的秒数，仅用于手动联调。 |
| `--http-timeout` | `120` 秒；有限超时，避免异常会话一直等待。 |
| `--expected-masked-padding` | `any`；也可用 `enabled` / `disabled` 校验实际开关。 |
| `--request-id` | 可选外部 ID，后续事件始终使用 start 返回的内部 ID。 |

手动联调的 `event-delay` 从上一条 ACK 起算；正式 replay 使用绝对时间线，见
[replay README](../replay/README.md)。最小测试的延迟不能用于代替正式性能实验配置。

## 回归测试

```shell
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv/bin/python -m pytest --noconftest \
  research/scripts/simple_test/tests -q
```

服务端 start 已补齐 `add_special_tokens` 到实际 engine 分词的传递，使实际输入与
长度镜像使用同一设置。这部分 Python 服务代码需要重启 P 后加载；原有默认值仍为 true。
