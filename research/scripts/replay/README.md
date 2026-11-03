# 科研数据回放

所有入口的请求级 QPS、闭环逻辑并发和客户端等待统计统一使用 [负载说明](LOAD.md)。正 QPS 默认独立发送，`max_concurrency` 只控制闭环；开环保护上限使用 `max_inflight`。

当前 P/D 交接由 D 生成全部回答；P-only 使用 `prefill_ready` 控制事件测量 P 侧 TTFT。协议、兼容字段与验证命令见 [P/D 交接说明](../server/PD-Handoff.md)。

本目录把离线证据转换为 REST 请求，比较普通整段 Prefill、分段流式 Prefill、
Update 合并、masked padding 和稳定 KV 提前传输。各实验保留独立入口与指标，
共同使用 `common/` 里的数据读取、模板渲染、请求时间线和 HTTP/SSE 实现。

## 目录与入口

| 目录 | 入口与职责 |
| --- | --- |
| `common/` | 数据校验、文本构造、时序、HTTP、结果持久化及统计校验；不直接运行。 |
| `configs/` | TriviaQA、HotpotQA 的独立数据与发送时间配置。 |
| `segmented_prefill/` | `replay_normal.py`、`replay_prompt_stream.py`、`replay_merged_update.py`，比较三种 Prefill 路径。 |
| `update_coalescing/` | `replay_update_coalescing.py`，复用 stream 发送，校验 P 实际合并配置与统计。 |
| `masked_padding/` | `run_masked_padding_sweep.sbatch`，复用 stream，比较带 mask 的完整 P→D 生成。 |
| `generation_quality/` | `normal/segmented/sparse` 保存完整答案，`evaluate_answers.py` 按 TriviaQA/HotpotQA 协议评分并比较分段与稀疏退化。 |
| `token_invalidation/` | 三个回放入口比较 Update-first、Append-first 整体建模与分段管理的实际失效 token，使用 P-only TTFT 路由。 |
| `query_probe/` | `replay.py`，经 P-only TTFT 路由验证联合 Append/Query prefill、块评分、固定比例选择及临时 KV 回收。 |
| `mixed_workload/` | 普通完整 prefill 与流式请求共用到达时间线，比较纯 Update 与分段 + 稀疏，并分别统计普通/流式 TTFT。 |
| `stable_kv_push/` | `replay_stable_kv_push.py`，分别观察 P 完成通知、D 首 token 与实际 KV 传输指标。 |
| `tests/` | CPU 回归测试，使用临时数据和模拟 HTTP 响应。 |

`segmented_prefill/replay_common.py` 和 `prompt_stream_stats_v2.py` 保留旧导入路径，
实际实现位于公共包。已有汇总、绘图入口和历史结果仍可读取；各实验使用独立结果目录。

Query probe 的启动脚本、数据配置与功能校验见 [query_probe/README.md](query_probe/README.md)。

普通请求与流式请求共存的混合负载、分配比例及配对报告见 [mixed_workload/README.md](mixed_workload/README.md)。

## Token 失效实验

新实验说明见 [token_invalidation/README.md](token_invalidation/README.md)。
三个入口共用原始事件计划，每次到达都发送，不合并同一时刻的事件。
主指标从服务端 schema v2 读取，并将抢占丢弃及 ACK 的逻辑后缀诊断单列。
离线工具只在三边共同有效的请求上比较失效总量。
`plot_token_invalidation.py` 读取配对结果，输出总失效数、均值/P95 和分布图；
回放与配对汇总默认保存到 `token_invalidation/results/`，图表保存在配对结果的 `plots/`。
`run_token_invalidation.sbatch` 用同一套数据和时间参数依次执行三种布局，
自动生成配对汇总与图表。

## 数据入口

新数据由三部分组成，**按 `sample_id` 对齐**，文件名无需是数字：

| 内容 | TriviaQA 默认路径 |
| --- | --- |
| 问题与评价答案 | `research/data/questions/triviaqa/web_unfiltered_dev_v1.jsonl` |
| Append chunks | `research/data/append_chunks/triviaqa/web_unfiltered_dev_v1/` |
| Update PipelinePool | `research/data/pipeline_pools/triviaqa/wiki_dev_k5_ef8_v1/` |

HotpotQA 使用 `hotpotqa`、`fullwiki_dev_context_v1`，Pool 默认仍为
`wiki_dev_k5_ef8_v1`。`--dataset triviaqa` 或 `--dataset hotpotqa` 可使用上述默认值；
也可传 `--questions-file`、`--append-dir`、`--pipeline-pools-dir` 指向其它版本。
配置示例已写入这三个路径。未指定独立数据时，`--data-dir` 继续读取旧
`requests/*.json`，保持 mixed-source v2 实验兼容。

读取器先验证问题清单 SHA256 与两份生成元数据，再通过各自 `manifest.jsonl`
找到样本，检查 ID、schema、状态、正文哈希及快照引用。选择顺序取自问题 JSONL；
旧 mixed-source 仍按 query ID 排序。不会为了回放少量请求而读取全部样本正文。

- Append `ready`：每个 chunk 原样成为一次 Append；`empty`：仍回放该问题，
  该来源只发送 `source_finished`。失败的 Append 或尚未就绪的 Pool 跳过。
- Pool 中的 `chunks` 是候选目录。每个 `snapshot.chunk_ids` 按顺序重建**完整替换文本**，
  形成一次 Update；最后必须有 `final`。普通模式只取最终快照。
- 评价答案只参与评分；证据选择、模板与发送正文不读取答案或 supporting facts。
- `--start-index` 跳过前 N 个可回放样本；`--max-requests 0` 选择全部。
  显式重复 `--sample-id` 或 `--query-id` 时回放所有指定 ID，不受默认数量限制。
  缺失索引、错版本、损坏正文和指定但未就绪的 ID 会报错。

## 原始文本与模型模板

**默认 `--input-transport text`**。`--tokenizer-model` 提供模型原生 chat template，
应与 P/D 加载的模型一致；默认 `~/Models/Qwen2.5-3B`。本地模型读取模板配置即可，
默认输入路径不会调用客户端 `encode()`。

| 请求阶段 | 默认发送的内容 |
| --- | --- |
| normal | `prompt` 字符串：模板前缀 + 全部 Append + 最终 Update + 问题 + 模板后缀。 |
| stream start | `prompt` 字符串：BOS、system message、user header 等固定前缀。 |
| Append event | 本块 `event.text`，只追加本来源正文。 |
| Update event | 本代完整 `event.text`，替换本来源上一代快照。 |
| stream finish | 问题与 `Answer:`，加 user 结束标记、assistant generation header。 |

已渲染的 normal/start 输入使用 `add_special_tokens=false`，由 P 直接分词，避免重复
BOS。启动 P 时需使用包含该 start 字段支持的当前代码。proxy 仍把 P 实际使用的
prompt token IDs 交给 D；这是内部 KV 对齐所需的正常传输。

normal 的整段分词与 stream 的逐块分词可能在边界处不同；本实验保留这种行为。
质量对照校验最终**文本**一致。`--input-transport token_ids` 保留客户端预分词的历史
实验模式，便于复现原先的输入方式；结果记录传输模式，跨模式的历史数值应分别比较。
`--return-token-ids` 控制返回输出 token IDs，与输入传输模式独立。

## 配置与发送时间

`--config` 读取扁平 JSON，字段对应 CLI 参数，将连字符写成下划线即可。
重复 ID 选项在 JSON 中写成 `sample_ids` / `query_ids` 字符串数组。
配置内文件路径相对 JSON 文件解析，`~` 展开为用户目录；显式 CLI 优先。
未知字段和错误类型直接报错。同一份配置可用于 normal、stream、merged、质量及
稳定 KV 入口；各入口的专用选项由 CLI 添加。normal 接受来源时间设置但不发送事件。

数据文件不保存实验到达时间。Pool 的 `search_elapsed_ms` 保留检索诊断，replay
不会自动据此安排发送。

| 参数 | 发送语义 |
| --- | --- |
| `qps` | 大于 0 时按 fixed/poisson 生成绝对请求到达时间；0 时按 `max_concurrency` 维持闭环请求数。 |
| `max_concurrency` | 闭环在途逻辑请求上限；正 QPS 不使用此值。 |
| `max_inflight` | 开环客户端保护上限；0 不限，到期后等槽位的时间单独记录。 |
| `append_initial_delay` / `update_initial_delay` | 从 start ACK 到该来源首条数据的目标延迟。 |
| `append_chunk_interval` / `update_chunk_interval` | 第 i 条数据目标为 `initial_delay + i * interval`。同源每条均等待 ACK；ACK 慢会导致迟发，不额外累加 interval。 |
| `source_finish_delay` | 最后一条数据 ACK 后到该来源 `source_finished` 的延迟；空来源从 initial delay 起算。 |
| `random_seed` | 请求到达过程随机种子，与模型采样 `seed` 分开。 |

Append 与 Update 来源并发发送。所有来源结束 ACK 后才发送全局 finish。
merged baseline 保留“同一计划时刻先更新本地 store，再发送合并快照”的语义。

## 用 TriviaQA 开始

先检查 16 个样本，无需启动服务：

```shell
.venv/bin/python research/scripts/replay/segmented_prefill/replay_prompt_stream.py \
  --config research/scripts/replay/configs/triviaqa.json \
  --dry-run --output-dir /tmp/replay-check --run-name triviaqa-check
```

输出的 `prompt_preview.jsonl` 包含每个请求的最终文本以及 start/finish 边界，
便于直接核对模型会看到的内容。dry-run 校验数据和模板，**不检查服务端 token 上限**。
正式运行前应给最大片段中间快照、全部 Append、模板、问题和输出留出上下文空间；
TriviaQA 的 Append 预算只覆盖 Append 本身。

启动当前 P/D 与对应 proxy 后，比较相同样本的 P-only TTFT：

```shell
.venv/bin/python research/scripts/replay/segmented_prefill/replay_normal.py \
  --config research/scripts/replay/configs/triviaqa.json \
  --proxy-url http://<PROXY_HOST>:8000 --ttft-only --run-name triviaqa-normal

.venv/bin/python research/scripts/replay/segmented_prefill/replay_prompt_stream.py \
  --config research/scripts/replay/configs/triviaqa.json \
  --proxy-url http://<PROXY_HOST>:8000 --ttft-only --run-name triviaqa-stream
```

merged 对照改用 `replay_merged_update.py`。完整生成去掉 `--ttft-only`。
HotpotQA 在对应数据生成完成后换用 `configs/hotpotqa.json`。
质量评分见 [generation_quality/README.md](generation_quality/README.md)，稳定 KV
实验需使用自己的专用 proxy，见 [stable_kv_push/README.md](stable_kv_push/README.md)。

批量入口 `run_ttft_qps_sweep.sbatch`、`run_policy_sweep.sbatch`、
`run_masked_padding_sweep.sbatch`、`run_generation_quality.sbatch` 和
`run_stable_kv_sweep.sbatch` 支持 `REPLAY_CONFIG` 与 `INPUT_TRANSPORT` 环境变量。
例如 `REPLAY_CONFIG=research/scripts/replay/configs/triviaqa.json`；脚本显式传入的
负载、时间、模型与输出参数仍优先于 JSON 默认值。稳定 KV 的受控长度 matrix
继续读取其专门生成的数据。

## 结果与验证

- 联网前写 `config.json`，保存解析后的数据路径、选中 ID 和内容指纹。
- normal/stream/merged 完成每个请求即刷写 `request_metrics.jsonl`；结束时写现有
  `request_metrics.csv`、`event_metrics.csv` 和 `summary.json`，失败或取消时也保存
  已完成请求。质量与稳定 KV 分别即时刷写 `generations.jsonl` 和 `requests.jsonl`。
- 完整 SSE 必须收到 `[DONE]`；收到 token 后断流仍记为失败。
  TTFT-only 在 P 完成通知到达后主动结束消费。
- 保留各实验统计校验：coalescing 的真实 P 配置与 recovery-aware 守恒、masked
  padding manifest，以及稳定 KV 的 P/D token 来源和传输字节守恒。

本地 CPU 回归命令：

```shell
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv/bin/python -m pytest --noconftest \
  research/scripts/replay/tests \
  tests/v1/test_prompt_stream_replay_metrics.py \
  tests/v1/test_stable_kv_replay.py -q
```
