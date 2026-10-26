# 分段与块稀疏的生成质量实验

本目录保存完整 P→D 生成答案，比较普通完整 prefill、分段 dense，以及 Query probe 块稀疏。使用当前 `research/data/questions`、`append_chunks`、`pipeline_pools`，按 `sample_id` 对齐。默认数据集为 TriviaQA，也支持 HotpotQA。

## 模式与主要对照

| `mode` 配置 / CLI | 行为 | 作用 |
| --- | --- | --- |
| `normal` | 将最终完整逻辑 Prompt 一次性提交 `/v1/completions` | 普通完整注意力基线 |
| `segmented` | 按来源发送 Append、Update；不注册 probe | 分段近似基线 |
| `sparse` | 同样分段发送，start 注册最终 Query，每个新 Append 执行 probe | 分段 + 块稀疏 |

三组使用相同模型、chat template、最终逻辑内容和生成参数。`segmented` 与 `sparse` 还使用相同的来源事件及时间配置。三组的 P 均固定为 `CHUNK_SPARSE_ATTN`，dense/sparse 使用同一个算子；D 均使用 `TRITON_ATTN` 的 `unified_attention`。最终 Query 和 D 的 decode 均完整读取历史 KV，D 只屏蔽 padding，不启用块稀疏。

**稀疏本身的主要质量对照是 `sparse_vs_segmented`。** `segmented_vs_normal` 衡量分段近似的影响，`sparse_vs_normal` 衡量两种近似的总体影响。

## 数据集与共享配置

| 配置 | 问题及 Append 版本 | PipelinePool 版本 |
| --- | --- | --- |
| [configs/triviaqa.json](configs/triviaqa.json) | `triviaqa/web_unfiltered_dev_v1` | `triviaqa/wiki_dev_k5_ef8_v1` |
| [configs/hotpotqa.json](configs/hotpotqa.json) | `hotpotqa/fullwiki_dev_context_v1` | `hotpotqa/wiki_dev_k5_ef8_v1` |

配置中的数据路径相对 JSON 所在目录解析。默认每组 100 条、串行回放、`qps=0`、每次请求后空闲 0.2 秒；Append 间隔 50 ms、Update 间隔 20 ms。默认最多生成 64 token，`temperature=0`、相同 seed，并用换行 stop 约束短答案。出现 `finish_reason=length` 时会单独统计，便于发现答案被截断。

`mode` 和 `probe_keep_ratio` 都可以写入 JSON，也可以用 CLI 覆盖：

```shell
.venv/bin/python research/scripts/replay/generation_quality/replay_generation_quality.py \
  --config research/scripts/replay/generation_quality/configs/triviaqa.json \
  --mode sparse --probe-keep-ratio 0.5 \
  --proxy-url http://<PROXY_HOST>:8000 --run-name triviaqa-sparse
```

`probe_keep_ratio` 范围是 `[0, 1]`，默认 0.5。保护块不参与该比例；设为 1 可以运行保留全部候选块的 probe 对照。`normal` / `segmented` 不执行 probe。

不提供数据输入时默认读取当前 TriviaQA 独立数据。旧 SQuAD 请求仍可显式使用 `--data-dir` 和 `--gold-manifest`，但不再是默认实验输入。

## P、D 与 proxy

本实验需要完整生成，因此要启动 **P 和 D**。P 的自定义后端当前要求单卡 TP=1、eager、同步调度、16-token KV block、auto KV dtype 和普通因果 full attention。D 使用原 Triton 后端；两端必须使用同一模型、KV dtype、block size 及 masked-padding 语义，P 关闭 Update 合并。

从仓库根目录分别提交，替换节点和模型路径：

```shell
mkdir -p research/scripts/logs research/scripts/replay/generation_quality/results

MODEL_PATH="$HOME/Models/Qwen2.5-3B" \
SERVED_MODEL_NAME=qwen2.5-3b \
GPU_MEMORY_UTILIZATION=0.8 \
ATTENTION_BACKEND=CHUNK_SPARSE_ATTN \
PROMPT_STREAM_MASKED_PADDING=1 \
PROMPT_STREAM_UPDATE_COALESCING_CONFIG='' \
ENABLE_KV_TRANSFER=1 MAX_MODEL_LEN=16384 \
sbatch --nodelist=<P_NODE> research/scripts/server/start_p_node.sbatch

MODEL_PATH="$HOME/Models/Qwen2.5-3B" \
SERVED_MODEL_NAME=qwen2.5-3b \
GPU_MEMORY_UTILIZATION=0.8 \
ATTENTION_BACKEND=TRITON_ATTN \
PROMPT_STREAM_MASKED_PADDING=1 MAX_MODEL_LEN=16384 \
sbatch --nodelist=<D_NODE> research/scripts/server/start_d_node.sbatch
```

确认两端模型加载完成后，启动 proxy：

```shell
.venv/bin/python research/scripts/server/proxy_server.py \
  --host 0.0.0.0 --port 8000 \
  --prefiller-hosts <P_NODE> --prefiller-ports 8100 \
  --decoder-hosts <D_NODE> --decoder-ports 8200
```

回放机通过 `http://<PROXY_HOST>:8000` 访问。`localhost` 只在回放与 proxy 位于同一台机器时适用。完整质量实验保留 KV transfer，不使用 `--disable-kv-transfer`，也不使用 `start_ttft` / `finish_ttft`。

## 批量运行与配置覆盖

Slurm 脚本默认依次运行 `normal segmented sparse`，然后自动评分。P/D/proxy 由上一步启动，批量脚本只运行客户端和离线评分，不申请 GPU。

```shell
DATASET=triviaqa PROXY_URL=http://<PROXY_HOST>:8000 \
sbatch research/scripts/replay/generation_quality/run_generation_quality.sbatch

DATASET=hotpotqa PROXY_URL=http://<PROXY_HOST>:8000 \
sbatch research/scripts/replay/generation_quality/run_generation_quality.sbatch
```

只对比稀疏引入的变化：

```shell
DATASET=triviaqa MODE_ORDER='segmented sparse' \
PROXY_URL=http://<PROXY_HOST>:8000 MAX_REQUESTS=100 \
PROBE_KEEP_RATIO=0.5 RUN_PREFIX=triviaqa-sparse-quality \
sbatch research/scripts/replay/generation_quality/run_generation_quality.sbatch
```

`REPLAY_CONFIG` 可指定自定义 JSON，`MODE_ORDER` 可调换执行顺序，且不得包含重复模式。默认 `QPS=0 MAX_CONCURRENCY=1` 串行；设为 `MAX_CONCURRENCY=4` 时按闭环维持最多四个在途请求，完成一个再补一个，各槽位遵守 `INTER_REQUEST_DELAY`。显式配置正 QPS 时改为按到达时钟发送，并受同一并发上限约束。

共享参数优先来自 JSON；只有显式设置的环境变量才追加为 CLI 覆盖，例如 `MODEL`、`TOKENIZER_MODEL`、`MAX_REQUESTS`、`MAX_TOKENS`、`SEED`、`PROBE_KEEP_RATIO`、事件间隔等。更换模型时同时设置 `MODEL` 和 `TOKENIZER_MODEL`，并与 P/D 保持一致。输出位置由 `OUTPUT_DIR` 控制，默认是本目录 `results/`。Slurm 总日志也写到该目录；提交前需要创建它。

### 同节点两张卡的一键完整测试

`run_full_quality.sbatch` 在一个 Slurm 作业里申请两张 GPU，分别启动 P/D，
再启动本地 proxy、三模式预热、正式回放和评分。默认使用 g2c13、
`~/Models/Llama-3.1-8B-Instruct`、TriviaQA 每组 1000 条、闭环并发 4，
显存比例 0.8、上下文上限 16384、调度 token 预算 1024。P 使用
`CHUNK_SPARSE_ATTN`，D 使用 `TRITON_ATTN`；均开启 masked padding，关闭 Update 合并。

```shell
mkdir -p research/scripts/replay/generation_quality/results
sbatch --time=7-00:00:00 \
  research/scripts/replay/generation_quality/run_full_quality.sbatch
```

该作业由 Slurm 后台管理，断开 SSH 后继续执行，结束时自动释放两张卡。
七天是当前分区的最长时限，脚本本身不额外限定为 24 小时。
结果在 `results/llama-triviaqa-job-<JOB_ID>/`，包含 `p.log`、`d.log`、
`proxy.log`、`warmup.log`、`quality.log` 以及三组结果和自动评分目录。
预热每组四条，单独存储，不计入正式的 1000 条。遇到执行错误立即停止，保留已落盘结果。

## 官方答案评分

评分使用每条问题中冻结的 `evaluation` 字段，官方答案只用于离线评分，不进入 Prompt 或 probe。

- **TriviaQA**：使用 `NormalizedAliases` 以及归一化的 `HumanAnswers`；标点与下划线替换为空格，去冠词并合并空白；多个参考答案分别取最高 EM、最高 token F1。依据 [官方评测脚本](https://github.com/mandarjoshi90/triviaqa/blob/master/evaluation/triviaqa_evaluation.py)。
- **HotpotQA**：使用官方答案归一化、EM 和 token F1；`yes/no/noanswer` 与另一答案不一致时 F1 为零。依据 [官方评测脚本](https://github.com/hotpotqa/hotpot/blob/master/hotpot_evaluate_v1.py)。当前只生成答案，不报告 supporting-fact 或 joint 指标。
- 不执行模型评分、语义匹配或事后答案抽取。正文中的解释也参与评分，因此三组统一要求只输出短答案。

`evaluate_answers.py` 支持运行目录或 `generations.jsonl`：

```shell
.venv/bin/python research/scripts/replay/generation_quality/evaluate_answers.py \
  --run normal=research/scripts/replay/generation_quality/results/<NORMAL_RUN> \
  --run segmented=research/scripts/replay/generation_quality/results/<SEGMENTED_RUN> \
  --run sparse=research/scripts/replay/generation_quality/results/<SPARSE_RUN> \
  --output-dir research/scripts/replay/generation_quality/results/<ANALYSIS_RUN>
```

也接受 `--normal-results`、`--segmented-results`、`--sparse-results`，或只传其中一组计算绝对分数。旧 `evaluate_squad.py` 命令转交同一个新版评分入口。

### 分母与配对口径

- 主指标 `exact_match`、`f1` 使用**全部预先选中的样本**为分母，范围 0–100；失败和缺失预测计零。配置清单使中断后尚未落盘的请求也能计入；只拿旧 JSONL 而没有选中清单时，只能使用已有记录作为分母。
- 另列成功率、失败数、缺失数及 `*_successful_only`，避免把服务错误和模型错误混为一谈。HTTP 成功但 SSE 断流、上游报错或 probe 未实际执行，都属于失败。
- 两组必须有相同样本集合；逐条校验官方答案、评分协议、最终逻辑 Prompt、来源事件及时间摘要、输入传输方式、模型/chat template 和采样配置。显式 token_ids 模式还检查逻辑 token 摘要。
- 每个配对报告 EM/F1 的百分点变化、四种精确匹配结局，以及“参考模式原本正确时，候选模式退化为错误”的比例。失败也纳入主退化统计，另列双侧成功的辅助差值。
- exact McNemar p-value 用于观察配对正确率差异；它不单独证明两组质量等价。

CSV 中的逐样本 EM/F1 与差值使用 0–1；JSON 汇总分数使用 0–100，汇总差值使用百分点。

报告中的 `sparse_vs_segmented` 以 segmented 为参考、sparse 为候选。`em_delta_percentage_points=-2` 表示 EM 下降 2 个百分点；`regression_rate_given_reference_exact=0.1` 表示参考原本答对的样本中有 10% 在候选模式下失去精确命中。

## 完整生成与执行证据

start 发送模型 chat prefix；Append/Update 按各来源的时钟发送；finish 发送最终问题、统一短答案约束和 assistant generation header。sparse 注册的 `probe_query` 是这段**实际 finish 文本**，保证临时 Query 与最终 Query 一致。

客户端完整消费 P 首 token 和 D 后续 SSE 到 `[DONE]`，并要求合法的 `stop` / `length` 终止原因。仅收到首 token 和 `[DONE]`、缺少终止原因时按失败处理，避免 D 加载失败被误计为成功答案。若 P 首 token 已满足 stop 条件，则允许合法地不再启动 D 续写。每条结果保存 P/D 输出 token 计数及 P 原始 `prompt_stream_stats`。

分段模式检查 masked-padding 元数据、Update 合并关闭、Update 接收数一致；sparse 额外检查每个有效 Append 已发布策略、完整独占块范围、评分和 Top-K、保留比例及私有 Query 页释放数。执行校验失败的答案不会被当作成功生成参与模型质量评分。

## 输出与离线检查

每组目录包含：

- `config.json`：选中样本、来源摘要、官方答案及配置，运行前写入。
- `generations.jsonl`：逐请求答案、失败原因、输入摘要、生成设置和实际 probe 统计，逐条落盘。
- `replay_summary.json`：本组成功和失败数。

评分目录包含 `evaluation_summary.json` 和 `per_request_scores.csv`。CSV 每行是一条样本在一组配对中的得分；三模式会产生三种配对。

先用真实数据离线验证路径和 Prompt：

```shell
.venv/bin/python research/scripts/replay/generation_quality/replay_generation_quality.py \
  --config research/scripts/replay/generation_quality/configs/hotpotqa.json \
  --mode sparse --max-requests 2 --dry-run --run-name hotpotqa-check
```

`--dry-run` 不访问服务，额外保存 `prompt_preview.jsonl`；它没有模型答案，评分器会拒绝将其作为质量结果。CPU 回归可运行：

```shell
.venv/bin/python -m pytest research/scripts/replay/tests -q
```
