# 生成质量与 SQuAD EM/F1 实验

本目录比较两种完整生成路径的答案质量：

- **普通 Prefill**：把最终完整 Prompt 一次性提交到 proxy 的
  `/v1/completions`，随后正常执行 P→D 分离。
- **混合来源分段流式 Prefill**：依次调用
  `/v1/prompt_stream/start`、`/event` 和 `/finish`，保留 Append 与 Update
  的独立来源语义，再由 P→D 链路完成生成。

实验不会使用 `finish_ttft` 或 `completions_ttft`。客户端消费完整 SSE，保存
最终答案，再按照 SQuAD v2 官方 Exact Match（EM）和 token-level F1 评分。

## 目录内容

| 文件 | 用途 |
| --- | --- |
| `replay_generation_quality.py` | 低压力逐请求回放；使用 `--mode normal` 或 `--mode segmented` 选择链路。 |
| `evaluate_squad.py` | 计算官方 EM/F1 和逐请求配对退化。 |
| `run_generation_quality.sbatch` | 顺序运行两种模式，结束后自动评分。 |
| `results/` | 默认结果目录；运行时创建。 |

## 评测口径

### SQuAD 官方 EM/F1

`evaluate_squad.py` 复现 SQuAD v2 的核心规则：

- 将答案转为小写，删除 ASCII 标点和英语冠词 `a/an/the`，再合并空白。
- **EM**：归一化预测与某个官方答案完全一致时记为 1。
- **F1**：根据归一化 token 的重合数量计算 precision、recall 和 F1。
- 多个官方答案分别评分后取最高值。
- 模式级 `exact_match` 和 `f1` 使用 `0--100` 百分数口径。

质量 Prompt 会在最终用户问题和 `Answer:` 之间插入统一约束，要求只返回最短
答案片段。该位置紧邻生成起点，不容易被前面较长的 RAG 内容稀释。默认还设置
`stop=["\n"]`，减少正确答案因为附带解释而被 F1 惩罚。两种模式使用完全相同
的约束、采样参数和最终逻辑内容。

回放器必须通过 `--tokenizer-model` 加载与 P/D 节点相同的 tokenizer，并调用
模型自己的 `apply_chat_template(..., add_generation_prompt=True)`。数据中的通用
模板作为 system message，检索内容和最终问题作为 user message。不能只把
`You are a helpful assistant` 与 `Answer:` 当作跨模型通用模板，因为 Qwen 和
Llama 使用不同的 assistant header 与结束 token。

### 普通模式不是标准答案生成器

网页检索可能有错误、噪声或无关内容，所以普通完整 Prompt 也可能答错。不能把
所有 `segmented` 错误都归因于流式近似。评分脚本同时报告：

- 两种模式各自的 EM、F1 和请求成功率；
- `both_exact`：两种模式都答对；
- `normal_only_exact`：普通模式答对、分段模式答错，是重点质量回退；
- `segmented_only_exact`：分段模式答对、普通模式答错；
- `neither_exact`：两种模式都没有精确命中；
- `segmented_em_regression_rate_given_normal_exact`：普通模式原本答对的请求中，
  分段模式退化为错误的比例；
- `mean_f1_delta_segmented_minus_normal`：逐请求配对的平均 F1 变化。

每条结果同时保存 `logical_prompt_sha256` 和
`logical_prompt_token_sha256`。若同一样本的两种模式构造了不同文本或 token
序列，评分脚本会直接报错，不生成不可比的统计。

## Prompt 与事件构造

普通模式先构造以下消息：

```text
system = 原始通用模板
user = 所有 Append chunk
     + 每个 Update 来源的最终完整快照
     + 最终用户问题
     + 短答案评测约束
     + Answer:
```

随后由目标 tokenizer 的 `apply_chat_template()` 加入 BOS、system/user role、
user 结束标记和 assistant generation header。回放器将渲染后的结构拆成固定
template prefix、动态来源内容和 finish suffix，并使用 tokenizer 预先生成 token
IDs。

分段模式使用相同逻辑内容，但保留 REST 事件形态：

1. `start` 发送 chat template 的 BOS、system message 和 user header token IDs。
2. 每个来源按自己的 `sequence_no` 串行发送预分词 token IDs。
3. Append 与 Update 来源按各自 interval 并发竞争。
4. 所有 `source_finished` 获得 ACK 后，在最终用户问题与 `Answer:` 之间插入
   短答案约束，并追加 user 结束标记与 assistant header，再发送
   `prompt_finished`。
5. `/finish` 先返回 P 首 token，proxy 再转发 D 的剩余 decode SSE。
6. 回放脚本消费到 `[DONE]`，拼接 P 与 D 的输出文本。

实验默认**一次只执行一个逻辑请求**，没有 QPS 或跨请求并发。参数
`--inter-request-delay` 还会在请求之间留出空闲时间。因此这里测输出质量，不测
高压力吞吐或 TTFT。

## 启动 P/D 节点

以下 `<MODEL_PATH>`、`<SERVED_MODEL_NAME>`、`<P_NODE>` 和 `<D_NODE>` 都是
需要替换的占位符。P、D 必须使用相同模型和 tokenizer。

```shell
mkdir -p research/scripts/logs
```

### P 节点

本实验不叠加 Update coalescing，应明确把对应配置设为空：

```shell
MODEL_PATH=<MODEL_PATH> \
SERVED_MODEL_NAME=<SERVED_MODEL_NAME> \
GPU_MEMORY_UTILIZATION=0.9 \
MAX_MODEL_LEN=16384 \
MAX_NUM_SEQS=64 \
PROMPT_STREAM_UPDATE_COALESCING_CONFIG='' \
sbatch --nodelist=<P_NODE> research/scripts/server/start_p_node.sbatch
```

P 默认监听 `8100`，NIXL side channel 使用 `5600`。启动脚本会自动发布实际
Slurm hostname，不应把 side channel host 写成 `localhost`。

### D 节点

```shell
MODEL_PATH=<MODEL_PATH> \
SERVED_MODEL_NAME=<SERVED_MODEL_NAME> \
GPU_MEMORY_UTILIZATION=0.9 \
MAX_MODEL_LEN=16384 \
MAX_NUM_SEQS=64 \
sbatch --nodelist=<D_NODE> research/scripts/server/start_d_node.sbatch
```

D 默认监听 `8200`，NIXL side channel 使用 `5601`。通过日志确认模型完成加载，
而不能只看 `squeue` 中的 `R` 状态：

```shell
tail -f research/scripts/logs/vllm-p-<P_JOB_ID>.out
```

```shell
tail -f research/scripts/logs/vllm-d-<D_JOB_ID>.out
```

`MAX_NUM_SEQS=64` 与通用服务启动默认值一致；质量回放仍严格串行，
不会因为提高服务上限而增加回放并发。它不是最大序列长度；单请求长度
仍受 `MAX_MODEL_LEN` 限制。当前 mixed-source 数据建议使用 16384；超长请求仍会
按错误记录落盘，需要提高 P、D 的 `MAX_MODEL_LEN` 或重新裁剪数据。

### 当前个人环境可直接复制的 P/D 示例（发布前删除）

下面保留当前个人环境的模型路径和常用节点，方便直接复制。节点被占用时通常只
需要修改 `P_NODE` 和 `D_NODE` 两行：

```shell
export MODEL_PATH=/workplace/home/huangxinying/Models/Llama-3.1-8B-Instruct
export SERVED_MODEL_NAME=llama-3.1-8b-instruct
export TOKENIZER_MODEL="${MODEL_PATH}"

P_NODE=g2c9
D_NODE=g2c9

mkdir -p research/scripts/logs

GPU_MEMORY_UTILIZATION=0.9 \
MAX_MODEL_LEN=16384 \
MAX_NUM_SEQS=64 \
PROMPT_STREAM_UPDATE_COALESCING_CONFIG="" \
sbatch --nodelist="${P_NODE}" research/scripts/server/start_p_node.sbatch

GPU_MEMORY_UTILIZATION=0.9 \
MAX_MODEL_LEN=16384 \
MAX_NUM_SEQS=64 \
sbatch --nodelist="${D_NODE}" research/scripts/server/start_d_node.sbatch
```

提交后使用以下命令记录 job id 和实际节点：

```shell
squeue -u "${USER}"
```

## 启动 proxy

P、D 完全启动后运行：

```shell
.venv/bin/python research/scripts/server/proxy_server.py \
  --host 0.0.0.0 \
  --port 8000 \
  --prefiller-hosts <P_NODE> \
  --prefiller-ports 8100 \
  --decoder-hosts <D_NODE> \
  --decoder-ports 8200
```

本机回放可以使用 `http://127.0.0.1:8000`。如果 replay 由 Slurm 分配到另一
节点，`127.0.0.1` 会指向 replay 节点自身，必须使用：

```text
http://<PROXY_HOST>:8000
```

## 单请求闭环

建议先用相同 `query_id` 分别验证两种模式。

### 普通 Prefill

```shell
.venv/bin/python \
  research/scripts/replay/generation_quality/replay_generation_quality.py \
  --mode normal \
  --data-dir research/data/mix_source \
  --query-id 0 \
  --proxy-url http://<PROXY_HOST>:8000 \
  --model <SERVED_MODEL_NAME> \
  --tokenizer-model <MODEL_PATH> \
  --max-tokens 32 \
  --temperature 0 \
  --seed 2026 \
  --run-name quality-query-0-normal
```

### 混合来源分段 Prefill

```shell
.venv/bin/python \
  research/scripts/replay/generation_quality/replay_generation_quality.py \
  --mode segmented \
  --data-dir research/data/mix_source \
  --query-id 0 \
  --proxy-url http://<PROXY_HOST>:8000 \
  --model <SERVED_MODEL_NAME> \
  --tokenizer-model <MODEL_PATH> \
  --append-chunk-interval 0.05 \
  --update-chunk-interval 0.20 \
  --max-tokens 32 \
  --temperature 0 \
  --seed 2026 \
  --run-name quality-query-0-segmented
```

执行评分：

```shell
.venv/bin/python \
  research/scripts/replay/generation_quality/evaluate_squad.py \
  --normal-results \
    research/scripts/replay/generation_quality/results/quality-query-0-normal \
  --segmented-results \
    research/scripts/replay/generation_quality/results/quality-query-0-segmented \
  --output-dir \
    research/scripts/replay/generation_quality/results/quality-query-0-analysis
```

## 批量 Slurm 回放

`run_generation_quality.sbatch` 默认选择前 100 个有官方答案的完整请求，先运行
`normal`，再运行 `segmented`，最后自动评分。两种模式都是逐请求串行执行。

```shell
PROXY_URL=http://g2c14:8000 \
MODEL=llama-3.1-8b-instruct \
TOKENIZER_MODEL=/workplace/home/huangxinying/Models/Llama-3.1-8B-Instruct \
START_INDEX=0 \
MAX_REQUESTS=0 \
MAX_TOKENS=32 \
TEMPERATURE=0 \
SEED=2026 \
INTER_REQUEST_DELAY=0.2 \
APPEND_CHUNK_INTERVAL=0.50 \
UPDATE_CHUNK_INTERVAL=0.20 \
RUN_PREFIX=squad-quality \
sbatch --nodelist=g2c12 \
  research/scripts/replay/generation_quality/run_generation_quality.sbatch
```

常用参数：

| 环境变量 | 默认值 | 含义 |
| --- | ---: | --- |
| `TOKENIZER_MODEL` | `MODEL_PATH` | 与 P/D 相同的本地模型目录或 Hugging Face tokenizer 标识；两者都未设置时退出。 |
| `TOKENIZER_TRUST_REMOTE_CODE` | `0` | 为 `1` 时允许 tokenizer 仓库自定义代码。 |
| `START_INDEX` | `0` | 在数值 `query_id` 排序后的起始位置。 |
| `MAX_REQUESTS` | `100` | 两种模式共同使用的请求数；`0` 表示全部合格请求。 |
| `MAX_TOKENS` | `32` | 每个答案最大生成 token 数。 |
| `INTER_REQUEST_DELAY` | `0.2` | 完整请求之间的空闲秒数。 |
| `APPEND_CHUNK_INTERVAL` | `0.50` | 同一 Append 来源的数据事件间隔。 |
| `UPDATE_CHUNK_INTERVAL` | `0.20` | 同一 Update 来源的快照间隔。 |
| `MODE_ORDER` | `normal segmented` | 执行顺序；正式配对必须同时包含二者。 |
| `STOP_AT_NEWLINE` | `1` | 为 `1` 时以换行停止，约束最短答案。 |
| `FAIL_FAST` | `0` | 为 `1` 时首个请求失败就终止。 |

若要交换顺序检查时间漂移：

```shell
MODE_ORDER='segmented normal' \
RUN_PREFIX=squad-quality-reversed \
PROXY_URL=http://<PROXY_HOST>:8000 \
MODEL=<SERVED_MODEL_NAME> \
TOKENIZER_MODEL=<MODEL_PATH> \
sbatch research/scripts/replay/generation_quality/run_generation_quality.sbatch
```

## 输出文件

每种模式的目录包含：

- `config.json`：完整配置、样本列表和请求文件哈希。
- `generations.jsonl`：官方答案、模型输出、错误、耗时、逻辑 Prompt 文本哈希
  和 token 序列哈希；每个请求完成后立即 `flush`，中途中止也能保留结果。
- `replay_summary.json`：成功与失败请求数。

评分目录包含：

- `evaluation_summary.json`：两种模式的 EM/F1、成功率和配对退化统计。
- `per_request_scores.csv`：每个问题的官方答案、两种预测、EM、F1 和配对结果。

查看汇总：

```shell
jq '.' \
  research/scripts/replay/generation_quality/results/<ANALYSIS_RUN>/evaluation_summary.json
```

筛选普通模式正确但分段模式错误的请求：

```shell
.venv/bin/python -c '
import csv
from pathlib import Path

path = Path("research/scripts/replay/generation_quality/results/<ANALYSIS_RUN>/per_request_scores.csv")
for row in csv.DictReader(path.open(encoding="utf-8")):
    if row["exact_outcome"] == "normal_only_exact":
        print(row["sample_id"], row["question"])
'
```

## 离线检查

`--dry-run` 只解析数据和构造逻辑 Prompt，不访问 proxy，也不生成可评分答案：

```shell
.venv/bin/python \
  research/scripts/replay/generation_quality/replay_generation_quality.py \
  --mode normal \
  --data-dir research/data/mix_source \
  --tokenizer-model <MODEL_PATH> \
  --max-requests 5 \
  --dry-run \
  --run-name quality-dry-run
```

## 当前限制

- EM/F1 衡量预测与 SQuAD 答案的一致性，不判断本次 RAG 上下文是否真的包含
  答案，也不衡量引用忠实性。
- 网页噪声会影响两种模式的绝对正确率，因此应同时报告模式级 EM/F1 和配对
  回退率，不能只看 `segmented` 单独分数。
- 未匹配 SQuAD 的问题会在回放前排除，不会被当成错误预测。
- 正式对比建议保持 `temperature=0`、相同 `seed`、相同模型与采样参数。
