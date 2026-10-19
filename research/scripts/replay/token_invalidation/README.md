# Token 失效统计

本实验用同一批问题、证据和原始事件时间线，比较三种上下文建模方式导致的
**已计算 token 失效数**。三个回放入口都使用普通 proxy 的 P-only TTFT 路由，不启动 D。

| 回放入口 | 服务端看到的正文 | 主指标覆盖的内容 |
| --- | --- | --- |
| `replay_update_first.py` | 单个 Update 来源，每次完整替换为 `当前 Update + 已到达 Append`。 | 整体正文中被快照替换失效的已计算 token。 |
| `replay_append_first.py` | 单个 Update 来源，每次完整替换为 `已到达 Append + 当前 Update`。 | 整体正文中被快照替换失效的已计算 token。 |
| `replay_segmented.py` | Append 增量与 Update 完整快照分别发送，保留独立 KV 域。 | Update 域中被快照替换失效的已计算 token。 |

**Update-first / Append-first 指正文的空间排列，不代表改变事件到达顺序。**
固定模板 T 仍在 start 提交，问题 Q 和模板后缀在 finish 提交；整体基线把动态正文
建模为一个完整 Update 来源。默认发送原始文本，P 负责分词。

## 统计定义

主列 `snapshot_invalidated_tokens` 来自 P 返回的
`update_tokens_invalidated_by_snapshot`。服务端每次选择 Update work 时累加：

$$
I = \max(0, C_{old} - \min(C_{old}, LCP))
$$

其中 `C_old` 是上一代 Update KV 的实际已计算进度，`LCP` 是新旧快照的 token
最长公共前缀。整体基线的 Update 域包括 Append 和 Update 两类证据，所以 Update
在前时，后面的 Append 也可能一起失效；Append 在前时，新 Append 插入旧 Update
之前也可能导致旧 Update 后缀失效。分段模式按服务端原有独立来源语义处理。

| 结果列 | 用途 |
| --- | --- |
| `snapshot_invalidated_tokens` | 服务端实际主指标。schema v2、守恒式、输入接收数和预期开关均有效后才赋值。 |
| `potential_replaced_suffix_tokens` | 从 ACK token IDs 计算的旧快照逻辑后缀长度，仅作诊断；未计算的后缀不代表实际计算损失。 |
| `preemption_discarded_tokens` | Update、Prefix/Append、Question 因抢占丢弃的 token 之和，单独报告，不加进快照失效主指标。 |
| `original_append_chunks` / `original_update_snapshots` | 原始数据事件数量。 |
| `outbound_append_events` / `outbound_update_events` | 实际 REST 数据事件的成功 ACK 数量；整体基线把两类原始事件都转换成 Update。 |
| `changed_update_events` | ACK 中 token 内容发生变化的 Update 数量。重复的 final 快照仍发送，但不产生新的服务端 admission。 |
| `formal_result_status` | 是否可以纳入正式失效统计；链路成功与统计有效分开记录。 |

**缺失和失败使用 null，有效的零失效保留为 0。** 单次请求可能多次失效并重新计算
同一位置，因此累计失效数可以超过最终 prompt 长度。抢占会改变实际缓存进度，
不能只根据数据文件离线推导主指标。

## 数据与事件时间

复用 [公共数据入口](../README.md)：支持 TriviaQA、HotpotQA 的独立 questions、
Append、PipelinePool，并兼容 `--data-dir` 下的 mixed-source v2。当前实验每请求
每类最多一个来源，与现有两个数据集的生成格式一致；空 Append 仍参与回放。

不传数据路径时默认 TriviaQA，`--max-requests` 默认 0，即遍历全部就绪样本。
复用 `../configs/triviaqa.json` 时，该配置内的 16 条限制仍生效，全量运行需显式
传 `--max-requests 0`。数据内容、时间配置与输出目录分别指定。

| 参数 | 默认值与含义 |
| --- | --- |
| `--append-initial-delay` / `--update-initial-delay` | `0` 秒；相对 start ACK 的首条数据目标时间。 |
| `--append-chunk-interval` | `0.05` 秒；Append 第 i 条目标为 `initial_delay + i × interval`。 |
| `--update-chunk-interval` | `0.02` 秒；Update 同样按原始快照序号计时。 |
| `--source-finish-delay` | `0` 秒；从各原始来源最后一条数据 ACK 起算。空来源从 initial delay 起算。 |
| `--qps` | `0`；默认闭环串行。大于 0 时使用公共 fixed/poisson 到达过程。 |
| `--max-concurrency` | `1`；使用既有 JSON 配置时可显式覆盖其并发默认值。 |
| `--expected-coalescing` | `disabled`；默认单独评估分段管理，检查 P 实际未启用 Update coalescing。 |
| `--expected-masked-padding` | `any`；可以显式要求 enabled 或 disabled，三种布局应保持同一设置。 |

三个模式使用同一个绝对时间计划。相同目标时刻先 Append、后 Update，逐条发送，
**不会将同时刻的多条原始事件折叠成一个快照**。所有模式都串行等待每条 REST ACK；
慢 ACK 导致后续事件迟发，时间线不重新起算。发送日志与 CSV 保存计划和实际偏移。

为保持本统计实验的控制条件一致，三个模式都在所有数据事件发送后才进入来源封口
阶段。分段模式分别发送各来源 `source_finished`；整体基线等所有原始来源达到封口
时间后，仅结束一个合并来源。所有结束 ACK 完成后才发送最终问题。
这与原性能 stream 回放的来源并发任务及早封口方式不同，不应把两种发送方式的结果混合。

ACK 表示事件入队，不代表 GPU 已完成对应计算。初始延迟决定谁先到达；正文排列
由三个独立脚本绑定，二者可以分别控制。

## 检查输入

无需启动服务，可先查看少量问题的完整事件正文和最终模型 prompt：

```shell
.venv/bin/python research/scripts/replay/token_invalidation/replay_update_first.py \
  --config research/scripts/replay/configs/triviaqa.json \
  --max-requests 2 --dry-run \
  --run-name update-first-check
```

`prompt_preview.jsonl` 包含 `event_plan`，可以检查合并快照的次序和目标时间。
dry-run 不生成服务端统计，不能据此计算失效数。正式运行时，上下文窗口应容纳
最大中间快照、固定模板、问题、分段 padding 和一个输出 token。

TriviaQA 的 v3 Append 默认最多 10 块，总预算 6,144 tokens，重建后仍使用
`web_unfiltered_dev_v1/`。使用这份数据时，P 节点建议设置 `MAX_MODEL_LEN=16384`，
不能继续沿用旧三块数据实验的 8,192 窗口。旧三块 Append 备份已清理。

## 遍历数据集

先启动 P 和普通 proxy；只做本实验时 P 可以用 `ENABLE_KV_TRANSFER=0`。
服务启动方式见 [server README](../../server/README.md)。P 的合并策略与 mask 开关
属于服务启动参数，下面的 expected 参数只校验实际值。

从仓库根目录分别运行三个回放。下面显式让 Update 延后 0.025 秒到达，
Append 间隔 0.05 秒、Update 间隔 0.02 秒；可按实验需要修改，并给重复轮次不同名称：

```shell
for layout in update_first append_first segmented; do
  .venv/bin/python "research/scripts/replay/token_invalidation/replay_${layout}.py" \
    --config research/scripts/replay/configs/triviaqa.json \
    --max-requests 0 --qps 0 --max-concurrency 1 \
    --append-initial-delay 0 --update-initial-delay 0.025 \
    --append-chunk-interval 0.05 --update-chunk-interval 0.02 \
    --source-finish-delay 0 \
    --expected-coalescing disabled --expected-masked-padding enabled \
    --run-name "triviaqa-${layout}-r1"
done
```

HotpotQA 对应 Pool 完成后，将配置替换为 `configs/hotpotqa.json`。第一次联调建议
把 `--max-requests 0` 改成 16；确认统计有效再遍历全部。`--max-tokens` 兼容旧配置，
实际 start 的输出预算始终为 1。默认不在客户端分词；历史输入方式可显式使用
`--input-transport token_ids`，同一次对比必须保持传输方式相同。

## Slurm 一次运行三个模式

[run_token_invalidation.sbatch](run_token_invalidation.sbatch) 用一个公共参数数组
依次运行 Update-first、Append-first、Segmented，随后自动配对汇总并生成 PNG/PDF。
这是 CPU 客户端作业，不申请 GPU；提交前启动 P 和普通 proxy。

从仓库根目录提交。Slurm 在脚本运行前打开输出日志，所以先创建 `results`：

```shell
mkdir -p research/scripts/replay/token_invalidation/results

DATASET=triviaqa \
PROXY_URL=http://router-node:8000 \
MAX_REQUESTS=0 \
APPEND_INITIAL_DELAY=0 \
UPDATE_INITIAL_DELAY=0.025 \
APPEND_CHUNK_INTERVAL=0.05 \
UPDATE_CHUNK_INTERVAL=0.02 \
SOURCE_FINISH_DELAY=0 \
EXPECTED_MASKED_PADDING=enabled \
sbatch --partition=gpu research/scripts/replay/token_invalidation/run_token_invalidation.sbatch
```

将 `router-node` 换成实际 proxy 所在节点。`PROXY_URL` 指向 proxy 的 8000 端口；
默认 localhost 仅适用于作业和 proxy 位于同一节点。资源、节点、时限可通过
`sbatch --nodelist ... --mem ... --time ...` 覆盖脚本中的声明。

同一套公共参数集中定义在脚本顶部，也可以像上面一样通过环境变量覆盖：

| 环境变量 | 默认值与含义 |
| --- | --- |
| `DATASET` | `triviaqa`；可改为 `hotpotqa`，自动选择相应数据默认路径。 |
| `QUESTIONS_FILE`、`APPEND_DIR`、`PIPELINE_POOLS_DIR` | 默认不设置；可覆盖数据版本。 |
| `MODEL`、`TOKENIZER_MODEL` | `qwen2.5-3b`、`~/Models/Qwen2.5-3B`。 |
| `START_INDEX`、`MAX_REQUESTS` | `0`、`0`；默认遍历全部就绪问题。 |
| `APPEND_INITIAL_DELAY`、`UPDATE_INITIAL_DELAY` | 均为 `0` 秒。 |
| `APPEND_CHUNK_INTERVAL`、`UPDATE_CHUNK_INTERVAL` | `0.05`、`0.02` 秒。 |
| `SOURCE_FINISH_DELAY` | `0` 秒。 |
| `QPS`、`ARRIVAL_PROCESS`、`MAX_CONCURRENCY` | `0`、`fixed`、`1`；默认串行，三边使用相同值。 |
| `RANDOM_SEED`、`SAMPLING_SEED`、`TEMPERATURE` | `2026`、`2026`、`0`。 |
| `EXPECTED_COALESCING`、`EXPECTED_MASKED_PADDING` | `disabled`、`any`；检查实际 P 设置。 |
| `FAIL_FAST`、`FORMAL_FAIL_FAST` | 均为 `0`；设为 `1` 则遇到对应失败停止作业。 |
| `DRY_RUN` | `0`；设为 `1` 只生成三种输入预览，跳过联网、配对与画图。 |
| `RUN_PLOTS`、`PLOT_X_SCALE`、`PLOT_DPI` | `1`、`linear`、`200`。 |
| `OUTPUT_DIR`、`RUN_NAME` | 本目录 `results`、`<dataset>-job-<Slurm job ID>`；重跑使用新名称。 |

脚本直接使用 `--dataset` 的公共数据入口，数量、时间等设置均取这一份公共参数，
不受 `configs/triviaqa.json` 的 16 条默认限制影响。

结果目录如下；三个模式的 `config.json` 分别保存同一套有效参数和输入指纹：

```shell
results/<RUN_NAME>/
  update_first/
  append_first/
  segmented/
  comparison/
    paired_requests.csv
    comparison_summary.json
    plots/
      token_invalidation_comparison.png
      token_invalidation_comparison.pdf
```

无需提交作业即可检查两个样本的三种布局，输出仍默认放到本目录的 `results`：

```shell
DRY_RUN=1 MAX_REQUESTS=2 \
bash research/scripts/replay/token_invalidation/run_token_invalidation.sbatch
```

首次连接服务或修改接口后，可先设 `MAX_REQUESTS=2 FAIL_FAST=1 FORMAL_FAIL_FAST=1`
验证三种模式的链路和统计，再扩大样本数。dry-run 仅检查输入，不验证服务端协议。
P 的 `MAX_MODEL_LEN` 须覆盖最大中间快照及分段 padding；超过窗口的请求会失败，
只能从三边共同有效样本中排除，不能把缺失统计记为零。

## 输出与配对比较

结果默认写入 `research/scripts/replay/token_invalidation/results/<run-name>/`：

| 文件 | 内容 |
| --- | --- |
| `config.json` | 有效配置、模板指纹、选中 ID、原始输入与计划指纹，联网前写入。 |
| `request_metrics.jsonl` | 每完成一个请求就刷写；中途失败也保留已完成记录。 |
| `request_metrics.csv` | 原始服务端计数、正式有效主指标及请求规模。 |
| `event_metrics.csv` | 每个事件的目标时间、实际发送时间、ACK、LCP 和错误。 |
| `summary.json` | 完成数、有效数、失效总数、均值、P50/P95/P99 和最大值。 |

`--fail-fast` 可在链路失败后停止，`--formal-fail-fast` 可在统计无效时停止，均默认
关闭。中断或 fail-fast 前的 JSONL、CSV 与摘要仍会保存。完整 P-only SSE 必须收到
`[DONE]`，只收到 token 后断流仍记为失败。

三个运行的单独总量可能基于不同有效样本，正式比较使用离线配对工具：

```shell
.venv/bin/python research/scripts/replay/token_invalidation/compare_results.py \
  --update-first-run research/scripts/replay/token_invalidation/results/triviaqa-update_first-r1 \
  --append-first-run research/scripts/replay/token_invalidation/results/triviaqa-append_first-r1 \
  --segmented-run research/scripts/replay/token_invalidation/results/triviaqa-segmented-r1 \
  --output-dir research/scripts/replay/token_invalidation/results/triviaqa-comparison-r1
```

工具检查同一批 ID、原始输入、证据、事件时间、模型模板与控制参数，并核对实际
P coalescing/mask 设置。Update-first 的最终文本排列不同是实验定义；Append-first
与分段模式的最终文本必须一致。

只将三边都正式有效的请求写入 `paired_requests.csv`，汇总到
`comparison_summary.json`。节省量为“基线失效数 − 分段失效数”，比例按共同有效
样本的总量计算；基线总量为 0 时比例为 null。负节省量保留原值。

配对汇总不传 `--output-dir` 时，默认写到本目录的 `results/comparison/`。
回放、汇总和画图的默认目录都通过脚本位置确定，不依赖启动命令时的工作目录。

## 绘图

运行 [plot_token_invalidation.py](plot_token_invalidation.py) 读取配对汇总，生成四联图：

- **总失效 token**：三种模式在共同有效请求上的累计值。
- **每请求失效 token**：三种模式的均值和 P95。
- **失效数分布**：展示每请求失效数的累积分布，包含有效的零失效请求。
- **分段管理节省量**：分别相对两种整体基线绘制 `基线 − 分段` 的累积分布，保留负节省量，并标注总量节省比例。

读取上面的自定义配对目录：

```shell
.venv/bin/python research/scripts/replay/token_invalidation/plot_token_invalidation.py \
  --comparison-dir research/scripts/replay/token_invalidation/results/triviaqa-comparison-r1
```

如果配对汇总使用默认 `results/comparison/`，画图时不需要传参：

```shell
.venv/bin/python research/scripts/replay/token_invalidation/plot_token_invalidation.py
```

默认生成 `comparison-dir/plots/token_invalidation_comparison.png` 和同名 PDF。
可通过 `--output-dir` 改变图表目录，`--dpi` 设置 PNG 分辨率，`--x-scale symlog`
展开跨度很大的 CDF 横轴，同时保留零值和负值。图片使用英文坐标标签，适合无中文
字体的服务器。没有共同有效样本或 CSV 与摘要不一致时会报错。
