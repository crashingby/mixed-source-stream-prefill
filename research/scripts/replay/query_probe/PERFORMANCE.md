# Query probe 正式性能实验

## 对照方案

两组统一使用 `--attention-backend CHUNK_SPARSE_ATTN`，共用新版 32×128 attention kernel、KV 分页布局和 padding 掩码。

| 模式 | Attention 读取范围 | Query probe |
| --- | --- | --- |
| dense | 完整历史页 | 不执行 |
| sparse | 按 Query 评分保留历史 Append 的候选块；probe 与最终 Query 完整读取 | 每次新 Append 执行 |

自定义后端沿用 Triton 的 KV 写入逻辑。`TRITON_ATTN` 仍使用原 `unified_attention`，不参与本次正式对照。

## 固定配置

| 项目 | 设置 |
| --- | --- |
| 模型 | Qwen2.5-3B，BF16，TP=1 |
| 资源 | 一张 GPU，仅启动 P 和 proxy，不启动 D |
| 执行 | eager、同步调度、分段 masked padding 开启 |
| Update 合并 | 关闭 |
| 上下文与调度预算 | 上限 16384；每步 2048 token；至少 32 个调度槽位 |
| 数据集 | TriviaQA，相同的前 1000 条可回放请求 |
| 并发 | 4、8、16；闭环负载，不额外固定 QPS |
| 每组预热 | 16 条，不进入正式统计 |
| 事件间隔 | Append 50 ms、Update 20 ms |
| 稀疏保留比例 | 每个 Append 保留 50% 完整独占候选块，保护模板与边界块 |

共 6 个正式组、6000 次请求；正式测试前进行 dense/sparse 各 2 条接口预检。并发 4、16 先 dense 后 sparse，并发 8 先 sparse 后 dense，各组串行使用同一张 GPU。

## 提交与后台运行

从仓库根目录提交，节点按空闲资源选择：

```shell
sbatch --parsable --nodelist=g2c13 --time=7-00:00:00 \
  research/scripts/replay/query_probe/run_performance.sbatch \
  --concurrency 4 8 16 --max-requests 1000 \
  --warmup-requests 16 --keep-ratio 0.5
```

`sbatch` 提交成功后，Slurm 在计算节点执行整个矩阵，包括 P/proxy、客户端、汇总绘图及结束清理。断开提交端 SSH 或结束交互会话不会终止作业，无需额外使用 tmux 或 nohup。时限按当前 gpu 分区的 7 天上限申请。

## 统计与结果

- 整体 TTFT 从提交 start 到收到 P 首 token，包含来源事件等待、排队、probe、评分和最终 prefill。
- 同时报告 finish TTFT、平均值、P50/P95/P99、失败数及完成吞吐。
- 对照核验样本、最终 prompt SHA256、事件计划 SHA256，并结合实际 Update 工作量和恢复统计排除少做工作产生的假加速。
- 保存配置、硬件信息、源码快照与 SHA256；临时缓存位于仓库 `tmp/`，结束后清理。
- 每组结果持续写盘；完成一对即更新 `comparison/` 的表格与图。任何一组失败即停止后续组并回收本作业创建的服务。

本轮 Slurm 作业为 **17074**，节点 **g2c13**。结果写入 `results/performance-17074/`，总日志为 `results/performance-17074.out`。具体状态以 Slurm 与该目录的 `matrix_status.json` 为准。

```shell
squeue -j 17074
tail -f research/scripts/replay/query_probe/results/performance-17074.out
```

本轮尚未形成正式性能结论。先前日志、结果及性能结论已清理；本轮完成后按实际结果更新此文档。回答质量与 D 节点性能需要单独测量。
