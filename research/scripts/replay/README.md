# Replay 实验目录

本目录按研究问题拆分批量回放实验。不同子目录可以读取同一份
`research/data/mix_source` 数据，但不应混用结果目录或实验配置。

## 目录结构

- `segmented_prefill/`：第一个创新点的实验包，比较分段流式 Prefill、合并
  Update baseline 与普通整段 Prompt。原有回放脚本、QPS sweep、绘图脚本和
  历史结果都已迁入该目录。
- `update_coalescing/`：第二个创新点的实验目录，用于评估 Update 合并策略。
  当前先提供完整实验设计，后续回放、批量提交、统计和绘图脚本均放在这里。

## 使用入口

- 复现原有分段 Prefill 实验：阅读
  [segmented_prefill/README.md](segmented_prefill/README.md)。
- 设计或执行 Update 合并实验：阅读
  [update_coalescing/README.md](update_coalescing/README.md)。

两个实验包应分别设置自己的 `--output-dir`。目录迁移后，第一个实验的默认
结果路径为：

```text
research/scripts/replay/segmented_prefill/results
```

Update 合并实验后续统一使用：

```text
research/scripts/replay/update_coalescing/results
```

迁移前生成的历史 `config.json` 和绘图 summary 仍记录旧的
`research/scripts/replay/results` 路径。它们属于当次运行的原始元数据，因此只移动
文件位置，不回写其中内容；新运行会记录新的默认路径。

## 共同实验约束

- 所有对比模式使用相同的数据样本、模型、随机种子和请求到达时间线。
- P 节点启动参数发生变化时，应重新启动 P 节点并重新预热；不能在同一个
  server 进程中把不同 coalescing 配置视为独立实验。
- TTFT-only 实验只测 P 节点生成首 token 的路径；完整生成质量和 D 节点传输
  应作为独立实验，不与 TTFT 主实验混在一起。
- 每次运行都保存完整配置、成功与失败请求数，以及能够复现样本选择和到达过程
  的随机种子。
