# 回放实验

`configs/` 预留实验配置，`results/` 预留实验输出，目前均只有 `.gitkeep`。
现有 replay 脚本及其输出路径继续使用原设置。

后续配置分别选择问题、Append chunks 与 Update PipelinePool 数据版本，
并定义请求到达过程、两路初始延迟和发送间隔。每次回放的配置快照与结果
保存在同一个实验输出目录，便于复现。

数据目录及现有数据位置见 [科研数据目录](../data/STRUCTURE.md)。
