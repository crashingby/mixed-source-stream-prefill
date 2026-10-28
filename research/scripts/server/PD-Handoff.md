# P/D 交接与 P 侧 TTFT

当前协议由 D 生成全部回答。P 仍执行 `max_tokens=1`，内部采样只用于确认首 token 已就绪；proxy 丢弃该采样结果。

## 执行流程

1. P 执行完整或分段 prefill，返回实际物理 prompt、KV 清单和研究统计。
2. SSE 客户端收到 `event=prefill_ready`、`measurement_scope=p_side_ttft` 的控制事件。该事件的 `choices=[]`，不携带回答文本或 token IDs。
3. 完整 P/D 请求把原始物理 prompt 和 KV 元数据交给 D，生成预算保持原值。D 读取全部有效 prompt KV，重算最后一个 prompt token 以取得 logits，随后生成完整回答。
4. 所有回答、停止条件和 usage 由 D 决定。P 内部的 stop 不会提前结束 D 请求，即使用户只请求一个输出 token，也会运行 D。

普通 proxy 的 `/v1/completions_ttft`、`/v1/prompt_stream/start_ttft` 和 `/v1/prompt_stream/finish_ttft` 继续支持 P-only 测试；完成通知后返回 `[DONE]`，不创建 D 请求或远端 KV 生命周期。完整生成客户端必须继续消费 D 输出并验证正常终止，不能把完成通知当作回答成功。

## KV 与分段约束

- D 的输入使用 P 实际执行的 token IDs，不重新拼接或分词，也不追加 P 采样出的 token。
- 分段物理 prompt 保留中间 padding；block manifest、逻辑位置和 padding mask 原样传递。
- `remote_num_tokens` 描述 P 实际完成的 KV 范围。`prompt_stream_prefill_output_tokens=0` 表示交接时没有追加 P 输出，而非 P 内部未执行采样。
- NIXL pull 的完整命中沿用 scheduler 的末 token 重算。Stable KV push 的收养路径校验原始 prompt，并设置 `num_computed_tokens=L-1`，重新计算最后一个 prompt token。
- KV 布局兼容性、exact-transfer 校验和独立缓存语义继续生效；未选中的稀疏 KV 仍完整传给 D。

更新后需要重启 P、D 和 proxy，并使用同版本 replay。启动参数保持原样；旧 P 的交接输出计数为 1，会被新 proxy 的清单校验拒绝。

## 指标口径

| 指标 | 终点 | 包含范围 |
| --- | --- | --- |
| P 侧 TTFT | 客户端收到 `prefill_ready` | 输入事件等待、P 排队、prefill、probe、评分与 HTTP 通知 |
| 用户侧 TTFT | 客户端收到 D 的首个回答 token | 额外包含交接关键路径上的 KV 传输和 D 接管 |
| finish 后 P 剩余延迟 | finish 发出至 `prefill_ready` | 输入到齐后的 P 侧剩余等待 |

共享 replay 为兼容现有绘图保留 `request_to_first_token_ms`、`finish_to_first_token_ms` 和成功标记 `first_token_received` 的列名；在 `--ttft-only` 下，这些字段表示完成通知的计时，`ttft_origin=prefill_ready`。完整生成的终点仍是真实回答，来源为 `decode`。P-only 的 `completion_tokens=0`，不报告输出 TPS。

Query probe replay 的 `first_token_origin=prefill_ready`、`ttft_scope=p_side`。Stable KV replay 保留历史字段 `finish_to_p_first_client_ms` 和 `p_first_to_d_first_client_ms`，当前分别表示 finish 至 P 完成通知、P 完成通知至 D 首 token；P 不再贡献回答 token。

2026-10-28 以前的结果可能使用“P 首 token 加 D 续写”的旧协议，应按各自配置和源码快照解释，不能把历史用户 TTFT 与当前 D 首 token 延迟直接混算。

## 正确性检查

已启动服务时，可执行块边界与预算检查：

```shell
.venv/bin/python -m research.scripts.simple_test.pd_handoff \
  --proxy-url http://127.0.0.1:18000 \
  --decode-url http://127.0.0.1:18200 \
  --model llama-3.1-8b-instruct \
  --output tmp/pd-handoff.json
```

脚本覆盖 31、32、33 token prompt，1 与 4 token 预算，JSON/SSE 两种响应，并与直接 D 生成比较。每次请求使用独立 cache salt，避免 D 的本地 prefix hit 掩盖实际 KV 交接。完整双 GPU 小规模验证复用质量测试 launcher：

```shell
MAX_REQUESTS=4 WARMUP_REQUESTS=1 PD_HANDOFF_SMOKE=1 \
RUN_PREFIX=pd-handoff \
sbatch --nodelist=g2c9 --time=01:00:00 \
  research/scripts/replay/generation_quality/run_full_quality.sbatch
```

节点按提交时的空闲资源选择。该作业还检查 P-only Query probe，并完成 TriviaQA 的 normal、segmented、sparse 三组 P/D 生成与评分。临时缓存位于仓库 `tmp/`，日志和结果保存至质量测试的 `results/`。

## 本次验证

2026-10-28 在 g2c9 的两张 GPU 上运行 Slurm 作业 `17948`，状态为 `COMPLETED`，退出码 `0:0`。模型为 Llama-3.1-8B-Instruct，P 使用 `CHUNK_SPARSE_ATTN`，D 使用 `TRITON_ATTN`，显存比例均为 0.8。

- 普通 P/D 的 12 个边界、预算与输出比对案例全部通过；另一个普通 P-only 控制事件案例通过。
- P-only sparse Query probe：2/2 成功，输出来源为 `prefill_ready`。
- TriviaQA 完整生成：normal、segmented、sparse 各预热 1 条、验证 4 条，全部成功；并发上限为 4，所有回答 token 均来自 D。
- 三组的小样本 EM/F1 均为 50%，用于确认评分链路可运行，不构成正式质量结论。
- Stable KV push 的 proxy、收养与 worker 协议通过 CPU 回归；本次真实双 GPU 验证使用 NIXL pull，未重跑 push 实验。

日志及原始结果见 [本次结果目录](../replay/generation_quality/results/pd-handoff-g2c9-20261028)，评分结果见 [evaluation_summary.json](../replay/generation_quality/results/pd-handoff-g2c9-20261028/pd-handoff-analysis-job-17948/evaluation_summary.json)。
