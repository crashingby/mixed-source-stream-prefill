# Prompt Stream 实验脚本目录

这个目录保存流式 prefill 原型相关的辅助脚本。现在脚本已经按用途拆分，根目录 README 只作为总览；具体启动参数和命令放在各子目录自己的 README 里。

## 目录结构

- `server/`：启动 PD 分离实验链路的脚本，包括 proxy 服务器和 Slurm 启动脚本。
- `simple_test/`：最小 RESTFUL 联调前端，用于快速验证 append、update 和混合事件链路。
- `data_processing/`：把原始 ANNS / crawl 数据构造成 mixed-source replay 数据集。
- `replay/`：按研究问题拆分的批量回放实验。`segmented_prefill/` 保存第一个
  创新点，`update_coalescing/` 保存 Update 合并实验设计与后续脚本。
- `logs/`：Slurm 启动 P/D 节点时输出的日志目录。

## 推荐阅读顺序

如果只是启动服务并跑最小测试，建议按这个顺序看：

1. `server/README.md`：先启动 P 节点、D 节点和 proxy。
2. `simple_test/README.md`：再跑 append-only、update-only 或 mixed 测试。
3. `replay/README.md`：服务稳定后，选择分段 Prefill 或 Update 合并实验。
4. `data_processing/README.md`：需要重新构造实验数据时再看。

## 通用约定

下面几个占位符会在各子目录 README 中反复出现：

| 占位符 | 含义 | 示例 |
| --- | --- | --- |
| `<REPO_ROOT>` | 当前 vLLM 仓库根目录 | `/path/to/vllm` |
| `<MODEL_PATH>` | 本地模型目录或 Hugging Face 模型名 | `/path/to/Qwen2.5-3B` |
| `<SERVED_MODEL_NAME>` | OpenAI API 请求体里的 `model` 名称 | `qwen2.5-3b` |
| `<P_HOST>` | P 节点 hostname 或 IP | `gpu-node-p` |
| `<D_HOST>` | D 节点 hostname 或 IP | `gpu-node-d` |
| `<PROXY_HOST>` | proxy 所在机器的 hostname 或 IP | `127.0.0.1` |
| `<P_PORT>` | P 节点 OpenAI server 端口 | `8100` |
| `<D_PORT>` | D 节点 OpenAI server 端口 | `8200` |
| `<PROXY_PORT>` | proxy 端口 | `8000` |
| `<GPU_UTIL>` | vLLM 可占用 GPU 显存比例 | `0.6` |
| `<MAX_MODEL_LEN>` | vLLM 最大上下文长度 | `4096` |
| `<MAX_NUM_SEQS>` | vLLM 最大并发序列数 | `16` |

注意：前端测试和 replay 脚本里的 `--model` 应该填写 `<SERVED_MODEL_NAME>`，不是 `<MODEL_PATH>`。`<MODEL_PATH>` 只用于启动 vLLM server 或提交 Slurm job。

## 当前环境示例

这一节是当前实验环境的可复制参数。对外发布 README 时可以删除或替换。

```shell
export MODEL_PATH=/workplace/home/huangxinying/Models/Qwen2.5-3B/
export SERVED_MODEL_NAME=qwen2.5-3b
export MODEL_PATH=/workplace/home/huangxinying/Models/Llama-3.1-8B-Instruct
export SERVED_MODEL_NAME=llama-3.1-8b-instruct
export GPU_MEMORY_UTILIZATION=0.6
export MAX_MODEL_LEN=4096
export MAX_NUM_SEQS=16
```
