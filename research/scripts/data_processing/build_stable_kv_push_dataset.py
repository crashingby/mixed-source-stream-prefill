# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""离线生成稳定 KV 提前传输的长度受控性能集，不用于问答质量评价。"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)
SCHEMA_VERSION = "prompt_stream_mixed/v2"
DATA_REVISION = "stable_kv_push_controlled/v1"
CONTENT_MARKER = "VLLM_PROMPT_STREAM_CONTENT_7CDA3F6B"

# 使用多主题自然语句，避免重复单个 token 人为制造长上下文或生成死循环。
FACTS = (
    "A library catalogs new books and keeps several copies of popular titles.",
    "The river station records rainfall and compares measurements every morning.",
    "A greenhouse regulates sunlight, humidity, and temperature for young plants.",
    "Engineers inspect bridges after heavy storms and document structural changes.",
    "The school garden contains herbs, vegetables, and a small irrigation system.",
    "A museum preserves maps, photographs, and notes describing local history.",
    "Researchers compare observations before revising their working hypotheses.",
    "A coastal town maintains warning signs and evacuation routes near the harbor.",
    "The observatory publishes weather summaries alongside astronomical records.",
    "A railway timetable accounts for maintenance, passenger demand, and connections.",
    "Solar panels provide electricity while batteries store energy for cloudy days.",
    "A restoration team records material samples before repairing an old building.",
    "Farmers rotate crops to preserve soil nutrients and reduce seasonal disease.",
    "Water filters remove sediment before a laboratory checks chemical composition.",
    "An archive uses multiple indexes so visitors can locate related documents.",
    "The community center offers language classes, concerts, and public lectures.",
)


@dataclass(frozen=True)
class DatasetConfig:
    """保存性能集长度、版本变化和物理序列预算，不保存事件到达时间。"""

    tokenizer_model: str  # tokenizer 身份必须与实际回放模型保持一致。
    output_dir: Path  # 每种 Append 配比在该目录下建立独立子目录。
    num_requests: int = 128  # 跨场景保持相同 query_id 和随机种子。
    knowledge_tokens: int = 2048  # Append 与最终完整 Update 的 token 总长。
    append_tokens: tuple[int, ...] = (256, 1024, 1792)
    append_chunks: int = 2  # 固定事件数量，避免消息开销成为配比实验的混杂项。
    update_snapshots: int = 5  # 每代 Update 都是等长完整快照。
    update_lcp_ratio: float = 0.75  # 相邻快照精确 LCP 的目标比例。
    seed: int = 2026  # 只影响内容，不影响回放的到达过程。
    block_size: int = 16  # Prefix+Append 的块对齐 padding 预算。
    max_model_len: int = 4096  # 生成时拒绝超出该物理序列上限的样本。
    max_tokens: int = 64  # 最终 prompt 之外预留的完整输出预算。
    overwrite: bool = False  # 默认拒绝覆盖已有场景，保护之前实验输入。


def encode(tokenizer: Any, text: str) -> list[int]:
    """独立编码一个流式分段，不自动插入 BOS/EOS。"""
    return [int(value) for value in tokenizer.encode(text, add_special_tokens=False)]


def decode(tokenizer: Any, token_ids: list[int]) -> str:
    """保留 token 边界原文，禁止 decode 的空白清理改变实验长度。"""
    return str(
        tokenizer.decode(
            token_ids,
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )
    )


def fit_text(tokenizer: Any, text: str, num_tokens: int) -> str:
    """截取可重新编码为精确目标长度的文本，无法满足时明确拒绝。"""
    token_ids = encode(tokenizer, text)
    if len(token_ids) < num_tokens:
        raise ValueError(f"Payload has {len(token_ids)} tokens; need {num_tokens}.")
    # 首次通常直接成功；邻近切点覆盖 Unicode 或 tokenizer 非严格可逆的边界。
    for offset in [0, *range(1, 65), *range(-1, -65, -1)]:
        candidate = decode(tokenizer, token_ids[: max(0, num_tokens + offset)])
        if len(encode(tokenizer, candidate)) == num_tokens:
            return candidate
    raise ValueError(f"Cannot construct an exact {num_tokens}-token text boundary.")


def payload_text(num_tokens: int, seed: int, label: str) -> str:
    """构造充分长的多主题文档，段落编号与打乱顺序避免单句循环。"""
    random_state = random.Random(seed)  # 每个调用不修改进程全局随机状态。
    paragraphs: list[str] = []
    # ASCII 文本单词数超过 token 预算，覆盖常用 Qwen/Llama tokenizer。
    for index in range(num_tokens // 8 + 32):
        facts = random_state.sample(FACTS, 3)
        paragraphs.append(f"{label} record {index}: {' '.join(facts)}\n")
    return "".join(paragraphs)


def split_append(tokenizer: Any, text: str, num_chunks: int) -> list[str]:
    """在可独立重编码的 token 边界切块，保持事件 token 总长精确不变。"""
    token_ids = encode(tokenizer, text)
    chunks: list[str] = []
    start = 0  # 下一块的起始 token，不按字符数近似 token 长度。
    for chunk_index in range(num_chunks - 1):
        desired = round(len(token_ids) * (chunk_index + 1) / num_chunks)
        candidates = sorted(
            range(start + 1, len(token_ids) - (num_chunks - chunk_index - 1) + 1),
            key=lambda boundary: abs(boundary - desired),
        )
        for end in candidates:
            left_text = decode(tokenizer, token_ids[start:end])
            right_text = decode(tokenizer, token_ids[end:])
            if (
                encode(tokenizer, left_text) == token_ids[start:end]
                and encode(tokenizer, right_text) == token_ids[end:]
                and "".join(chunks) + left_text + right_text == text
            ):
                chunks.append(left_text)
                start = end
                break
        else:
            raise ValueError(
                f"Cannot split Append into {num_chunks} token-safe chunks."
            )
    chunks.append(decode(tokenizer, token_ids[start:]))
    if "".join(chunks) != text or sum(len(encode(tokenizer, c)) for c in chunks) != len(
        token_ids
    ):
        raise ValueError("Append splitting changed the transmitted token length.")
    return chunks


def lcp_length(previous: list[int], current: list[int]) -> int:
    """返回两个完整 Update 快照的 source-local LCP token 数。"""
    return next(
        (
            index
            for index, pair in enumerate(zip(previous, current))
            if pair[0] != pair[1]
        ),
        min(len(previous), len(current)),
    )


def update_texts(
    tokenizer: Any, num_tokens: int, num_snapshots: int, lcp_ratio: float, seed: int
) -> tuple[list[str], list[int]]:
    """生成等长快照，逐代验证实际 LCP 恰好等于 floor(length*ratio)。"""
    desired_lcp = math.floor(num_tokens * lcp_ratio)
    initial = fit_text(tokenizer, payload_text(num_tokens, seed, "Initial"), num_tokens)
    snapshots = [initial]
    previous = encode(tokenizer, initial)
    common_ids = previous[:desired_lcp]  # 共享前缀不因后续修订或调度而移动。
    common_text = decode(tokenizer, common_ids)
    actual_lcps: list[int] = []
    for version in range(1, num_snapshots):
        # 不同边界前导符避免 tokenizer 合并 shared prefix 最后一个 token。
        for separator in ("\n", " ", "; ", ": ", ". ", "! ", "? ", "\t"):
            body = payload_text(num_tokens, seed + version, f"Revision{version}")
            candidate = fit_text(tokenizer, common_text + separator + body, num_tokens)
            current = encode(tokenizer, candidate)
            if (
                current[:desired_lcp] == common_ids
                and lcp_length(previous, current) == desired_lcp
            ):
                snapshots.append(candidate)
                actual_lcps.append(desired_lcp)
                previous = current
                break
        else:
            raise ValueError(
                f"Cannot build snapshot={version} with tokens={num_tokens}, "
                f"exact_lcp={desired_lcp}; choose another ratio or tokenizer."
            )
    return snapshots, actual_lcps


def event_record(
    source_id: str, source_type: str, sequence_no: int, text: str | None
) -> dict[str, Any]:
    """创建 v2 数据事件或来源结束事件，保持 source 内序号连续。"""
    event: dict[str, Any] = {
        "source_id": source_id,
        "source_type": source_type,
        "event_type": source_type if text is not None else "source_finished",
        "chunk_id": f"{source_id}-{sequence_no}",
        "version": sequence_no if text is not None else max(0, sequence_no - 1),
        "sequence_no": sequence_no,
    }
    if text is not None:
        event["text"] = text  # 现有 renderer 读取 text 后再编码，不混放 token_ids。
    return {"event": event}


def build_sample(
    tokenizer: Any, config: DatasetConfig, query_index: int, num_append_tokens: int
) -> dict[str, Any]:
    """构造一条配比样本并校验 chat template、事件长度与物理 padding 预算。"""
    sample_id = f"anns-query-{query_index:06d}"  # 沿用回放 --query-id 的文件名约定。
    identity = hashlib.sha256(f"{config.seed}:{query_index}".encode()).hexdigest()[:16]
    system_message = (
        f"Request identifier {identity}. You are a research assistant. Read the "
        "provided evidence and produce a factual, concise multi-point summary."
    )
    question = "Summarize four different observations from the supplied evidence."
    finish_text = f"\nUser question: {question}\nAnswer:"
    num_update_tokens = config.knowledge_tokens - num_append_tokens
    append_text = fit_text(
        tokenizer,
        payload_text(num_append_tokens, config.seed + query_index, "Stable"),
        num_append_tokens,
    )
    append_parts = split_append(tokenizer, append_text, config.append_chunks)
    snapshots, actual_lcps = update_texts(
        tokenizer,
        num_update_tokens,
        config.update_snapshots,
        config.update_lcp_ratio,
        config.seed + query_index + 100000,
    )
    rendered = tokenizer.apply_chat_template(
        [
            {"role": "system", "content": system_message},
            {"role": "user", "content": CONTENT_MARKER},
        ],
        tokenize=False,
        add_generation_prompt=True,
    )
    if not isinstance(rendered, str) or rendered.count(CONTENT_MARKER) != 1:
        raise ValueError("Chat template must preserve the content marker exactly once.")
    prefix, suffix = rendered.split(CONTENT_MARKER)
    final_content = append_text + snapshots[-1] + finish_text
    full_prompt = tokenizer.apply_chat_template(
        [
            {"role": "system", "content": system_message},
            {"role": "user", "content": final_content},
        ],
        tokenize=False,
        add_generation_prompt=True,
    )
    if prefix + final_content + suffix != full_prompt:
        raise ValueError("Chat template cannot be split into a fixed envelope.")
    prefix_len = len(encode(tokenizer, prefix))
    question_len = len(encode(tokenizer, finish_text)) + len(encode(tokenizer, suffix))
    padding_len = (-(prefix_len + num_append_tokens)) % config.block_size
    logical_tokens = prefix_len + config.knowledge_tokens + question_len
    # 输出预算包含 P 的首 token，不能另加一次，物理 padding 则必须计入上限。
    required_tokens = logical_tokens + padding_len + config.max_tokens
    if required_tokens > config.max_model_len:
        raise ValueError(
            f"sample={sample_id} needs {required_tokens} physical sequence tokens "
            f"including output; --max-model-len={config.max_model_len}."
        )
    append_id, update_id = "controlled_append", "controlled_update"
    events = [
        event_record(append_id, "append", i, t) for i, t in enumerate(append_parts)
    ]
    events.append(event_record(append_id, "append", len(append_parts), None))
    events.extend(
        event_record(update_id, "update", i, t) for i, t in enumerate(snapshots)
    )
    events.append(event_record(update_id, "update", len(snapshots), None))
    return {
        "schema_version": SCHEMA_VERSION,
        "data_revision": DATA_REVISION,
        "sample_id": sample_id,
        "query_id": str(query_index),
        "question": question,
        "status": "complete",
        "start": {
            "api": "completions",
            "prompt": system_message,
            "stream": True,
            "max_tokens": config.max_tokens,
            "temperature": 0,
        },
        "sources": [
            {"source_id": append_id, "source_type": "append"},
            {"source_id": update_id, "source_type": "update"},
        ],
        "event_order": {
            "storage_layout": "grouped_by_source",
            "within_source": "sequence_no",
            "across_sources": "unspecified",
        },
        "events": events,
        "finish": {
            "after_source_ids": [append_id, update_id],
            "event": {
                "source_id": "final_question",
                "source_type": None,
                "event_type": "prompt_finished",
                "chunk_id": "final-question",
                "version": 0,
                "sequence_no": 0,
                "text": finish_text,
            },
        },
        "controlled_workload": {
            "knowledge_tokens": config.knowledge_tokens,
            "append_tokens": num_append_tokens,
            "append_chunk_tokens": [
                len(encode(tokenizer, part)) for part in append_parts
            ],
            "update_tokens": num_update_tokens,
            "update_lcp_tokens": actual_lcps,
            "append_fraction": num_append_tokens / config.knowledge_tokens,
            "prefix_tokens": prefix_len,
            "question_tokens": question_len,
            "padding_tokens": padding_len,
            "logical_prompt_tokens": logical_tokens,
            "physical_prompt_tokens": logical_tokens + padding_len,
            "required_sequence_tokens": required_tokens,
            "request_identifier": identity,
        },
        "provenance": {
            "kind": "synthetic_performance_only",
            "seed": config.seed,
            "tokenizer_model": config.tokenizer_model,
        },
    }


def validate_config(config: DatasetConfig) -> None:
    """在写文件前拒绝无效长度、全相同快照与不明确的覆盖范围。"""
    for name in (
        "num_requests",
        "knowledge_tokens",
        "append_chunks",
        "update_snapshots",
        "block_size",
        "max_model_len",
        "max_tokens",
    ):
        if getattr(config, name) <= 0:
            raise ValueError(f"{name} must be positive.")
    if not 0 <= config.update_lcp_ratio < 1:
        raise ValueError("--update-lcp-ratio must be in [0, 1).")
    if not config.append_tokens or len(set(config.append_tokens)) != len(
        config.append_tokens
    ):
        raise ValueError("--append-tokens must contain distinct scenarios.")
    for num_append_tokens in config.append_tokens:
        if not config.append_chunks <= num_append_tokens < config.knowledge_tokens:
            raise ValueError(
                "Each Append budget must fit nonempty chunks and leave Update."
            )


def run(config: DatasetConfig, tokenizer: Any) -> list[Path]:
    """生成独立场景目录及索引/长度汇总；不触碰原始混合数据或联网。"""
    validate_config(config)
    output_dirs = [
        config.output_dir
        / f"append-{length:04d}-update-{config.knowledge_tokens - length:04d}"
        for length in config.append_tokens
    ]
    # 覆盖只允许本生成器所属场景，禁止误把 SQuAD 质量集当作输出目录清理。
    for output_dir in output_dirs:
        if output_dir.exists() and any(output_dir.iterdir()):
            summary_path = output_dir / "dataset_summary.json"
            if not config.overwrite:
                raise FileExistsError(
                    f"Existing scenario {output_dir}; use --overwrite."
                )
            if (
                not summary_path.exists()
                or json.loads(summary_path.read_text()).get("data_revision")
                != DATA_REVISION
            ):
                raise ValueError(f"Refusing to overwrite foreign dataset {output_dir}.")
    for num_append_tokens, output_dir in zip(config.append_tokens, output_dirs):
        request_dir = output_dir / "requests"
        # 先构造全部样本，长度或边界失败时不留下半份请求集合。
        samples = [
            build_sample(tokenizer, config, i, num_append_tokens)
            for i in range(config.num_requests)
        ]
        request_dir.mkdir(parents=True, exist_ok=True)
        for obsolete in request_dir.glob("anns-query-*.json"):
            obsolete.unlink()  # 只删除同一场景的旧生成文件，避免减量重建留残余。
        manifests: list[str] = []
        for sample in samples:
            path = request_dir / f"{sample['sample_id']}.json"
            path.write_text(
                json.dumps(sample, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            manifests.append(
                json.dumps(
                    {
                        "sample_id": sample["sample_id"],
                        "query_id": sample["query_id"],
                        "request_path": f"requests/{path.name}",
                    }
                )
            )
        (output_dir / "manifest.jsonl").write_text(
            "\n".join(manifests) + "\n", encoding="utf-8"
        )
        settings = asdict(config)
        settings["output_dir"] = str(config.output_dir)
        summary = {
            "schema_version": SCHEMA_VERSION,
            "data_revision": DATA_REVISION,
            "num_samples": len(samples),
            "tokenizer_model": config.tokenizer_model,
            "seed": config.seed,
            "configuration": settings,
            "scenario": {
                "knowledge_tokens": config.knowledge_tokens,
                "append_tokens": num_append_tokens,
                "update_tokens": config.knowledge_tokens - num_append_tokens,
                "append_fraction": num_append_tokens / config.knowledge_tokens,
                "append_chunks": config.append_chunks,
                "update_snapshots": config.update_snapshots,
                "update_lcp_ratio": config.update_lcp_ratio,
            },
            "length_bounds": {
                name: {
                    "min": min(s["controlled_workload"][name] for s in samples),
                    "max": max(s["controlled_workload"][name] for s in samples),
                }
                for name in (
                    "logical_prompt_tokens",
                    "physical_prompt_tokens",
                    "required_sequence_tokens",
                    "padding_tokens",
                )
            },
            "purpose": "Controlled transfer performance only; no gold answers.",
        }
        (output_dir / "dataset_summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        logger.info("Generated %d samples in %s", len(samples), output_dir)
    return output_dirs


def parse_args() -> DatasetConfig:
    """解析长度配比和生成选项，事件间隔仍完全由 replay 控制。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--tokenizer-model",
        required=True,
        help="本地 tokenizer 路径；不联网且必须具备 chat template。",
    )
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--num-requests", type=int, default=128)
    parser.add_argument("--knowledge-tokens", type=int, default=2048)
    parser.add_argument(
        "--append-tokens", nargs="+", type=int, default=[256, 1024, 1792]
    )
    parser.add_argument("--append-chunks", type=int, default=2)
    parser.add_argument("--update-snapshots", type=int, default=5)
    parser.add_argument("--update-lcp-ratio", type=float, default=0.75)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--overwrite", action="store_true")
    parsed = vars(parser.parse_args())
    parsed["append_tokens"] = tuple(parsed["append_tokens"])
    return DatasetConfig(**parsed)


def main() -> None:
    """加载本地 tokenizer 并生成所有场景，不启动推理服务或访问网络。"""
    from transformers import AutoTokenizer

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s"
    )
    config = parse_args()
    validate_config(config)
    tokenizer = AutoTokenizer.from_pretrained(
        config.tokenizer_model, local_files_only=True
    )
    run(config, tokenizer)


if __name__ == "__main__":
    main()
