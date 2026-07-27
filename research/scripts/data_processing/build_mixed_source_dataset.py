# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""从 Stream2LLM ANNS trace 构造 Append/Update 混合来源数据。

脚本以 ANNS 原始问题为查询：

1. 将每一行 ``PipelinePool`` 物化为完整的 Update 文本快照；
2. 使用 DDGS 检索相同问题，并用 Trafilatura 抽取网页正文；
3. 分别保存 ANNS Update 和网页 Append 的来源内事件序列；
4. 输出与 prompt-stream RESTFUL 事件同构的一请求一 JSON 数据。

数据集不保存真实检索耗时或事件间隔。跨来源交错顺序和发送等待时间由后续
实验回放脚本控制。网页搜索和抓取结果会按 query 缓存在输出目录的
``.cache/web`` 中，已有请求默认跳过，因此中断后可以直接续跑。
"""

from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import ipaddress
import json
import logging
import os
import random
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal
from urllib.parse import SplitResult, urlsplit, urlunsplit

import httpx
import trafilatura
from ddgs import DDGS

logger = logging.getLogger(__name__)

SCHEMA_VERSION = "prompt_stream_mixed/v2"
WEB_CACHE_SCHEMA_VERSION = "mixed_source_web_cache/v2"
ANNS_SOURCE_TYPE: Literal["update"] = "update"
WEB_SOURCE_TYPE: Literal["append"] = "append"
DEFAULT_PREFIX = (
    "You are a helpful assistant. Read the streamed retrieval context first. "
    "Some sources append stable evidence, while other sources publish mutable "
    "snapshots. Answer the final user question using the latest available "
    "evidence.\n\n"
)
DEFAULT_USER_AGENT = (
    "vLLM-MixedSource-DatasetBuilder/1.0 "
    "(research data collection; contact the dataset maintainer)"
)
HTML_CONTENT_TYPES = (
    "text/html",
    "application/xhtml+xml",
    "text/plain",
)

SourceType = Literal["append", "update"]


@dataclass(frozen=True)
class AnnsTraceRow:
    """一条 ANNS PipelinePool 完整快照。

    Args:
        row_index: CSV 中从零开始的原始行号。
        start_iteration: 当前区间的起始迭代。
        end_iteration: 当前区间的结束迭代。
        document_ids: PipelinePool 中按原顺序排列的文档 id。
    """

    row_index: int
    start_iteration: int
    end_iteration: int
    document_ids: tuple[str, ...]


@dataclass(frozen=True)
class SearchHit:
    """DDGS 返回的一条候选网页。"""

    rank: int
    title: str
    url: str
    snippet: str


@dataclass(frozen=True)
class WebPage:
    """成功抓取并抽取正文的一条网页结果。"""

    rank: int
    title: str
    url: str
    final_url: str
    snippet: str
    text: str
    status_code: int
    content_sha256: str


@dataclass(frozen=True)
class WebFailure:
    """一条未能形成 Append chunk 的网页抓取结果。"""

    rank: int
    title: str
    url: str
    snippet: str
    reason: str


@dataclass(frozen=True)
class WebCollection:
    """一个问题对应的完整在线网页采集结果。"""

    query: str
    search_provider: str
    configuration: dict[str, Any]
    search_hits: int
    pages: tuple[WebPage, ...]
    failures: tuple[WebFailure, ...]


def parse_args() -> argparse.Namespace:
    """解析数据生成参数。

    Returns:
        命令行参数命名空间。
    """
    parser = argparse.ArgumentParser(
        description=(
            "Build prompt-stream mixed-source samples from ANNS traces and "
            "fresh web search results."
        )
    )
    parser.add_argument(
        "--anns-dir",
        type=Path,
        default=Path("research/data/anns"),
        help="Directory containing query_trace_map_5k.json, res/, and corpus.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("research/data/mix_source"),
        help="Dataset output directory.",
    )
    parser.add_argument(
        "--query-id",
        action="append",
        dest="query_ids",
        default=None,
        help="ANNS query id to build. Repeat this option to select several ids.",
    )
    parser.add_argument(
        "--start-index",
        type=int,
        default=0,
        help="Start position in numerically sorted query ids.",
    )
    parser.add_argument(
        "--max-queries",
        type=int,
        default=1,
        help=(
            "Maximum queries to build when --query-id is omitted. Use 0 for "
            "all remaining queries; default 1 prevents accidental mass crawl."
        ),
    )
    parser.add_argument(
        "--max-web-results",
        type=int,
        default=5,
        help="Maximum DDGS search results fetched for each question.",
    )
    parser.add_argument(
        "--min-successful-web-pages",
        type=int,
        default=1,
        help="Minimum extracted pages required for a sample to be complete.",
    )
    parser.add_argument(
        "--fetch-workers",
        type=int,
        default=4,
        help="Maximum concurrent page downloads within one query.",
    )
    parser.add_argument(
        "--search-timeout",
        type=float,
        default=20.0,
        help="DDGS search timeout in seconds.",
    )
    parser.add_argument(
        "--fetch-timeout",
        type=float,
        default=20.0,
        help="Timeout for one page HTTP request in seconds.",
    )
    parser.add_argument(
        "--max-search-retries",
        type=int,
        default=2,
        help="Additional DDGS attempts after the initial search fails.",
    )
    parser.add_argument(
        "--max-fetch-retries",
        type=int,
        default=1,
        help="Additional HTTP attempts after the initial page fetch fails.",
    )
    parser.add_argument(
        "--retry-backoff",
        type=float,
        default=1.0,
        help="Initial exponential retry backoff in seconds.",
    )
    parser.add_argument(
        "--query-delay",
        type=float,
        default=1.0,
        help=(
            "Collection-only delay between live searches in seconds; never "
            "stored as an experiment event interval."
        ),
    )
    parser.add_argument(
        "--region",
        type=str,
        default="wt-wt",
        help="DDGS search region.",
    )
    parser.add_argument(
        "--safesearch",
        choices=("on", "moderate", "off"),
        default="moderate",
        help="DDGS SafeSearch mode.",
    )
    parser.add_argument(
        "--search-backend",
        type=str,
        default="auto",
        help="DDGS text-search backend.",
    )
    parser.add_argument(
        "--max-page-bytes",
        type=int,
        default=5 * 1024 * 1024,
        help="Reject a response after this many downloaded bytes.",
    )
    parser.add_argument(
        "--min-web-chars",
        type=int,
        default=200,
        help="Reject extracted page text shorter than this many characters.",
    )
    parser.add_argument(
        "--max-web-chars",
        type=int,
        default=12_000,
        help="Truncate each extracted web page to this many characters; 0 disables.",
    )
    parser.add_argument(
        "--tokenizer-model",
        type=str,
        default=None,
        help=(
            "Tokenizer path or HuggingFace repo used for token-level truncation. "
            "Required when any --max-*-tokens content budget is enabled."
        ),
    )
    parser.add_argument(
        "--max-web-chunk-tokens",
        type=int,
        default=0,
        help=(
            "Truncate each formatted web Append chunk to this many tokenizer "
            "tokens; 0 disables token-level web truncation."
        ),
    )
    parser.add_argument(
        "--max-anns-document-tokens",
        type=int,
        default=0,
        help=(
            "Truncate each formatted ANNS document inside Update snapshots to "
            "this many tokenizer tokens; 0 disables token-level document "
            "truncation."
        ),
    )
    parser.add_argument(
        "--max-update-snapshot-tokens",
        type=int,
        default=0,
        help=(
            "Truncate each complete ANNS Update snapshot to this many tokenizer "
            "tokens after per-document formatting; 0 disables snapshot-level "
            "truncation."
        ),
    )
    parser.add_argument(
        "--max-anns-document-chars",
        type=int,
        default=0,
        help="Truncate each ANNS document to this many characters; 0 disables.",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=64,
        help="Default decode max_tokens stored in each start request.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Default decode temperature stored in each start request.",
    )
    parser.add_argument(
        "--prefix",
        type=str,
        default=DEFAULT_PREFIX,
        help="Fixed prompt prefix stored in each start request.",
    )
    parser.add_argument(
        "--offline",
        action="store_true",
        help="Never access the network; require an existing per-query web cache.",
    )
    parser.add_argument(
        "--only-cached-web",
        action="store_true",
        help=(
            "Only build queries whose per-query web cache already exists. This "
            "is useful after a partial crawl when rebuilding short request JSON."
        ),
    )
    parser.add_argument(
        "--refresh-web",
        action="store_true",
        help=("Ignore web caches and search/fetch again; requires --overwrite."),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Rebuild request JSON files that already exist.",
    )
    parser.add_argument(
        "--require-web",
        action="store_true",
        help=("Fail when successful pages are below --min-successful-web-pages."),
    )
    parser.add_argument(
        "--fail-fast",
        action="store_true",
        help="Stop at the first query failure.",
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=0,
        help="Seed used only for deterministic retry jitter.",
    )
    parser.add_argument(
        "--user-agent",
        type=str,
        default=DEFAULT_USER_AGENT,
        help="User-Agent used for page downloads.",
    )
    parser.add_argument(
        "--log-level",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
        default="INFO",
        help="Logging verbosity.",
    )
    return parser.parse_args()


def _atomic_write_text(path: Path, text: str) -> None:
    """原子写入文本文件，避免中断留下半个 JSON。

    Args:
        path: 最终文件路径。
        text: 待写入的完整文本。
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary_path.write_text(text, encoding="utf-8")
    os.replace(temporary_path, path)


def _atomic_write_json(path: Path, value: Any) -> None:
    """以稳定格式原子写入 JSON。

    Args:
        path: 最终 JSON 文件路径。
        value: 可 JSON 序列化的对象。
    """
    serialized = json.dumps(
        value,
        ensure_ascii=False,
        indent=2,
        sort_keys=False,
    )
    _atomic_write_text(path, f"{serialized}\n")


def _load_json(path: Path) -> Any:
    """读取 UTF-8 JSON 文件。"""
    return json.loads(path.read_text(encoding="utf-8"))


def _numeric_query_sort_key(query_id: str) -> tuple[int, int | str]:
    """让纯数字 query id 按数值排序，其它 id 按字符串排序。"""
    if query_id.isdigit():
        return (0, int(query_id))
    return (1, query_id)


def _select_query_ids(
    query_map: dict[str, dict[str, str]],
    args: argparse.Namespace,
) -> list[str]:
    """根据显式 id 或切片参数选择本次处理的问题。

    Args:
        query_map: ANNS query-to-trace 映射。
        args: 命令行参数。

    Returns:
        按稳定顺序排列的 query id。

    Raises:
        ValueError: 参数越界或显式 query id 不存在。
    """
    if args.query_ids:
        missing_ids = [
            query_id for query_id in args.query_ids if query_id not in query_map
        ]
        if missing_ids:
            raise ValueError(f"Unknown ANNS query ids: {missing_ids}.")
        return list(dict.fromkeys(args.query_ids))

    if args.start_index < 0:
        raise ValueError("--start-index must be non-negative.")
    if args.max_queries < 0:
        raise ValueError("--max-queries must be non-negative.")

    query_ids = sorted(query_map, key=_numeric_query_sort_key)
    query_ids = query_ids[args.start_index :]
    if args.max_queries:
        query_ids = query_ids[: args.max_queries]
    return query_ids


def _parse_pipeline_pool(raw_pool: str) -> tuple[str, ...]:
    """解析 ANNS ``PipelinePool``，包括空池和单元素池。

    Args:
        raw_pool: CSV 中形如 ``(12,34)``、``(12)`` 或 ``()`` 的字段。

    Returns:
        保持原始顺序的字符串文档 id。

    Raises:
        ValueError: 字段不是合法的整数 id 序列。
    """
    normalized = raw_pool.strip()
    if normalized in ("", "()"):
        return ()

    # 优先按逗号解析，避免 Python 把 ``(123)`` 当成整数而不是单元素 tuple。
    inner = normalized
    if normalized.startswith("(") and normalized.endswith(")"):
        inner = normalized[1:-1]
    parts = [part.strip() for part in inner.split(",") if part.strip()]
    if parts and all(re.fullmatch(r"[+-]?\d+", part) for part in parts):
        return tuple(str(int(part)) for part in parts)

    # 保留兼容性：若未来 trace 改成 JSON/Python list，仍给出清晰的解析结果。
    try:
        parsed = ast.literal_eval(normalized)
    except (SyntaxError, ValueError) as exc:
        raise ValueError(f"Invalid PipelinePool value: {raw_pool!r}.") from exc
    if isinstance(parsed, int):
        return (str(parsed),)
    if not isinstance(parsed, (list, tuple)):
        raise ValueError(f"Invalid PipelinePool value: {raw_pool!r}.")
    return tuple(str(int(document_id)) for document_id in parsed)


def _load_trace_rows(anns_dir: Path, trace_file: str) -> list[AnnsTraceRow]:
    """读取一个问题的全部 ANNS Update 快照。

    Args:
        anns_dir: ANNS 数据根目录。
        trace_file: query map 中记录的 trace 文件名。

    Returns:
        按 CSV 原始顺序排列的快照。真实时间列不写入生成数据。
    """
    trace_path = anns_dir / "res" / trace_file
    rows: list[AnnsTraceRow] = []
    with trace_path.open(encoding="utf-8", newline="") as trace_stream:
        reader = csv.DictReader(trace_stream)
        for row_index, row in enumerate(reader):
            rows.append(
                AnnsTraceRow(
                    row_index=row_index,
                    start_iteration=int(row["StartIteration"]),
                    end_iteration=int(row["EndIteration"]),
                    document_ids=_parse_pipeline_pool(row["PipelinePool"]),
                )
            )

    return rows


def _find_corpus_shards(anns_dir: Path) -> list[Path]:
    """只选择两套重复 corpus 中的一套。

    Args:
        anns_dir: ANNS 数据根目录。

    Returns:
        排序后的四个 corpus shard 路径。
    """
    shards = sorted(anns_dir.glob("retrieved_corpus_content.part.*.json"))
    if not shards:
        shards = sorted(anns_dir.glob("retrieved_corpus_content.[0-9]*.json"))
    if not shards:
        raise FileNotFoundError(f"No ANNS corpus shards found in {anns_dir}.")
    return shards


def _load_required_documents(
    anns_dir: Path,
    required_document_ids: set[str],
) -> dict[str, str]:
    """从 corpus shard 中只保留本批问题实际引用的文档。

    Args:
        anns_dir: ANNS 数据根目录。
        required_document_ids: 所有待处理 trace 引用的文档 id。

    Returns:
        文档 id 到原始正文的映射。

    Raises:
        KeyError: corpus 未覆盖全部 trace 引用。
    """
    remaining_ids = set(required_document_ids)
    documents: dict[str, str] = {}
    for shard_path in _find_corpus_shards(anns_dir):
        logger.info("Loading required documents from %s.", shard_path)
        shard = _load_json(shard_path)
        for document_id in remaining_ids.intersection(shard):
            documents[document_id] = str(shard[document_id])
        remaining_ids.difference_update(shard)
        if not remaining_ids:
            break

    if remaining_ids:
        example_ids = sorted(remaining_ids)[:10]
        raise KeyError(
            f"ANNS corpus is missing {len(remaining_ids)} document ids, "
            f"for example {example_ids}."
        )
    return documents


def _normalize_text(text: str) -> str:
    """压缩行内空白，同时保留正文段落边界。"""
    normalized_lines = []
    for line in text.replace("\x00", " ").splitlines():
        normalized_line = re.sub(r"\s+", " ", line).strip()
        if normalized_line:
            normalized_lines.append(normalized_line)
    return "\n".join(normalized_lines)


def _truncate_text(text: str, max_chars: int) -> str:
    """按字符上限截断文本，并显式标记截断。"""
    if max_chars <= 0 or len(text) <= max_chars:
        return text
    marker = "\n[truncated]"
    content_length = max(0, max_chars - len(marker))
    return f"{text[:content_length].rstrip()}{marker}"


def _load_tokenizer(tokenizer_model: str | None) -> Any | None:
    """按需加载 tokenizer，用于离线构造更短的实验数据。

    Args:
        tokenizer_model: 本地 tokenizer 路径或 HuggingFace repo 名称。

    Returns:
        transformers tokenizer；未启用 token 预算时返回 ``None``。
    """
    if tokenizer_model is None:
        return None

    # 延迟导入，避免只做字符裁剪或复用旧数据时强制依赖 transformers。
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(
        tokenizer_model,
        trust_remote_code=True,
    )


def _encode_text(tokenizer: Any, text: str) -> list[int]:
    """用运行时 tokenizer 计算不含 special token 的 token 序列。"""
    return list(tokenizer.encode(text, add_special_tokens=False))


def _decode_token_ids(tokenizer: Any, token_ids: list[int]) -> str:
    """把截断后的 token 序列还原为文本。"""
    return str(
        tokenizer.decode(
            token_ids,
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )
    )


def _truncate_text_by_tokens(
    text: str,
    max_tokens: int,
    tokenizer: Any | None,
) -> str:
    """按 tokenizer token 上限截断文本，并保留显式截断标记。

    Args:
        text: 待截断文本。
        max_tokens: 最大 token 数；小于等于零表示不启用。
        tokenizer: 运行时 tokenizer。

    Returns:
        不超过 token 预算的文本。若预算太小，会只保留截断标记自身。
    """
    if max_tokens <= 0:
        return text
    if tokenizer is None:
        raise ValueError("--tokenizer-model is required for token budgets.")

    token_ids = _encode_text(tokenizer, text)
    if len(token_ids) <= max_tokens:
        return text

    marker = "\n[truncated]"
    marker_ids = _encode_text(tokenizer, marker)
    if len(marker_ids) >= max_tokens:
        return _decode_token_ids(tokenizer, marker_ids[:max_tokens]).rstrip()
    content_budget = max(0, max_tokens - len(marker_ids))
    truncated_ids = token_ids[:content_budget] + marker_ids
    return _decode_token_ids(tokenizer, truncated_ids).rstrip()


def _truncate_wrapped_body_by_tokens(
    header: str,
    body: str,
    footer: str,
    max_tokens: int,
    tokenizer: Any | None,
) -> str:
    """在固定包装文本内裁剪正文，使整个 chunk 尽量满足 token 预算。"""
    if max_tokens <= 0:
        return f"{header}{body}{footer}"
    if tokenizer is None:
        raise ValueError("--tokenizer-model is required for token budgets.")

    wrapper_tokens = len(_encode_text(tokenizer, f"{header}{footer}"))
    body_budget = max(0, max_tokens - wrapper_tokens)
    if body_budget == 0:
        truncated_body = ""
    else:
        truncated_body = _truncate_text_by_tokens(
            body,
            body_budget,
            tokenizer,
        )
    return f"{header}{truncated_body}{footer}"


def _limit_joined_text_by_tokens(
    parts: list[str],
    separator: str,
    max_tokens: int,
    tokenizer: Any | None,
) -> str:
    """按 token 预算拼接若干文本片段，保留能放下的前缀片段。

    这个函数用于 Update 快照整体预算。它优先保留完整文档片段；当第一个或
    下一个文档超过剩余预算时，只截断该文档的前缀，避免继续塞入更多来源。
    """
    if not parts:
        return ""
    if max_tokens <= 0:
        return separator.join(parts)
    if tokenizer is None:
        raise ValueError("--tokenizer-model is required for token budgets.")

    selected_parts: list[str] = []
    used_tokens = 0
    separator_tokens = len(_encode_text(tokenizer, separator))
    for part in parts:
        part_tokens = len(_encode_text(tokenizer, part))
        extra_separator_tokens = separator_tokens if selected_parts else 0
        required_tokens = extra_separator_tokens + part_tokens
        if used_tokens + required_tokens <= max_tokens:
            selected_parts.append(part)
            used_tokens += required_tokens
            continue

        remaining_tokens = max_tokens - used_tokens - extra_separator_tokens
        if remaining_tokens > 0:
            selected_parts.append(
                _truncate_text_by_tokens(
                    part,
                    remaining_tokens,
                    tokenizer,
                )
            )
        break

    if not selected_parts:
        return _truncate_text_by_tokens(parts[0], max_tokens, tokenizer)
    return separator.join(selected_parts)


def _format_anns_document(
    document_id: str,
    text: str,
    max_document_chars: int,
    max_document_tokens: int,
    tokenizer: Any | None,
) -> str:
    """把单个 ANNS 文档格式化为稳定的快照片段。

    文档标签只包含稳定 id，不包含 snapshot 版本或动态排名，避免无谓破坏
    Update 源内 LCP。
    """
    normalized_text = _truncate_text(
        _normalize_text(text),
        max_document_chars,
    )
    header = f'<retrieved_document id="{document_id}">\n'
    footer = "\n</retrieved_document>"
    return _truncate_wrapped_body_by_tokens(
        header=header,
        body=normalized_text,
        footer=footer,
        max_tokens=max_document_tokens,
        tokenizer=tokenizer,
    )


def _materialize_anns_snapshot(
    row: AnnsTraceRow,
    documents: dict[str, str],
    max_document_chars: int,
    max_document_tokens: int,
    max_snapshot_tokens: int,
    tokenizer: Any | None,
) -> str:
    """按 PipelinePool 原顺序物化完整 Update 快照。

    空 PipelinePool 返回空字符串。它不是坏数据，而是“清空当前 Update
    source”的合法事件。
    """
    formatted_documents = [
        _format_anns_document(
            document_id,
            documents[document_id],
            max_document_chars,
            max_document_tokens,
            tokenizer,
        )
        for document_id in row.document_ids
    ]
    return _limit_joined_text_by_tokens(
        parts=formatted_documents,
        separator="\n\n",
        max_tokens=max_snapshot_tokens,
        tokenizer=tokenizer,
    )


def _canonicalize_url(raw_url: str) -> str:
    """规范化公开 HTTP(S) URL，并拒绝明显的本地地址。

    Args:
        raw_url: 搜索结果中的 URL。

    Returns:
        去除 fragment 后的 URL。

    Raises:
        ValueError: URL scheme/host 非法或指向本地保留地址。
    """
    parsed = urlsplit(raw_url.strip())
    if parsed.scheme.lower() not in ("http", "https"):
        raise ValueError("unsupported_url_scheme")
    if not parsed.hostname:
        raise ValueError("missing_url_host")
    if parsed.username is not None or parsed.password is not None:
        raise ValueError("url_credentials_not_allowed")

    hostname = parsed.hostname.lower()
    if hostname == "localhost" or hostname.endswith(".localhost"):
        raise ValueError("local_url_not_allowed")
    try:
        address = ipaddress.ip_address(hostname)
    except ValueError:
        pass
    else:
        if (
            address.is_private
            or address.is_loopback
            or address.is_link_local
            or address.is_multicast
            or address.is_reserved
            or address.is_unspecified
        ):
            raise ValueError("non_public_ip_not_allowed")

    host = hostname
    if parsed.port is not None:
        is_default_port = (parsed.scheme.lower() == "http" and parsed.port == 80) or (
            parsed.scheme.lower() == "https" and parsed.port == 443
        )
        if not is_default_port:
            host = f"{host}:{parsed.port}"
    normalized = SplitResult(
        scheme=parsed.scheme.lower(),
        netloc=host,
        path=parsed.path or "/",
        query=parsed.query,
        fragment="",
    )
    return urlunsplit(normalized)


def _web_collection_config(args: argparse.Namespace) -> dict[str, Any]:
    """返回会影响网页 chunk 内容的缓存配置。"""
    return {
        "max_web_results": args.max_web_results,
        "region": args.region,
        "safesearch": args.safesearch,
        "search_backend": args.search_backend,
        "max_page_bytes": args.max_page_bytes,
        "min_web_chars": args.min_web_chars,
        "max_web_chars": args.max_web_chars,
    }


def _search_web(
    query: str,
    args: argparse.Namespace,
    rng: random.Random,
) -> list[SearchHit]:
    """调用 DDGS，并对临时搜索失败进行有限重试。

    Args:
        query: ANNS 原始问题。
        args: 命令行参数。
        rng: 确定性 retry jitter 随机源。

    Returns:
        按搜索排名排列、URL 去重后的候选网页。

    Raises:
        RuntimeError: 全部搜索尝试均失败。
    """
    last_error: Exception | None = None
    for attempt in range(args.max_search_retries + 1):
        try:
            results = DDGS(timeout=args.search_timeout).text(
                query,
                region=args.region,
                safesearch=args.safesearch,
                max_results=args.max_web_results,
                backend=args.search_backend,
            )
            hits: list[SearchHit] = []
            seen_urls: set[str] = set()
            for result in results:
                raw_url = str(result.get("href") or result.get("url") or "")
                try:
                    url = _canonicalize_url(raw_url)
                except ValueError as exc:
                    logger.debug("Skipping search URL %r: %s.", raw_url, exc)
                    continue
                if url in seen_urls:
                    continue
                seen_urls.add(url)
                hits.append(
                    SearchHit(
                        rank=len(hits),
                        title=str(result.get("title") or ""),
                        url=url,
                        snippet=str(result.get("body") or ""),
                    )
                )
            return hits
        except Exception as exc:
            last_error = exc
            if attempt >= args.max_search_retries:
                break
            delay = args.retry_backoff * (2**attempt)
            delay += rng.uniform(0, max(0.0, args.retry_backoff * 0.2))
            logger.warning(
                "Web search attempt %d failed: %s; retrying in %.2fs.",
                attempt + 1,
                exc,
                delay,
            )
            time.sleep(delay)
    raise RuntimeError(f"DDGS search failed for query {query!r}.") from last_error


def _read_response_bytes(
    response: httpx.Response,
    max_page_bytes: int,
) -> bytes:
    """流式读取响应，并在页面过大时立即停止。"""
    content_length = response.headers.get("content-length")
    if content_length is not None:
        try:
            declared_length = int(content_length)
        except ValueError:
            declared_length = None
        if declared_length is not None and declared_length > max_page_bytes:
            raise ValueError("page_too_large")

    chunks: list[bytes] = []
    total_bytes = 0
    for chunk in response.iter_bytes():
        total_bytes += len(chunk)
        if total_bytes > max_page_bytes:
            raise ValueError("page_too_large")
        chunks.append(chunk)
    return b"".join(chunks)


def _extract_page_text(
    html: str,
    final_url: str,
    args: argparse.Namespace,
) -> str:
    """使用 Trafilatura 提取并裁剪网页正文。"""
    extracted = trafilatura.extract(
        html,
        url=final_url,
        output_format="txt",
        include_comments=False,
        include_tables=True,
        deduplicate=True,
        favor_precision=True,
    )
    if extracted is None:
        raise ValueError("no_extractable_content")
    normalized = _normalize_text(extracted)
    if len(normalized) < args.min_web_chars:
        raise ValueError("extracted_content_too_short")
    return _truncate_text(normalized, args.max_web_chars)


def _fetch_search_hit(
    hit: SearchHit,
    args: argparse.Namespace,
) -> WebPage | WebFailure:
    """下载并解析一个搜索结果。

    Args:
        hit: 规范化后的搜索结果。
        args: 命令行参数。

    Returns:
        成功的网页正文，或带结构化原因的失败记录。数据中不保存抓取耗时。
    """
    last_error: Exception | None = None
    for attempt in range(args.max_fetch_retries + 1):
        try:
            with (
                httpx.Client(
                    follow_redirects=True,
                    max_redirects=5,
                    timeout=args.fetch_timeout,
                    headers={"User-Agent": args.user_agent},
                ) as client,
                client.stream("GET", hit.url) as response,
            ):
                response.raise_for_status()
                content_type = response.headers.get(
                    "content-type",
                    "",
                ).lower()
                if content_type and not any(
                    content_type.startswith(allowed_type)
                    for allowed_type in HTML_CONTENT_TYPES
                ):
                    raise ValueError("unsupported_content_type")
                final_url = _canonicalize_url(str(response.url))
                body = _read_response_bytes(
                    response,
                    args.max_page_bytes,
                )
                encoding = response.encoding or "utf-8"
                html = body.decode(encoding, errors="replace")
                text = _extract_page_text(html, final_url, args)
                status_code = response.status_code

            return WebPage(
                rank=hit.rank,
                title=hit.title,
                url=hit.url,
                final_url=final_url,
                snippet=hit.snippet,
                text=text,
                status_code=status_code,
                content_sha256=hashlib.sha256(text.encode("utf-8")).hexdigest(),
            )
        except Exception as exc:
            last_error = exc
            if attempt >= args.max_fetch_retries:
                break
            time.sleep(args.retry_backoff * (2**attempt))

    reason = type(last_error).__name__
    if isinstance(last_error, ValueError) and str(last_error):
        reason = str(last_error)
    elif isinstance(last_error, httpx.HTTPStatusError):
        reason = f"http_status_{last_error.response.status_code}"
    elif last_error is not None and str(last_error):
        reason = f"{type(last_error).__name__}: {last_error}"
    return WebFailure(
        rank=hit.rank,
        title=hit.title,
        url=hit.url,
        snippet=hit.snippet,
        reason=reason,
    )


def _collect_web(
    query: str,
    args: argparse.Namespace,
    rng: random.Random,
) -> WebCollection:
    """搜索并并发抓取一个问题的网页结果。

    并发只用于加快数据构造。最终 Append 顺序固定为搜索排名，与网页真实
    完成时间无关，因此相同缓存可以得到确定性的 source-local 事件序列。
    """
    hits = _search_web(query, args, rng)
    pages: list[WebPage] = []
    failures: list[WebFailure] = []

    with ThreadPoolExecutor(max_workers=args.fetch_workers) as executor:
        futures = [
            executor.submit(
                _fetch_search_hit,
                hit,
                args,
            )
            for hit in hits
        ]
        for future in as_completed(futures):
            result = future.result()
            if isinstance(result, WebPage):
                pages.append(result)
            else:
                failures.append(result)

    # 并发完成顺序不进入数据集；先恢复搜索排名，再按正文 hash 去重。
    pages.sort(key=lambda page: page.rank)
    unique_pages: list[WebPage] = []
    seen_content_hashes: set[str] = set()
    for page in pages:
        if page.content_sha256 in seen_content_hashes:
            failures.append(
                WebFailure(
                    rank=page.rank,
                    title=page.title,
                    url=page.url,
                    snippet=page.snippet,
                    reason="duplicate_extracted_content",
                )
            )
            continue
        seen_content_hashes.add(page.content_sha256)
        unique_pages.append(page)

    failures.sort(key=lambda failure: failure.rank)
    return WebCollection(
        query=query,
        search_provider="ddgs",
        configuration=_web_collection_config(args),
        search_hits=len(hits),
        pages=tuple(unique_pages),
        failures=tuple(failures),
    )


def _serialize_web_collection(collection: WebCollection) -> dict[str, Any]:
    """把网页采集结果转成不含时间字段的缓存对象。"""
    return {
        "schema_version": WEB_CACHE_SCHEMA_VERSION,
        "query": collection.query,
        "search_provider": collection.search_provider,
        "configuration": collection.configuration,
        "search_hits": collection.search_hits,
        "pages": [asdict(page) for page in collection.pages],
        "failures": [asdict(failure) for failure in collection.failures],
    }


def _deserialize_web_collection(
    value: dict[str, Any],
    configuration: dict[str, Any],
) -> WebCollection:
    """从新旧 query 级缓存恢复网页结果，并忽略 v1 时间字段。"""
    pages = tuple(
        WebPage(
            rank=int(page["rank"]),
            title=str(page["title"]),
            url=str(page["url"]),
            final_url=str(page["final_url"]),
            snippet=str(page["snippet"]),
            text=str(page["text"]),
            status_code=int(page["status_code"]),
            content_sha256=str(page["content_sha256"]),
        )
        for page in value["pages"]
    )
    failures = tuple(
        WebFailure(
            rank=int(failure["rank"]),
            title=str(failure["title"]),
            url=str(failure["url"]),
            snippet=str(failure["snippet"]),
            reason=str(failure["reason"]),
        )
        for failure in value["failures"]
    )
    return WebCollection(
        query=str(value["query"]),
        search_provider=str(value["search_provider"]),
        configuration=configuration,
        search_hits=int(value["search_hits"]),
        pages=pages,
        failures=failures,
    )


def _get_web_collection(
    query_id: str,
    question: str,
    args: argparse.Namespace,
    rng: random.Random,
) -> WebCollection:
    """优先读取 query 级网页缓存，否则执行在线采集。"""
    cache_path = args.output_dir / ".cache" / "web" / f"query_{query_id}.json"
    expected_config = _web_collection_config(args)
    if cache_path.exists() and not args.refresh_web:
        cached_value = _load_json(cache_path)
        cached_config = cached_value.get("configuration")
        if cached_config is not None and cached_config != expected_config:
            raise ValueError(
                f"Web cache configuration mismatch for query {query_id}. "
                "Use the original options or pass --refresh-web --overwrite."
            )
        cached = _deserialize_web_collection(cached_value, expected_config)
        if cached.query != question:
            raise ValueError(
                f"Cached query text does not match ANNS query id {query_id}."
            )
        # 读取 v1 缓存时立即迁移，确保 output-dir 中也不再保留时间字段。
        if cached_value.get("schema_version") != WEB_CACHE_SCHEMA_VERSION:
            _atomic_write_json(cache_path, _serialize_web_collection(cached))
        logger.info("Using cached web collection for query %s.", query_id)
        return cached
    if args.offline:
        raise FileNotFoundError(f"Offline mode requires web cache {cache_path}.")

    collection = _collect_web(question, args, rng)
    _atomic_write_json(cache_path, _serialize_web_collection(collection))
    return collection


def _filter_query_ids_with_web_cache(
    query_ids: list[str],
    output_dir: Path,
) -> list[str]:
    """只保留已有网页缓存的问题 id。

    Args:
        query_ids: 参数选择出的候选问题 id。
        output_dir: mixed-source 数据输出目录。

    Returns:
        仍然保持原顺序、且存在 `.cache/web/query_<id>.json` 的问题 id。
    """
    cached_query_ids: list[str] = []
    for query_id in query_ids:
        cache_path = output_dir / ".cache" / "web" / f"query_{query_id}.json"
        if cache_path.exists():
            cached_query_ids.append(query_id)
    return cached_query_ids


def _format_web_chunk(
    page: WebPage,
    max_chunk_tokens: int,
    tokenizer: Any | None,
) -> str:
    """把网页正文格式化为一个独立 Append chunk。"""
    title = _normalize_text(page.title)
    header = f"<web_document>\nTitle: {title}\nURL: {page.final_url}\nContent:\n"
    footer = "\n</web_document>\n"
    return _truncate_wrapped_body_by_tokens(
        header=header,
        body=page.text,
        footer=footer,
        max_tokens=max_chunk_tokens,
        tokenizer=tokenizer,
    )


def _build_source_finished_event(
    source_id: str,
    source_type: SourceType,
    sequence_no: int,
    version: int,
) -> dict[str, Any]:
    """构造与 RESTFUL 协议同构的 source_finished 事件。"""
    return {
        "event": {
            "source_id": source_id,
            "event_type": "source_finished",
            "source_type": source_type,
            "chunk_id": f"{source_id}-finished",
            "version": version,
            "sequence_no": sequence_no,
            "text": None,
        },
        "provenance": {
            "kind": "source_barrier",
        },
    }


def _build_sample(
    query_id: str,
    question: str,
    trace_file: str,
    trace_rows: list[AnnsTraceRow],
    documents: dict[str, str],
    web_collection: WebCollection,
    args: argparse.Namespace,
    tokenizer: Any | None,
) -> dict[str, Any]:
    """把 ANNS 与网页内容保存为两个独立的 source-local 事件序列。

    ``events[].event`` 可直接作为 ``/v1/prompt_stream/event`` 的内层 event。
    列表按来源分组仅用于稳定存储；source 内顺序由 ``sequence_no`` 定义，
    跨来源交错和发送间隔必须由实验回放策略决定。
    """
    sample_id = (
        f"anns-query-{int(query_id):06d}"
        if query_id.isdigit()
        else (f"anns-query-{query_id}")
    )
    anns_source_id = f"anns:{query_id}"
    web_source_id = f"web:{query_id}"
    anns_events: list[dict[str, Any]] = []
    web_events: list[dict[str, Any]] = []

    # ANNS CSV 行序就是 Update 来源内部版本顺序；每行始终物化完整快照。
    for sequence_no, row in enumerate(trace_rows):
        snapshot_text = _materialize_anns_snapshot(
            row,
            documents,
            args.max_anns_document_chars,
            args.max_anns_document_tokens,
            args.max_update_snapshot_tokens,
            tokenizer,
        )
        anns_events.append(
            {
                "event": {
                    "source_id": anns_source_id,
                    "event_type": "update",
                    "source_type": ANNS_SOURCE_TYPE,
                    "chunk_id": f"{anns_source_id}-snapshot-{sequence_no}",
                    "version": sequence_no,
                    "sequence_no": sequence_no,
                    "text": snapshot_text,
                },
                "provenance": {
                    "kind": "anns_snapshot",
                    "trace_file": trace_file,
                    "trace_row": row.row_index,
                    "start_iteration": row.start_iteration,
                    "end_iteration": row.end_iteration,
                    "document_ids": list(row.document_ids),
                },
            }
        )
    anns_events.append(
        _build_source_finished_event(
            source_id=anns_source_id,
            source_type=ANNS_SOURCE_TYPE,
            sequence_no=len(trace_rows),
            version=max(0, len(trace_rows) - 1),
        )
    )

    # 网页按搜索 rank 保存。并发抓取的实际完成先后不会影响数据顺序。
    for sequence_no, page in enumerate(web_collection.pages):
        web_events.append(
            {
                "event": {
                    "source_id": web_source_id,
                    "event_type": "append",
                    "source_type": WEB_SOURCE_TYPE,
                    "chunk_id": f"{web_source_id}-page-{sequence_no}",
                    "version": sequence_no,
                    "sequence_no": sequence_no,
                    "text": _format_web_chunk(
                        page,
                        args.max_web_chunk_tokens,
                        tokenizer,
                    ),
                },
                "provenance": {
                    "kind": "web_page",
                    "search_rank": page.rank,
                    "title": page.title,
                    "url": page.url,
                    "final_url": page.final_url,
                    "snippet": page.snippet,
                    "status_code": page.status_code,
                    "content_sha256": page.content_sha256,
                },
            }
        )
    web_events.append(
        _build_source_finished_event(
            source_id=web_source_id,
            source_type=WEB_SOURCE_TYPE,
            sequence_no=len(web_collection.pages),
            version=max(0, len(web_collection.pages) - 1),
        )
    )

    num_web_pages = len(web_collection.pages)
    sample_status = (
        "complete" if num_web_pages >= args.min_successful_web_pages else "partial"
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "sample_id": sample_id,
        "query_id": query_id,
        "question": question,
        "status": sample_status,
        "start": {
            "api": "completions",
            "prompt": args.prefix,
            "stream": False,
            "max_tokens": args.max_tokens,
            "temperature": args.temperature,
        },
        "sources": [
            {
                "source_id": anns_source_id,
                "source_type": ANNS_SOURCE_TYPE,
                "semantics": "full_snapshot",
            },
            {
                "source_id": web_source_id,
                "source_type": WEB_SOURCE_TYPE,
                "semantics": "monotonic_append",
            },
        ],
        "event_order": {
            "storage_layout": "grouped_by_source",
            "within_source": "sequence_no",
            "across_sources": "unspecified",
        },
        "events": anns_events + web_events,
        "finish": {
            "after_source_ids": [anns_source_id, web_source_id],
            "event": {
                "source_id": "final_question",
                "event_type": "prompt_finished",
                "source_type": None,
                "chunk_id": "final-question",
                "version": 0,
                "sequence_no": 0,
                "text": f"\nUser question: {question}\nAnswer:",
            },
        },
        "web_collection": {
            "provider": web_collection.search_provider,
            "configuration": web_collection.configuration,
            "search_hits": web_collection.search_hits,
            "successful_pages": num_web_pages,
            "failed_pages": [asdict(failure) for failure in web_collection.failures],
        },
        "provenance": {
            "base_dataset": "stream2llm-data/anns",
            "base_query_id": query_id,
            "trace_file": trace_file,
        },
    }


def _validate_no_timing_fields(value: Any, path: str = "$") -> None:
    """递归拒绝 v1 中遗留的真实时间或耗时字段。"""
    forbidden_keys = {
        "ready_offset_ms",
        "fetch_duration_ms",
        "search_duration_ms",
        "collection_duration_ms",
        "start_time_us",
        "end_time_us",
        "collected_at",
        "generated_at",
        "updated_at",
    }
    if isinstance(value, dict):
        for key, child in value.items():
            if key in forbidden_keys:
                raise ValueError(f"Timing field {path}.{key} is not allowed in v2.")
            _validate_no_timing_fields(child, f"{path}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _validate_no_timing_fields(child, f"{path}[{index}]")


def _validate_sample(sample: dict[str, Any]) -> None:
    """检查回放所依赖的核心 schema 与 source 内顺序约束。

    Raises:
        ValueError: 样本缺少关键字段、事件 id 重复或 source 序号不连续。
    """
    if sample.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("Unexpected mixed-source schema version.")
    if not sample.get("start", {}).get("prompt"):
        raise ValueError("Sample start.prompt must be non-empty.")
    if not sample.get("finish", {}).get("event", {}).get("text"):
        raise ValueError("Sample finish question must be non-empty.")
    event_order = sample.get("event_order", {})
    if event_order.get("across_sources") != "unspecified":
        raise ValueError("Cross-source event order must remain unspecified.")
    _validate_no_timing_fields(sample)

    seen_chunk_ids: set[str] = set()
    source_events: dict[str, list[dict[str, Any]]] = {}
    for record in sample["events"]:
        event = record["event"]
        chunk_id = event["chunk_id"]
        if chunk_id in seen_chunk_ids:
            raise ValueError(f"Duplicate chunk_id={chunk_id}.")
        seen_chunk_ids.add(chunk_id)
        source_events.setdefault(event["source_id"], []).append(event)
        if event["event_type"] == "append" and not event["text"]:
            raise ValueError(f"Append chunk {chunk_id} must be non-empty.")
        # Update 的空字符串具有“清空当前 snapshot”的明确语义，必须允许。
        if event["event_type"] == "update" and event["text"] is None:
            raise ValueError(f"Update snapshot {chunk_id} must carry text.")

    for source_id, events in source_events.items():
        sequences = [int(event["sequence_no"]) for event in events]
        if sequences != list(range(len(events))):
            raise ValueError(
                f"Non-contiguous sequence numbers for {source_id}: {sequences}."
            )
        if events[-1]["event_type"] != "source_finished":
            raise ValueError(f"Source {source_id} must end with source_finished.")
        versions = [
            int(event["version"])
            for event in events
            if event["event_type"] in ("append", "update")
        ]
        if versions != list(range(len(versions))):
            raise ValueError(
                f"Non-contiguous data versions for {source_id}: {versions}."
            )


def _request_path(output_dir: Path, query_id: str) -> Path:
    """返回一个 query 的稳定请求文件路径。"""
    filename = (
        f"anns-query-{int(query_id):06d}.json"
        if query_id.isdigit()
        else f"anns-query-{query_id}.json"
    )
    return output_dir / "requests" / filename


def _rebuild_manifest(output_dir: Path) -> int:
    """从全部请求文件确定性重建 manifest.jsonl。

    Returns:
        manifest 中的请求数量。
    """
    entries: list[dict[str, Any]] = []
    for request_path in sorted((output_dir / "requests").glob("*.json")):
        sample = _load_json(request_path)
        _validate_sample(sample)
        num_append_events = sum(
            record["event"]["event_type"] == "append" for record in sample["events"]
        )
        num_update_events = sum(
            record["event"]["event_type"] == "update" for record in sample["events"]
        )
        entries.append(
            {
                "sample_id": sample["sample_id"],
                "query_id": sample["query_id"],
                "path": request_path.relative_to(output_dir).as_posix(),
                "status": sample["status"],
                "question": sample["question"],
                "num_append_events": num_append_events,
                "num_update_events": num_update_events,
            }
        )
    entries.sort(key=lambda entry: _numeric_query_sort_key(entry["query_id"]))
    manifest_text = "".join(
        f"{json.dumps(entry, ensure_ascii=False)}\n" for entry in entries
    )
    _atomic_write_text(output_dir / "manifest.jsonl", manifest_text)
    return len(entries)


def _write_dataset_metadata(
    output_dir: Path,
    args: argparse.Namespace,
    num_samples: int,
) -> None:
    """写入不包含机器绝对路径的数据集级元信息。"""
    metadata = {
        "schema_version": SCHEMA_VERSION,
        "description": (
            "Mixed prompt-stream dataset using ANNS full snapshots as Update "
            "events and freshly collected web pages as Append events."
        ),
        "num_samples": num_samples,
        "event_order": {
            "within_source": "sequence_no",
            "across_sources": "controlled_by_replay",
        },
        "layout": {
            "manifest": "manifest.jsonl",
            "requests": "requests/*.json",
            "web_cache": ".cache/web/query_<id>.json",
        },
        "generation_defaults": {
            "max_web_results": args.max_web_results,
            "min_successful_web_pages": args.min_successful_web_pages,
            "max_web_chars": args.max_web_chars,
            "max_web_chunk_tokens": args.max_web_chunk_tokens,
            "max_anns_document_chars": args.max_anns_document_chars,
            "max_anns_document_tokens": args.max_anns_document_tokens,
            "max_update_snapshot_tokens": args.max_update_snapshot_tokens,
            "tokenizer_model": args.tokenizer_model,
            "max_tokens": args.max_tokens,
            "temperature": args.temperature,
        },
    }
    _atomic_write_json(output_dir / "dataset.json", metadata)


def _validate_args(args: argparse.Namespace) -> None:
    """在访问大文件或网络前检查参数范围。"""
    positive_fields = (
        "max_web_results",
        "min_successful_web_pages",
        "fetch_workers",
        "search_timeout",
        "fetch_timeout",
        "max_page_bytes",
        "min_web_chars",
        "max_tokens",
    )
    for field_name in positive_fields:
        if getattr(args, field_name) <= 0:
            raise ValueError(f"--{field_name.replace('_', '-')} must be positive.")
    non_negative_fields = (
        "max_search_retries",
        "max_fetch_retries",
        "retry_backoff",
        "query_delay",
        "max_web_chars",
        "max_anns_document_chars",
        "max_web_chunk_tokens",
        "max_anns_document_tokens",
        "max_update_snapshot_tokens",
    )
    for field_name in non_negative_fields:
        if getattr(args, field_name) < 0:
            raise ValueError(f"--{field_name.replace('_', '-')} must be non-negative.")
    token_budget_fields = (
        "max_web_chunk_tokens",
        "max_anns_document_tokens",
        "max_update_snapshot_tokens",
    )
    if (
        any(getattr(args, field_name) > 0 for field_name in token_budget_fields)
        and args.tokenizer_model is None
    ):
        raise ValueError(
            "--tokenizer-model is required when token-level content "
            "budgets are enabled."
        )
    if args.min_successful_web_pages > args.max_web_results:
        raise ValueError("--min-successful-web-pages cannot exceed --max-web-results.")
    if args.offline and args.refresh_web:
        raise ValueError("--offline cannot be combined with --refresh-web.")
    if args.only_cached_web and args.refresh_web:
        raise ValueError("--only-cached-web cannot be combined with --refresh-web.")
    if args.refresh_web and not args.overwrite:
        raise ValueError("--refresh-web requires --overwrite.")


def run(args: argparse.Namespace) -> None:
    """执行一次可恢复的小批量混合数据生成。

    Args:
        args: 已解析并校验的命令行参数。
    """
    _validate_args(args)
    tokenizer = _load_tokenizer(args.tokenizer_model)
    query_map_path = args.anns_dir / "query_trace_map_5k.json"
    query_map: dict[str, dict[str, str]] = _load_json(query_map_path)
    query_ids = _select_query_ids(query_map, args)
    if args.only_cached_web:
        original_num_query_ids = len(query_ids)
        query_ids = _filter_query_ids_with_web_cache(query_ids, args.output_dir)
        logger.info(
            "Selected %d/%d queries with existing web cache.",
            len(query_ids),
            original_num_query_ids,
        )
    if not query_ids:
        raise ValueError("No ANNS queries selected.")

    # 先解析本批 trace，再一次性加载所需 corpus 文档，避免把约 200 MiB
    # 的完整 corpus 长期驻留在 Python 字典中。
    traces: dict[str, list[AnnsTraceRow]] = {}
    required_document_ids: set[str] = set()
    for query_id in query_ids:
        trace_file = query_map[query_id]["trace_file"]
        trace_rows = _load_trace_rows(args.anns_dir, trace_file)
        traces[query_id] = trace_rows
        for row in trace_rows:
            required_document_ids.update(row.document_ids)
    documents = _load_required_documents(
        args.anns_dir,
        required_document_ids,
    )

    succeeded = 0
    failed = 0
    for query_index, query_id in enumerate(query_ids):
        request_path = _request_path(args.output_dir, query_id)
        if request_path.exists() and not args.overwrite:
            existing_sample = _load_json(request_path)
            if existing_sample.get("schema_version") == SCHEMA_VERSION:
                logger.info(
                    "Skipping existing request %s; use --overwrite to rebuild.",
                    request_path,
                )
                succeeded += 1
                continue
            logger.info(
                "Rebuilding legacy request %s as schema %s.",
                request_path,
                SCHEMA_VERSION,
            )

        question = query_map[query_id]["query"]
        trace_file = query_map[query_id]["trace_file"]
        logger.info(
            "Building query %s (%d/%d): %s",
            query_id,
            query_index + 1,
            len(query_ids),
            question,
        )
        try:
            rng_seed = args.random_seed + (
                int(query_id) if query_id.isdigit() else query_index
            )
            web_collection = _get_web_collection(
                query_id,
                question,
                args,
                random.Random(rng_seed),
            )
            if (
                args.require_web
                and len(web_collection.pages) < args.min_successful_web_pages
            ):
                raise RuntimeError(
                    f"Only {len(web_collection.pages)} usable web pages were "
                    f"collected for query {query_id}; "
                    f"{args.min_successful_web_pages} required."
                )
            sample = _build_sample(
                query_id=query_id,
                question=question,
                trace_file=trace_file,
                trace_rows=traces[query_id],
                documents=documents,
                web_collection=web_collection,
                args=args,
                tokenizer=tokenizer,
            )
            _validate_sample(sample)
            _atomic_write_json(request_path, sample)
            succeeded += 1
            logger.info(
                "Wrote %s with %d ANNS snapshots and %d web pages (%s).",
                request_path,
                len(traces[query_id]),
                len(web_collection.pages),
                sample["status"],
            )
        except Exception:
            failed += 1
            logger.exception("Failed to build query %s.", query_id)
            if args.fail_fast:
                raise

        is_last_query = query_index == len(query_ids) - 1
        if not args.offline and not is_last_query and args.query_delay > 0:
            time.sleep(args.query_delay)

    num_samples = _rebuild_manifest(args.output_dir)
    _write_dataset_metadata(args.output_dir, args, num_samples)
    logger.info(
        "Finished: %d selected, %d succeeded/skipped, %d failed, "
        "%d total samples in manifest.",
        len(query_ids),
        succeeded,
        failed,
        num_samples,
    )
    if failed:
        raise RuntimeError(f"Failed to build {failed} selected queries.")


def main() -> None:
    """脚本入口。"""
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
    run(args)


if __name__ == "__main__":
    main()
