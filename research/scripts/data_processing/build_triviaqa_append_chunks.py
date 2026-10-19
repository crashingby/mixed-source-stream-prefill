# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""按问题用 BM25 筛选 TriviaQA 原文段落，生成有 token 预算的 Append。"""

from __future__ import annotations

import argparse
import json
import logging
from collections.abc import Iterator
from functools import lru_cache, partial
from pathlib import Path
from typing import Any

from append_chunks_common import (
    DATA_ROOT,
    PROJECT_ROOT,
    EvidenceDocument,
    InputRecord,
    add_common_arguments,
    fingerprint_file,
    generate_dataset,
    normalize_body,
    output_version,
    source_path,
    text_sha256,
    validate_arguments,
)
from triviaqa_passages import PassageSelector

DEFAULT_TOKENIZER = PROJECT_ROOT.parents[1] / "Models/Qwen2.5-3B"
SELECTION_DEFAULTS = {
    "append_tokens": 6144,
    "chunk_tokens": 640,
    "max_chunks": 10,
    "passage_tokens": 512,
    "overlap_tokens": 80,
    "min_matched_terms": 2,
    "min_query_coverage": 0.3,
    "min_score_ratio": 0.2,
}


@lru_cache(maxsize=256)
def read_evidence(path: Path) -> str:
    """读取证据并统一换行，缓存重复使用的页面。"""
    return normalize_body(path.read_text(encoding="utf-8"))


def iter_questions(files: list[tuple[str, Path]]) -> Iterator[InputRecord]:
    """流式读取 TriviaQA 大型 JSON，保存完整评价对象。"""
    try:
        import ijson
    except ImportError as error:
        raise RuntimeError(
            "缺少 ijson，请用 uv 安装 data_processing/requirements.txt"
        ) from error
    for split, path in files:
        with path.open("rb") as handle:
            for item in ijson.items(handle, "Data.item", use_float=True):
                yield InputRecord(
                    question_id=str(item.get("QuestionId", "")),
                    question=item["Question"],
                    split=split,
                    source_file=path,
                    raw=item,
                    evaluation={
                        "protocol": "triviaqa_question_level_em_f1",
                        "answer": item.get("Answer"),
                    },
                )


def extract_documents(
    record: InputRecord, evidence_dir: Path, variant: str
) -> tuple[list[EvidenceDocument], list[dict[str, Any]]]:
    """按历史排名读取本地正文，缺失候选保留诊断记录。"""
    if variant == "wikipedia-rc":
        entries = list(enumerate(record.raw.get("EntityPages", [])))
        folder = "wikipedia"
    else:
        entries = sorted(
            enumerate(record.raw.get("SearchResults", [])),
            key=lambda indexed_entry: int(indexed_entry[1].get("Rank", 2**31)),
        )
        folder = "web"
    documents: list[EvidenceDocument] = []
    issues: list[dict[str, Any]] = []
    base = (evidence_dir / folder).resolve()
    for index, entry in entries:
        provenance = {
            "qa_file": source_path(record.source_file),
            "question_id": record.question_id,
            "candidate_index": index,
            "rank": entry.get("Rank"),
            "url": entry.get("Url"),
        }
        filename = entry.get("Filename")
        if not filename:
            issues.append({**provenance, "reason": "missing_filename"})
            continue
        path = (base / filename).resolve()
        if not path.is_relative_to(base):
            raise ValueError(f"证据文件越出目录：{filename}")
        provenance["source_file"] = source_path(path)
        if not path.is_file():
            issues.append({**provenance, "reason": "missing_file"})
            continue
        body = read_evidence(path)
        if not body.strip():
            issues.append({**provenance, "reason": "empty_body"})
            continue
        documents.append(
            EvidenceDocument(
                document_id=f"triviaqa:{folder}:{filename}@{text_sha256(body)}",
                title=entry.get("Title") or Path(filename).stem,
                url=entry.get("Url"),
                body=body,
                source_type="web" if folder == "web" else "wikipedia",
                provenance=provenance,
            )
        )
    return documents, issues


def resolve_inputs(data_dir: Path, variant: str, split: str) -> list[tuple[str, Path]]:
    """定位已经下载的官方 QA 文件，缺失时直接报告路径。"""
    files = []
    for current in ["train", "dev", "test"] if split == "all" else [split]:
        suffix = "test-without-answers" if current == "test" else current
        if variant == "web-unfiltered":
            path = (
                data_dir
                / "unfiltered/triviaqa-unfiltered"
                / f"unfiltered-web-{suffix}.json"
            )
        else:
            domain = "wikipedia" if variant == "wikipedia-rc" else "web"
            path = data_dir / "rc/qa" / f"{domain}-{suffix}.json"
        if not path.is_file():
            raise FileNotFoundError(f"缺少 QA 文件：{path}")
        files.append((current, path))
    return files


def load_selector(args: argparse.Namespace) -> tuple[PassageSelector, dict[str, Any]]:
    """校验筛选参数并加载本地 tokenizer，记录配置和 tokenizer 文件指纹。"""
    for name in [
        "append_tokens",
        "chunk_tokens",
        "max_chunks",
        "passage_tokens",
        "min_matched_terms",
    ]:
        if getattr(args, name) <= 0:
            raise ValueError(f"--{name.replace('_', '-')} 必须为正数")
    if not 0 <= args.overlap_tokens < args.passage_tokens:
        raise ValueError("--overlap-tokens 必须非负且小于 --passage-tokens")
    for name in ["min_query_coverage", "min_score_ratio"]:
        if not 0 <= getattr(args, name) <= 1:
            raise ValueError(f"--{name.replace('_', '-')} 必须介于 0 和 1")
    if args.chunk_chars:
        raise ValueError(
            "TriviaQA 每个选定摘录已是一块，请使用 --chunk-tokens 控制长度"
        )
    tokenizer_path = args.tokenizer.expanduser().resolve()
    if not tokenizer_path.is_dir():
        raise FileNotFoundError(
            f"缺少本地 tokenizer 目录：{tokenizer_path}；请设置 --tokenizer"
        )
    tokenizer_file = tokenizer_path / "tokenizer.json"
    if not tokenizer_file.is_file():
        raise FileNotFoundError(f"缺少 fast tokenizer 文件：{tokenizer_file}")
    try:
        from tokenizers import Tokenizer
    except ImportError as error:
        raise RuntimeError(
            "缺少 tokenizers，请用 uv 安装 data_processing/requirements.txt"
        ) from error
    tokenizer = Tokenizer.from_file(str(tokenizer_file))
    # 文件可能保存了训练时的 padding/truncation，预算必须计数完整证据文本。
    tokenizer.no_padding()
    tokenizer.no_truncation()
    selection = {name: getattr(args, name) for name in SELECTION_DEFAULTS}
    selector = PassageSelector(tokenizer, **selection, max_documents=args.max_documents)
    settings = {
        **selection,
        "algorithm": "query_bm25_context_windows/v3",
        "candidate_layout": "continuous_body_sentence_or_line_boundaries",
        "short_tail": "expand_left_when_overlap_enabled",
        "bm25_k1": 1.5,
        "bm25_b": 0.75,
        "numeric_query_bonus": 0.5,
        "index_scope": "per_question_candidate_bodies",
        "term_matching": "unicode_words_casefold_stopwords/v1",
        "source_diversity_multiplier": 0.7,
        "near_duplicate_trigram_jaccard": 0.8,
        "same_source_overlap_ratio": 0.5,
        "budget_scope": "rendered_text_including_headers_urls_separators",
        "tokenizer": {
            "path": str(tokenizer_path),
            "class": type(tokenizer).__name__,
            "files": [
                fingerprint_file(path)
                for path in sorted(tokenizer_path.iterdir())
                if path.is_file()
                and path.name
                in {
                    "tokenizer.json",
                    "tokenizer_config.json",
                    "special_tokens_map.json",
                    "added_tokens.json",
                    "vocab.json",
                    "merges.txt",
                    "config.json",
                    "tokenizer.model",
                    "spiece.model",
                }
            ],
        },
    }
    return selector, settings


def selection_version(args: argparse.Namespace) -> str:
    """默认沿用 v1 路径，自定义筛选配置以指纹后缀区分。"""
    base = f"{args.variant.replace('-', '_')}_{args.split}_v1"
    changed = {
        name: getattr(args, name)
        for name, default in SELECTION_DEFAULTS.items()
        if getattr(args, name) != default
    }
    if args.tokenizer.expanduser().resolve() != DEFAULT_TOKENIZER.resolve():
        changed["tokenizer"] = str(args.tokenizer.expanduser().resolve())
    if changed:
        base += "_cfg" + text_sha256(json.dumps(changed, sort_keys=True))[:10]
    return output_version(base, args)


def main(argv: list[str] | None = None) -> int:
    """默认离线生成 dev 的 BM25 摘录与独立问题清单，支持覆盖原 v1 输出。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DATA_ROOT / "raw/triviaqa")
    parser.add_argument(
        "--variant",
        choices=["web-unfiltered", "web-rc", "wikipedia-rc"],
        default="web-unfiltered",
    )
    parser.add_argument(
        "--split", choices=["dev", "train", "test", "all"], default="dev"
    )
    add_common_arguments(parser)
    parser.add_argument(
        "--tokenizer",
        type=Path,
        default=DEFAULT_TOKENIZER,
        help="本地 fast tokenizer 目录；只读取分词文件，不加载模型",
    )
    help_text = {
        "append_tokens": "每问题完整 Append 的 token 上限，包含标题、URL 和分隔符",
        "chunk_tokens": "每块渲染文本的 token 上限",
        "max_chunks": "每问题最多保留的摘录数，不要求填满",
        "passage_tokens": "候选正文窗口的 token 上限",
        "overlap_tokens": "相邻正文窗口的重叠目标 token 数；末尾短窗口可增加重叠",
        "min_matched_terms": "最少匹配的不同问题词数，问题更短时自动降低",
        "min_query_coverage": "最少匹配的问题词比例，与匹配词数下限同时应用",
        "min_score_ratio": "相对于最高合格加权分数的最低比例",
    }
    for name, default in SELECTION_DEFAULTS.items():
        parser.add_argument(
            "--" + name.replace("_", "-"),
            type=type(default),
            default=default,
            help=help_text[name],
        )
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    validate_arguments(args)
    read_evidence.cache_clear()
    files = resolve_inputs(args.data_dir, args.variant, args.split)
    evidence_dir = args.data_dir / "rc/evidence"
    if not evidence_dir.is_dir():
        raise FileNotFoundError(f"缺少正文目录：{evidence_dir}")
    selector, selection_settings = load_selector(args)
    version = selection_version(args)
    output_dir = args.output_dir or DATA_ROOT / "append_chunks/triviaqa" / version
    questions_file = (
        args.questions_file or DATA_ROOT / "questions/triviaqa" / f"{version}.jsonl"
    )
    metadata = generate_dataset(
        records=iter_questions(files),
        extractor=partial(
            extract_documents, evidence_dir=evidence_dir, variant=args.variant
        ),
        dataset="triviaqa",
        generator="build_triviaqa_append_chunks",
        source_files=[path for _, path in files],
        args=args,
        output_dir=output_dir,
        questions_file=questions_file,
        settings={
            "variant": args.variant,
            "split": args.split,
            "evidence_dir": source_path(evidence_dir),
            "selection": selection_settings,
            "source_order": "historical_rank_local_body"
            if args.variant != "wikipedia-rc"
            else "entity_pages_local_body",
            "deduplication": "source_filename_and_body_sha256",
            "evaluation_granularity": "question",
            "live_web_search": False,
            "uses_gold_answers": False,
        },
        sample_builder=selector.build_sample,
        generator_version=3,
    )
    return 1 if metadata["completion_status"] == "complete_with_errors" else 0


if __name__ == "__main__":
    raise SystemExit(main())
