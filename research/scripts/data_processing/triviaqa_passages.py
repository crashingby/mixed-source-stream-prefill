# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""使用纯 CPU 的 BM25 选择可追溯、具有 token 预算的 TriviaQA 原文摘录。"""

from __future__ import annotations

import math
from bisect import bisect_left
from collections import Counter, OrderedDict
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any

import regex as re
from append_chunks_common import (
    EvidenceDocument,
    InputRecord,
    build_sample,
    text_sha256,
)

if TYPE_CHECKING:
    from tokenizers import Tokenizer

STOP_WORDS = frozenset(
    [
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "been",
        "being",
        "by",
        "did",
        "do",
        "does",
        "for",
        "from",
        "had",
        "has",
        "have",
        "how",
        "in",
        "into",
        "is",
        "it",
        "its",
        "of",
        "on",
        "or",
        "s",
        "that",
        "the",
        "their",
        "there",
        "these",
        "they",
        "this",
        "those",
        "to",
        "was",
        "were",
        "what",
        "when",
        "where",
        "which",
        "who",
        "whom",
        "whose",
        "why",
        "will",
        "with",
        "would",
    ]
)
# 显式限定字母和数字，保持原先标准库 Unicode 单词去除下划线的匹配语义。
WORD_PATTERN = re.compile(r"[\p{L}\p{N}]+")
LINE_END = re.compile(r"\n+")
SENTENCE_END = re.compile(r"[.!?](?:[\"'’”)]*)\s+")


def terms(text: str) -> list[str]:
    """按大小写无关的单词匹配，去除固定英语停用词。"""
    return [
        word for word in WORD_PATTERN.findall(text.casefold()) if word not in STOP_WORDS
    ]


@dataclass(frozen=True)
class Passage:
    """一段未经改写的原文，偏移相对于统一换行后的完整正文。"""

    document: EvidenceDocument
    start: int
    end: int
    counts: Counter[str]
    score: float = 0.0
    selection_score: float = 0.0
    matched_terms: int = 0

    @property
    def body(self) -> str:
        """返回原文字符区间对应的正文。"""
        return self.document.body[self.start : self.end]


class PassageSelector:
    """在每问题的本地证据中排名并渲染摘录，不读取评价字段。

    Args:
        tokenizer: 仅用于分段和计数的 fast tokenizer，不加载模型权重。
        append_tokens: 拼接所有渲染块后的总 token 上限。
        chunk_tokens: 包含标题、URL 和分隔符的单块上限。
        max_chunks: 每问题最多保留的摘录数，不要求填满。
        passage_tokens: 候选正文窗口的 token 上限。
        overlap_tokens: 相邻正文窗口的重叠目标；末尾短窗口可增加重叠。
        min_matched_terms: 最少匹配的不同问题词数。
        min_query_coverage: 最少匹配的问题词比例，与词数下限同时应用。
        min_score_ratio: 相对于最高合格加权分数的保留阈值。
        max_documents: 最多采用的不同来源数，0 表示不额外限制。
    """

    def __init__(
        self,
        tokenizer: Tokenizer,
        append_tokens: int = 6144,
        chunk_tokens: int = 640,
        max_chunks: int = 10,
        passage_tokens: int = 512,
        overlap_tokens: int = 80,
        min_matched_terms: int = 2,
        min_query_coverage: float = 0.3,
        min_score_ratio: float = 0.2,
        max_documents: int = 0,
    ) -> None:
        self.tokenizer = tokenizer
        self.append_tokens = append_tokens
        self.chunk_tokens = chunk_tokens
        self.max_chunks = max_chunks
        self.passage_tokens = passage_tokens
        self.overlap_tokens = overlap_tokens
        self.min_matched_terms = min_matched_terms
        self.min_query_coverage = min_query_coverage
        self.min_score_ratio = min_score_ratio
        self.max_documents = max_documents
        self._window_cache: OrderedDict[str, tuple[tuple[int, int], ...]] = (
            OrderedDict()
        )

    def count_tokens(self, text: str) -> int:
        """计数完整字符串，不把分别编码后的块长度当成最终总长。"""
        return len(self.tokenizer.encode(text, add_special_tokens=False).ids)

    def window_spans(self, body: str) -> tuple[tuple[int, int], ...]:
        """缓存少量网页的窗口偏移，避免重复网页再次分词。"""
        if body not in self._window_cache:
            self._window_cache[body] = self._split_body(body)
            if len(self._window_cache) > 64:
                self._window_cache.popitem(last=False)
        self._window_cache.move_to_end(body)
        return self._window_cache[body]

    def _split_body(self, body: str) -> tuple[tuple[int, int], ...]:
        """把相邻短行纳入连续窗口，优先在句末或换行处结束，保留原文偏移。"""
        offsets = self.tokenizer.encode(body, add_special_tokens=False).offsets
        if not offsets:
            return ()
        if len(offsets) <= self.passage_tokens:
            return ((0, len(body)),)
        token_starts = [span[0] for span in offsets]
        boundaries = sorted(
            {match.end() for match in SENTENCE_END.finditer(body)}
            | {match.end() for match in LINE_END.finditer(body)}
        )
        spans = []
        token_start = 0
        while token_start < len(offsets):
            # 末尾不足半窗时向前补上下文，避免标题、页脚独自成为很短的候选。
            if (
                self.overlap_tokens
                and 0 < len(offsets) - token_start < self.passage_tokens // 2
            ):
                tail_start = max(0, len(offsets) - self.passage_tokens)
                if not spans or offsets[tail_start][0] > spans[-1][0]:
                    token_start = tail_start
            start = offsets[token_start][0]
            token_end = min(token_start + self.passage_tokens, len(offsets))
            end = len(body) if token_end == len(offsets) else offsets[token_end - 1][1]
            if token_end < len(offsets):
                lower = offsets[token_start + (token_end - token_start) // 2][0]
                boundary = bisect_left(boundaries, end + 1) - 1
                if boundary >= 0 and boundaries[boundary] > lower:
                    end = boundaries[boundary]
            # 子串独立编码可能与全文中的 BPE 边界不同，因此再检查实际窗口长度。
            while self.count_tokens(body[start:end]) > self.passage_tokens:
                end -= 1
            if end <= start:
                raise ValueError(
                    "候选窗口无法容纳一个原文字符，请增大 --passage-tokens"
                )
            spans.append((start, end))
            if end == len(body):
                break
            end_index = bisect_left(token_starts, end)
            next_start = max(token_start + 1, end_index - self.overlap_tokens)
            # 多个 token 可能对应同一 Unicode 字符，不能因此生成重复起点。
            while next_start < len(offsets) and token_starts[next_start] <= start:
                next_start += 1
            if self.overlap_tokens == 0:
                while next_start < len(offsets) and token_starts[next_start] < end:
                    next_start += 1
            token_start = next_start
        return tuple(spans)

    def candidates(self, documents: list[EvidenceDocument]) -> list[Passage]:
        """跨短行构造有上下文的候选，每份摘录仍是网页正文的连续子串。"""
        passages = []
        for document in documents:
            for start, end in self.window_spans(document.body):
                body = document.body[start:end]
                passages.append(Passage(document, start, end, Counter(terms(body))))
        return passages

    def rank(self, question: str, passages: list[Passage]) -> list[Passage]:
        """计算 Okapi BM25，索引范围是当前问题的候选正文，k1=1.5、b=0.75。"""
        query_terms = set(terms(question))
        numeric_terms = {word for word in query_terms if word.isdecimal()}
        if not passages or not query_terms:
            return []
        lengths = [sum(passage.counts.values()) for passage in passages]
        average_length = sum(lengths) / len(passages) or 1.0
        frequencies: Counter[str] = Counter()
        for passage in passages:
            frequencies.update(query_terms.intersection(passage.counts))
        idfs = {
            word: math.log(1 + (len(passages) - count + 0.5) / (count + 0.5))
            for word, count in frequencies.items()
        }
        ranked = []
        for passage, length in zip(passages, lengths):
            matches = query_terms.intersection(passage.counts)
            normalization = 1.5 * (1 - 0.75 + 0.75 * length / average_length)
            score = sum(
                idfs[word]
                * passage.counts[word]
                * 2.5
                / (passage.counts[word] + normalization)
                for word in matches
            )
            # 年份、数量通常是问题的具体约束；仅做软加权，缺少数字的证据仍可入选。
            numeric_bonus = (
                1 + 0.5 * len(matches & numeric_terms) / len(numeric_terms)
                if numeric_terms
                else 1.0
            )
            ranked.append(
                replace(
                    passage,
                    score=score,
                    selection_score=score * numeric_bonus,
                    matched_terms=len(matches),
                )
            )
        # 同分时沿用历史来源顺序与正文顺序，保证重复生成结果一致。
        return sorted(ranked, key=lambda passage: -passage.selection_score)

    @staticmethod
    def is_duplicate(passage: Passage, selected: list[Passage]) -> bool:
        """排除同源重叠窗口以及跨网页的近重复原文。"""
        words = WORD_PATTERN.findall(passage.body.casefold())
        normalized = " ".join(words)
        shingles = {tuple(words[index : index + 3]) for index in range(len(words) - 2)}
        for previous in selected:
            if passage.document.document_id == previous.document.document_id:
                overlap = max(
                    0,
                    min(passage.end, previous.end) - max(passage.start, previous.start),
                )
                shorter = min(
                    passage.end - passage.start, previous.end - previous.start
                )
                if overlap / shorter >= 0.5:
                    return True
            previous_words = WORD_PATTERN.findall(previous.body.casefold())
            if normalized == " ".join(previous_words):
                return True
            previous_shingles = {
                tuple(previous_words[index : index + 3])
                for index in range(len(previous_words) - 2)
            }
            union = shingles | previous_shingles
            if union and len(shingles & previous_shingles) / len(union) >= 0.8:
                return True
        return False

    @staticmethod
    def excerpt(passage: Passage) -> EvidenceDocument:
        """以原始文档 ID 和字符区间标识摘录，保存来源及筛选分数。"""
        document = passage.document
        return replace(
            document,
            document_id=f"{document.document_id}#chars={passage.start}:{passage.end}",
            body=passage.body,
            sentence_spans=[],
            provenance={
                **document.provenance,
                "source_document_id": document.document_id,
                "source_body_sha256": text_sha256(document.body),
                "source_body_span": [passage.start, passage.end],
                "bm25_score": passage.score,
                "selection_score": passage.selection_score,
                "matched_query_terms": passage.matched_terms,
            },
        )

    def build_sample(
        self,
        sample_id: str,
        record: InputRecord,
        documents: list[EvidenceDocument],
        issues: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """只用问题与证据筛选摘录，再以最终渲染文本检查两层预算。"""
        unique: dict[str, EvidenceDocument] = {}
        duplicates: dict[str, list[dict[str, Any]]] = {}
        for document in documents:
            if document.document_id not in unique:
                unique[document.document_id] = document
            else:
                duplicates.setdefault(document.document_id, []).append(
                    document.provenance
                )
        candidates = self.candidates(list(unique.values()))
        ranked = self.rank(record.question, candidates)
        query_size = len(set(terms(record.question)))
        required_terms = min(
            query_size,
            max(
                self.min_matched_terms, math.ceil(query_size * self.min_query_coverage)
            ),
        )
        eligible = [
            passage
            for passage in ranked
            if passage.score > 0 and passage.matched_terms >= required_terms
        ]
        threshold = (
            eligible[0].selection_score * self.min_score_ratio if eligible else 0.0
        )
        remaining = [
            passage for passage in eligible if passage.selection_score >= threshold
        ]
        selected: list[Passage] = []
        excerpts: list[EvidenceDocument] = []
        excluded_duplicates = excluded_budget = 0
        while remaining and len(selected) < self.max_chunks:
            source_ids = {passage.document.document_id for passage in selected}
            # 软性优先不同网页；同一来源仍可贡献第二段有用证据。
            best = max(
                range(len(remaining)),
                key=lambda index: (
                    remaining[index].selection_score
                    * (
                        0.7
                        if remaining[index].document.document_id in source_ids
                        else 1.0
                    )
                ),
            )
            passage = remaining.pop(best)
            if (
                self.max_documents
                and len(source_ids) >= self.max_documents
                and passage.document.document_id not in source_ids
            ):
                continue
            if self.is_duplicate(passage, selected):
                excluded_duplicates += 1
                continue
            excerpt = self.excerpt(passage)
            trial = build_sample(sample_id, [*excerpts, excerpt], issues)
            append_text = "".join(chunk["text"] for chunk in trial["chunks"])
            chunk_text = trial["chunks"][-1]["text"]
            if (
                self.count_tokens(chunk_text) > self.chunk_tokens
                or self.count_tokens(append_text) > self.append_tokens
            ):
                excluded_budget += 1
                continue
            selected.append(passage)
            excerpts.append(excerpt)
        sample = build_sample(sample_id, excerpts, issues)
        for document in sample["documents"]:
            source_id = document["provenance"]["source_document_id"]
            if source_id in duplicates:
                document["duplicate_provenance"] = duplicates[source_id]
        sample["append_tokens"] = self.count_tokens(
            "".join(chunk["text"] for chunk in sample["chunks"])
        )
        for chunk in sample["chunks"]:
            chunk["text_tokens"] = self.count_tokens(chunk["text"])
        sample["diagnostics"].update(
            num_candidates=len(documents) + len(issues),
            num_duplicates=len(documents) - len(unique),
            num_source_documents=len(unique),
            num_passage_candidates=len(candidates),
            num_qualified_passages=len(eligible),
            num_excluded_duplicate_passages=excluded_duplicates,
            num_excluded_by_token_budget=excluded_budget,
            selection_empty_reason=(
                "no_local_evidence"
                if not unique
                else "no_passage_within_selection_rules"
            )
            if not selected
            else None,
        )
        return sample
