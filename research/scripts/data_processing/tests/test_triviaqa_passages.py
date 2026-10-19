# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""检查查询相关性、预算、去重和原文追溯，避免整页进入 Append。"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

PROCESSING_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROCESSING_DIR))

from append_chunks_common import (  # noqa: E402
    EvidenceDocument,
    InputRecord,
    text_sha256,
)
from test_append_chunks import make_tokenizer  # noqa: E402
from triviaqa_passages import PassageSelector  # noqa: E402


class TriviaQAPassagesTest(unittest.TestCase):
    """以合成网页检查选择行为，不要求固定 BM25 分数或具体窗口数。"""

    def setUp(self) -> None:
        """各用例使用独立的本地分词器与查询。"""
        self.tokenizer = make_tokenizer()
        self.selector = PassageSelector(self.tokenizer)
        self.record = InputRecord(
            "q",
            "Which album did Bowie release in 1974?",
            "dev",
            Path(__file__),
            {"Answer": "GOLD_SENTINEL"},
            {"answer": "GOLD_SENTINEL"},
        )

    @staticmethod
    def document(
        body: str, document_id: str = "page", title: str = "Music"
    ) -> EvidenceDocument:
        """创建保留来源的网页正文。"""
        return EvidenceDocument(
            document_id,
            title,
            body,
            "web",
            {"source_file": "web/example.txt"},
            "https://example.org/music",
        )

    def test_selects_evidence_after_long_irrelevant_prefix(self) -> None:
        """答案段落在网页末尾时仍能入选，不以网页开头充当摘录。"""
        body = (
            "unrelated navigation " * 1200
            + "\nBowie released the album Diamond Dogs in 1974."
        )
        document = self.document(body)
        sample = self.selector.build_sample("q", self.record, [document], [])
        self.assertEqual(len(sample["chunks"]), 1)
        self.assertIn("Diamond Dogs", sample["chunks"][0]["text"])
        provenance = sample["documents"][0]["provenance"]
        start, end = provenance["source_body_span"]
        self.assertIn("Diamond Dogs", body[start:end])
        self.assertLessEqual(self.selector.count_tokens(body[start:end]), 512)
        self.assertNotEqual(body[start:end], body)
        self.assertEqual(provenance["source_body_sha256"], text_sha256(body))

    def test_budget_includes_source_headers_and_joined_text(self) -> None:
        """正文能容纳但标题过长时跳过，剩余块满足单块和最终字符串预算。"""
        selector = PassageSelector(
            self.tokenizer,
            append_tokens=72,
            chunk_tokens=40,
            passage_tokens=20,
            overlap_tokens=4,
        )
        documents = [
            self.document("Bowie album 1974 Diamond Dogs.", "huge", "title " * 100),
            self.document("Bowie released an album in 1974.", "a"),
            self.document("Bowie recorded the 1974 album in London.", "b"),
            self.document("Bowie promoted the album in 1974 with a tour.", "c"),
        ]
        sample = selector.build_sample("q", self.record, documents, [])
        self.assertEqual(sample["status"], "ready")
        text = "".join(chunk["text"] for chunk in sample["chunks"])
        self.assertEqual(sample["append_tokens"], selector.count_tokens(text))
        self.assertLessEqual(sample["append_tokens"], 72)
        for chunk in sample["chunks"]:
            self.assertEqual(chunk["text_tokens"], selector.count_tokens(chunk["text"]))
            self.assertLessEqual(chunk["text_tokens"], 40)
        self.assertNotIn(
            "huge",
            [doc["provenance"]["source_document_id"] for doc in sample["documents"]],
        )
        self.assertGreater(sample["diagnostics"]["num_excluded_by_token_budget"], 0)

    def test_duplicate_pages_and_weak_reviews_do_not_fill_three_slots(self) -> None:
        """重复事实只保留一次，仅匹配通用 album 的评论不为凑块数入选。"""
        fact = "Bowie released the album Diamond Dogs in 1974."
        documents = [
            self.document(fact, "a"),
            self.document(fact, "b"),
            self.document("Great album, excellent music and lovely sound!", "review"),
        ]
        sample = self.selector.build_sample("q", self.record, documents, [])
        self.assertEqual(len(sample["chunks"]), 1)
        self.assertGreater(sample["diagnostics"]["num_excluded_duplicate_passages"], 0)

    def test_multiple_passages_from_one_source_and_source_limit(self) -> None:
        """同一页可保留两段不同证据，来源数限制不等于摘录数限制。"""
        selector = PassageSelector(
            self.tokenizer, max_documents=1, passage_tokens=12, overlap_tokens=0
        )
        documents = [
            self.document(
                "Bowie released the album Diamond Dogs in 1974.\n"
                "The 1974 Bowie tour included album recordings "
                "and rehearsals in London.",
                "a",
            ),
            self.document(
                "Bowie celebrated the 1974 album at a New York concert.", "b"
            ),
        ]
        sample = selector.build_sample("q", self.record, documents, [])
        self.assertEqual(len(sample["chunks"]), 2)
        self.assertEqual(
            {doc["provenance"]["source_document_id"] for doc in sample["documents"]},
            {"a"},
        )
        self.assertEqual(len({doc["document_id"] for doc in sample["documents"]}), 2)

    def test_short_lines_share_context_and_tail_is_not_an_orphan(self) -> None:
        """网页短行共同组成窗口，末尾事实带有上下文且所有偏移可追溯。"""
        selector = PassageSelector(self.tokenizer, passage_tokens=30, overlap_tokens=5)
        body = "Bowie album recording notes\n" * 15 + "Released in 1974."
        passages = selector.candidates([self.document(body)])
        self.assertLess(len(passages), len(body.splitlines()))
        self.assertEqual(passages[-1].end, len(body))
        self.assertIn("Released in 1974.", passages[-1].body)
        self.assertGreaterEqual(selector.count_tokens(passages[-1].body), 15)
        for passage in passages:
            self.assertEqual(passage.body, body[passage.start : passage.end])
            self.assertLessEqual(selector.count_tokens(passage.body), 30)

    def test_selection_is_independent_of_gold_answers(self) -> None:
        """改变答案标注不影响模型可见文本、排序或筛选诊断。"""
        documents = [self.document("Bowie released an album in 1974.")]
        before = self.selector.build_sample("q", self.record, documents, [])
        self.record.raw["Answer"] = "CHANGED"
        self.record.evaluation = {"answer": "CHANGED"}
        after = self.selector.build_sample("q", self.record, documents, [])
        self.assertEqual(before, after)
        self.assertNotIn("GOLD_SENTINEL", str(before))

    def test_unicode_long_sentences_and_overlapping_windows(self) -> None:
        """无换行长句和 Unicode 正文均保留原文偏移，超预算窗口不能漏过检查。"""
        selector = PassageSelector(self.tokenizer, passage_tokens=12, overlap_tokens=3)
        body = "中文 🎵 Bowie " + "word " * 70 + "album 1974. End of paragraph."
        document = self.document(body)
        candidates = selector.candidates([document])
        self.assertGreater(len(candidates), 2)
        self.assertEqual(candidates[0].start, 0)
        self.assertEqual(candidates[-1].end, len(body))
        for passage in candidates:
            self.assertEqual(passage.body, body[passage.start : passage.end])
            self.assertLessEqual(selector.count_tokens(passage.body), 12)
        starts = [passage.start for passage in candidates]
        self.assertEqual(starts, sorted(set(starts)))

    def test_empty_evidence_and_no_query_match_remain_explicit(self) -> None:
        """没有本地正文和没有相关摘录分别记录原因，均保留空请求。"""
        empty = self.selector.build_sample("q", self.record, [], [])
        unmatched = self.selector.build_sample(
            "q", self.record, [self.document("Football scores.")], []
        )
        self.assertEqual(empty["status"], "empty")
        self.assertEqual(empty["append_tokens"], 0)
        self.assertEqual(
            empty["diagnostics"]["selection_empty_reason"], "no_local_evidence"
        )
        self.assertEqual(unmatched["status"], "empty")
        self.assertEqual(
            unmatched["diagnostics"]["selection_empty_reason"],
            "no_passage_within_selection_rules",
        )


if __name__ == "__main__":
    unittest.main()
