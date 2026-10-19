# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""使用合成证据验证 Append 生成、分段与安全重建。"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from typing import TYPE_CHECKING, Any
from unittest.mock import patch

if TYPE_CHECKING:
    from tokenizers import Tokenizer

PROCESSING_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROCESSING_DIR))

from append_chunks_common import (  # noqa: E402
    EvidenceDocument,
    InputRecord,
    build_sample,
    generate_dataset,
    text_sha256,
)
from build_hotpotqa_append_chunks import (  # noqa: E402
    extract_documents,
    normalize_pairs,
)


def make_tokenizer() -> Tokenizer:
    """使用本地小型分词器验证预算，测试不读取或下载大模型。"""
    from tokenizers import Tokenizer, models, pre_tokenizers

    tokenizer = Tokenizer(models.WordLevel({"[UNK]": 0}, unk_token="[UNK]"))
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    return tokenizer


class AppendChunksTest(unittest.TestCase):
    """检查最终输入不受发送粒度影响以及两个数据适配器的行为。"""

    def test_segmentation_preserves_unicode_and_source_header(self) -> None:
        """短块、长标题和多字节字符均须无损拼接。"""
        body = "第一段包含中文。\n\nSecond paragraph. " + "word " * 50
        document = EvidenceDocument(
            "doc:v1", "Very long title " * 10, body, "web", {}, "https://example.org"
        )
        whole = build_sample("q1", [document], [])
        segmented = build_sample("q1", [document], [], chunk_chars=40)
        reconstructed = "".join(chunk["text"] for chunk in segmented["chunks"])
        self.assertEqual(reconstructed, whole["chunks"][0]["text"])
        self.assertEqual(segmented["append_sha256"], whole["append_sha256"])
        self.assertEqual(reconstructed.count("### [A1]"), 1)
        offset = 0
        for chunk in segmented["chunks"]:
            self.assertEqual(
                chunk["rendered_span"], [offset, offset + len(chunk["text"])]
            )
            offset += len(chunk["text"])
        self.assertEqual(
            text_sha256(reconstructed), segmented["documents"][0]["rendered_sha256"]
        )

    def test_duplicate_sources_and_empty_evidence(self) -> None:
        """去重保留其余来源，空证据保留可回放的空样本。"""
        documents = [
            EvidenceDocument("same", "Title", "Body", "web", {"rank": rank})
            for rank in [1, 3]
        ]
        sample = build_sample("q", documents, [])
        self.assertEqual(len(sample["documents"]), 1)
        self.assertEqual(sample["documents"][0]["duplicate_provenance"], [{"rank": 3}])
        self.assertEqual(build_sample("empty", [], [])["status"], "empty")

    def test_hotpot_sentence_mapping_and_gold_independence(self) -> None:
        """列式 context 兼容空句子，证据选择与答案标注无关。"""
        raw = {
            "context": {
                "title": ["Topic"],
                "sentences": [["Alpha", "", "Beta", " Gamma"]],
            },
            "answer": "GOLD",
            "supporting_facts": {"title": ["Topic"], "sent_id": [2]},
        }
        record = InputRecord("q", "Question?", "dev", Path(__file__), raw, {})
        documents, issues = extract_documents(record)
        self.assertEqual(issues, [])
        self.assertEqual(documents[0].body, "Alpha Beta Gamma")
        self.assertEqual(documents[0].sentence_spans[2]["body_span"], [6, 10])
        before = build_sample("q", documents, [])
        raw["answer"] = "CHANGED"
        raw["supporting_facts"] = {"title": [], "sent_id": []}
        after = build_sample("q", extract_documents(record)[0], [])
        self.assertEqual(before, after)
        with self.assertRaises(ValueError):
            normalize_pairs({"title": ["Topic"], "sent_id": []}, "sent_id")

    def test_triviaqa_cli_missing_documents_and_evaluation_separation(self) -> None:
        """端到端验证历史排名、缺失文件、去重及评价字段隔离。"""
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            qa_dir = root / "raw/unfiltered/triviaqa-unfiltered"
            evidence_dir = root / "raw/rc/evidence/web"
            qa_dir.mkdir(parents=True)
            evidence_dir.mkdir(parents=True)
            (evidence_dir / "a.txt").write_text(
                "Topic fact Alpha\r\n", encoding="utf-8"
            )
            (evidence_dir / "b.txt").write_text("Topic fact Beta", encoding="utf-8")
            tokenizer_dir = root / "tokenizer"
            tokenizer_dir.mkdir()
            tokenizer = make_tokenizer()
            tokenizer.enable_truncation(max_length=2)
            tokenizer.enable_padding(length=50)
            tokenizer.save(str(tokenizer_dir / "tokenizer.json"))
            rows = [
                {
                    "QuestionId": "q/1",
                    "Question": "Which Topic fact QUESTION_SENTINEL?",
                    "Answer": {"NormalizedAliases": ["GOLD_SENTINEL"]},
                    "SearchResults": [
                        {"Rank": 4, "Title": "Alpha", "Filename": "a.txt"},
                        {"Rank": 1, "Title": "Beta", "Filename": "b.txt"},
                        {"Rank": 2, "Title": "Unavailable"},
                        {"Rank": 3, "Title": "Duplicate", "Filename": "b.txt"},
                        {"Rank": 5, "Filename": "missing.txt"},
                    ],
                },
                {"QuestionId": "empty", "Question": "Empty?", "SearchResults": []},
            ]
            (qa_dir / "unfiltered-web-dev.json").write_text(
                json.dumps({"Data": rows}), encoding="utf-8"
            )
            output, questions = root / "append", root / "questions.jsonl"
            command = [
                sys.executable,
                str(PROCESSING_DIR / "build_triviaqa_append_chunks.py"),
                "--data-dir",
                str(root / "raw"),
                "--output-dir",
                str(output),
                "--questions-file",
                str(questions),
                "--tokenizer",
                str(tokenizer_dir),
            ]
            subprocess.run(command, cwd=temporary, check=True, capture_output=True)
            manifest = [
                json.loads(line)
                for line in (output / "manifest.jsonl").read_text().splitlines()
            ]
            sample_text = (output / manifest[0]["path"]).read_text()
            sample = json.loads(sample_text)
            self.assertEqual(
                [doc["title"] for doc in sample["documents"]], ["Beta", "Alpha"]
            )
            self.assertEqual(sample["documents"][0]["provenance"]["candidate_index"], 1)
            self.assertEqual(
                sample["documents"][0]["duplicate_provenance"][0]["candidate_index"], 3
            )
            self.assertLess(sample["append_tokens"], 50)
            self.assertEqual(manifest[0]["num_unavailable_candidates"], 2)
            self.assertEqual(manifest[1]["status"], "empty")
            self.assertNotIn("GOLD_SENTINEL", sample_text)
            self.assertNotIn("QUESTION_SENTINEL", sample_text)
            self.assertIn("GOLD_SENTINEL", questions.read_text())
            original = (output / "manifest.jsonl").read_bytes()
            result = subprocess.run(command, cwd=temporary, capture_output=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertEqual(original, (output / "manifest.jsonl").read_bytes())
            subprocess.run(command + ["--overwrite"], check=True, capture_output=True)
            rebuilt = json.loads((output / manifest[0]["path"]).read_text())
            self.assertEqual(rebuilt["append_sha256"], sample["append_sha256"])
            # 重现用户只删 Append、保留问题清单后直接运行的情况。
            original_questions = questions.read_bytes()
            question_mtime = questions.stat().st_mtime_ns
            shutil.rmtree(output)
            subprocess.run(command, check=True, capture_output=True)
            self.assertEqual(original_questions, questions.read_bytes())
            self.assertEqual(question_mtime, questions.stat().st_mtime_ns)
            self.assertEqual(original, (output / "manifest.jsonl").read_bytes())

    def test_hotpot_parquet_cli(self) -> None:
        """读取真实 parquet 容器，答案和 supporting facts 单独输出。"""
        import pyarrow as pa
        import pyarrow.parquet as pq

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            qa = root / "raw/qa/fullwiki"
            qa.mkdir(parents=True)
            row = {
                "id": "abc",
                "question": "Question?",
                "answer": "GOLD_SENTINEL",
                "supporting_facts": {"title": ["Other"], "sent_id": [1]},
                "context": {
                    "title": ["First", "Other"],
                    "sentences": [["First body."], ["One.", " Two."]],
                },
            }
            pq.write_table(
                pa.Table.from_pylist([row]), qa / "validation-00000-of-00001.parquet"
            )
            output, questions = root / "append", root / "questions.jsonl"
            subprocess.run(
                [
                    sys.executable,
                    str(PROCESSING_DIR / "build_hotpotqa_append_chunks.py"),
                    "--data-dir",
                    str(root / "raw"),
                    "--output-dir",
                    str(output),
                    "--questions-file",
                    str(questions),
                ],
                cwd=temporary,
                check=True,
                capture_output=True,
            )
            entry = json.loads((output / "manifest.jsonl").read_text())
            sample_text = (output / entry["path"]).read_text()
            sample = json.loads(sample_text)
            self.assertEqual(
                [doc["title"] for doc in sample["documents"]], ["First", "Other"]
            )
            self.assertNotIn("GOLD_SENTINEL", sample_text)
            self.assertEqual(
                json.loads(questions.read_text())["evaluation"]["supporting_facts"],
                [["Other", 1]],
            )

    def test_existing_questions_must_match_before_creating_append(self) -> None:
        """独立 QA 可以复用，但内容、数量或生成期间发生变化时不能发布 Append。"""
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source.json"
            source.write_text("{}")
            record = InputRecord(
                "q", "Question\u2028with separator?", "dev", source, {}, {}
            )
            document = EvidenceDocument("d", "Title", "Body", "web", {})
            output, questions = root / "append", root / "questions.jsonl"
            args = argparse.Namespace(
                max_requests=0,
                max_documents=0,
                chunk_chars=0,
                log_every=0,
                overwrite=False,
            )
            kwargs = dict(
                extractor=lambda record: ([document], []),
                dataset="fixture",
                generator="fixture",
                source_files=[source],
                args=args,
                output_dir=output,
                questions_file=questions,
                settings={},
            )
            generate_dataset([record], **kwargs)
            original = questions.read_bytes()
            shutil.rmtree(output)
            for content in [
                b"",
                original.replace(b"Question", b"Changed"),
                original * 2,
            ]:
                with self.subTest(content=content):
                    questions.write_bytes(content)
                    with self.assertRaisesRegex(ValueError, "已有问题清单"):
                        generate_dataset([record], **kwargs)
                    self.assertFalse(output.exists())
                    self.assertEqual(questions.read_bytes(), content)
            questions.write_bytes(original)

            def change_questions(
                record: InputRecord,
            ) -> tuple[list[EvidenceDocument], list[dict[str, Any]]]:
                """模拟正文处理期间外部写入已有问题清单。"""
                questions.write_bytes(b"external change")
                return [document], []

            kwargs["extractor"] = change_questions
            with self.assertRaisesRegex(ValueError, "问题清单在生成期间发生变化"):
                generate_dataset([record], **kwargs)
            self.assertFalse(output.exists())
            self.assertEqual(questions.read_bytes(), b"external change")

    def test_failed_rebuild_preserves_published_dataset(self) -> None:
        """生成失败或发布问题清单失败时，应恢复已有输出。"""
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source.json"
            source.write_text("{}")
            record = InputRecord("q", "Question", "dev", source, {}, {})
            document = EvidenceDocument("d", "Title", "Body", "web", {})
            args = argparse.Namespace(
                max_requests=0,
                max_documents=0,
                chunk_chars=0,
                log_every=0,
                overwrite=False,
            )
            kwargs = dict(
                extractor=lambda record: ([document], []),
                dataset="fixture",
                generator="fixture",
                source_files=[source],
                args=args,
                output_dir=root / "append",
                questions_file=root / "questions.jsonl",
                settings={},
            )
            generate_dataset([record], **kwargs)
            original_manifest = (root / "append/manifest.jsonl").read_bytes()
            original_questions = (root / "questions.jsonl").read_bytes()
            args.overwrite = True
            with self.assertRaisesRegex(ValueError, "重复问题 ID"):
                generate_dataset([record, record], **kwargs)
            self.assertEqual(
                original_manifest, (root / "append/manifest.jsonl").read_bytes()
            )
            rename = Path.rename

            def fail_question_publish(path: Path, target: Path) -> Path:
                """模拟问题清单发布失败。"""
                if path.name == "questions.jsonl":
                    raise OSError("simulated publication failure")
                return rename(path, target)

            with (
                patch.object(Path, "rename", fail_question_publish),
                self.assertRaises(OSError),
            ):
                generate_dataset([record], **kwargs)
            self.assertEqual(
                original_manifest, (root / "append/manifest.jsonl").read_bytes()
            )
            self.assertEqual(
                original_questions, (root / "questions.jsonl").read_bytes()
            )


if __name__ == "__main__":
    unittest.main()
