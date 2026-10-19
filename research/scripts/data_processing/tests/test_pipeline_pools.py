# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""用本机 HTTP 服务验证入库续跑、完整候选快照和失败恢复。"""

from __future__ import annotations

import bz2
import io
import json
import subprocess
import sys
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from unittest.mock import patch

PROCESSING_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROCESSING_DIR))

import build_hotpotqa_pipeline_pools as hotpot  # noqa: E402
import build_triviaqa_pipeline_pools as trivia  # noqa: E402
import pipeline_pools_common as common  # noqa: E402


class FakeService:
    """只实现采集器使用的 HTTP 协议，不加载真实模型。"""

    def __init__(self) -> None:
        self.documents: list[dict[str, Any]] = []
        self.queries: list[dict[str, Any]] = []
        self.uploads = 0
        self.batch_requests = 0
        self.lookup_requests = 0
        self.bad_ack = False
        self.stream_mode = "normal"
        self.upload_mode = "normal"
        service = self

        class Handler(BaseHTTPRequestHandler):
            """模拟服务器的 JSON 与 SSE 响应。"""

            def log_message(self, format: str, *args: Any) -> None:
                """测试不输出逐请求访问日志。"""

            def reply(self, payload: Any, status: int = 200) -> None:
                """回复一个正常结束的 JSON 响应。"""
                body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def do_GET(self) -> None:
                """返回健康信息、文章列表或指定文章。"""
                if self.path == "/health":
                    self.reply(
                        {
                            "status": "ok",
                            "model": "fixture/bge-m3",
                            "dimension": 4,
                            "document_count": len(service.documents),
                            "chunk_count": len(service.documents),
                        }
                    )
                elif self.path.startswith("/documents?"):
                    self.reply({"documents": service.documents})
                elif self.path.startswith("/documents/"):
                    for document in service.documents:
                        if document["id"] == self.path.rsplit("/", 1)[1]:
                            self.reply(document)
                            return
                    self.reply({"detail": "not found"}, 404)
                else:
                    self.reply({}, 404)

            def do_POST(self) -> None:
                """模拟候选池改变、断流及上传成功但响应丢失。"""
                payload = json.loads(
                    self.rfile.read(int(self.headers["Content-Length"]))
                )
                if self.path == "/documents":
                    service.uploads += 1
                    if service.upload_mode != "drop_uncommitted":
                        document = {"id": f"doc{len(service.documents)}", **payload}
                        service.documents.append(document)
                    if service.upload_mode != "normal":
                        self.close_connection = True
                        return
                    self.reply(document, 201)
                    return
                if self.path == "/documents/lookup":
                    service.lookup_requests += 1
                    results = []
                    for item in payload["documents"]:
                        matches = [
                            document
                            for document in service.documents
                            if document["name"] == item["name"]
                        ]
                        if (
                            matches
                            and common.text_sha256(matches[0]["text"])
                            != item["body_sha256"]
                        ):
                            self.reply({"detail": "body conflict"}, 409)
                            return
                        results.append(
                            {
                                "ingest_key": item["ingest_key"],
                                "id": matches[0]["id"] if matches else None,
                            }
                        )
                    self.reply({"documents": results})
                    return
                if self.path == "/documents/batch":
                    service.batch_requests += 1
                    if service.upload_mode == "drop_uncommitted":
                        self.close_connection = True
                        return
                    results = []
                    for item in payload["documents"]:
                        matches = [
                            document
                            for document in service.documents
                            if document["name"] == item["name"]
                        ]
                        reused = bool(matches)
                        if reused:
                            document = matches[0]
                            if document["text"] != item["text"]:
                                self.reply({"detail": "body conflict"}, 409)
                                return
                        else:
                            document = {"id": f"doc{len(service.documents)}", **item}
                            service.documents.append(document)
                            service.uploads += 1
                        results.append(
                            {
                                "ingest_key": item["ingest_key"],
                                "id": document["id"],
                                "name": document["name"],
                                "body_sha256": common.text_sha256(document["text"]),
                                "reused": reused,
                            }
                        )
                    if service.upload_mode == "drop_committed":
                        service.upload_mode = "normal"
                        self.close_connection = True
                        return
                    if service.bad_ack:
                        results[-1]["body_sha256"] = "wrong"
                    self.reply({"documents": results}, 201)
                    return
                if self.path != "/search/stream":
                    self.reply({}, 404)
                    return
                service.queries.append(payload)
                chunks = [
                    {
                        "label": index,
                        "document_id": document["id"],
                        "name": document["name"],
                        "chunk_index": 0,
                        "start": 0,
                        "end": len(document["text"]),
                        "text": document["text"],
                        "distance": 0.1 + index / 10,
                    }
                    for index, document in enumerate(service.documents)
                ]
                request_id = f"query{len(service.queries)}"
                events: list[tuple[str, dict[str, Any]]] = []
                for sequence, pool in enumerate([chunks[-1:], chunks], 1):
                    events.append(
                        (
                            "candidates",
                            {
                                "request_id": request_id,
                                "query_index": 0,
                                "sequence": sequence,
                                "elapsed_ms": sequence / 10,
                                "chunks": pool,
                            },
                        )
                    )
                events.append(
                    (
                        "final",
                        {
                            "request_id": request_id,
                            "updates": 2,
                            "ef": payload["ef"],
                            "k": min(payload["k"], len(chunks)),
                            "chunks": chunks[: payload["k"]],
                            "search_elapsed_ms": 0.4,
                            "request_elapsed_ms": 1.4,
                        },
                    )
                )
                events.append(("done", {"request_id": request_id, "updates": 2}))
                if service.stream_mode == "truncated":
                    events = events[:-1]
                elif service.stream_mode == "error":
                    events = [("error", {"message": "fixture error"})]
                elif service.stream_mode == "wrong_id":
                    events[-1][1]["request_id"] = "other"
                body = "".join(
                    f"event: {event}\ndata: {json.dumps(item, ensure_ascii=False)}\n\n"
                    for event, item in events
                ).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self.server.server_port}"
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()

    def close(self) -> None:
        """停止测试线程并释放监听端口。"""
        self.server.shutdown()
        self.server.server_close()
        self.thread.join()


class PipelinePoolsTest(unittest.TestCase):
    """验证两种语料适配、实际 HTTP 传输与可恢复的输出。"""

    def setUp(self) -> None:
        """每个测试使用独立临时目录和空服务库。"""
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.service = FakeService()
        self.addCleanup(self.service.close)
        self.addCleanup(self.temporary.cleanup)

    def arguments(self, dataset: str) -> list[str]:
        """构造不触碰科研真实数据的参数及合成输入。"""
        raw = self.root / "raw" / dataset
        if dataset == "triviaqa":
            wiki = raw / "rc/evidence/wikipedia"
            wiki.mkdir(parents=True)
            (wiki / "Alpha.txt").write_text(
                "Alpha body\u2028continued", encoding="utf-8"
            )
            (wiki / "Beta.txt").write_text("Beta body", encoding="utf-8")
            (wiki / "Empty.txt").write_text(" \n", encoding="utf-8")
        else:
            wiki = (
                raw / "corpus/enwiki-20171001-pages-meta-current-withlinks-abstracts/AA"
            )
            wiki.mkdir(parents=True)
            with bz2.open(wiki / "wiki_00.bz2", "wt", encoding="utf-8") as handle:
                for index, text in enumerate([["Alpha", "body"], [["Beta", "body"]]]):
                    handle.write(
                        json.dumps(
                            {
                                "id": index,
                                "title": f"Topic {index}",
                                "url": f"https://en.wikipedia.org/wiki?curid={index}",
                                "text": text,
                            }
                        )
                        + "\n"
                    )
        questions = self.root / f"{dataset}.jsonl"
        with questions.open("w", encoding="utf-8") as handle:
            for index in range(2):
                handle.write(
                    json.dumps(
                        {
                            "sample_id": f"{dataset}:dev:q{index}",
                            "dataset": dataset,
                            "question": f"Question\u2028{index}?",
                            "evaluation": {"answer": "GOLD_SENTINEL"},
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
        return [
            "--server-url",
            self.service.url,
            "--data-dir",
            str(raw),
            "--questions-file",
            str(questions),
            "--output-dir",
            str(self.root / "pools"),
            "--state-db",
            str(self.root / "ingest.sqlite3"),
            "--retries",
            "0",
        ]

    def sample(self) -> dict[str, Any]:
        """读取 manifest 指向的首个样本。"""
        with (self.root / "pools/manifest.jsonl").open(encoding="utf-8") as handle:
            entry = json.loads(next(handle))
        return json.loads((self.root / "pools" / entry["path"]).read_text())

    def test_triviaqa_full_snapshots_and_resume(self) -> None:
        """候选池是替换快照；成功样本、文章在续跑时均不重复提交。"""
        arguments = self.arguments("triviaqa") + ["--k", "1", "--ef", "2"]
        self.assertEqual(trivia.main(arguments), 0)
        sample = self.sample()
        self.assertEqual(
            [len(item["chunk_ids"]) for item in sample["snapshots"]], [1, 2, 1]
        )
        self.assertEqual(
            [item["event"] for item in sample["snapshots"]],
            [
                "candidates",
                "candidates",
                "final",
            ],
        )
        self.assertEqual(len(sample["chunks"]), 2)
        catalog = {chunk["chunk_id"]: chunk for chunk in sample["chunks"]}
        for snapshot in sample["snapshots"]:
            text = "".join(catalog[key]["text"] for key in snapshot["chunk_ids"])
            self.assertEqual(common.text_sha256(text), snapshot["text_sha256"])
            self.assertNotIn("elapsed_ms", text)
            self.assertNotIn("distance", text)
        self.assertNotIn("GOLD_SENTINEL", json.dumps(sample))
        self.assertTrue(
            all(set(query) == {"query", "k", "ef"} for query in self.service.queries)
        )
        self.assertEqual(trivia.main(arguments), 0)
        self.assertEqual(self.service.uploads, 2)
        self.assertEqual(self.service.batch_requests, 1)
        self.assertEqual(len(self.service.queries), 2)
        with self.assertRaisesRegex(ValueError, "输出配置"):
            trivia.main(arguments + ["--ef", "3"])

    def test_hotpot_default_output_and_parameters(self) -> None:
        """验证默认输出路径、k/ef 和全语料导入与问题标注无关。"""
        arguments = self.arguments("hotpotqa")
        output_index = arguments.index("--output-dir")
        del arguments[output_index : output_index + 2]
        with patch.object(common, "DATA_ROOT", self.root / "data"):
            self.assertEqual(hotpot.main(arguments), 0)
        output = self.root / "data/pipeline_pools/hotpotqa/wiki_dev_k5_ef8_v1"
        metadata = json.loads((output / "metadata.json").read_text())
        self.assertEqual(metadata["settings"]["k"], 5)
        self.assertEqual(metadata["settings"]["ef"], 8)
        self.assertEqual(metadata["num_ready"], 2)
        self.assertEqual(self.service.documents[0]["text"], "Alpha body")
        self.assertEqual(self.service.documents[1]["text"], "Beta body")
        self.assertEqual(
            metadata["settings"]["corpus"]["scope"], "all_available_wikipedia"
        )

    def test_stream_failure_is_retried_on_next_run(self) -> None:
        """缺少 done 的请求不能保存为成功，恢复后只重新检索。"""
        arguments = self.arguments("triviaqa")
        self.service.stream_mode = "truncated"
        self.assertEqual(trivia.main(arguments), 1)
        self.assertEqual(self.sample()["status"], "error")
        self.service.stream_mode = "normal"
        self.assertEqual(trivia.main(arguments), 0)
        self.assertEqual(self.sample()["status"], "ready")
        self.assertEqual(self.service.uploads, 2)

    def test_upload_response_lost_reconciles_before_retry(self) -> None:
        """HTTP 上传已提交但响应丢失时，应复用服务文章。"""
        arguments = self.arguments("triviaqa") + ["--retries", "1", "--import-only"]
        self.service.upload_mode = "drop_committed"
        with patch.object(common.time, "sleep"):
            self.assertEqual(trivia.main(arguments), 0)
        self.assertEqual(self.service.uploads, 2)
        self.assertEqual(len(self.service.documents), 2)
        self.assertEqual(self.service.batch_requests, 2)

    def test_uncommitted_batch_retries_and_resumes_safely(self) -> None:
        """幂等批量接口可以重试；耗尽次数后保留 pending 供续跑。"""
        arguments = self.arguments("triviaqa") + ["--retries", "2", "--import-only"]
        self.service.upload_mode = "drop_uncommitted"
        with (
            patch.object(common.time, "sleep"),
            self.assertRaisesRegex(RuntimeError, "批量上传未确认"),
        ):
            trivia.main(arguments)
        self.assertEqual(self.service.batch_requests, 3)
        self.assertEqual(self.service.uploads, 0)
        self.service.upload_mode = "normal"
        self.assertEqual(trivia.main(arguments), 0)
        self.assertEqual(len(self.service.documents), 2)

    def test_bad_batch_ack_recovers_legacy_pending_without_reupload(self) -> None:
        """整批确认前必须验证全部响应；旧版名称可恢复 pending。"""
        arguments = self.arguments("triviaqa") + ["--import-only"]
        self.service.bad_ack = True
        with self.assertRaisesRegex(RuntimeError, "批量上传未确认"):
            trivia.main(arguments)
        path = self.root / "ingest.sqlite3"
        with common.sqlite3.connect(path) as database:
            self.assertEqual(
                database.execute(
                    "SELECT count(*) FROM documents WHERE status='ready'"
                ).fetchone()[0],
                0,
            )
            database.execute(
                "UPDATE documents SET status='ready', server_id='doc0' WHERE name=?",
                (self.service.documents[0]["name"],),
            )
        for document in self.service.documents:
            document.pop("ingest_key")
        self.service.bad_ack = False
        self.assertEqual(trivia.main(arguments), 0)
        self.assertEqual(self.service.batch_requests, 1)
        self.assertEqual(self.service.lookup_requests, 1)
        self.assertEqual(self.service.uploads, 2)

    def test_batch_size_and_byte_budget_split_requests(self) -> None:
        """按文章数或 UTF-8 正文字节拆批，最后一批也必须提交。"""
        arguments = self.arguments("triviaqa") + [
            "--import-only",
            "--import-batch-bytes",
            "25",
        ]
        self.assertEqual(trivia.main(arguments), 0)
        self.assertEqual(self.service.batch_requests, 2)
        self.assertEqual(self.service.uploads, 2)
        self.assertEqual(trivia.main(arguments + ["--import-batch-size", "1"]), 0)
        self.assertEqual(self.service.batch_requests, 2)

    def test_changed_or_foreign_database_is_rejected(self) -> None:
        """不能在复用状态时连接不同服务库，也不能混入其他文档。"""
        arguments = self.arguments("triviaqa")
        self.assertEqual(trivia.main(arguments), 0)
        self.service.documents.append(
            {"id": "foreign", "name": "foreign", "text": "other"}
        )
        with self.assertRaisesRegex(ValueError, "未记录的文档"):
            trivia.main(arguments)
        self.service.documents.clear()
        with self.assertRaisesRegex(RuntimeError, "HTTP 404"):
            trivia.main(arguments)

    def test_subset_import_can_continue_to_full_corpus(self) -> None:
        """只入库部分文章后，可复用状态补齐，再独立检索。"""
        arguments = self.arguments("triviaqa")
        self.assertEqual(
            trivia.main(arguments + ["--import-only", "--max-documents", "1"]), 0
        )
        self.assertEqual(self.service.uploads, 1)
        self.assertEqual(trivia.main(arguments + ["--import-only"]), 0)
        self.assertEqual(self.service.uploads, 2)
        self.assertEqual(trivia.main(arguments + ["--skip-import"]), 0)
        self.assertEqual(len(self.service.queries), 2)
        metadata = json.loads((self.root / "pools/metadata.json").read_text())
        self.assertEqual(
            metadata["settings"]["corpus"]["scope"], "all_available_wikipedia"
        )

    def test_sse_protocol_rejects_error_wrong_id_and_incomplete_event(self) -> None:
        """覆盖错误事件、混入别的请求及不完整的 SSE 边界。"""
        trivia.main(self.arguments("triviaqa") + ["--import-only"])
        client = common.RetrievalClient(self.service.url, 10)
        for mode in ["error", "wrong_id"]:
            self.service.stream_mode = mode
            with self.assertRaises((RuntimeError, ValueError)):
                client.search("Question", 5, 8)
        with self.assertRaisesRegex(ValueError, "事件结束之前"):
            list(common.iter_sse(io.BytesIO(b"event: done\ndata: {}\n")))

    def test_cli_help_from_other_directory_and_invalid_ef(self) -> None:
        """入口可从其他目录启动，非法 ef 在联网或写入前拒绝。"""
        for dataset in ["triviaqa", "hotpotqa"]:
            result = subprocess.run(
                [
                    sys.executable,
                    str(PROCESSING_DIR / f"build_{dataset}_pipeline_pools.py"),
                    "--help",
                ],
                cwd=self.root,
                capture_output=True,
                text=True,
                check=True,
            )
            self.assertIn("--skip-import", result.stdout)
        with self.assertRaisesRegex(ValueError, "k <= ef"):
            trivia.main(["--k", "5", "--ef", "4"])
        for arguments in [
            ["--import-batch-size", "0"],
            ["--import-batch-size", "513"],
            ["--import-batch-bytes", "0"],
            ["--import-batch-bytes", "8388609"],
        ]:
            with self.assertRaisesRegex(ValueError, "import-batch"):
                trivia.main(arguments)


if __name__ == "__main__":
    unittest.main()
