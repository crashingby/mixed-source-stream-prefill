# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""向量服务的文档入库、SSE 采集和独立 PipelinePool 写入。"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import logging
import math
import sqlite3
import tempfile
import time
from collections.abc import Callable, Iterator
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, BinaryIO
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import ProxyHandler, Request, build_opener

from append_chunks_common import DATA_ROOT, fingerprint_file, text_sha256

logger = logging.getLogger(__name__)
SCHEMA_VERSION = "pipeline_pools/v1"
RENDERER_VERSION = "markdown_update_sources/v1"
RETRY_ERRORS = (OSError, URLError, ValueError)


@dataclass
class WikiDocument:
    """一篇待上传文章；来源信息仅在本地保存。"""

    source_id: str
    title: str
    text: str
    url: str
    source_file: str


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    """在同一目录写临时文件，再原子替换，避免中断留下半个样本。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, delete=False
    ) as handle:
        temporary = Path(handle.name)
        try:
            json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
            handle.write("\n")
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
    try:
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


@contextmanager
def exclusive_file(path: Path) -> Iterator[None]:
    """锁定本脚本的状态或输出，阻止多个采集器并发改写。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+b") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError(f"另一个采集器正在使用 {path}") from error
        yield


def iter_sse(stream: BinaryIO) -> Iterator[tuple[str, dict[str, Any]]]:
    """按字节换行解析 SSE，正文里的 Unicode 分隔符不会拆开 JSON。"""
    event = "message"
    parts: list[bytes] = []
    for raw in stream:
        line = raw.rstrip(b"\r\n")
        if not line:
            if parts:
                payload = json.loads(b"\n".join(parts).decode("utf-8"))
                if not isinstance(payload, dict):
                    raise ValueError("SSE data 必须是 JSON 对象")
                yield event, payload
            event, parts = "message", []
        elif not line.startswith(b":"):
            field, _, value = line.partition(b":")
            value = value.removeprefix(b" ")
            if field == b"event":
                event = value.decode("utf-8")
            elif field == b"data":
                parts.append(value)
    if parts:
        raise ValueError("SSE 在事件结束之前断开")


class RetrievalClient:
    """与单个检索服务通信；本机连接绕过代理。"""

    def __init__(self, url: str, timeout: float) -> None:
        self.url = url.rstrip("/")
        self.timeout = timeout
        parsed = urlsplit(self.url)
        if parsed.scheme not in {"http", "https"} or not parsed.hostname:
            raise ValueError(f"无效的服务地址：{url}")
        handlers = (
            [ProxyHandler({})]
            if parsed.hostname in {"localhost", "127.0.0.1", "::1"}
            else []
        )
        self.opener = build_opener(*handlers)

    def open(self, path: str, body: dict[str, Any] | None = None) -> Any:
        """发送一次请求；有副作用的 POST 不在这里盲目重试。"""
        data = (
            None
            if body is None
            else json.dumps(body, ensure_ascii=False).encode("utf-8")
        )
        request = Request(
            self.url + path,
            data=data,
            headers={"Content-Type": "application/json", "Accept": "*/*"},
        )
        try:
            return self.opener.open(request, timeout=self.timeout)
        except HTTPError as error:
            detail = error.read(4096).decode("utf-8", errors="replace")
            raise RuntimeError(f"{path}: HTTP {error.code}: {detail}") from error

    def json(self, path: str, body: dict[str, Any] | None = None) -> Any:
        """读取非流式接口的 JSON。"""
        with self.open(path, body) as response:
            return json.load(response)

    def health(self) -> dict[str, Any]:
        """确认服务已就绪，并取得模型、文档和 chunk 数量。"""
        health = self.json("/health")
        if health.get("status") != "ok":
            raise ValueError(f"服务未就绪：{health}")
        for key in ["model", "dimension", "document_count", "chunk_count"]:
            if key not in health:
                raise ValueError(f"health 缺少 {key}")
        return health

    @staticmethod
    def checked_documents(
        result: Any, expected_keys: list[str], *, allow_missing: bool = False
    ) -> list[dict[str, Any]]:
        """整批验证服务响应，拒绝遗漏、重复或错配的确认记录。"""
        if not isinstance(result, dict) or not isinstance(
            result.get("documents"), list
        ):
            raise ValueError("批量接口缺少 documents")
        documents = result["documents"]
        if len(documents) != len(expected_keys):
            raise ValueError("批量响应的文章数量不一致")
        ids = []
        for document, key in zip(documents, expected_keys):
            if not isinstance(document, dict) or document.get("ingest_key") != key:
                raise ValueError("批量响应的 ingest_key 不一致")
            document_id = document.get("id")
            if document_id is None and allow_missing:
                continue
            if not isinstance(document_id, str) or not document_id:
                raise ValueError("批量响应缺少有效文档 ID")
            ids.append(document_id)
        if len(set(ids)) != len(ids):
            raise ValueError("批量响应包含重复文档 ID")
        return documents

    def lookup_documents(self, items: list[dict[str, str]]) -> list[dict[str, Any]]:
        """按索引核对稳定标识、旧名称与哈希，恢复结果不明的上传。"""
        return self.checked_documents(
            self.json("/documents/lookup", {"documents": items}),
            [item["ingest_key"] for item in items],
            allow_missing=True,
        )

    def search(self, query: str, k: int, ef: int) -> list[tuple[str, dict[str, Any]]]:
        """及时收完整条候选流；断流、乱序和服务错误均拒绝标记成功。"""
        events: list[tuple[str, dict[str, Any]]] = []
        request_id: str | None = None
        sequence = 0
        final: dict[str, Any] | None = None
        body = {"query": query, "k": k, "ef": ef}
        with self.open("/search/stream", body) as response:
            if "text/event-stream" not in response.headers.get("Content-Type", ""):
                raise ValueError("检索接口未返回 SSE")
            for event, payload in iter_sse(response):
                if event == "error":
                    raise RuntimeError(f"检索失败：{payload.get('message', payload)}")
                if event not in {"candidates", "final", "done"}:
                    raise ValueError(f"未知 SSE 事件：{event}")
                current_id = payload.get("request_id")
                if not isinstance(current_id, str) or not current_id:
                    raise ValueError("SSE 缺少 request_id")
                request_id = request_id or current_id
                if current_id != request_id:
                    raise ValueError("一条检索流包含不同 request_id")
                if event != "done":
                    chunks = payload.get("chunks")
                    if not isinstance(chunks, list):
                        raise ValueError("候选池 chunks 必须是列表")
                    fields = {
                        "label",
                        "document_id",
                        "chunk_index",
                        "start",
                        "end",
                        "text",
                        "distance",
                    }
                    if any(
                        not isinstance(chunk, dict) or not fields <= chunk.keys()
                        for chunk in chunks
                    ):
                        raise ValueError("候选块缺少必需字段")
                if event == "candidates":
                    sequence += 1
                    if final is not None or payload.get("sequence") != sequence:
                        raise ValueError("候选池快照乱序")
                    elapsed = payload.get("elapsed_ms")
                    if (
                        not isinstance(elapsed, (int, float))
                        or not math.isfinite(elapsed)
                        or elapsed < 0
                    ):
                        raise ValueError("候选池 elapsed_ms 无效")
                    if len(payload["chunks"]) > ef:
                        raise ValueError("候选池超过配置的 ef")
                elif event == "final":
                    if final is not None or payload.get("updates") != sequence:
                        raise ValueError("final 重复或更新数量不一致")
                    if payload.get("ef") != ef or len(payload["chunks"]) > k:
                        raise ValueError("final 与查询参数不一致")
                    final = payload
                else:
                    if final is None or payload.get("updates") != sequence:
                        raise ValueError("done 缺少对应的完整 final")
                    events.append((event, payload))
                    return events
                events.append((event, payload))
        raise ValueError("检索流缺少 final/done，不能保存为成功样本")


class IngestState:
    """SQLite 只记录客户端入库进度和来源映射，不保存向量索引。"""

    def __init__(self, path: Path, settings: dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(path)
        self.db.row_factory = sqlite3.Row
        self.db.executescript(
            "CREATE TABLE IF NOT EXISTS metadata (key TEXT PRIMARY KEY, value TEXT);"
            "CREATE TABLE IF NOT EXISTS documents ("
            "source_id TEXT PRIMARY KEY, name TEXT UNIQUE, body_sha256 TEXT, "
            "provenance TEXT, server_id TEXT UNIQUE, status TEXT);"
        )
        previous = self.get("settings")
        if previous is not None and previous != settings:
            self.close()
            raise ValueError("入库状态的语料、服务或模型不一致，请使用新的状态文件")
        self.set("settings", settings)

    def close(self) -> None:
        """关闭客户端状态数据库。"""
        self.db.close()

    def get(self, key: str) -> Any:
        """读取持久化元数据。"""
        row = self.db.execute(
            "SELECT value FROM metadata WHERE key=?", (key,)
        ).fetchone()
        return None if row is None else json.loads(row[0])

    def set(self, key: str, value: Any) -> None:
        """立即提交状态，支持进程中断后续跑。"""
        with self.db:
            self.db.execute(
                "INSERT OR REPLACE INTO metadata VALUES (?, ?)",
                (key, json.dumps(value, ensure_ascii=False)),
            )

    def count(self) -> int:
        """返回已确认上传成功的文章数量。"""
        return self.db.execute(
            "SELECT count(*) FROM documents WHERE status='ready'"
        ).fetchone()[0]

    def recover(self, client: RetrievalClient) -> None:
        """恢复上传结果不明的文章，并验证服务器仍是同一个数据库。"""
        pending = self.db.execute(
            "SELECT * FROM documents WHERE status='pending'"
        ).fetchall()
        dataset = self.get("settings")["dataset"]
        for start in range(0, len(pending), 256):
            rows = pending[start : start + 256]
            found = client.lookup_documents(
                [
                    {
                        "ingest_key": self.ingest_key(dataset, row["source_id"]),
                        "name": row["name"],
                        "body_sha256": row["body_sha256"],
                    }
                    for row in rows
                ]
            )
            self.mark_ready_many(
                [
                    (row["source_id"], result["id"])
                    for row, result in zip(rows, found)
                    if result["id"] is not None
                ]
            )
        first = self.db.execute(
            "SELECT * FROM documents WHERE status='ready' LIMIT 1"
        ).fetchone()
        if first is not None:
            document = client.json(f"/documents/{first['server_id']}")
            if text_sha256(document["text"]) != first["body_sha256"]:
                raise ValueError("服务器文章与客户端入库记录不一致")
        health = client.health()
        if health["document_count"] != self.count():
            raise ValueError(
                "服务器含未记录的文档；请复用正确的 --state-db，"
                "或换一个空的服务数据库，避免重复入库"
            )
        frozen = self.get("frozen_health")
        if frozen is not None and frozen != health:
            raise ValueError("服务语料在采集之外发生了变化，请使用新的数据库和状态")

    @staticmethod
    def ingest_key(dataset: str, source_id: str) -> str:
        """生成与路径、批次和服务生成 ID 无关的稳定入库标识。"""
        return f"wiki:{dataset}:{text_sha256(source_id)}"

    def mark_ready_many(self, pairs: list[tuple[str, str]]) -> None:
        """一笔事务确认整批上传；pairs 为来源 ID、服务 ID。"""
        if not pairs:
            return
        with self.db:
            self.db.executemany(
                "UPDATE documents SET status='ready', server_id=? WHERE source_id=?",
                [(server_id, source_id) for source_id, server_id in pairs],
            )

    def upload_batch(
        self,
        client: RetrievalClient,
        documents: list[WikiDocument],
        dataset: str,
        retries: int,
    ) -> int:
        """先记录一批 pending，再幂等上传并整批确认；重跑跳过 ready。"""
        if not documents:
            return 0
        placeholders = ",".join("?" for _ in documents)
        existing = {
            row["source_id"]: row
            for row in self.db.execute(
                f"SELECT * FROM documents WHERE source_id IN ({placeholders})",
                [document.source_id for document in documents],
            )
        }
        pending = []
        items = []
        source_ids = []
        seen: dict[str, str] = {}
        for document in documents:
            digest = text_sha256(document.text)
            previous = seen.setdefault(document.source_id, digest)
            row = existing.get(document.source_id)
            if previous != digest or (row is not None and row["body_sha256"] != digest):
                raise ValueError(f"原始文章已改变：{document.source_id}")
            if document.source_id in source_ids or (
                row is not None and row["status"] == "ready"
            ):
                continue
            name = f"wiki:{dataset}:{text_sha256(document.source_id)[:32]}|"
            name = (name + " ".join(document.title.split()))[:255]
            if row is not None:
                name = row["name"]
            provenance = {
                "source_id": document.source_id,
                "title": document.title,
                "url": document.url,
                "source_file": document.source_file,
                "body_sha256": digest,
            }
            pending.append((document.source_id, name, digest, json.dumps(provenance)))
            source_ids.append(document.source_id)
            items.append(
                {
                    "ingest_key": self.ingest_key(dataset, document.source_id),
                    "name": name,
                    "text": document.text,
                }
            )
        if not items:
            return 0
        # 先提交全部来源映射；即使 HTTP 响应丢失，下一次仍能精确恢复。
        with self.db:
            self.db.executemany(
                "INSERT OR IGNORE INTO documents VALUES (?, ?, ?, ?, NULL, 'pending')",
                pending,
            )
        for attempt in range(retries + 1):
            try:
                result = client.json("/documents/batch", {"documents": items})
                confirmed = client.checked_documents(
                    result, [item["ingest_key"] for item in items]
                )
                for item, record, row in zip(items, confirmed, pending):
                    if (
                        record.get("name") != item["name"]
                        or record.get("body_sha256") != row[2]
                    ):
                        raise ValueError("批量响应的名称或正文哈希不一致")
                self.mark_ready_many(
                    [(key, record["id"]) for key, record in zip(source_ids, confirmed)]
                )
                timings = result.get("timings", {})
                logger.info(
                    "批量确认 %d 篇（复用 %d），编码 %.0f ms，索引 %.0f ms，"
                    "数据库 %.0f ms",
                    len(items),
                    sum(record.get("reused", False) for record in confirmed),
                    timings.get("embedding_ms", 0),
                    timings.get("index_ms", 0),
                    timings.get("database_ms", 0),
                )
                return len(items)
            except RETRY_ERRORS:
                if attempt == retries:
                    raise RuntimeError(
                        f"批量上传未确认（{len(items)} 篇），进度已保存；"
                        "重新运行会核对 pending，并用相同 ingest_key 安全续传。"
                    ) from None
                logger.warning(
                    "批量上传中断，使用相同 ingest_key 重试：%d", attempt + 1
                )
                time.sleep(min(2**attempt, 10))
        raise AssertionError("上传重试未正常结束")

    def provenance(self, server_id: str) -> dict[str, Any]:
        """将检索结果的服务 ID 还原为原始文章来源。"""
        row = self.db.execute(
            "SELECT provenance FROM documents WHERE server_id=? AND status='ready'",
            (server_id,),
        ).fetchone()
        if row is None:
            raise ValueError(f"检索返回未导入的文章：{server_id}")
        return json.loads(row[0])

    def fingerprint(self) -> str:
        """流式计算文章内容及服务 ID 指纹，区分重建后的数据库。"""
        digest = hashlib.sha256()
        for row in self.db.execute(
            "SELECT source_id, body_sha256, server_id FROM documents "
            "WHERE status='ready' ORDER BY source_id"
        ):
            digest.update((json.dumps(list(row)) + "\n").encode("utf-8"))
        return digest.hexdigest()


def build_pool(
    sample_id: str,
    events: list[tuple[str, dict[str, Any]]],
    state: IngestState,
) -> dict[str, Any]:
    """共享块正文，快照保存完整的有序引用；最终 Top-k 也是一个快照。"""
    catalog: dict[str, dict[str, Any]] = {}
    sources: dict[str, dict[str, Any]] = {}
    snapshots: list[dict[str, Any]] = []
    final: dict[str, Any] = {}
    for event, payload in events:
        if event == "done":
            continue
        chunk_ids: list[str] = []
        distances: list[float] = []
        texts: list[str] = []
        for chunk in payload["chunks"]:
            document_id = chunk["document_id"]
            if document_id not in sources:
                sources[document_id] = state.provenance(document_id)
            provenance = sources[document_id]
            identity = [
                provenance["source_id"],
                provenance["body_sha256"],
                chunk["chunk_index"],
                chunk["start"],
                chunk["end"],
            ]
            chunk_id = "u" + text_sha256(json.dumps(identity))[:24]
            distance = chunk["distance"]
            if not isinstance(distance, (int, float)) or not math.isfinite(distance):
                raise ValueError("候选块距离不是有限数值")
            if not isinstance(chunk["text"], str) or not chunk["text"].strip():
                raise ValueError("候选块没有正文")
            title = " ".join(provenance["title"].split())
            text = f"\n\n### [{chunk_id}] {title}\n"
            if provenance["url"]:
                text += f"Source: {provenance['url']}\n"
            text += "\n" + chunk["text"] + "\n\n---\n"
            entry = {
                "chunk_id": chunk_id,
                "document_id": chunk["document_id"],
                "label": chunk["label"],
                "source": provenance,
                "chunk_index": chunk["chunk_index"],
                "body_span": [chunk["start"], chunk["end"]],
                "body": chunk["text"],
                "text": text,
            }
            if chunk_id in catalog and catalog[chunk_id] != entry:
                raise ValueError(f"同一候选块在流中改变了内容：{chunk_id}")
            catalog[chunk_id] = entry
            chunk_ids.append(chunk_id)
            distances.append(distance)
            texts.append(text)
        if len(set(chunk_ids)) != len(chunk_ids):
            raise ValueError("同一个快照包含重复候选块")
        rendered = "".join(texts)
        snapshots.append(
            {
                "sequence_no": len(snapshots),
                "event": event,
                "source_sequence": payload.get("sequence"),
                "source_elapsed_ms": payload.get("elapsed_ms"),
                "chunk_ids": chunk_ids,
                "distances": distances,
                "text_chars": len(rendered),
                "text_sha256": text_sha256(rendered),
            }
        )
        if event == "final":
            final = {key: value for key, value in payload.items() if key != "chunks"}
    if not final or not snapshots[-1]["chunk_ids"]:
        raise ValueError("检索未得到非空最终结果")
    return {
        "schema": SCHEMA_VERSION,
        "renderer": RENDERER_VERSION,
        "sample_id": sample_id,
        "status": "ready",
        "request_id": final["request_id"],
        "chunks": list(catalog.values()),
        "snapshots": snapshots,
        "final": final,
    }


def add_arguments(parser: argparse.ArgumentParser, dataset: str, port: int) -> None:
    """添加两个入口共用的服务、输出和续跑选项。"""
    parser.add_argument("--server-url", default=f"http://127.0.0.1:{port}")
    parser.add_argument("--data-dir", type=Path, default=DATA_ROOT / "raw" / dataset)
    parser.add_argument("--questions-file", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--state-db", type=Path, help="客户端入库状态，不是向量库")
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--ef", type=int, default=8)
    parser.add_argument("--timeout", type=float, default=300, help="网络等待秒数")
    parser.add_argument("--retries", type=int, default=3, help="首次失败后的重试次数")
    parser.add_argument("--max-documents", type=int, default=0, help="0 为全部语料")
    parser.add_argument("--max-requests", type=int, default=0, help="0 为全部问题")
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument(
        "--import-batch-size",
        type=int,
        default=256,
        help="每次批量入库的文章数，最多 512",
    )
    parser.add_argument(
        "--import-batch-bytes",
        type=int,
        default=4 * 1024 * 1024,
        help="每批正文的 UTF-8 字节上限，最多 8 MiB",
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--import-only", action="store_true", help="只入库，不检索")
    mode.add_argument("--skip-import", action="store_true", help="复用已记录的入库结果")


def import_corpus(
    args: argparse.Namespace,
    dataset: str,
    documents: Callable[[], Iterator[WikiDocument]],
    state: IngestState,
    client: RetrievalClient,
) -> dict[str, Any]:
    """先完成语料导入，再固定此次采集的内容指纹与服务计数。"""
    state.recover(client)
    logger.info("已有入库进度：%d 篇已确认", state.count())
    if args.max_documents and state.count() > args.max_documents:
        raise ValueError("已入库文档多于 --max-documents；子集试跑请使用独立服务数据库")
    if not args.skip_import and not state.get("import_complete"):
        state.set("frozen_health", None)
        count, uploaded = 0, 0
        batch: list[WikiDocument] = []
        batch_bytes, reported = 0, 0
        started = time.perf_counter()

        def flush_batch() -> None:
            """确认一批后再继续读取语料，失败时保留 pending。"""
            nonlocal uploaded, batch_bytes, reported
            uploaded += state.upload_batch(client, batch, dataset, args.retries)
            batch.clear()
            batch_bytes = 0
            if args.log_every and count - reported >= args.log_every:
                elapsed = time.perf_counter() - started
                logger.info(
                    "入库：已遍历 %d，本轮确认 %d，平均 %.1f 篇/秒",
                    count,
                    uploaded,
                    uploaded / elapsed if elapsed else 0,
                )
                reported = count

        for document in documents():
            if args.max_documents and count >= args.max_documents:
                break
            if not document.text.strip():
                continue
            size = len(document.text.encode("utf-8"))
            if size > args.import_batch_bytes:
                raise ValueError(
                    f"文章 {document.source_id} 有 {size} 字节，超过 "
                    "--import-batch-bytes；请增大批量字节上限"
                )
            if batch and (
                len(batch) >= args.import_batch_size
                or batch_bytes + size > args.import_batch_bytes
            ):
                flush_batch()
            batch.append(document)
            batch_bytes += size
            count += 1
        flush_batch()
        state.set("import_complete", not bool(args.max_documents))
        logger.info("本轮入库：已遍历 %d，新上传 %d", count, uploaded)
    health = client.health()
    if not health["chunk_count"] or health["document_count"] != state.count():
        raise ValueError("服务库为空或文档数量与入库状态不一致")
    state.set("frozen_health", health)
    return {
        "fingerprint": state.fingerprint(),
        "scope": "all_available_wikipedia"
        if state.get("import_complete")
        else "subset",
        "health": health,
    }


def collect_questions(
    args: argparse.Namespace,
    dataset: str,
    corpus: dict[str, Any],
    output: Path,
    state: IngestState,
    client: RetrievalClient,
) -> int:
    """采集独立问题清单；已有成功样本复用，失败样本在下次运行重试。"""
    settings = {
        "dataset": dataset,
        "server_url": client.url,
        "k": args.k,
        "ef": args.ef,
        "max_requests": args.max_requests,
        "max_documents": args.max_documents,
        "questions": fingerprint_file(args.questions_file),
        "corpus": corpus,
    }
    metadata_path = output / "metadata.json"
    if metadata_path.exists():
        previous = json.loads(metadata_path.read_text(encoding="utf-8"))
        if (
            previous.get("schema") != SCHEMA_VERSION
            or previous.get("settings") != settings
        ):
            raise ValueError("输出配置或语料已改变，请使用新的 --output-dir")
    elif output.exists() and any(path.name != ".lock" for path in output.iterdir()):
        raise ValueError(f"拒绝写入没有本生成器 metadata 的非空目录：{output}")
    metadata = {
        "schema": SCHEMA_VERSION,
        "renderer": RENDERER_VERSION,
        "generator": f"build_{dataset}_pipeline_pools",
        "settings": settings,
        "state_db": str(args.state_db),
        "uses_gold_for_selection": False,
        "completion_status": "running",
    }
    atomic_json(metadata_path, metadata)
    ready = errors = processed = resumed = 0
    seen: set[str] = set()
    # manifest 逐轮重建，避免续跑重复追加；每个样本成功后独立原子落盘。
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=output, delete=False
    ) as manifest:
        temporary = Path(manifest.name)
        try:
            with args.questions_file.open(encoding="utf-8") as questions:
                for line in questions:
                    if not line.strip():
                        continue
                    if args.max_requests and processed >= args.max_requests:
                        break
                    question = json.loads(line)
                    sample_id, query = question["sample_id"], question["question"]
                    if question.get("dataset") != dataset or sample_id in seen:
                        raise ValueError(f"问题数据集不符或 ID 重复：{sample_id}")
                    if not isinstance(query, str) or not query.strip():
                        raise ValueError(f"空问题：{sample_id}")
                    seen.add(sample_id)
                    path = Path("samples") / f"{text_sha256(sample_id)}.json"
                    destination = output / path
                    sample: dict[str, Any] = {}
                    if destination.exists():
                        sample = json.loads(destination.read_text(encoding="utf-8"))
                        if sample.get("sample_id") != sample_id:
                            raise ValueError(f"样本 ID 不一致：{destination}")
                    if sample.get("status") == "ready":
                        resumed += 1
                    else:
                        for attempt in range(args.retries + 1):
                            try:
                                events = client.search(query, args.k, args.ef)
                                sample = build_pool(sample_id, events, state)
                                break
                            except (RuntimeError, *RETRY_ERRORS) as error:
                                if attempt < args.retries:
                                    time.sleep(min(2**attempt, 10))
                                else:
                                    logger.error("采集失败 %s：%s", sample_id, error)
                                    sample = {
                                        "schema": SCHEMA_VERSION,
                                        "sample_id": sample_id,
                                        "status": "error",
                                        "error": str(error),
                                    }
                        atomic_json(destination, sample)
                    processed += 1
                    ready += sample["status"] == "ready"
                    errors += sample["status"] == "error"
                    entry = {
                        "sample_id": sample_id,
                        "path": path.as_posix(),
                        "status": sample["status"],
                        "num_snapshots": len(sample.get("snapshots", [])),
                    }
                    manifest.write(json.dumps(entry, ensure_ascii=False) + "\n")
                    if args.log_every and processed % args.log_every == 0:
                        logger.info(
                            "检索：%d 请求，成功 %d，失败 %d", processed, ready, errors
                        )
            if client.health() != corpus["health"]:
                raise ValueError("采集期间服务语料发生了变化，本轮结果不能发布")
            if not processed:
                raise ValueError("问题清单为空")
            manifest.close()
            temporary.replace(output / "manifest.jsonl")
        finally:
            temporary.unlink(missing_ok=True)
    metadata.update(
        completion_status="complete_with_errors" if errors else "complete",
        num_requests=processed,
        num_ready=ready,
        num_errors=errors,
        num_resumed=resumed,
    )
    atomic_json(metadata_path, metadata)
    logger.info(
        "采集结束：%s，成功 %d，失败 %d，复用 %d", output, ready, errors, resumed
    )
    return 1 if errors else 0


def run_pipeline(
    args: argparse.Namespace,
    dataset: str,
    question_version: str,
    corpus_dir: Path,
    documents: Callable[[], Iterator[WikiDocument]],
) -> int:
    """校验默认路径，锁定状态并顺序执行入库和逐问题检索。"""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    if not 1 <= args.k <= 1000 or not args.k <= args.ef <= 10000:
        raise ValueError("需要 1 <= k <= 1000 且 k <= ef <= 10000")
    for name in ["max_documents", "max_requests", "retries", "log_every"]:
        if getattr(args, name) < 0:
            raise ValueError(f"--{name.replace('_', '-')} 不能为负数")
    if not math.isfinite(args.timeout) or args.timeout <= 0:
        raise ValueError("--timeout 必须是有限正数")
    if not 1 <= args.import_batch_size <= 512:
        raise ValueError("--import-batch-size 必须在 1～512 之间")
    if not 1 <= args.import_batch_bytes <= 8 * 1024 * 1024:
        raise ValueError("--import-batch-bytes 必须在 1 字节～8 MiB 之间")
    args.questions_file = args.questions_file or (
        DATA_ROOT / "questions" / dataset / f"{question_version}.jsonl"
    )
    if not args.import_only and not args.questions_file.is_file():
        raise FileNotFoundError(f"缺少问题清单，请先生成 Append：{args.questions_file}")
    if not args.skip_import and not corpus_dir.is_dir():
        raise FileNotFoundError(f"缺少 Wikipedia 语料目录：{corpus_dir}")
    client = RetrievalClient(args.server_url, args.timeout)
    try:
        health = client.health()
    except (RuntimeError, *RETRY_ERRORS) as error:
        raise RuntimeError(f"请先启动对应向量服务：{client.url}；{error}") from error
    version = f"wiki_dev_k{args.k}_ef{args.ef}_v1"
    for name, label in [("max_documents", "docs"), ("max_requests", "first")]:
        if value := getattr(args, name):
            version += f"_{label}{value}"
    output = args.output_dir or DATA_ROOT / "pipeline_pools" / dataset / version
    args.state_db = args.state_db or (
        DATA_ROOT
        / "pipeline_pools"
        / dataset
        / "_state"
        / f"wiki_v1_{text_sha256(client.url)[:12]}.sqlite3"
    )
    if not args.import_only:
        for input_path in [args.state_db, args.questions_file]:
            if input_path.resolve().is_relative_to(output.resolve()):
                raise ValueError("问题清单和入库状态必须位于输出版本目录之外")
        metadata_path = output / "metadata.json"
        if metadata_path.exists():
            previous = json.loads(metadata_path.read_text(encoding="utf-8"))
            expected = {
                "dataset": dataset,
                "server_url": client.url,
                "k": args.k,
                "ef": args.ef,
                "max_requests": args.max_requests,
                "max_documents": args.max_documents,
                "questions": fingerprint_file(args.questions_file),
            }
            saved = previous.get("settings", {})
            if previous.get("schema") != SCHEMA_VERSION or any(
                saved.get(key) != value for key, value in expected.items()
            ):
                raise ValueError("输出配置或问题清单已改变，请使用新的 --output-dir")
        elif output.exists() and any(path.name != ".lock" for path in output.iterdir()):
            raise ValueError(f"拒绝写入没有本生成器 metadata 的非空目录：{output}")
    state_settings = {
        "schema": "wikipedia_ingest/v1",
        "dataset": dataset,
        "corpus_dir": str(corpus_dir.resolve()),
        "server_url": client.url,
        "model": health["model"],
        "dimension": health["dimension"],
    }
    with ExitStack() as stack:
        stack.enter_context(exclusive_file(Path(str(args.state_db) + ".lock")))
        state = IngestState(args.state_db, state_settings)
        stack.callback(state.close)
        if not args.import_only:
            stack.enter_context(exclusive_file(output / ".lock"))
        corpus = import_corpus(args, dataset, documents, state, client)
        logger.info(
            "语料就绪：%d 文档，%d 块", state.count(), corpus["health"]["chunk_count"]
        )
        if args.import_only:
            return 0
        return collect_questions(args, dataset, corpus, output, state, client)
