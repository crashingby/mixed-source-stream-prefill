# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""稳定 KV 的区间级 NIXL WRITE；状态由既有 Push writer 线程独占。"""

from __future__ import annotations

import queue
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import msgspec

from vllm.distributed.kv_transfer.kv_connector.v1.nixl.metadata import (
    STABLE_KV_NOTIF_PREFIX,
    StableKVWorkerMetadata,
)
from vllm.logger import init_logger

if TYPE_CHECKING:
    from vllm.distributed.kv_transfer.kv_connector.v1.nixl.push_worker import (
        NixlPushConnectorWorker,
    )

logger = init_logger(__name__)


def validate_stable_kv_write(command: dict[str, Any]) -> None:
    """拒绝范围、组数或块数不一致的 WRITE，不允许静默裁短。"""
    for field_name in ("session_id", "range_id"):
        if not isinstance(command.get(field_name), str) or not command[field_name]:
            raise ValueError(f"Stable KV {field_name} must be a nonempty string")
    for field_name in ("generation", "start_block", "end_block"):
        value = command.get(field_name)
        if type(value) is not int or value < 0:
            raise ValueError(f"Stable KV {field_name} must be a nonnegative int")
    num_blocks = command["end_block"] - command["start_block"]
    if num_blocks <= 0:
        raise ValueError("Stable KV WRITE requires a nonempty block interval")
    source = command.get("source_block_ids")
    target = command.get("target_block_ids")
    if not isinstance(source, (list, tuple)) or not isinstance(target, (list, tuple)):
        raise ValueError("Stable KV block IDs must be grouped sequences")
    if not source or len(source) != len(target):
        raise ValueError("Stable KV source/target group counts must match")
    for groups in (source, target):
        for group in groups:
            if not isinstance(group, (list, tuple)) or len(group) != num_blocks:
                raise ValueError("Stable KV each group must exactly cover the interval")
            if any(type(block_id) is not int or block_id < 0 for block_id in group):
                raise ValueError("Stable KV block IDs must be nonnegative integers")
            if len(set(group)) != len(group):
                raise ValueError("Stable KV block IDs within a group must be unique")


@dataclass
class _StableWrite:
    """保存尚未回收的 NIXL handle 和区间身份。"""

    handle: int  # 仅 writer 线程提交、查询和回收。
    command: dict[str, Any]  # 冻结源/目标映射直到 WRITE 完成。
    submitted_at: float  # 本机单调时钟，用于观测完成延迟。


class StableKVPushWorker:
    """在 Push writer 内执行增量传输，通过线程安全队列返回事件。

    主线程只调用 enqueue()/drain_events()；其余方法只由 writer 调用。
    drain 仅表示本地 WRITE 已停止，跨端资源释放仍需 Scheduler 的握手。
    """

    def __init__(self, worker: NixlPushConnectorWorker) -> None:
        self.worker = worker  # 复用已注册 GPU 内存和 NIXL peer 握手。
        self.commands: queue.Queue[dict[str, Any]] = queue.Queue()
        self.events: queue.Queue[dict[str, Any]] = queue.Queue()
        self._writes: dict[tuple[str, int, str], _StableWrite] = {}
        self._submitted: dict[tuple[str, int, str], bytes] = {}  # 冻结已提交映射。
        self._completed: dict[tuple[str, int, str], dict[str, Any]] = {}
        self._draining: set[tuple[str, int]] = set()  # 禁止取消之后的迟到 WRITE。
        self._drain_reported: set[tuple[str, int]] = set()

    def enqueue(self, commands: list[dict[str, Any]]) -> None:
        """从 worker 主线程排入命令，执行仍由 writer 线程负责。"""
        for command in commands:
            self.commands.put(command)

    def drain_events(self) -> StableKVWorkerMetadata | None:
        """将已发生的控制/区间事件交给本轮 worker 输出。"""
        events: list[dict[str, Any]] = []
        while True:
            try:
                events.append(self.events.get_nowait())
            except queue.Empty:
                break
        return StableKVWorkerMetadata(events) if events else None

    def has_work(self) -> bool:
        """只有实际在途传输或待处理命令才让 writer 自轮询。"""
        return bool(self._writes) or not self.commands.empty()

    @staticmethod
    def _identity(command: dict[str, Any]) -> dict[str, Any]:
        """生成不含 block 地址的完成事件身份，避免混入请求级通知。"""
        return {
            name: command[name]
            for name in (
                "session_id",
                "generation",
                "range_id",
                "start_block",
                "end_block",
                "early",
            )
            if name in command
        }

    def _ensure_peer(self, peer: dict[str, Any], session_id: str) -> str:
        """握手并验证首版只连接单 rank 节点，返回目标 NIXL agent。"""
        if peer.get("tp_size", 1) != 1:
            raise ValueError("Stable KV Push currently requires peer TP=1")
        engine_id = peer["engine_id"]
        if not self.worker._ensure_d_handshake(
            engine_id, peer["host"], int(peer["port"]), 1, session_id
        ):
            raise RuntimeError(f"Stable KV handshake failed: {engine_id}")
        agents = self.worker._remote_agents[engine_id]
        if len(agents) != 1:
            raise ValueError("Stable KV Push requires exactly one remote agent")
        return next(iter(agents.values()))

    def _send_control(self, command: dict[str, Any]) -> None:
        """发送 Scheduler 产生的控制消息；不在后台分配或释放 KV。"""
        payload = command["payload"]
        agent = self._ensure_peer(command["peer"], payload.get("session_id", ""))
        message = STABLE_KV_NOTIF_PREFIX + msgspec.msgpack.encode(
            {"kind": "control", "payload": payload}
        )
        self.worker.nixl_wrapper.send_notif(agent, notif_msg=message)

    def process_commands(self) -> None:
        """按 FIFO 执行控制、WRITE 与本地 drain 命令。"""
        while True:
            try:
                command = self.commands.get_nowait()
            except queue.Empty:
                break
            try:
                kind = command["kind"]
                if kind == "control":
                    self._send_control(command)
                elif kind == "write":
                    self._submit_write(command)
                elif kind == "drain":
                    key = (command["session_id"], command["generation"])
                    self._draining.add(key)
                else:
                    raise ValueError(f"Unknown stable KV command kind: {kind}")
            except Exception as error:
                # 控制消息的会话身份位于 payload；失败也必须能返回到所属请求。
                identity = command.get("payload", command)
                self.events.put(
                    {
                        **self._identity(identity),
                        "kind": "error",
                        "error": str(error),
                        "operation": command.get("kind"),
                    }
                )
                logger.exception("Stable KV command failed: %s", command.get("kind"))
        self._report_drained()

    def handle_notification(self, notification: bytes) -> None:
        """接收分批协议消息，不触发 finished_sending/finished_recving。"""
        try:
            event = msgspec.msgpack.decode(notification[len(STABLE_KV_NOTIF_PREFIX) :])
            if not isinstance(event, dict) or event.get("kind") not in (
                "control",
                "received",
            ):
                raise ValueError("Malformed stable KV notification")
            self.events.put(event)
        except Exception:
            logger.exception("Invalid stable KV notification")

    def _submit_write(self, command: dict[str, Any]) -> None:
        """精确匹配各层 block descriptors，提交一个区间级 WRITE。"""
        validate_stable_kv_write(command)
        key = (command["session_id"], command["generation"], command["range_id"])
        if key[:2] in self._draining:
            raise ValueError(f"Stable KV session is draining: {key[:2]}")
        # 固定字段顺序，HTTP/msgpack 字典键顺序不影响幂等判断。
        mapping = msgspec.msgpack.encode(
            [
                command["peer"]["engine_id"],
                command["start_block"],
                command["end_block"],
                command["source_block_ids"],
                command["target_block_ids"],
                command.get("early", False),
            ]
        )
        if key in self._submitted:
            if self._submitted[key] != mapping:
                raise ValueError(
                    f"Stable KV range reused with different mapping: {key}"
                )
            if key in self._completed:
                self.events.put(self._completed[key])
            return  # 相同 range_id 幂等，重复 ACK 由 Scheduler 按区间去重。
        self._ensure_peer(command["peer"], command["session_id"])
        worker = self.worker
        engine_id = command["peer"]["engine_id"]
        assert worker.transfer_topo is not None
        remote_info = worker.transfer_topo.get_engine_info(engine_id)
        if remote_info.remote_block_size != worker.block_size:
            raise ValueError("Stable KV Push requires equal P/D kernel block sizes")
        if remote_info.remote_physical_blocks_per_logical != (
            worker._physical_blocks_per_logical_kv_block
        ):
            raise ValueError("Stable KV Push requires equal P/D logical block sizes")
        # 两侧独立展开逻辑块；绝不沿用旧 Push 的裁短对齐逻辑。
        source_ids = worker._logical_to_kernel_block_ids(command["source_block_ids"])
        target_ids = worker._logical_to_remote_kernel_block_ids(
            command["target_block_ids"], remote_info.remote_physical_blocks_per_logical
        )
        if len(source_ids) != len(worker._group_spec_types):
            raise ValueError("Stable KV WRITE must cover every configured KV group")
        for groups, engine in ((source_ids, worker.engine_id), (target_ids, engine_id)):
            if any(
                block >= worker.dst_num_blocks[engine]
                for group in groups
                for block in group
            ):
                raise ValueError(f"Stable KV block outside registered pool: {engine}")
        local_desc = worker._compute_desc_ids(
            block_ids=source_ids,
            dst_num_blocks=worker.dst_num_blocks[worker.engine_id],
            block_size_ratio=1,
            physical_blocks_per_logical=worker._physical_blocks_per_logical_kv_block,
        )
        remote_desc = worker._compute_desc_ids(
            block_ids=target_ids,
            dst_num_blocks=worker.dst_num_blocks[engine_id],
            block_size_ratio=None,
            physical_blocks_per_logical=remote_info.remote_physical_blocks_per_logical,
        )
        if len(local_desc) != len(remote_desc) or not len(local_desc):
            raise ValueError("Stable KV descriptor counts must match and be nonzero")
        ranks = worker.dst_xfer_side_handles[engine_id]
        if len(ranks) != 1:
            raise ValueError("Stable KV Push requires one destination rank")
        notification = STABLE_KV_NOTIF_PREFIX + msgspec.msgpack.encode(
            {**self._identity(command), "kind": "received"}
        )
        handle = worker.nixl_wrapper.make_prepped_xfer(
            "WRITE",
            worker.src_xfer_handles_by_block_size[remote_info.remote_block_size],
            local_desc,
            next(iter(ranks.values())),
            remote_desc,
            notif_msg=notification,
        )
        try:
            worker.nixl_wrapper.transfer(handle)
        except Exception:
            worker.nixl_wrapper.release_xfer_handle(handle)
            raise
        self._writes[key] = _StableWrite(handle, command, time.perf_counter())
        self._submitted[key] = mapping

    def poll_transfers(self) -> None:
        """回收完成句柄并分别报告带宽统计和源块可解除 pin 的时刻。"""
        worker = self.worker
        for key, transfer in list(self._writes.items()):
            try:
                state = worker.nixl_wrapper.check_xfer_state(transfer.handle)
                if state == "PROC":
                    continue
                if state != "DONE":
                    raise RuntimeError(f"NIXL stable WRITE state={state}")
                telemetry = worker.nixl_wrapper.get_xfer_telemetry(transfer.handle)
                # NIXL telemetry 的 duration 单位为微秒；与网络时钟无关。
                event = {
                    **self._identity(transfer.command),
                    "kind": "sent",
                    "bytes": int(telemetry.totalBytes),
                    "transfer_ms": float(telemetry.xferDuration) / 1000.0,
                    "post_ms": float(telemetry.postDuration) / 1000.0,
                    "completion_ms": (time.perf_counter() - transfer.submitted_at)
                    * 1000,
                }
                worker.xfer_stats.record_transfer(telemetry)
            except Exception as error:
                event = {
                    **self._identity(transfer.command),
                    "kind": "error",
                    "operation": "write",
                    "error": str(error),
                }
                worker.xfer_stats.record_failed_transfer()
            # 必须先释放 NIXL 句柄，之后 Scheduler 才能接收到安全回收事件。
            try:
                worker.nixl_wrapper.release_xfer_handle(transfer.handle)
            except Exception:
                logger.exception("Cannot release stable WRITE; keeping blocks pinned")
                continue
            del self._writes[key]
            self._completed[key] = event
            self.events.put(event)
        self._report_drained()

    def _report_drained(self) -> None:
        """所有本地 WRITE 消失后报告 drain；远端释放由控制协议决定。"""
        for key in self._draining - self._drain_reported:
            if any(write_key[:2] == key for write_key in self._writes):
                continue
            self._drain_reported.add(key)
            # 保留会话 tombstone 拦截迟到命令，释放大块的区间映射记录。
            for range_key in list(self._submitted):
                if range_key[:2] == key:
                    self._submitted.pop(range_key)
                    self._completed.pop(range_key, None)
            self.events.put(
                {"kind": "drained", "session_id": key[0], "generation": key[1]}
            )
