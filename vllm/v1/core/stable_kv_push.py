# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Stable KV Push 的接收会话、范围准入与原子 block table 收养。"""

import math
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, cast

from vllm.logger import init_logger
from vllm.v1.engine.prompt_stream import parse_prompt_stream_padding_range
from vllm.v1.kv_cache_interface import FullAttentionSpec
from vllm.v1.request import (
    PromptStreamExecutionPhase,
    PromptStreamKVSegment,
    RequestStatus,
)

if TYPE_CHECKING:
    from vllm.distributed.kv_transfer.kv_connector.v1.nixl.connector import (
        NixlPushConnector,
    )
    from vllm.v1.core.kv_cache_utils import KVCacheBlock
    from vllm.v1.core.sched.output import SchedulerOutput
    from vllm.v1.core.sched.scheduler import Scheduler
    from vllm.v1.request import Request

logger = init_logger(__name__)


@dataclass
class StableKVRange:
    """固定一次 WRITE 的序列块区间及源/目标映射。"""

    range_id: str
    start_block: int
    end_block: int
    source_block_ids: list[list[int]]
    early: bool
    target_block_ids: list[list[int]] = field(default_factory=list)
    pinned_blocks: list["KVCacheBlock"] = field(default_factory=list)
    granted: bool = False  # 目标地址已经授予，不能再改变。
    completed: bool = False  # P 指发送完成，D 指接收完成。
    stats_recorded: bool = False  # 网络统计按 range_id 只累加一次。


@dataclass
class StableKVSession:
    """长期接收关系；D 在 adopted 前不属于模型生成请求。"""

    session_id: str
    request_id: str
    peer: dict[str, Any]
    producer: bool
    early_push: bool
    generation: int = 0
    state: str = "receiving"
    error: str | None = None
    ranges: dict[str, StableKVRange] = field(default_factory=dict)
    offered_blocks: int = 0  # P 已发布的连续范围，不等于已传完。
    final_tokens: list[int] | None = None
    final_params: dict[str, Any] | None = None
    adopted_request_id: str | None = None
    local_drained: bool = False
    peer_drained: bool = False
    last_activity: float = field(default_factory=time.monotonic)
    closed_at: float | None = None
    metrics: dict[str, int | float] = field(
        default_factory=lambda: {
            "early_transferred_bytes": 0,
            "final_transferred_bytes": 0,
            "total_transferred_bytes": 0,
            "early_transferred_blocks": 0,
            "final_transferred_blocks": 0,
            "early_transfer_ms": 0.0,
            "final_transfer_ms": 0.0,
        }
    )

    @property
    def reservation_id(self) -> str:
        """独立于用户 request ID 的内部块所有权 key。"""
        return f"stable-kv:{self.session_id}:{self.generation}"


class StableKVPushController:
    """仅在显式启用时维护增量 WRITE 与接收会话。

    所有会话及 KV pool 修改均发生于 EngineCore 主线程。Worker 线程只通过
    connector metadata 返回消息，避免与正常 Scheduler 争用同一张 block 表。
    """

    def __init__(self, scheduler: "Scheduler") -> None:
        self.scheduler = scheduler
        self.manager = scheduler.kv_cache_manager
        self.connector = cast("NixlPushConnector", scheduler.connector)
        config = scheduler.vllm_config
        transfer = config.kv_transfer_config
        assert transfer is not None and self.connector is not None
        extra = transfer.kv_connector_extra_config
        self.producer = transfer.kv_role == "kv_producer"
        self.block_size = scheduler.block_size
        self.num_groups = len(scheduler.kv_cache_config.kv_cache_groups)
        self.max_sessions = int(extra.get("stable_kv_max_sessions", 128))
        self.timeout_s = float(extra.get("stable_kv_session_timeout_s", 600))
        self.poll_interval_s = float(extra.get("stable_kv_poll_interval_ms", 5)) / 1000
        self.max_reserved_fraction = float(
            extra.get("stable_kv_max_reserved_fraction", 0.5)
        )
        if (
            self.max_sessions < 1
            or not math.isfinite(self.timeout_s)
            or self.timeout_s <= 0
            or not math.isfinite(self.poll_interval_s)
            or self.poll_interval_s <= 0
            or not 0 < self.max_reserved_fraction <= 1
        ):
            raise ValueError("Invalid stable_kv session/poll/reservation limits")
        # 首版通过同步步的返回证明 KV 写入完成；不接受异步或多流水线配置。
        if (
            transfer.kv_connector != "NixlPushConnector"
            or transfer.kv_role not in ("kv_producer", "kv_consumer")
            or config.scheduler_config.async_scheduling
            or config.parallel_config.tensor_parallel_size != 1
            or config.parallel_config.pipeline_parallel_size != 1
            or config.parallel_config.data_parallel_size != 1
            or config.speculative_config is not None
            or config.model_config.is_encoder_decoder
            or config.lora_config is not None
        ):
            raise ValueError(
                "stable_kv_push requires NixlPushConnector, TP=PP=DP=1, "
                "--no-async-scheduling, decoder-only and no speculation/LoRA"
            )
        for group in scheduler.kv_cache_config.kv_cache_groups:
            if (
                type(group.kv_cache_spec) is not FullAttentionSpec
                or group.kv_cache_spec.block_size != self.block_size
            ):
                raise ValueError("stable_kv_push requires uniform full attention")
        self.sessions: dict[str, StableKVSession] = {}
        self.request_sessions: dict[str, str] = {}
        # 同步调度仍固定提交时的 phase，回包不读取已经切换的 active view。
        self._scheduled_spans: dict[int, list[tuple[str, int, int]]] = {}

    def describe(self) -> dict[str, Any]:
        """返回会话绑定需要的 NIXL peer 身份与逻辑块尺寸。"""
        return dict(
            self.connector.stable_kv_peer_info(),
            enabled=True,
            role="kv_producer" if self.producer else "kv_consumer",
        )

    def has_work(self) -> bool:
        """开放会话需要有限频率的零 forward 轮询来接收 worker 控制消息。"""
        return any(
            s.state not in ("aborted", "adopted") for s in self.sessions.values()
        )

    def _session(self, session_id: Any) -> StableKVSession:
        if not isinstance(session_id, str) or session_id not in self.sessions:
            raise ValueError(f"Unknown stable KV session: {session_id!r}")
        return self.sessions[session_id]

    def status(self, session: StableKVSession) -> dict[str, Any]:
        """导出 CPU 状态快照，不把本机单调时钟暴露为跨机时间戳。"""
        return {
            "session_id": session.session_id,
            "state": session.state,
            "ready": session.state in ("ready", "adopted"),
            "error": session.error,
            "metrics": dict(session.metrics),
            "peer": self.describe(),
            "received_blocks": sum(
                r.end_block - r.start_block
                for r in session.ranges.values()
                if r.completed
            ),
        }

    def control(self, command: dict[str, Any]) -> dict[str, Any]:
        """处理 REST 转来的 ack 型控制消息；不会创建生成 output queue。"""
        op = command.get("op")
        if op == "describe":
            return self.describe()
        if op in ("reserve", "bind"):
            return self._open(command, producer=op == "bind")
        session = self._session(command.get("session_id"))
        session.last_activity = time.monotonic()
        if op == "finalize":
            self._finalize(session, command)
        elif op == "abort":
            self._abort(session)
        elif op != "status":
            raise ValueError(f"Unknown stable KV operation: {op!r}")
        return self.status(session)

    def _open(self, command: dict[str, Any], producer: bool) -> dict[str, Any]:
        """验证并登记接收/发送绑定，重复绑定只能使用完全相同身份。"""
        if producer != self.producer:
            raise ValueError("Stable KV operation does not match engine role")
        sid, rid, peer = (
            command.get("session_id"),
            command.get("request_id"),
            command.get("peer"),
        )
        if not isinstance(sid, str) or not sid or len(sid) > 128:
            raise ValueError("stable KV session_id must contain 1..128 characters")
        if not isinstance(rid, str) or not rid or not isinstance(peer, dict):
            raise ValueError("stable KV request_id and peer are required")
        for key in ("engine_id", "host", "port"):
            if not peer.get(key):
                raise ValueError(f"Missing stable KV peer field: {key}")
        if (
            peer.get("tp_size") != 1
            or peer.get("block_sizes") != [self.block_size] * self.num_groups
        ):
            raise ValueError("Stable KV peer TP/block sizes do not match")
        early = command.get("early_push", True)
        if type(early) is not bool:
            raise ValueError("early_push must be a boolean")
        if sid in self.sessions:
            old = self.sessions[sid]
            if (old.request_id, old.peer, old.early_push) != (rid, peer, early):
                raise ValueError("Conflicting stable KV session rebind")
            return self.status(old)
        if (
            sum(s.state != "aborted" for s in self.sessions.values())
            >= self.max_sessions
        ):
            raise ValueError("Stable KV session capacity exhausted")
        if producer:
            request = self.scheduler.requests.get(rid)
            if (
                request is None
                or request.is_finished()
                or not request.is_prompt_streaming()
            ):
                raise ValueError(f"No live prompt-stream Request for binding: {rid}")
            if rid in self.request_sessions:
                raise ValueError(
                    "Prompt-stream Request already has a stable KV session"
                )
        session = StableKVSession(sid, rid, dict(peer), producer, early)
        self.sessions[sid] = session
        if producer:
            self.request_sessions[rid] = sid
            # bind 可以晚于模板完成；只有没有在途写入时才使用现有提交水位。
            request = self.scheduler.requests[rid]
            if not self.scheduler._prompt_stream_inflight_steps.get(rid):
                self._offer_prefix(request)
        return self.status(session)

    def _send(self, session: StableKVSession, payload: dict[str, Any]) -> None:
        """发送带有会话代次的 NIXL 控制消息。"""
        self.connector.enqueue_stable_kv_command(
            {
                "kind": "control",
                "peer": session.peer,
                "payload": dict(
                    payload,
                    session_id=session.session_id,
                    generation=session.generation,
                ),
            }
        )

    def capture_scheduled(self, output: "SchedulerOutput") -> None:
        """在调度提前推进水位前固定本轮 PA 实际计算范围。"""
        spans = []
        for rid, count in output.num_scheduled_tokens.items():
            if rid not in self.request_sessions:
                continue
            request = self.scheduler.requests[rid]
            view = request.prompt_stream_execution_view
            if view is None or view.phase != PromptStreamExecutionPhase.PREFIX_APPEND:
                continue
            assert request.prompt_stream is not None  # PREFIX_APPEND 的请求状态已建立。
            work = request.prompt_stream.active_work_item
            if work is None:
                continue
            end = min(
                request.num_computed_tokens + count, work.prefix_append_logical_len
            )
            spans.append((rid, end, request.num_preemptions))
        if spans:
            self._scheduled_spans[id(output)] = spans

    def complete_scheduled(self, output: "SchedulerOutput") -> None:
        """同步 worker 回包后发布完整真实块，排除 padding 和旧代次结果。"""
        for rid, end, preemptions in self._scheduled_spans.pop(id(output), []):
            request = self.scheduler.requests.get(rid)
            if (
                request is not None
                and not request.is_finished()
                and request.num_preemptions == preemptions
            ):
                self._offer_prefix(request, end)

    def _offer_prefix(self, request: "Request", completed: int | None = None) -> None:
        session = self.sessions[self.request_sessions[request.request_id]]
        if not session.early_push or session.state != "receiving":
            return
        if completed is None:
            assert request.prompt_stream is not None
            completed = request.prompt_stream.prefix_append_committed_logical_len
        end = completed // self.block_size
        if end <= session.offered_blocks:
            return
        key = request.get_prompt_stream_segment_key(PromptStreamKVSegment.PREFIX_APPEND)
        groups = self.manager.coordinator.get_blocks(key)
        self._offer(session, groups, end, early=True)

    def _offer(
        self, session: StableKVSession, groups: Any, end: int, early: bool
    ) -> None:
        """固定源块并增加传输引用；仅传未曾发布的连续后缀。"""
        start = session.offered_blocks
        if end <= start:
            return
        slices = [list(group[start:end]) for group in groups]
        if len(slices) != self.num_groups or any(len(g) != end - start for g in slices):
            self._abort(session, "Source block table does not cover offered range")
            return
        block_ids = [[b.block_id for b in group] for group in slices]
        pinned = [b for group in slices for b in group]
        self.manager.block_pool.touch(pinned)
        range_id = f"{session.generation}:{start}:{end}"
        span = StableKVRange(
            range_id, start, end, block_ids, early, pinned_blocks=pinned
        )
        session.ranges[range_id] = span
        session.offered_blocks = end
        session.last_activity = time.monotonic()
        self._send(
            session,
            {
                "kind": "offer",
                "range_id": range_id,
                "start_block": start,
                "end_block": end,
                "source_block_ids": block_ids,
                "early": early,
            },
        )

    def finish_request(
        self, request: "Request", block_ids: Any
    ) -> tuple[bool, dict | None]:
        """终态发布尾块和 UPDATE；普通 connector 不再重复发送整张表。"""
        session = self.sessions[self.request_sessions[request.request_id]]
        if request.status not in (
            RequestStatus.FINISHED_STOPPED,
            RequestStatus.FINISHED_LENGTH_CAPPED,
        ) or session.state in ("aborting", "aborted", "failed"):
            self._abort(session, session.error)
            return False, None
        # 该水位只计主线程已经观测到完成的 WRITE，区别于“提前发布”的分类。
        # 尚在 worker 回包中的完成不计入，因此是保守的已隐藏传输量。
        session.metrics["completed_bytes_before_p_finish"] = session.metrics[
            "total_transferred_bytes"
        ]
        session.metrics["completed_blocks_before_p_finish"] = sum(
            span.end_block - span.start_block
            for span in session.ranges.values()
            if span.completed
        )
        groups = self.manager.get_prompt_stream_blocks(request).blocks
        self._offer(session, groups, len(block_ids[0]), early=False)
        if session.state in ("aborting", "aborted", "failed"):
            return False, None  # _offer 失败不可被下面的终态更新重新激活。
        session.state = "finalizing"
        session.final_tokens = request.get_prompt_stream_physical_prompt_token_ids()
        peer = self.describe()
        # Scheduler._connector_finished 会追加既有严格 manifest 与 mask 字段。
        params = {
            "do_remote_prefill": True,
            "do_remote_decode": False,
            "remote_block_ids": block_ids,
            "remote_num_tokens": len(session.final_tokens),
            "remote_engine_id": peer["engine_id"],
            "remote_host": peer["host"],
            "remote_port": peer["port"],
            "remote_request_id": request.request_id,
            "tp_size": 1,
            "stable_kv_session_id": session.session_id,
            "stable_kv_generation": session.generation,
        }
        # 每个区间有独立引用，正常请求自身可以释放；不复用整请求 finished_sending。
        return False, params

    def _finalize(self, session: StableKVSession, command: dict[str, Any]) -> None:
        """固定最终 manifest；只有全部范围映射匹配才能标记 ready。"""
        if session.producer or session.state in ("aborting", "aborted", "failed"):
            raise ValueError("Cannot finalize this stable KV session")
        tokens, params = (
            command.get("physical_prompt_token_ids"),
            command.get("kv_transfer_params"),
        )
        if (
            not isinstance(tokens, list)
            or not tokens
            or not all(type(t) is int and t >= 0 for t in tokens)
            or len(tokens) > self.scheduler.max_model_len
            or not isinstance(params, dict)
        ):
            raise ValueError("Invalid stable KV final token view")
        expected = (len(tokens) + self.block_size - 1) // self.block_size
        groups = params.get("remote_block_ids")
        if (
            params.get("stable_kv_session_id") != session.session_id
            or params.get("stable_kv_generation") != session.generation
            or params.get("remote_engine_id") != session.peer["engine_id"]
            or params.get("remote_request_id") != session.request_id
            or params.get("remote_num_tokens") != len(tokens)
            or params.get("prompt_stream_exact_transfer") is not True
            or params.get("prompt_stream_physical_prompt_len") != len(tokens)
            or type(params.get("prompt_stream_prefill_output_tokens")) is not int
            or params.get("prompt_stream_prefill_output_tokens") != 0
            or params.get("prompt_stream_remote_block_sizes")
            != [self.block_size] * self.num_groups
            or params.get("prompt_stream_remote_block_counts")
            != [expected] * self.num_groups
            or not isinstance(groups, (list, tuple))
            or len(groups) != self.num_groups
            or any(
                not isinstance(g, (list, tuple))
                or len(g) != expected
                or any(type(b) is not int or b < 0 for b in g)
                or len(set(g)) != len(g)
                for g in groups
            )
        ):
            raise ValueError("Stable KV final manifest does not match session/layout")
        padding = parse_prompt_stream_padding_range(params, len(tokens))
        if (
            padding is not None
            and not self.scheduler.scheduler_config.prompt_stream_masked_padding
        ):
            raise ValueError("Enable masked padding on D before finalize")
        logical_len = params.get("prompt_stream_logical_prompt_len")
        if type(logical_len) is not int or not 0 < logical_len <= len(tokens):
            raise ValueError("Invalid stable KV logical prompt length")
        if session.final_tokens is not None:
            if session.final_tokens != tokens or session.final_params != params:
                raise ValueError("Conflicting stable KV finalize retry")
            return
        session.final_tokens, session.final_params = list(tokens), dict(params)
        session.state = "finalizing"
        session.metrics["received_blocks_at_finalize"] = sum(
            r.end_block - r.start_block for r in session.ranges.values() if r.completed
        )
        self._check_ready(session)

    def _check_ready(self, session: StableKVSession) -> None:
        if session.state != "finalizing" or session.producer:
            return
        assert session.final_tokens is not None and session.final_params is not None
        expected = (len(session.final_tokens) + self.block_size - 1) // self.block_size
        cursor = 0
        for span in sorted(session.ranges.values(), key=lambda r: r.start_block):
            if span.start_block != cursor or not span.completed:
                return
            for index, group in enumerate(span.source_block_ids):
                if group != list(
                    session.final_params["remote_block_ids"][index][
                        span.start_block : span.end_block
                    ]
                ):
                    self._abort(
                        session, "Received source mapping differs from final manifest"
                    )
                    return
            cursor = span.end_block
        if cursor == expected:
            session.state = "ready"
        elif cursor > expected:
            self._abort(session, "Received range exceeds final manifest")

    def adopt(self, request: "Request") -> None:
        """D 收养完整 prompt KV，重算最后一个 prompt token 后开始生成。"""
        params = request.kv_transfer_params or {}
        sid = params.get("stable_kv_session_id")
        if sid is None:
            return
        session = self._session(sid)
        if session.producer or session.state != "ready":
            raise ValueError(f"Stable KV session is not ready for adoption: {sid}")
        assert session.final_tokens is not None and session.final_params is not None
        tokens = request.prompt_token_ids
        if (
            tokens is None
            or not tokens
            or tokens != session.final_tokens
            # 不允许收养时去掉 exact-transfer 或改写有效长度等约束，
            # 否则后续普通 Request 可能误把接收的近似 KV 发布到 prefix cache。
            or any(
                params.get(key) != value
                for key, value in session.final_params.items()
                if key not in ("do_remote_prefill", "do_remote_decode")
            )
        ):
            raise ValueError("D prompt or KV manifest differs from finalized view")
        self.manager.adopt_stable_kv_blocks(session.reservation_id, request)
        # 完整命中仍需重算最后一个 prompt token，以获得采样 logits。
        request.num_computed_tokens = len(session.final_tokens) - 1
        params["do_remote_prefill"] = False
        params["do_remote_decode"] = False
        session.adopted_request_id = request.request_id
        session.state = "adopted"

    def _accept_offer(self, session: StableKVSession, payload: dict[str, Any]) -> None:
        """接收源块声明，拒绝范围重叠和形状冲突。"""
        if session.producer or session.state in (
            "adopted",
            "aborting",
            "aborted",
            "failed",
        ):
            return
        start, end, rid = (
            payload.get("start_block"),
            payload.get("end_block"),
            payload.get("range_id"),
        )
        groups = payload.get("source_block_ids")
        if (
            type(start) is not int
            or type(end) is not int
            or not 0 <= start < end
            or end * self.block_size
            > math.ceil(self.scheduler.max_model_len / self.block_size)
            * self.block_size
            or not isinstance(rid, str)
            or not isinstance(groups, list)
            or len(groups) != self.num_groups
            or any(
                not isinstance(g, list)
                or len(g) != end - start
                or any(type(b) is not int or b < 0 for b in g)
                for g in groups
            )
        ):
            raise ValueError("Invalid stable KV offer range or block IDs")
        old = session.ranges.get(rid)
        if old is not None:
            if (old.start_block, old.end_block, old.source_block_ids) != (
                start,
                end,
                groups,
            ):
                raise ValueError("Conflicting duplicate stable KV offer")
            return
        if any(
            start < r.end_block and r.start_block < end for r in session.ranges.values()
        ):
            raise ValueError("Overlapping stable KV offers")
        if session.final_tokens is not None:
            expected = (
                len(session.final_tokens) + self.block_size - 1
            ) // self.block_size
            if end > expected:
                raise ValueError("Offer exceeds finalized stable KV manifest")
        session.ranges[rid] = StableKVRange(
            rid, start, end, groups, bool(payload.get("early"))
        )

    def _grant_pending(self, session: StableKVSession) -> None:
        """容量允许时扩展 D 表并发送 registration；不足则保留待处理范围。"""
        if session.producer or session.state not in ("receiving", "finalizing"):
            return
        for span in sorted(session.ranges.values(), key=lambda r: r.start_block):
            if span.granted:
                continue
            # 所有未收养会话的目标占用受同一上限约束，不挤占整个 D 显存池。
            reserved = sum(
                len(g)
                for s in self.sessions.values()
                if not s.producer and s.state not in ("adopted", "aborted")
                for g in self.manager.coordinator.get_blocks(s.reservation_id)
            )
            current = self.manager.get_blocks(session.reservation_id).blocks
            needed = sum(max(0, span.end_block - len(g)) for g in current)
            limit = int(
                self.scheduler.kv_cache_config.num_blocks * self.max_reserved_fraction
            )
            if reserved + needed > limit:
                return
            blocks = self.manager.reserve_stable_kv_blocks(
                session.reservation_id,
                min(span.end_block * self.block_size, self.scheduler.max_model_len),
            )
            if blocks is None:
                return
            span.target_block_ids = [
                [b.block_id for b in group[span.start_block : span.end_block]]
                for group in blocks.blocks
            ]
            span.granted = True
            self._send(
                session,
                {
                    "kind": "grant",
                    "range_id": span.range_id,
                    "target_block_ids": span.target_block_ids,
                },
            )

    def _apply_event(self, event: dict[str, Any]) -> None:
        """在主线程应用来自 NIXL worker 的区间与控制事件。"""
        kind = event.get("kind")
        payload = event.get("payload", {}) if kind == "control" else event
        session = self.sessions.get(payload.get("session_id", ""))
        if session is None or payload.get("generation") != session.generation:
            return  # 关闭后的迟到消息不允许重建同名会话。
        session.last_activity = time.monotonic()
        if kind == "control":
            action = payload.get("kind")
            if action == "offer":
                self._accept_offer(session, payload)
            elif (
                action == "grant"
                and session.producer
                and session.state in ("receiving", "finalizing")
            ):
                span = session.ranges[payload["range_id"]]
                targets = payload["target_block_ids"]
                if len(targets) != self.num_groups or any(
                    len(g) != span.end_block - span.start_block for g in targets
                ):
                    raise ValueError("Stable KV grant shape mismatch")
                if span.granted:
                    if targets != span.target_block_ids:
                        raise ValueError("Conflicting stable KV grant retry")
                    return
                span.granted, span.target_block_ids = True, targets
                self.connector.enqueue_stable_kv_command(
                    {
                        "kind": "write",
                        "peer": session.peer,
                        "session_id": session.session_id,
                        "generation": session.generation,
                        "range_id": span.range_id,
                        "start_block": span.start_block,
                        "end_block": span.end_block,
                        "source_block_ids": span.source_block_ids,
                        "target_block_ids": targets,
                        "early": span.early,
                    }
                )
            elif action == "abort":
                self._abort(session, payload.get("error"), notify=False)
            elif action == "drained":
                session.peer_drained = True
            elif action == "transfer_stats":
                self._record_stats(session, payload)
        elif kind in ("sent", "received"):
            if session.state in ("aborted", "adopted"):
                return
            span = session.ranges[event["range_id"]]
            if (kind == "sent") != session.producer:
                raise ValueError("Stable KV completion role mismatch")
            if not span.granted or (
                event.get("start_block") != span.start_block
                or event.get("end_block") != span.end_block
            ):
                raise ValueError("Stable KV completion does not match granted range")
            span.completed = True
            if span.pinned_blocks:
                self.manager.block_pool.free_blocks(reversed(span.pinned_blocks))
                span.pinned_blocks.clear()
            if kind == "sent":
                self._record_stats(session, event)
                self._send(session, dict(event, kind="transfer_stats"))
            self._check_ready(session)
        elif kind == "drained":
            session.local_drained = True
            self._send(session, {"kind": "drained"})
        elif kind == "error":
            self._abort(session, str(event.get("error", "NIXL transfer failed")))

    def _record_stats(self, session: StableKVSession, payload: dict[str, Any]) -> None:
        """按真实 WRITE handle 完成耗时统计，不混入 HTTP/目标分配等待。"""
        span = session.ranges.get(payload.get("range_id", ""))
        if span is None or span.stats_recorded:
            return
        span.stats_recorded = True
        prefix = "early" if span.early else "final"
        num_bytes = int(payload.get("bytes", payload.get("num_bytes", 0)))
        session.metrics[f"{prefix}_transferred_bytes"] += num_bytes
        session.metrics["total_transferred_bytes"] += num_bytes
        session.metrics[f"{prefix}_transferred_blocks"] += (
            span.end_block - span.start_block
        )
        session.metrics[f"{prefix}_transfer_ms"] += float(payload.get("transfer_ms", 0))

    def _abort(
        self, session: StableKVSession, error: str | None = None, notify: bool = True
    ) -> None:
        """双侧停止新写入；收到双方 drained 前不归还 D 地址。"""
        if session.state in ("aborted", "aborting", "failed"):
            return
        if error:
            session.error = error
            logger.warning("Stable KV session %s failed: %s", session.session_id, error)
        session.state = "failed" if error else "aborting"
        if notify:
            self._send(session, {"kind": "abort", "error": error})
        self.connector.enqueue_stable_kv_command(
            {
                "kind": "drain",
                "session_id": session.session_id,
                "generation": session.generation,
            }
        )
        # 正常生成路径的 blocks 由正常 Request 释放，不能由 reservation 再释放。
        rid = session.request_id if session.producer else session.adopted_request_id
        if rid is not None:
            request = self.scheduler.requests.get(rid)
            if request is not None and not request.is_finished():
                self.scheduler.finish_requests(rid, RequestStatus.FINISHED_ABORTED)

    def invalidate_request(self, request: "Request") -> None:
        """首版遇到 P 抢占显式终止会话，防止旧地址写入新代次布局。"""
        sid = self.request_sessions.get(request.request_id)
        if sid is not None:
            session = self.sessions[sid]
            # 抢占过程尚在修改 running；延迟终止请求，先标记失败并停止传输。
            if session.state in ("receiving", "finalizing"):
                session.error = (
                    "P KV was preempted; stable push session must be retried"
                )

    def poll(self) -> None:
        """驱动控制消息、待分配范围、超时与双侧安全释放。"""
        for event in self.connector.pop_stable_kv_events():
            try:
                self._apply_event(event)
            except (KeyError, TypeError, ValueError) as exc:
                payload = event.get("payload", event)
                session = self.sessions.get(payload.get("session_id", ""))
                if session is not None:
                    self._abort(session, f"Invalid stable KV protocol event: {exc}")
                else:
                    logger.warning("Rejected unbound stable KV event: %s", exc)
        now = time.monotonic()
        for sid, session in list(self.sessions.items()):
            if session.state == "aborted":
                if (
                    session.closed_at is not None
                    and now - session.closed_at > self.timeout_s
                ):
                    del self.sessions[sid]  # 有界 tombstone 保留窗口。
                continue
            if session.error and session.state not in ("failed", "aborting"):
                self._abort(session, session.error)
            if now - session.last_activity > self.timeout_s and session.state not in (
                "failed",
                "aborting",
                "adopted",
            ):
                self._abort(session, "Stable KV session lease expired")
            self._grant_pending(session)
            if (
                session.state in ("failed", "aborting")
                and session.local_drained
                and (
                    session.peer_drained
                    # reserve 成功而 P bind 失败时，对端还没有会话可回复。
                    # 从未授予任何地址就不可能有远端 WRITE，无需永久隔离空表。
                    or (
                        not session.producer
                        and not any(r.granted for r in session.ranges.values())
                    )
                )
            ):
                for span in session.ranges.values():
                    if span.pinned_blocks:
                        self.manager.block_pool.free_blocks(
                            reversed(span.pinned_blocks)
                        )
                        span.pinned_blocks.clear()
                if not session.producer and session.adopted_request_id is None:
                    self.manager.coordinator.free(session.reservation_id)
                self.request_sessions.pop(session.request_id, None)
                session.state, session.closed_at = "aborted", now
