# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Stable KV 状态机的纯 CPU 验证，使用真实 block pool 和模拟跨节点消息。"""

from types import SimpleNamespace
from typing import Any

import pytest
import torch

from vllm.sampling_params import SamplingParams
from vllm.v1.core.kv_cache_manager import KVCacheManager
from vllm.v1.core.stable_kv_push import StableKVPushController
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
)
from vllm.v1.request import PromptStreamExecutionPhase, Request


class FakeConnector:
    """只模拟控制队列，GPU WRITE 的完成由测试显式推进。"""

    def __init__(self, engine_id: str) -> None:
        self.engine_id = engine_id
        self.commands: list[dict[str, Any]] = []
        self.events: list[dict[str, Any]] = []

    def stable_kv_peer_info(self) -> dict[str, Any]:
        return {
            "engine_id": self.engine_id,
            "host": "g2c9",
            "port": 5600 if self.engine_id == "P" else 5601,
            "tp_size": 1,
            "block_sizes": [4],
        }

    def enqueue_stable_kv_command(self, command: dict[str, Any]) -> None:
        self.commands.append(command)

    def pop_stable_kv_events(self) -> list[dict[str, Any]]:
        events, self.events = self.events, []
        return events


def _controller(producer: bool) -> StableKVPushController:
    """构造单组 full-attention KV pool，不分配任何 GPU tensor。"""
    config = KVCacheConfig(
        num_blocks=32,
        kv_cache_tensors=[],
        kv_cache_groups=[
            KVCacheGroupSpec(
                ["layer"],
                FullAttentionSpec(
                    block_size=4,
                    num_kv_heads=1,
                    head_size=1,
                    dtype=torch.float32,
                ),
            )
        ],
    )
    manager = KVCacheManager(
        kv_cache_config=config,
        max_model_len=64,
        scheduler_block_size=4,
        hash_block_size=4,
        max_num_batched_tokens=64,
        enable_caching=False,
    )
    scheduler = SimpleNamespace(
        kv_cache_manager=manager,
        kv_cache_config=config,
        block_size=4,
        max_model_len=64,
        scheduler_config=SimpleNamespace(prompt_stream_masked_padding=True),
        connector=FakeConnector("P" if producer else "D"),
        requests={},
        _prompt_stream_inflight_steps={},
        aborted_requests=[],
        vllm_config=SimpleNamespace(
            kv_transfer_config=SimpleNamespace(
                kv_connector="NixlPushConnector",
                kv_role="kv_producer" if producer else "kv_consumer",
                kv_connector_extra_config={"stable_kv_push": True},
            ),
            scheduler_config=SimpleNamespace(async_scheduling=False),
            parallel_config=SimpleNamespace(
                tensor_parallel_size=1,
                pipeline_parallel_size=1,
                data_parallel_size=1,
            ),
            speculative_config=None,
            lora_config=None,
            model_config=SimpleNamespace(is_encoder_decoder=False),
        ),
    )

    def finish_requests(request_id: str, _status: Any) -> None:
        scheduler.aborted_requests.append(request_id)
        scheduler.requests.pop(request_id, None)

    scheduler.finish_requests = finish_requests
    return StableKVPushController(scheduler)


def _pair(
    early_push: bool = True,
) -> tuple[StableKVPushController, StableKVPushController]:
    """完成 D reserve 与 P bind，模板此时尚未产生完整块。"""
    producer, consumer = _controller(True), _controller(False)
    producer.scheduler.requests["request-P"] = SimpleNamespace(
        request_id="request-P",
        is_finished=lambda: False,
        is_prompt_streaming=lambda: True,
        prompt_stream=SimpleNamespace(prefix_append_committed_logical_len=0),
    )
    consumer.control(
        {
            "op": "reserve",
            "session_id": "session",
            "request_id": "request-P",
            "peer": producer.describe(),
            "early_push": early_push,
        }
    )
    producer.control(
        {
            "op": "bind",
            "session_id": "session",
            "request_id": "request-P",
            "peer": consumer.describe(),
            "early_push": early_push,
        }
    )
    return producer, consumer


def _exchange(
    producer: StableKVPushController, consumer: StableKVPushController
) -> list[dict[str, Any]]:
    """传递 offer/grant/stats，不自动完成 WRITE 或取消 drain。"""
    writes = []
    for _ in range(8):
        progress = False
        for source, target in ((producer, consumer), (consumer, producer)):
            commands, source.connector.commands = source.connector.commands, []
            for command in commands:
                progress = True
                if command["kind"] == "control":
                    target.connector.events.append(command)
                elif command["kind"] == "write":
                    writes.append(command)
            target.poll()
        if not progress:
            return writes
    raise AssertionError("Control messages did not quiesce")


def _complete(
    producer: StableKVPushController,
    consumer: StableKVPushController,
    write: dict[str, Any],
) -> None:
    """明确推进 WRITE 完成，模拟 NIXL 给两侧的独立通知。"""
    event = {
        "session_id": "session",
        "generation": 0,
        "range_id": write["range_id"],
        "start_block": write["start_block"],
        "end_block": write["end_block"],
        "bytes": (write["end_block"] - write["start_block"]) * 32,
        "transfer_ms": 0.5,
    }
    producer.connector.events.append(dict(event, kind="sent"))
    consumer.connector.events.append(dict(event, kind="received"))
    producer.poll()
    consumer.poll()
    _exchange(producer, consumer)


def _manifest(producer: StableKVPushController) -> dict[str, Any]:
    """构造包含中间 padding 与零交接输出计数的最终严格 manifest。"""
    return {
        "stable_kv_session_id": "session",
        "stable_kv_generation": 0,
        "remote_engine_id": "P",
        "remote_request_id": "request-P",
        "remote_num_tokens": 10,
        "remote_block_ids": [
            [b.block_id for b in group]
            for group in producer.manager.get_blocks("source").blocks
        ],
        "prompt_stream_exact_transfer": True,
        "prompt_stream_remote_block_sizes": [4],
        "prompt_stream_remote_block_counts": [3],
        "prompt_stream_physical_prompt_len": 10,
        "prompt_stream_logical_prompt_len": 8,
        "prompt_stream_prefill_output_tokens": 0,
        "prompt_stream_masked_padding": {"version": 1, "start": 6, "end": 8},
    }


@pytest.mark.parametrize("early_push", [False, True])
def test_incremental_handoff_adopts_exact_same_blocks(early_push: bool) -> None:
    """最终屏障前不能生成；收养后保留地址与引用，并重算最后一个 prompt token。"""
    producer, consumer = _pair(early_push)
    groups = producer.manager.reserve_stable_kv_blocks("source", 10).blocks
    if early_push:
        producer._offer(producer.sessions["session"], groups, 1, early=True)
        early = _exchange(producer, consumer)
        assert len(early) == 1
        assert all(block.ref_cnt == 2 for group in groups for block in group[:1])
        _complete(producer, consumer, early[0])
        assert consumer.sessions["session"].state == "receiving"
    producer._offer(producer.sessions["session"], groups, 3, early=False)
    final = _exchange(producer, consumer)
    assert len(final) == 1
    params = _manifest(producer)
    consumer.control(
        {
            "op": "finalize",
            "session_id": "session",
            "physical_prompt_token_ids": list(range(10)),
            "kv_transfer_params": params,
        }
    )
    assert consumer.sessions["session"].state == "finalizing"
    assert not consumer.status(consumer.sessions["session"])["ready"]
    _complete(producer, consumer, final[0])
    assert consumer.sessions["session"].state == "ready"
    reservation = consumer.sessions["session"].reservation_id
    before = consumer.manager.get_blocks(reservation).blocks
    free_before = consumer.manager.block_pool.get_num_free_blocks()
    request = Request(
        request_id="request-D",
        prompt_token_ids=list(range(10)),
        sampling_params=SamplingParams(
            max_tokens=3, extra_args={"kv_transfer_params": dict(params)}
        ),
        pooling_params=None,
    )
    consumer.adopt(request)
    after = consumer.manager.get_blocks(request.request_id).blocks
    assert all(
        a is b for left, right in zip(before, after) for a, b in zip(left, right)
    )
    assert consumer.manager.block_pool.get_num_free_blocks() == free_before
    assert request.num_computed_tokens == 9
    assert request.num_tokens == 10
    assert request.skip_reading_prefix_cache
    assert request.prompt_stream_masked_padding
    assert consumer.sessions["session"].state == "adopted"
    assert producer.sessions["session"].metrics["early_transferred_blocks"] == (
        1 if early_push else 0
    )
    assert producer.sessions["session"].metrics["final_transferred_blocks"] == (
        2 if early_push else 3
    )
    with pytest.raises(ValueError, match="not ready"):
        consumer.adopt(request)


@pytest.mark.parametrize(
    ("field_name", "replacement"),
    [
        ("prompt_stream_exact_transfer", False),
        ("prompt_stream_exact_transfer", None),
        ("remote_num_tokens", 9),
        ("remote_num_tokens", None),
        ("stable_kv_generation", 1),
        ("stable_kv_generation", None),
    ],
)
def test_adoption_rejects_changed_or_missing_final_manifest_fields(
    field_name: str, replacement: Any
) -> None:
    """收养时篡改或删除最终约束必须拒绝，并保留接收表及全部块引用。"""
    producer, consumer = _pair()
    groups = producer.manager.reserve_stable_kv_blocks("source", 10).blocks
    producer._offer(producer.sessions["session"], groups, 3, early=False)
    write = _exchange(producer, consumer)[0]
    _complete(producer, consumer, write)
    params = _manifest(producer)
    consumer.control(
        {
            "op": "finalize",
            "session_id": "session",
            "physical_prompt_token_ids": list(range(10)),
            "kv_transfer_params": params,
        }
    )
    session = consumer.sessions["session"]
    assert session.state == "ready"
    incoming_params = dict(params)
    if replacement is None:
        incoming_params.pop(field_name)
    else:
        incoming_params[field_name] = replacement
    request = Request(
        request_id="invalid-request-D",
        prompt_token_ids=list(range(10)),
        sampling_params=SamplingParams(
            max_tokens=3, extra_args={"kv_transfer_params": incoming_params}
        ),
        pooling_params=None,
    )
    before = consumer.manager.get_blocks(session.reservation_id).blocks
    refs_before = [[block.ref_cnt for block in group] for group in before]
    free_before = consumer.manager.block_pool.get_num_free_blocks()
    with pytest.raises(ValueError, match="KV manifest differs"):
        consumer.adopt(request)
    after = consumer.manager.get_blocks(session.reservation_id).blocks
    assert session.state == "ready"
    assert session.adopted_request_id is None
    assert request.num_computed_tokens == 0
    assert all(
        a is b for left, right in zip(before, after) for a, b in zip(left, right)
    )
    assert [[block.ref_cnt for block in group] for group in after] == refs_before
    assert consumer.manager.block_pool.get_num_free_blocks() == free_before
    assert all(
        request.request_id not in manager.req_to_blocks
        for manager in consumer.manager.coordinator.single_type_managers
    )


def test_final_manifest_rejects_received_source_mismatch() -> None:
    """同样块数仍不足以认领缓存，源 block 映射必须逐项一致。"""
    producer, consumer = _pair()
    groups = producer.manager.reserve_stable_kv_blocks("source", 10).blocks
    producer._offer(producer.sessions["session"], groups, 3, early=False)
    write = _exchange(producer, consumer)[0]
    _complete(producer, consumer, write)
    params = _manifest(producer)
    params["remote_block_ids"][0][0] += 100
    consumer.control(
        {
            "op": "finalize",
            "session_id": "session",
            "physical_prompt_token_ids": list(range(10)),
            "kv_transfer_params": params,
        }
    )
    assert consumer.sessions["session"].state == "failed"
    assert "mapping differs" in consumer.sessions["session"].error


def test_duplicate_offer_is_idempotent_and_overlap_fails() -> None:
    """重复同一声明不重复分配，重叠声明拒绝进入可生成状态。"""
    producer, consumer = _pair()
    groups = producer.manager.reserve_stable_kv_blocks("source", 10).blocks
    producer._offer(producer.sessions["session"], groups, 1, early=True)
    offer = producer.connector.commands[0]
    _exchange(producer, consumer)
    free_before = consumer.manager.block_pool.get_num_free_blocks()
    consumer.connector.events.append(offer)
    consumer.poll()
    assert consumer.manager.block_pool.get_num_free_blocks() == free_before
    assert len(consumer.sessions["session"].ranges) == 1
    overlap = dict(offer, payload=dict(offer["payload"], range_id="other"))
    consumer.connector.events.append(overlap)
    consumer.poll()
    assert consumer.sessions["session"].state == "failed"
    assert "Overlapping" in consumer.sessions["session"].error


def test_abort_requires_both_drains_before_releasing_target_blocks() -> None:
    """取消中已有注册仍可能被 WRITE 使用，双方 drained 前不回收地址。"""
    producer, consumer = _pair()
    initial_free = consumer.manager.block_pool.get_num_free_blocks()
    groups = producer.manager.reserve_stable_kv_blocks("source", 10).blocks
    producer._offer(producer.sessions["session"], groups, 3, early=False)
    _exchange(producer, consumer)
    allocated_free = consumer.manager.block_pool.get_num_free_blocks()
    assert allocated_free == initial_free - 3
    consumer.control({"op": "abort", "session_id": "session"})
    consumer.connector.events.append(
        {"kind": "drained", "session_id": "session", "generation": 0}
    )
    consumer.poll()
    assert consumer.sessions["session"].state == "aborting"
    assert consumer.sessions["session"].local_drained
    assert not consumer.sessions["session"].peer_drained
    assert any(span.granted for span in consumer.sessions["session"].ranges.values())
    assert consumer.manager.block_pool.get_num_free_blocks() == allocated_free
    consumer.connector.events.append(
        {
            "kind": "control",
            "payload": {"kind": "drained", "session_id": "session", "generation": 0},
        }
    )
    consumer.poll()
    assert consumer.sessions["session"].state == "aborted"
    assert consumer.manager.block_pool.get_num_free_blocks() == initial_free


@pytest.mark.parametrize("has_ungranted_offer", [False, True])
def test_partial_start_abort_without_grants_needs_only_local_drain(
    has_ungranted_offer: bool,
) -> None:
    """D 已 reserve 而 P bind 失败时，不等待从未存在的 P 会话回复。"""
    producer, consumer = _controller(True), _controller(False)
    initial_free = consumer.manager.block_pool.get_num_free_blocks()
    consumer.control(
        {
            "op": "reserve",
            "session_id": "partial-start",
            "request_id": "unbound-P-request",
            "peer": producer.describe(),
        }
    )
    session = consumer.sessions["partial-start"]
    assert not producer.sessions
    if has_ungranted_offer:
        # 即便已看到块声明，只要尚未授予地址，远端仍不可能 WRITE 本地内存。
        consumer._accept_offer(
            session,
            {
                "range_id": "0:0:1",
                "start_block": 0,
                "end_block": 1,
                "source_block_ids": [[11]],
                "early": True,
            },
        )
    consumer.control({"op": "abort", "session_id": "partial-start"})
    consumer.poll()
    assert session.state == "aborting"
    assert not session.local_drained
    consumer.connector.events.append(
        {"kind": "drained", "session_id": "partial-start", "generation": 0}
    )
    consumer.poll()
    assert session.local_drained
    assert not session.peer_drained
    assert session.state == "aborted"
    assert not consumer.has_work()
    assert consumer.manager.block_pool.get_num_free_blocks() == initial_free
    # start 回滚和客户端显式 abort 可重复到达，不产生额外分配或重复 free。
    consumer.control({"op": "abort", "session_id": "partial-start"})
    assert consumer.manager.block_pool.get_num_free_blocks() == initial_free


def test_old_generation_completion_does_not_unlock_final_barrier() -> None:
    """迟到代次的成功通知不能提升当前接收范围。"""
    producer, consumer = _pair()
    groups = producer.manager.reserve_stable_kv_blocks("source", 10).blocks
    producer._offer(producer.sessions["session"], groups, 3, early=False)
    write = _exchange(producer, consumer)[0]
    consumer.connector.events.append(
        {
            "kind": "received",
            "session_id": "session",
            "generation": -1,
            "range_id": write["range_id"],
        }
    )
    consumer.poll()
    assert not consumer.sessions["session"].ranges[write["range_id"]].completed


@pytest.mark.parametrize(
    ("field_name", "bad_value"),
    [
        ("prompt_stream_exact_transfer", False),
        ("prompt_stream_physical_prompt_len", 9),
        ("prompt_stream_prefill_output_tokens", 1),
        ("remote_num_tokens", 11),
        ("prompt_stream_remote_block_counts", [2]),
        ("remote_block_ids", [[1, 1, 2]]),
        ("prompt_stream_masked_padding", {"version": 1, "start": 6, "end": 11}),
    ],
)
def test_malformed_final_manifest_never_marks_ready(
    field_name: str, bad_value: Any
) -> None:
    """不同类型的最终布局错误在收养前拒绝，不能污染普通 Request。"""
    producer, consumer = _pair()
    producer.manager.reserve_stable_kv_blocks("source", 10)
    params = _manifest(producer)
    params[field_name] = bad_value
    with pytest.raises(ValueError):
        consumer.control(
            {
                "op": "finalize",
                "session_id": "session",
                "physical_prompt_token_ids": list(range(10)),
                "kv_transfer_params": params,
            }
        )
    assert consumer.sessions["session"].state == "receiving"
    assert consumer.sessions["session"].final_tokens is None


def test_receive_completion_without_registration_fails_closed() -> None:
    """有 offer 但未注册目标时收到完成通知，不能据此制造有效 KV。"""
    producer, consumer = _pair()
    groups = producer.manager.reserve_stable_kv_blocks("source", 10).blocks
    producer._offer(producer.sessions["session"], groups, 1, early=True)
    offer = producer.connector.commands[0]["payload"]
    consumer._accept_offer(consumer.sessions["session"], offer)
    consumer.connector.events.append(
        {
            "kind": "received",
            "session_id": "session",
            "generation": 0,
            "range_id": offer["range_id"],
            "start_block": 0,
            "end_block": 1,
        }
    )
    consumer.poll()
    assert consumer.sessions["session"].state == "failed"
    assert "granted range" in consumer.sessions["session"].error


@pytest.mark.parametrize("preempted", [False, True])
def test_only_completed_real_full_blocks_are_published(preempted: bool) -> None:
    """调度水位不等于传输就绪，padding 与旧抢占代次都不能提前发送。"""
    producer, _ = _pair()
    producer.manager.reserve_stable_kv_blocks("source", 8)
    request = producer.scheduler.requests["request-P"]
    request.num_computed_tokens = 0
    request.num_preemptions = 0
    request.prompt_stream_execution_view = SimpleNamespace(
        phase=PromptStreamExecutionPhase.PREFIX_APPEND
    )
    request.prompt_stream.active_work_item = SimpleNamespace(
        prefix_append_logical_len=6
    )
    request.get_prompt_stream_segment_key = lambda _segment: "source"
    output = SimpleNamespace(num_scheduled_tokens={"request-P": 8})
    producer.capture_scheduled(output)
    assert not producer.connector.commands
    # 模拟调度提前推进并清空 work；完成处理必须使用捕获的真实范围。
    request.prompt_stream.active_work_item = None
    request.num_computed_tokens = 8
    request.num_preemptions = int(preempted)
    producer.complete_scheduled(output)
    if preempted:
        assert not producer.connector.commands
    else:
        offer = producer.connector.commands[0]["payload"]
        assert offer["start_block"] == 0
        assert offer["end_block"] == 1  # 六个真实 token 只有一块稳定完整块。


def test_reservation_pressure_does_not_grant_unallocated_addresses() -> None:
    """目标显存额度不足时保留 offer，待额度可用后才发布目标地址。"""
    producer, consumer = _pair()
    consumer.max_reserved_fraction = 1 / 32  # 目前只允许预留一块。
    groups = producer.manager.reserve_stable_kv_blocks("source", 10).blocks
    producer._offer(producer.sessions["session"], groups, 3, early=False)
    assert not _exchange(producer, consumer)
    span = next(iter(consumer.sessions["session"].ranges.values()))
    assert not span.granted
    assert not span.target_block_ids
    consumer.max_reserved_fraction = 0.5
    consumer.poll()
    writes = _exchange(producer, consumer)
    assert len(writes) == 1
    assert len(writes[0]["target_block_ids"][0]) == 3
