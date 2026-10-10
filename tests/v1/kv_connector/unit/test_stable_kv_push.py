# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""无需 GPU 的区间 WRITE、完成隔离和取消排空测试。"""

from copy import deepcopy
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import msgspec
import pytest

from vllm.distributed.kv_transfer.kv_connector.v1.nixl.metadata import (
    STABLE_KV_NOTIF_PREFIX,
)
from vllm.distributed.kv_transfer.kv_connector.v1.nixl.stable_push import (
    StableKVPushWorker,
    validate_stable_kv_write,
)


def _command() -> dict[str, Any]:
    """构造两个完整块的传输，源和目标使用不同物理地址。"""
    return {
        "kind": "write",
        "session_id": "session-a",
        "generation": 0,
        "range_id": "range-0",
        "start_block": 0,
        "end_block": 2,
        "early": True,
        "source_block_ids": [[1, 2]],
        "target_block_ids": [[5, 6]],
        "peer": {"engine_id": "d", "host": "localhost", "port": 5601, "tp_size": 1},
    }


def _worker() -> MagicMock:
    """模拟 NIXL 句柄和缓存 descriptor，保留真实 transport 状态机。"""
    worker = MagicMock()
    worker.engine_id = "p"
    worker.block_size = 16
    worker._physical_blocks_per_logical_kv_block = 1
    worker._group_spec_types = [object]
    worker._remote_agents = {"d": {0: "d-agent"}}
    worker._ensure_d_handshake.return_value = True
    worker.transfer_topo.get_engine_info.return_value = SimpleNamespace(
        remote_block_size=16, remote_physical_blocks_per_logical=1
    )
    worker._logical_to_kernel_block_ids.side_effect = lambda blocks: blocks
    worker._logical_to_remote_kernel_block_ids.side_effect = lambda blocks, ratio: (
        blocks
    )
    worker.dst_num_blocks = {"p": 100, "d": 100}
    worker.src_xfer_handles_by_block_size = {16: 10}
    worker.dst_xfer_side_handles = {"d": {0: 20}}
    worker._compute_desc_ids.side_effect = lambda **kw: kw["block_ids"][0]
    worker.nixl_wrapper.make_prepped_xfer.return_value = 123
    worker.nixl_wrapper.check_xfer_state.return_value = "PROC"
    worker.nixl_wrapper.get_xfer_telemetry.return_value = SimpleNamespace(
        totalBytes=4096, xferDuration=2000, postDuration=100, descCount=2
    )
    return worker


@pytest.mark.parametrize(
    "field,value",
    [
        ("target_block_ids", [[5]]),
        ("target_block_ids", [[5, 6], [7, 8]]),
        ("target_block_ids", [[5, 5]]),
        ("source_block_ids", [[-1, 2]]),
        ("source_block_ids", [[True, 2]]),
        ("source_block_ids", [1, 2]),
        ("end_block", 0),
        ("generation", -1),
        ("range_id", ""),
    ],
)
def test_write_rejects_ambiguous_mapping(field: str, value: Any) -> None:
    """错误分组和范围必须在任何 NIXL 写入之前拒绝。"""
    command = _command()
    command[field] = value
    with pytest.raises(ValueError):
        validate_stable_kv_write(command)


def test_write_uses_exact_descriptors_and_range_notification() -> None:
    """提前 WRITE 使用独立协议通知，目标块次序与 grant 完全一致。"""
    worker = _worker()
    transport = StableKVPushWorker(worker)
    transport.enqueue([_command()])
    transport.process_commands()
    call = worker.nixl_wrapper.make_prepped_xfer.call_args
    assert call.args == ("WRITE", 10, [1, 2], 20, [5, 6])
    notification = call.kwargs["notif_msg"]
    assert notification.startswith(STABLE_KV_NOTIF_PREFIX)
    event = msgspec.msgpack.decode(notification[len(STABLE_KV_NOTIF_PREFIX) :])
    assert event["kind"] == "received"
    assert event["session_id"] == "session-a"
    assert event["range_id"] == "range-0"
    assert "finished_recving" not in event
    assert transport.drain_events() is None


def test_drain_waits_for_write_and_preserves_generation_tombstone() -> None:
    """取消不能先释放在途源块；drained 之后也拒绝迟到的新 WRITE。"""
    worker = _worker()
    transport = StableKVPushWorker(worker)
    transport.enqueue(
        [_command(), {"kind": "drain", "session_id": "session-a", "generation": 0}]
    )
    transport.process_commands()
    transport.poll_transfers()
    assert transport.drain_events() is None
    worker.nixl_wrapper.check_xfer_state.return_value = "DONE"
    transport.poll_transfers()
    metadata = transport.drain_events()
    assert metadata is not None
    assert [event["kind"] for event in metadata.events] == ["sent", "drained"]
    assert metadata.events[0]["transfer_ms"] == 2
    assert metadata.events[0]["bytes"] == 4096
    worker.nixl_wrapper.release_xfer_handle.assert_called_once_with(123)
    transport.enqueue([_command()])
    transport.process_commands()
    metadata = transport.drain_events()
    assert metadata is not None and metadata.events[0]["kind"] == "error"
    worker.nixl_wrapper.transfer.assert_called_once()


def test_duplicate_range_is_idempotent_but_mapping_change_fails() -> None:
    """重复区间不重复传输，相同区间身份不能绑定另一组目标块。"""
    worker = _worker()
    transport = StableKVPushWorker(worker)
    command = _command()
    transport.enqueue([command, deepcopy(command)])
    transport.process_commands()
    worker.nixl_wrapper.transfer.assert_called_once()
    changed = deepcopy(command)
    changed["target_block_ids"] = [[7, 8]]
    transport.enqueue([changed])
    transport.process_commands()
    metadata = transport.drain_events()
    assert metadata is not None
    assert metadata.events[0]["kind"] == "error"
    assert "different mapping" in metadata.events[0]["error"]


def test_receive_event_does_not_complete_normal_request() -> None:
    """接收端只回传范围事件，不修改旧 Push 的请求完成容器。"""
    transport = StableKVPushWorker(_worker())
    event = {"kind": "received", "session_id": "session-a", "range_id": "range-0"}
    transport.handle_notification(
        STABLE_KV_NOTIF_PREFIX + msgspec.msgpack.encode(event)
    )
    metadata = transport.drain_events()
    assert metadata is not None and metadata.events == [event]


def test_blocks_outside_registered_pool_fail_before_write() -> None:
    """拒绝越界 block ID，避免 descriptor 地址落到未注册内存。"""
    worker = _worker()
    transport = StableKVPushWorker(worker)
    command = _command()
    command["target_block_ids"] = [[99, 100]]
    transport.enqueue([command])
    transport.process_commands()
    worker.nixl_wrapper.make_prepped_xfer.assert_not_called()
    metadata = transport.drain_events()
    assert metadata is not None
    assert "outside registered pool" in metadata.events[0]["error"]


def test_failed_release_never_reports_drained() -> None:
    """NIXL 尚不能释放句柄时不能通知 Scheduler 回收目标或源 blocks。"""
    worker = _worker()
    transport = StableKVPushWorker(worker)
    transport.enqueue(
        [
            _command(),
            {"kind": "drain", "session_id": "session-a", "generation": 0},
        ]
    )
    transport.process_commands()
    worker.nixl_wrapper.check_xfer_state.return_value = "ERR"
    worker.nixl_wrapper.release_xfer_handle.side_effect = RuntimeError("still active")
    transport.poll_transfers()
    assert transport.drain_events() is None
    assert transport.has_work()
    worker.nixl_wrapper.release_xfer_handle.side_effect = None
    transport.poll_transfers()
    metadata = transport.drain_events()
    assert metadata is not None
    assert [event["kind"] for event in metadata.events] == ["error", "drained"]


def test_received_notification_keeps_control_payload_separate() -> None:
    """容量申请只能回到主线程，writer 不把控制消息当作数据完成。"""
    transport = StableKVPushWorker(_worker())
    message = {
        "kind": "control",
        "payload": {"kind": "offer", "session_id": "session-a", "generation": 0},
    }
    transport.handle_notification(
        STABLE_KV_NOTIF_PREFIX + msgspec.msgpack.encode(message)
    )
    metadata = transport.drain_events()
    assert metadata is not None and metadata.events == [message]
