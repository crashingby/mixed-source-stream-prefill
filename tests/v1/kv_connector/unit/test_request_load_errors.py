# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import queue
from unittest.mock import Mock

import pytest

from vllm.distributed.kv_transfer.kv_connector.utils import KVOutputAggregator
from vllm.distributed.kv_transfer.kv_connector.v1.multi_connector import (
    MultiConnector,
)
from vllm.distributed.kv_transfer.kv_connector.v1.nixl.base_worker import (
    NixlBaseConnectorWorker,
)

from .utils import create_model_runner_output

pytestmark = pytest.mark.cpu_test


def test_nixl_worker_drains_request_level_load_errors() -> None:
    """HMA 传输失败保留 request identity，并且 drain 只返回一次。"""
    worker = object.__new__(NixlBaseConnectorWorker)
    worker._recving_metadata = {}
    worker._is_hma_required = True
    worker._invalid_block_ids = queue.Queue()
    worker._failed_recv_reqs = queue.Queue()
    worker._failed_load_req_ids = queue.Queue()
    worker.xfer_stats = Mock()

    worker._handle_failed_transfer("request-0", handle=None)

    assert worker._failed_recv_reqs.get_nowait() == "request-0"
    assert worker.get_block_ids_with_load_errors() == set()
    assert worker.get_request_ids_with_load_errors() == {"request-0"}
    assert worker.get_request_ids_with_load_errors() == set()
    worker.xfer_stats.record_failed_transfer.assert_called_once_with()


def test_kv_output_aggregator_defers_failure_until_all_workers_finish() -> None:
    """任一 rank 的失败标记必须等待所有 rank 完成 receive 后再上报。"""
    request_id = "request-0"
    aggregator = KVOutputAggregator(expected_finished_count=2)

    first_output = aggregator.aggregate(
        [
            create_model_runner_output(
                reqs=[],
                finished_recving={request_id},
                failed_recving={request_id},
            ),
            create_model_runner_output(reqs=[]),
        ]
    )

    assert first_output is not None
    assert first_output.kv_connector_output is not None
    assert first_output.kv_connector_output.finished_recving is None
    assert first_output.kv_connector_output.failed_recving is None

    final_output = aggregator.aggregate(
        [
            create_model_runner_output(reqs=[]),
            create_model_runner_output(
                reqs=[],
                finished_recving={request_id},
            ),
        ]
    )

    assert final_output is not None
    assert final_output.kv_connector_output is not None
    assert final_output.kv_connector_output.finished_recving == {request_id}
    assert final_output.kv_connector_output.failed_recving == {request_id}


def test_multi_connector_unions_request_level_load_errors() -> None:
    """MultiConnector 合并所有子 connector 的失败请求集合。"""
    first_connector = Mock()
    first_connector.get_request_ids_with_load_errors.return_value = {
        "request-0",
        "request-shared",
    }
    second_connector = Mock()
    second_connector.get_request_ids_with_load_errors.return_value = {
        "request-1",
        "request-shared",
    }
    connector = object.__new__(MultiConnector)
    connector._connectors = [first_connector, second_connector]

    assert connector.get_request_ids_with_load_errors() == {
        "request-0",
        "request-1",
        "request-shared",
    }
