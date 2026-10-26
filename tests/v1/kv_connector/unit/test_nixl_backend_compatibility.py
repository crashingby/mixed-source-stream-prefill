# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""验证自定义 attention 与 Triton 的 KV 兼容识别，不加载真实模型。"""

from types import SimpleNamespace
from typing import cast

import pytest

from vllm.config import VllmConfig
from vllm.distributed.kv_transfer.kv_connector.v1.nixl.metadata import (
    compute_nixl_compatibility_hash,
)

pytestmark = pytest.mark.cpu_test


def make_config() -> VllmConfig:
    """提供哈希所需的模型和缓存元数据，避免下载权重或初始化 GPU。"""
    return cast(
        VllmConfig,
        SimpleNamespace(
            model_config=SimpleNamespace(
                model="llama-3.1-8b-instruct",
                dtype="bfloat16",
                get_total_num_kv_heads=lambda: 8,
                get_head_size=lambda: 128,
                get_total_num_hidden_layers=lambda: 32,
            ),
            cache_config=SimpleNamespace(cache_dtype="auto"),
            scheduler_config=SimpleNamespace(disable_hybrid_kv_cache_manager=True),
        ),
    )


def test_chunk_sparse_and_triton_share_kv_compatibility() -> None:
    """仅为共用 KV 写入格式的两个后端建立兼容关系。"""
    config = make_config()
    triton = compute_nixl_compatibility_hash(config, "TRITON_ATTN", False)
    assert compute_nixl_compatibility_hash(config, "CHUNK_SPARSE_ATTN", False) == triton
    assert compute_nixl_compatibility_hash(config, "FLASH_ATTN", False) != triton
    assert compute_nixl_compatibility_hash(config, "TRITON_ATTN", True) != triton


@pytest.mark.parametrize("field", ["model", "dtype", "cache_dtype"])
def test_mixed_backend_still_rejects_incompatible_config(field: str) -> None:
    """后端别名不绕过模型、计算 dtype 或 KV dtype 的检查。"""
    local, remote = make_config(), make_config()
    if field == "cache_dtype":
        remote.cache_config.cache_dtype = "fp8"
    else:
        setattr(remote.model_config, field, "different")
    assert compute_nixl_compatibility_hash(
        local, "TRITON_ATTN", False
    ) != compute_nixl_compatibility_hash(remote, "CHUNK_SPARSE_ATTN", False)
