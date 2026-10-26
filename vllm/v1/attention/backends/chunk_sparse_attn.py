# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""让完整读取与 Query probe 块稀疏读取共用同一个分页 attention 算子。"""

from typing import ClassVar

import torch

from vllm.config.cache import CacheDType
from vllm.model_executor.layers.quantization.utils.quant_utils import QuantKey
from vllm.platforms.interface import DeviceCapability
from vllm.v1.attention.backend import AttentionCGSupport, AttentionType, MultipleOf
from vllm.v1.attention.backends.triton_attn import (
    TritonAttentionBackend,
    TritonAttentionImpl,
    TritonAttentionMetadata,
    TritonAttentionMetadataBuilder,
)
from vllm.v1.attention.ops.query_probe_attention import query_probe_attention


class ChunkSparseAttentionMetadataBuilder(TritonAttentionMetadataBuilder):
    """复用分页布局；动态页列表由同步 runner 为整批准备，不支持图捕获。"""

    _cudagraph_support: ClassVar[AttentionCGSupport] = AttentionCGSupport.NEVER


class ChunkSparseAttentionBackend(TritonAttentionBackend):
    """显式选择的实验后端；全量、稀疏及最终 Query 均使用 32×128 tile。"""

    supported_dtypes: ClassVar[list[torch.dtype]] = [torch.float16, torch.bfloat16]
    supported_kv_cache_dtypes: ClassVar[list[CacheDType]] = ["auto"]

    @staticmethod
    def get_name() -> str:
        return "CHUNK_SPARSE_ATTN"

    @staticmethod
    def get_impl_cls() -> type["ChunkSparseAttentionImpl"]:
        return ChunkSparseAttentionImpl

    @staticmethod
    def get_builder_cls() -> type[ChunkSparseAttentionMetadataBuilder]:
        return ChunkSparseAttentionMetadataBuilder

    @staticmethod
    def get_supported_kernel_block_sizes() -> list[int | MultipleOf]:
        return [16]

    @classmethod
    def supports_block_size(cls, block_size: int | None) -> bool:
        return block_size in (None, 16)

    @classmethod
    def supports_head_size(cls, head_size: int) -> bool:
        return head_size in (32, 64, 128, 256)

    @classmethod
    def supports_attn_type(cls, attn_type: str) -> bool:
        return attn_type == AttentionType.DECODER

    @classmethod
    def supports_non_causal(cls) -> bool:
        return False

    @classmethod
    def supports_batch_invariance(cls) -> bool:
        return False

    @classmethod
    def supports_mm_prefix(cls) -> bool:
        return False

    @classmethod
    def supports_sink(cls) -> bool:
        return False

    @classmethod
    def supports_alibi_sqrt(cls) -> bool:
        return False

    @classmethod
    def supports_compute_capability(cls, capability: DeviceCapability) -> bool:
        return capability.to_int() >= 80


class ChunkSparseAttentionImpl(TritonAttentionImpl):
    """沿用 Triton KV 写入和布局，仅替换 attention 读取与可选评分。"""

    def __init__(
        self,
        num_heads: int,
        head_size: int,
        scale: float,
        num_kv_heads: int,
        alibi_slopes: list[float] | None,
        sliding_window: int | None,
        kv_cache_dtype: str,
        logits_soft_cap: float | None = None,
        attn_type: AttentionType = AttentionType.DECODER,
        kv_sharing_target_layer_name: int | None = None,
        sinks: torch.Tensor | None = None,
        use_alibi_sqrt: bool = False,
        chunk_lookback: int = -1,
    ) -> None:
        """在加载模型时拒绝本算子未实现的注意力语义。"""
        if (
            kv_cache_dtype != "auto"
            or alibi_slopes is not None
            or sliding_window is not None
            or logits_soft_cap not in (None, 0)
            or attn_type != AttentionType.DECODER
            or sinks is not None
            or use_alibi_sqrt
            or chunk_lookback != -1
        ):
            raise ValueError(
                "CHUNK_SPARSE_ATTN requires unquantized causal full attention."
            )
        super().__init__(
            num_heads,
            head_size,
            scale,
            num_kv_heads,
            alibi_slopes,
            sliding_window,
            kv_cache_dtype,
            logits_soft_cap,
            attn_type,
            kv_sharing_target_layer_name,
            sinks,
            use_alibi_sqrt,
            chunk_lookback,
        )
        self.supports_quant_query_input = False

    def fused_output_quant_supported(self, quant_key: QuantKey) -> bool:
        """当前 kernel 直接写入 FP16/BF16 输出。"""
        return False

    def forward(
        self,
        layer: torch.nn.Module,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: TritonAttentionMetadata,
        output: torch.Tensor,
        output_scale: torch.Tensor | None = None,
        output_block_scale: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """整批执行一次新版 kernel；dense 使用完整页列表，不触发 probe。"""
        if output_scale is not None or output_block_scale is not None:
            raise NotImplementedError("CHUNK_SPARSE_ATTN has no output quantization.")
        if attn_metadata is None:
            # 模型显存测量没有真实请求，不读取 KV。
            return output.fill_(0)
        assert attn_metadata.query_probe is not None
        key_cache, value_cache = kv_cache.unbind(1)
        query_probe_attention(
            query,
            key_cache,
            value_cache,
            output,
            attn_metadata.block_table,
            attn_metadata.query_probe,
            self.scale,
            layer.layer_name,
        )
        return output
