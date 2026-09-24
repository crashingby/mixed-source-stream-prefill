# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Pull-specific scheduler-side logic for the NIXL connector."""

import time
from typing import TYPE_CHECKING, Any

from vllm.distributed.kv_transfer.kv_connector.utils import BlockIds
from vllm.distributed.kv_transfer.kv_connector.v1.nixl.base_scheduler import (
    NixlBaseConnectorScheduler,
)
from vllm.logger import init_logger
from vllm.utils.math_utils import cdiv
from vllm.v1.engine.prompt_stream import parse_prompt_stream_padding_range

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.v1.core.kv_cache_manager import KVCacheBlocks
    from vllm.v1.kv_cache_interface import KVCacheConfig
    from vllm.v1.request import Request

logger = init_logger(__name__)


def _normalize_prompt_stream_manifest_ints(
    value: Any,
    field_name: str,
    *,
    minimum: int,
) -> list[int]:
    """严格解析 prompt-stream manifest 的整数数组。"""
    if not isinstance(value, (list, tuple)):
        raise ValueError(f"{field_name} must be a list or tuple")
    normalized: list[int] = []
    for index, item in enumerate(value):
        if type(item) is not int or item < minimum:
            raise ValueError(
                f"{field_name}[{index}] must be an integer >= {minimum}, "
                f"but got {item!r}"
            )
        normalized.append(item)
    return normalized


def _normalize_prompt_stream_remote_block_ids(value: Any) -> BlockIds:
    """严格解析 prompt-stream manifest 中按 KV group 排列的 block IDs。"""
    if not isinstance(value, (list, tuple)):
        raise ValueError("remote_block_ids must be a list or tuple")
    normalized: list[list[int]] = []
    for group_id, group in enumerate(value):
        if not isinstance(group, (list, tuple)):
            raise ValueError(f"remote_block_ids[{group_id}] must be a list or tuple")
        normalized_group: list[int] = []
        for block_index, block_id in enumerate(group):
            if type(block_id) is not int or block_id < 0:
                raise ValueError(
                    "remote_block_ids"
                    f"[{group_id}][{block_index}] must be a non-negative "
                    f"integer, but got {block_id!r}"
                )
            normalized_group.append(block_id)
        normalized.append(normalized_group)
    return tuple(normalized)


def _prompt_stream_block_counts(value: Any) -> list[int] | None:
    """仅用于错误日志，安全读取可能畸形的 block groups。"""
    if not isinstance(value, (list, tuple)):
        return None
    if not all(isinstance(group, (list, tuple)) for group in value):
        return None
    return [len(group) for group in value]


def _trim_prompt_stream_local_blocks(
    block_ids: BlockIds,
    kv_cache_config: "KVCacheConfig",
    num_external_tokens: int,
) -> BlockIds:
    """只保留 D 侧实际需要从 P 加载的前缀 blocks。"""
    if len(block_ids) != len(kv_cache_config.kv_cache_groups):
        raise ValueError(
            "Prompt stream local block groups do not match KV cache groups: "
            f"blocks={len(block_ids)}, "
            f"groups={len(kv_cache_config.kv_cache_groups)}."
        )
    return tuple(
        group[: cdiv(num_external_tokens, kv_group.kv_cache_spec.block_size)]
        for group, kv_group in zip(
            block_ids,
            kv_cache_config.kv_cache_groups,
        )
    )


def _trim_prompt_stream_local_blocks_to_manifest(
    block_ids: BlockIds,
    expected_block_counts: list[int] | tuple[int, ...],
) -> BlockIds:
    """按 P 侧权威 manifest 裁剪 D 的逻辑 block table。

    D prompt 会额外包含 P 采样出的首 token，所以它最终可能比 remote KV
    多一个 tail block。manifest 只描述需要从 P 读取的物理 prompt blocks。
    """
    if len(block_ids) != len(expected_block_counts):
        raise ValueError(
            "Prompt stream local block groups do not match manifest: "
            f"local_groups={len(block_ids)}, "
            f"manifest_groups={len(expected_block_counts)}."
        )

    trimmed_groups: list[list[int]] = []
    for group_id, (group, expected_count) in enumerate(
        zip(block_ids, expected_block_counts)
    ):
        if expected_count < 0 or len(group) < expected_count:
            raise ValueError(
                "Prompt stream local block count is smaller than manifest: "
                f"group_id={group_id}, local_blocks={len(group)}, "
                f"expected_blocks={expected_count}."
            )
        trimmed_groups.append(group[:expected_count])
    return tuple(trimmed_groups)


class NixlPullConnectorScheduler(NixlBaseConnectorScheduler):
    """Pull-specific scheduler logic (READ-based KV transfer)."""

    def __init__(
        self,
        vllm_config: "VllmConfig",
        engine_id: str,
        kv_cache_config: "KVCacheConfig",
    ):
        super().__init__(vllm_config, engine_id, kv_cache_config)

    def get_num_new_matched_tokens(
        self, request: "Request", num_computed_tokens: int
    ) -> tuple[int, bool]:
        """
        For remote prefill, pull all prompt blocks from remote
        asynchronously relative to engine execution.

        Args:
            request (Request): the request object.
            num_computed_tokens (int): the number of locally
                computed tokens for this request
        Returns:
            * the number of tokens that can be loaded from the
              external KV cache beyond what is already computed.
            * true if the external KV cache tokens will be loaded
              asynchronously (between scheduler steps).
        """

        params = request.kv_transfer_params
        logger.debug(
            "NIXLConnector get_num_new_matched_tokens: "
            "num_computed_tokens=%s, kv_transfer_params=%s",
            num_computed_tokens,
            params,
        )

        if (
            params is not None
            and params.get("prompt_stream_exact_transfer")
            and params.get("do_remote_prefill")
            and not params.get("_remote_blocks_processed")
        ):
            # P 首 token 已追加到 D prompt，但它没有 remote KV。必须先于
            # 通用 do_remote_prefill 分支按 manifest 覆盖范围计算，否则会把
            # D-only tail token 错算成外部 token，并多申请一个 block。
            remote_num_tokens = params.get("remote_num_tokens")
            if not isinstance(remote_num_tokens, int):
                logger.error(
                    "Prompt stream exact transfer has invalid remote token "
                    "count: request_id=%s, remote_num_tokens=%r",
                    request.request_id,
                    remote_num_tokens,
                )
                return 0, False
            count = (
                min(remote_num_tokens, request.num_prompt_tokens) - num_computed_tokens
            )
            if count > 0:
                # 短 prompt-stream 也必须实际读取 P KV，不能套用通用的
                # kv_recompute_threshold 后退化成 D 本地 full prefill。
                return count, True
            return 0, False

        if params is not None and params.get("do_remote_prefill"):
            # Remote prefill: get all prompt blocks from remote.
            token_ids = request.prompt_token_ids or []
            actual = self._mamba_prefill_token_count(len(token_ids))
            count = actual - num_computed_tokens
            if count > 0:
                return count, True

        if params is not None and params.get("do_remote_decode") and self._has_mamba:
            self._truncate_mamba_request_for_prefill(request)

        if (
            params is not None
            and params.get("do_remote_decode")
            and params.get("remote_block_ids")
            and all(
                p in params
                for p in (
                    "remote_engine_id",
                    "remote_request_id",
                    "remote_host",
                    "remote_port",
                )
            )
        ):
            # Decode node has kv blocks for part of prefill request, so, provide them
            # as an external token count to scheduler.
            # The tokens will be loaded if not already present
            # in the prefill node local cache
            remote_num_tokens = params.get("remote_num_tokens") or 0
            count = (
                min(remote_num_tokens, request.num_prompt_tokens) - num_computed_tokens
            )
            if count > 0:
                # Check kv_recompute_threshold: skip pull if
                # remote tokens are below the threshold.
                if (
                    self.kv_recompute_threshold > 0
                    and count < self.kv_recompute_threshold
                ):
                    logger.debug(
                        "Skipping remote pull for %s: %d remote tokens < threshold %d",
                        request.request_id,
                        count,
                        self.kv_recompute_threshold,
                    )
                    return 0, False
                return count, True

        # No remote prefill for this request.
        return 0, False

    def update_state_after_alloc(
        self, request: "Request", blocks: "KVCacheBlocks", num_external_tokens: int
    ):
        params = request.kv_transfer_params
        logger.debug(
            "NIXLConnector update_state_after_alloc: "
            "num_external_tokens=%s, kv_transfer_params=%s",
            num_external_tokens,
            params,
        )

        if not params:
            return

        if params.get("do_remote_decode") or (
            params.get("do_remote_prefill") and self.is_bidirectional_kv_xfer_enabled
        ):
            self._reqs_in_batch.add(request.request_id)
        if self.use_host_buffer and params.get("do_remote_decode"):
            # NOTE: when accelerator is not directly supported by Nixl,
            # prefilled blocks need to be saved to host memory before transfer.
            self._reqs_need_save[request.request_id] = request
        elif params.get("do_remote_prefill") or (
            params.get("do_remote_decode")
            and self.is_bidirectional_kv_xfer_enabled
            and not params.get("_remote_blocks_processed")
        ):
            if params.get("prompt_stream_exact_transfer"):
                # Exact prompt-stream requests never use the generic
                # prefix-caching alignment path. Validate the P manifest before
                # the worker sees it, then register even invalid requests so the
                # worker can report a request-scoped transfer failure.
                unhashed_local_block_ids: BlockIds = (
                    blocks.get_unhashed_block_ids_all_groups()
                    if num_external_tokens > 0
                    else tuple([] for _ in self.kv_cache_config.kv_cache_groups)
                )
                remote_block_ids: Any = params.get("remote_block_ids")
                expected_counts: list[int] = []
                expected_sizes: list[int] = []
                manifest_error: str | None = None
                try:
                    expected_counts = _normalize_prompt_stream_manifest_ints(
                        params.get("prompt_stream_remote_block_counts"),
                        "prompt_stream_remote_block_counts",
                        minimum=0,
                    )
                    expected_sizes = _normalize_prompt_stream_manifest_ints(
                        params.get("prompt_stream_remote_block_sizes"),
                        "prompt_stream_remote_block_sizes",
                        minimum=1,
                    )
                    local_sizes = [
                        group.kv_cache_spec.block_size
                        for group in self.kv_cache_config.kv_cache_groups
                    ]
                    num_groups = len(local_sizes)
                    if len(expected_counts) != num_groups:
                        raise ValueError(
                            "manifest block-count groups differ from D KV groups: "
                            f"manifest={len(expected_counts)}, local={num_groups}"
                        )
                    if expected_sizes != local_sizes:
                        raise ValueError(
                            "P/D block sizes differ: "
                            f"remote={expected_sizes}, local={local_sizes}"
                        )

                    physical_prompt_len = params.get(
                        "prompt_stream_physical_prompt_len"
                    )
                    if type(physical_prompt_len) is not int or physical_prompt_len < 0:
                        raise ValueError(
                            "prompt_stream_physical_prompt_len must be a "
                            "non-negative integer"
                        )
                    prefill_output_tokens = params.get(
                        "prompt_stream_prefill_output_tokens"
                    )
                    if (
                        type(prefill_output_tokens) is not int
                        or prefill_output_tokens < 0
                        or prefill_output_tokens > 1
                    ):
                        raise ValueError(
                            "prompt_stream_prefill_output_tokens must be 0 or 1"
                        )
                    if request.num_prompt_tokens != (
                        physical_prompt_len + prefill_output_tokens
                    ):
                        raise ValueError(
                            "D prompt length differs from P physical prompt plus "
                            "prefill output: "
                            f"decode={request.num_prompt_tokens}, "
                            f"physical={physical_prompt_len}, "
                            f"prefill_output={prefill_output_tokens}"
                        )
                    # 掩码只改变 attention 的可见范围；KV 仍按完整物理块
                    # 搬运。验证区间，不能依据逻辑长度裁剪 remote block IDs。
                    padding_range = parse_prompt_stream_padding_range(
                        params, request.num_prompt_tokens
                    )
                    if padding_range is not None:
                        pad_start, pad_end = padding_range
                        if any(
                            pad_end % block_size != 0
                            or pad_end - pad_start >= block_size
                            for block_size in expected_sizes
                        ):
                            raise ValueError(
                                "masked padding range does not match the "
                                "PREFIX_APPEND segment boundary"
                            )
                    if num_external_tokens != physical_prompt_len:
                        raise ValueError(
                            "external token count differs from physical prompt: "
                            f"external={num_external_tokens}, "
                            f"physical={physical_prompt_len}"
                        )
                    derived_counts = [
                        cdiv(physical_prompt_len, block_size)
                        for block_size in expected_sizes
                    ]
                    if expected_counts != derived_counts:
                        raise ValueError(
                            "manifest block counts do not cover the physical "
                            f"prompt exactly: manifest={expected_counts}, "
                            f"derived={derived_counts}"
                        )

                    normalized_remote_block_ids = (
                        _normalize_prompt_stream_remote_block_ids(remote_block_ids)
                    )
                    if len(normalized_remote_block_ids) != num_groups:
                        raise ValueError(
                            "remote block groups differ from D KV groups: "
                            f"remote={len(normalized_remote_block_ids)}, "
                            f"local={num_groups}"
                        )
                    remote_counts = [
                        len(group) for group in normalized_remote_block_ids
                    ]
                    if remote_counts != expected_counts:
                        raise ValueError(
                            "remote block ids differ from manifest: "
                            f"remote={remote_counts}, manifest={expected_counts}"
                        )
                    for field_name in (
                        "remote_engine_id",
                        "remote_request_id",
                        "remote_host",
                    ):
                        if not isinstance(
                            params.get(field_name), str
                        ) or not params.get(field_name):
                            raise ValueError(f"missing or invalid {field_name}")
                    remote_port = params.get("remote_port")
                    if type(remote_port) is not int or remote_port <= 0:
                        raise ValueError("remote_port must be a positive integer")
                    tp_size = params.get("tp_size", 1)
                    if type(tp_size) is not int or tp_size <= 0:
                        raise ValueError("tp_size must be a positive integer")

                    params["remote_block_ids"] = normalized_remote_block_ids
                    unhashed_local_block_ids = (
                        _trim_prompt_stream_local_blocks_to_manifest(
                            unhashed_local_block_ids,
                            expected_counts,
                        )
                    )
                except (TypeError, ValueError) as err:
                    manifest_error = str(err)

                # Keep metadata construction safe even if the incoming JSON had
                # None or another non-iterable value.
                params["prompt_stream_remote_block_counts"] = expected_counts
                params["prompt_stream_remote_block_sizes"] = expected_sizes
                if manifest_error is None:
                    params.pop("prompt_stream_manifest_error", None)
                else:
                    params["prompt_stream_manifest_error"] = manifest_error
                    logger.error(
                        "Prompt stream KV manifest mismatch before transfer: "
                        "request_id=%s, error=%s, num_external_tokens=%s, "
                        "local_block_counts=%s, remote_block_counts=%s",
                        request.request_id,
                        manifest_error,
                        num_external_tokens,
                        _prompt_stream_block_counts(unhashed_local_block_ids),
                        _prompt_stream_block_counts(remote_block_ids),
                    )

                local_block_ids = self.get_sw_clipped_blocks(unhashed_local_block_ids)
                self._reqs_need_recv[request.request_id] = (
                    request,
                    local_block_ids,
                )
            elif params.get("remote_block_ids"):
                if all(
                    p in params
                    for p in (
                        "remote_engine_id",
                        "remote_request_id",
                        "remote_host",
                        "remote_port",
                    )
                ):
                    # If remote_blocks and num_external_tokens = 0, we have
                    # a full prefix cache hit on the local node. We need to call
                    # send_notif in _read_blocks to free the memory on the remote node.

                    unhashed_local_block_ids: BlockIds = (
                        blocks.get_unhashed_block_ids_all_groups()
                        if num_external_tokens > 0
                        else ()
                    )
                    if (
                        num_external_tokens > 0
                        and params.get("prompt_stream_physical_prompt_len") is not None
                    ):
                        # Prompt stream proxy 会把 P 采样的首 token 接到 D prompt。
                        # 当 remote prefix 恰好 block 对齐时，D 会为这个本地
                        # token 多申请一块；NIXL 只能加载覆盖 external tokens 的
                        # 前缀 blocks，尾块留给 D 本地 forward。
                        unhashed_local_block_ids = _trim_prompt_stream_local_blocks(
                            unhashed_local_block_ids,
                            self.kv_cache_config,
                            num_external_tokens,
                        )
                    local_block_ids = self.get_sw_clipped_blocks(
                        unhashed_local_block_ids
                    )

                    # Get unhashed blocks to pull from remote. Mind that a full prefix
                    # cache hit is indicated with an empty list.
                    self._reqs_need_recv[request.request_id] = (
                        request,
                        local_block_ids,
                    )

                else:
                    logger.warning(
                        "Got invalid KVTransferParams: %s. This "
                        "request will not utilize KVTransfer",
                        params,
                    )
            else:
                assert num_external_tokens == 0
            # Only trigger 1 KV transfer per request.
            params["do_remote_prefill"] = False
            params["_remote_blocks_processed"] = True

    def request_finished(
        self,
        request: "Request",
        block_ids: "BlockIds",
    ) -> tuple[bool, dict[str, Any] | None]:
        """
        Once a request is finished, determine whether request blocks
        should be freed now or will be sent asynchronously and freed later.
        """
        from vllm.v1.request import RequestStatus

        params = request.kv_transfer_params
        logger.debug(
            "NIXLConnector request_finished(%s), request_status=%s, "
            "kv_transfer_params=%s",
            request.request_id,
            request.status,
            params,
        )
        if not params:
            return False, None

        is_p_node = bool(params.get("do_remote_decode"))
        is_d_node = not is_p_node

        # Stop heartbeating for aborted requests that never reached finished_recving:
        # normal path cleans up in update_connector_output.
        self._stop_heartbeat(request.request_id)

        if params.get("do_remote_prefill"):
            # If do_remote_prefill is still True when the request is finished,
            # update_state_after_alloc must not have been called (the request
            # must have been aborted before it was scheduled, e.g. via the
            # abort_immediately path used to clean up KV-transfer requests
            # rejected at the D-side serving layer).
            # To avoid stranding the prefill blocks in the prefill instance,
            # we must add empty block_ids to _reqs_need_recv so that our
            # worker side will notify and free blocks in the prefill instance.
            self._reqs_need_recv[request.request_id] = (request, [])
            params["do_remote_prefill"] = False
            return False, None

        if is_d_node and not self.is_bidirectional_kv_xfer_enabled:
            return False, None

        if request.status not in (
            RequestStatus.FINISHED_LENGTH_CAPPED,
            RequestStatus.FINISHED_STOPPED,
        ):
            # Also include the case of a P/D Prefill request with immediate
            # block free (eg abort). Stop tracking this request.
            self._reqs_not_processed.add(request.request_id)
            # Clear _reqs_need_save if a request is aborted as partial prefill.
            self._reqs_need_save.pop(request.request_id, None)
            return False, None

        # 同一个业务 request_id 曾经失败过时，成功的新生命周期不能继续携带
        # stale not-processed 标记，否则 worker 会先登记、随后又删除该请求。
        self._reqs_not_processed.discard(request.request_id)

        # TODO: check whether block_ids actually ever be 0. If not we could
        # remove the conditional below
        delay_free_blocks = any(len(group) > 0 for group in block_ids)
        remote_num_tokens = 0
        if delay_free_blocks:
            # Prefill request on remote. It will be read from D upon completion
            request_kv_blocks_ttl = self._kv_lease_duration
            if is_d_node:
                # For blocks pinned on D, use a simpler timeout for now instead of a
                # lease mechanism as turn2 request is client-driven.
                request_kv_blocks_ttl = self.decoder_kv_blocks_ttl
            logger.debug(
                "NIXLConnector request_finished(%s) waiting for %d seconds "
                "before releasing blocks",
                request.request_id,
                request_kv_blocks_ttl,
            )
            self._reqs_need_send[request.request_id] = (
                time.perf_counter() + request_kv_blocks_ttl
            )
            # NOTE HMA will "mark" empty/null blocks in groups with 0s (eg SWA ones),
            # trimming down after allocating for the whole sequence length. Empty
            # blocks are always at the start of the list.
            # Here we "unpad" blocks to send the actual remote blocks to be read.
            block_ids = self.get_sw_clipped_blocks(block_ids)

            remote_num_tokens = request.num_computed_tokens

        if getattr(request, "is_prompt_streaming", lambda: False)():
            logger.info(
                "NIXL prompt stream request_finished: request_id=%s "
                "is_p_node=%s num_prompt_tokens=%s num_computed_tokens=%s "
                "num_output_tokens=%s remote_num_tokens=%s block_counts=%s",
                request.request_id,
                is_p_node,
                request.num_prompt_tokens,
                request.num_computed_tokens,
                request.num_output_tokens,
                remote_num_tokens,
                [len(group) for group in block_ids],
            )

        return delay_free_blocks, dict(
            do_remote_prefill=is_p_node,
            do_remote_decode=is_d_node,
            remote_block_ids=block_ids,
            remote_engine_id=self.engine_id,
            remote_request_id=request.request_id,
            remote_host=self.side_channel_host,
            remote_port=self.side_channel_port,
            tp_size=self.vllm_config.parallel_config.tensor_parallel_size,
            remote_num_tokens=remote_num_tokens,
        )
