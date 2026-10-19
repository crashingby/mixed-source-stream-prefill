# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""历史导入路径兼容层；公共实现位于 replay/common。"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

from research.scripts.replay.common.config import DEFAULT_DATA_DIR as DEFAULT_DATA_DIR
from research.scripts.replay.common.config import (
    DEFAULT_EVENT_PROGRESS_LOG_INTERVAL as DEFAULT_EVENT_PROGRESS_LOG_INTERVAL,
)
from research.scripts.replay.common.config import (
    DEFAULT_OUTPUT_DIR as DEFAULT_OUTPUT_DIR,
)
from research.scripts.replay.common.config import DEFAULT_PROXY_URL as DEFAULT_PROXY_URL
from research.scripts.replay.common.config import (
    STREAM_TIMING_MODES as STREAM_TIMING_MODES,
)
from research.scripts.replay.common.config import parse_args as parse_args
from research.scripts.replay.common.config import validate_args as _validate_args
from research.scripts.replay.common.dataset import (
    MIX_SOURCE_SCHEMA_VERSION as MIX_SOURCE_SCHEMA_VERSION,
)
from research.scripts.replay.common.dataset import (
    extract_source_streams as _extract_source_streams,
)
from research.scripts.replay.common.dataset import get_data_events as _get_data_events
from research.scripts.replay.common.dataset import (
    limit_selected_samples as _limit_selected_samples,
)
from research.scripts.replay.common.dataset import load_json as _load_json
from research.scripts.replay.common.dataset import (
    numeric_query_sort_key as _numeric_query_sort_key,
)
from research.scripts.replay.common.dataset import parse_sample as _parse_sample
from research.scripts.replay.common.dataset import request_limit as _request_limit
from research.scripts.replay.common.dataset import select_samples as _select_samples
from research.scripts.replay.common.metrics import (
    EVENT_METRIC_FIELDS as EVENT_METRIC_FIELDS,
)
from research.scripts.replay.common.metrics import (
    REQUEST_METRIC_FIELDS as REQUEST_METRIC_FIELDS,
)
from research.scripts.replay.common.metrics import (
    apply_prompt_stream_stats as apply_prompt_stream_stats,
)
from research.scripts.replay.common.metrics import (
    apply_prompt_stream_stats_or_raise as _apply_prompt_stream_stats_or_raise,
)
from research.scripts.replay.common.metrics import percentile as _percentile
from research.scripts.replay.common.metrics import (
    request_metric_row as _request_metric_row,
)
from research.scripts.replay.common.metrics import (
    require_non_negative_stat as _require_non_negative_stat,
)
from research.scripts.replay.common.metrics import (
    require_optional_non_negative_stat as _require_optional_non_negative_stat,
)
from research.scripts.replay.common.metrics import (
    summarize_metrics as _summarize_metrics,
)
from research.scripts.replay.common.models import ArrivalProcess as ArrivalProcess
from research.scripts.replay.common.models import EventMetric as EventMetric
from research.scripts.replay.common.models import (
    RenderedReplayPrompt as RenderedReplayPrompt,
)
from research.scripts.replay.common.models import ReplayMode as ReplayMode
from research.scripts.replay.common.models import ReplaySample as ReplaySample
from research.scripts.replay.common.models import RequestMetric as RequestMetric
from research.scripts.replay.common.models import SampleReplayInput as SampleReplayInput
from research.scripts.replay.common.models import SourceEventStream as SourceEventStream
from research.scripts.replay.common.models import SseMeasurement as SseMeasurement
from research.scripts.replay.common.prompt import (
    CHAT_TEMPLATE_CONTENT_MARKER as CHAT_TEMPLATE_CONTENT_MARKER,
)
from research.scripts.replay.common.prompt import (
    ChatTemplateRenderer as ChatTemplateRenderer,
)
from research.scripts.replay.common.prompt import (
    override_sampling_params as _override_sampling_params,
)
from research.scripts.replay.common.prompt import (
    render_sample_prompt as _render_sample_prompt,
)
from research.scripts.replay.common.results import json_default as _json_default
from research.scripts.replay.common.results import (
    make_run_directory as _make_run_directory,
)
from research.scripts.replay.common.results import utc_timestamp as _utc_timestamp
from research.scripts.replay.common.results import write_csv as _write_csv
from research.scripts.replay.common.results import write_json as _write_json
from research.scripts.replay.common.results import (
    write_run_outputs as _write_run_outputs,
)
from research.scripts.replay.common.runner import main as main
from research.scripts.replay.common.runner import replay_requests as _replay_requests
from research.scripts.replay.common.runner import run as run
from research.scripts.replay.common.timeline import (
    planned_arrival_offsets as _planned_arrival_offsets,
)
from research.scripts.replay.common.timeline import (
    should_log_event_progress as should_log_event_progress,
)
from research.scripts.replay.common.timeline import sleep_until as _sleep_until
from research.scripts.replay.common.transport import (
    LegacyStatsSemanticsIncompatible as LegacyStatsSemanticsIncompatible,
)
from research.scripts.replay.common.transport import ReplayHttpError as ReplayHttpError
from research.scripts.replay.common.transport import chunk_text as _chunk_text
from research.scripts.replay.common.transport import (
    consume_sse_body as _consume_sse_response,
)
from research.scripts.replay.common.transport import (
    consume_sse_response as consume_sse_response,
)
from research.scripts.replay.common.transport import (
    is_first_token_choice as _is_first_token_choice,
)
from research.scripts.replay.common.transport import (
    normalize_v1_url as _normalize_v1_url,
)
from research.scripts.replay.common.transport import post_json as _post_json
from research.scripts.replay.common.transport import (
    prompt_stream_finish_endpoint as _prompt_stream_finish_endpoint,
)
from research.scripts.replay.common.transport import (
    prompt_stream_start_endpoint as _prompt_stream_start_endpoint,
)

__all__ = [
    "SourceEventStream",
    "SampleReplayInput",
    "RenderedReplayPrompt",
    "EventMetric",
    "RequestMetric",
    "SseMeasurement",
    "ReplaySample",
    "ReplayMode",
    "ArrivalProcess",
    "MIX_SOURCE_SCHEMA_VERSION",
    "_numeric_query_sort_key",
    "_load_json",
    "_extract_source_streams",
    "_parse_sample",
    "_select_samples",
    "_get_data_events",
    "_request_limit",
    "_limit_selected_samples",
    "CHAT_TEMPLATE_CONTENT_MARKER",
    "ChatTemplateRenderer",
    "_render_sample_prompt",
    "_override_sampling_params",
    "DEFAULT_DATA_DIR",
    "DEFAULT_OUTPUT_DIR",
    "DEFAULT_PROXY_URL",
    "STREAM_TIMING_MODES",
    "DEFAULT_EVENT_PROGRESS_LOG_INTERVAL",
    "parse_args",
    "_validate_args",
    "_sleep_until",
    "should_log_event_progress",
    "_planned_arrival_offsets",
    "ReplayHttpError",
    "LegacyStatsSemanticsIncompatible",
    "_normalize_v1_url",
    "_prompt_stream_finish_endpoint",
    "_prompt_stream_start_endpoint",
    "_post_json",
    "_is_first_token_choice",
    "_chunk_text",
    "_consume_sse_response",
    "consume_sse_response",
    "REQUEST_METRIC_FIELDS",
    "EVENT_METRIC_FIELDS",
    "_require_non_negative_stat",
    "_require_optional_non_negative_stat",
    "_apply_prompt_stream_stats_or_raise",
    "apply_prompt_stream_stats",
    "_percentile",
    "_summarize_metrics",
    "_request_metric_row",
    "_utc_timestamp",
    "_json_default",
    "_write_json",
    "_write_csv",
    "_write_run_outputs",
    "_make_run_directory",
    "_replay_requests",
    "run",
    "main",
]
