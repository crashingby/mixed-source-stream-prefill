# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""构造模型模板文本；默认将分词交给 P 节点。"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from datetime import datetime
from functools import partial
from pathlib import Path
from typing import Any

from jinja2.sandbox import ImmutableSandboxedEnvironment

from .dataset import get_data_events
from .models import RenderedReplayPrompt, SampleReplayInput

CHAT_TEMPLATE_CONTENT_MARKER = "VLLM_PROMPT_STREAM_CONTENT_7CDA3F6B"


class LocalTemplateTokenizer:
    """读取本地模板配置；仅在显式预分词模式中加载 tokenizer.json。"""

    def __init__(self, directory: Path) -> None:
        self.directory = directory
        self.config = json.loads((directory / "tokenizer_config.json").read_text())
        template_path = directory / "chat_template.jinja"
        template = (
            template_path.read_text()
            if template_path.is_file()
            else self.config.get("chat_template")
        )
        if isinstance(template, list):
            templates = {item["name"]: item["template"] for item in template}
            template = templates.get("default")
        if isinstance(template, dict):
            template = template.get("default")
        if not isinstance(template, str) or not template:
            raise ValueError(f"本地 tokenizer 没有默认 chat template：{directory}")
        self.chat_template = template
        environment = ImmutableSandboxedEnvironment(
            trim_blocks=True, lstrip_blocks=True, extensions=["jinja2.ext.loopcontrols"]
        )
        environment.globals["raise_exception"] = self._raise_exception
        environment.globals["strftime_now"] = lambda format: datetime.now().strftime(
            format
        )
        # 与 Hugging Face 文本模板保持 JSON 过滤器的非 HTML 转义行为一致。
        environment.filters["tojson"] = partial(json.dumps, ensure_ascii=False)
        self._template = environment.from_string(template)
        self._tokenizer: Any = None

    @staticmethod
    def _raise_exception(message: str) -> None:
        raise ValueError(message)

    def get_chat_template(self) -> str:
        """返回模板原文，用于本轮配置指纹。"""
        return self.chat_template

    def apply_chat_template(
        self,
        messages: list[dict[str, str]],
        *,
        tokenize: bool,
        add_generation_prompt: bool,
    ) -> str:
        """渲染 system/user 的模型原生文本，不做分词。"""
        if tokenize:
            raise ValueError("文本模板渲染不接受 tokenize=True")
        tokens = {
            name: value.get("content") if isinstance(value, dict) else value
            for name, value in self.config.items()
            if name.endswith("_token")
        }
        return self._template.render(
            messages=messages,
            add_generation_prompt=add_generation_prompt,
            tools=None,
            **tokens,
        )

    def encode(self, text: str, *, add_special_tokens: bool = False) -> list[int]:
        """按需加载本地分词文件，供兼容模式或显式长度诊断使用。"""
        if self._tokenizer is None:
            from tokenizers import Tokenizer

            self._tokenizer = Tokenizer.from_file(
                str(self.directory / "tokenizer.json")
            )
        return self._tokenizer.encode(text, add_special_tokens=add_special_tokens).ids


@dataclass(frozen=True)
class ChatTemplateRenderer:
    """共享模型模板；文本模式只渲染，不在客户端分词。"""

    tokenizer: Any
    tokenizer_model: str
    chat_template_sha256: str

    @classmethod
    def from_pretrained(
        cls,
        tokenizer_model: str,
        trust_remote_code: bool = False,
    ) -> ChatTemplateRenderer:
        """读取本地模板；非本地标识保留 Hugging Face 加载路径。"""
        directory = Path(tokenizer_model).expanduser()
        if (directory / "tokenizer_config.json").is_file():
            tokenizer = LocalTemplateTokenizer(directory)
        else:
            from transformers import AutoTokenizer

            tokenizer = AutoTokenizer.from_pretrained(
                tokenizer_model, trust_remote_code=trust_remote_code
            )
        template = tokenizer.get_chat_template()
        if not isinstance(template, str) or not template:
            raise ValueError(f"模型没有可用 chat template：{tokenizer_model}")
        return cls(
            tokenizer, tokenizer_model, hashlib.sha256(template.encode()).hexdigest()
        )

    def encode_content(self, text: str) -> tuple[int, ...]:
        """显式预分词或诊断调用；默认 HTTP 文本输入不调用此方法。"""
        return tuple(
            int(token)
            for token in self.tokenizer.encode(text, add_special_tokens=False)
        )

    def render_chat_prompt(self, system_message: str, user_content: str) -> str:
        """渲染完整 prompt 文本。"""
        value = self.tokenizer.apply_chat_template(
            [
                {"role": "system", "content": system_message},
                {"role": "user", "content": user_content},
            ],
            tokenize=False,
            add_generation_prompt=True,
        )
        if not isinstance(value, str):
            raise ValueError("chat template 必须返回文本")
        return value

    def render_envelope(self, system_message: str) -> tuple[str, str]:
        """提取 start 与 finish 的模板边界，保留 BOS 和 assistant header。"""
        value = self.render_chat_prompt(system_message, CHAT_TEMPLATE_CONTENT_MARKER)
        if value.count(CHAT_TEMPLATE_CONTENT_MARKER) != 1:
            raise ValueError("chat template 必须完整保留一次 user 内容 marker")
        return tuple(value.split(CHAT_TEMPLATE_CONTENT_MARKER, 1))


def render_sample_prompt(
    sample: SampleReplayInput,
    renderer: ChatTemplateRenderer,
    *,
    input_transport: str = "text",
) -> RenderedReplayPrompt:
    """构造同一份最终文本；stream 的各块由服务端独立分词。

    Args:
        sample: 已按 ID 对齐的证据和问题，不从评价字段构造输入。
        renderer: 模型模板渲染器。
        input_transport: 默认 text；token_ids 仅供历史实验复现。
    """
    append_parts, update_parts = [], []
    for source in sample.source_streams:
        records = get_data_events(source)
        if source.source_type == "append":
            append_parts.extend(str(row["event"].get("text") or "") for row in records)
        elif records:
            update_parts.append(str(records[-1]["event"].get("text") or ""))
    append_text, update_text = "".join(append_parts), "".join(update_parts)
    question_text = str(sample.finish_event.get("text") or "")
    prefix, suffix = renderer.render_envelope(
        str(sample.start_request["prompt"]).strip()
    )
    user_content = append_text + update_text + question_text
    full_text = prefix + user_content + suffix
    native_text = renderer.render_chat_prompt(
        str(sample.start_request["prompt"]).strip(), user_content
    )
    # Llama 模板会 trim 用户内容；流式事件保留边缘空白，三组共用同一拼接文本。
    if native_text not in (full_text, prefix + user_content.strip() + suffix):
        raise ValueError(f"模板无法无损拆分：{sample.sample_id}")
    # 模板后缀必须随 finish 文本传输，不能仅保存在客户端 token 列表中。
    finish_text = question_text + suffix
    start_ids = finish_ids = full_ids = ()
    if input_transport == "token_ids":
        start_ids = renderer.encode_content(prefix)
        finish_ids = renderer.encode_content(question_text) + renderer.encode_content(
            suffix
        )
        full_ids = (
            start_ids
            + renderer.encode_content(append_text)
            + renderer.encode_content(update_text)
            + finish_ids
        )
    elif input_transport != "text":
        raise ValueError(f"未知输入传输方式：{input_transport}")
    return RenderedReplayPrompt(
        start_token_ids=start_ids,
        finish_text=finish_text,
        finish_token_ids=finish_ids,
        logical_prompt_token_ids=full_ids,
        logical_prompt_text=full_text,
        append_chars=len(append_text),
        final_update_chars=len(update_text),
        start_text=prefix,
        input_transport=input_transport,
    )


def event_payload(
    event: dict[str, Any],
    renderer: ChatTemplateRenderer,
    input_transport: str = "text",
) -> dict[str, Any]:
    """复制事件并确定唯一输入模式，避免遗留 token_ids 覆盖新的正文。"""
    outbound = dict(event)
    outbound.pop("token_ids", None)
    if input_transport == "token_ids" and outbound.get("text") is not None:
        outbound["token_ids"] = list(renderer.encode_content(str(outbound["text"])))
    return outbound


def override_sampling_params(request: dict[str, Any], args: argparse.Namespace) -> None:
    """将实验采样参数应用于请求。"""
    for name in ("max_tokens", "temperature", "seed"):
        value = getattr(args, name, None)
        if value is not None:
            request[name] = value
