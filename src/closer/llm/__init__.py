"""LLM package factory."""

from __future__ import annotations

from pathlib import Path

from closer.config import CloserConfig
from closer.llm.base import LLMClient
from closer.llm.openai_chat import OpenAIChatClient
from closer.llm.openai_responses import OpenAIResponsesClient


def build_llm_client(config: CloserConfig, *, trace_dir: Path | None = None) -> LLMClient:
    config.require_base_model()
    if config.base_model.api_style == "chat":
        return OpenAIChatClient(config, trace_dir=trace_dir)
    return OpenAIResponsesClient(config, trace_dir=trace_dir)
