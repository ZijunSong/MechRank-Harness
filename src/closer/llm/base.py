"""LLM client abstraction."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from closer.config import CloserConfig
from closer.orchestrator.budget import BudgetManager
from closer.schemas.trace import LLMCallTrace


class LLMClient(ABC):
    def __init__(self, config: CloserConfig, *, trace_dir: Path | None = None) -> None:
        self.config = config
        self.trace_dir = trace_dir

    @abstractmethod
    async def generate_text(
        self,
        *,
        stage: str,
        system: str,
        user: str,
        budget: BudgetManager | None = None,
        max_output_tokens: int | None = None,
    ) -> tuple[str, LLMCallTrace]: ...

    @abstractmethod
    async def generate_structured(
        self,
        *,
        stage: str,
        schema: type[BaseModel],
        system: str,
        user: str,
        budget: BudgetManager | None = None,
    ) -> tuple[BaseModel, LLMCallTrace]: ...

    async def aclose(self) -> None:
        return None


def extract_usage(payload: dict[str, Any]) -> dict[str, int | float | None]:
    usage = payload.get("usage") or {}
    input_tokens = usage.get("input_tokens", usage.get("prompt_tokens"))
    output_tokens = usage.get("output_tokens", usage.get("completion_tokens"))
    total_tokens = usage.get("total_tokens")
    details = usage.get("output_tokens_details") or usage.get("completion_tokens_details") or {}
    input_details = usage.get("input_tokens_details") or {}
    cost = usage.get("cost") or usage.get("total_cost")
    if total_tokens is None and input_tokens is not None and output_tokens is not None:
        total_tokens = int(input_tokens) + int(output_tokens)
    return {
        "input_tokens": int(input_tokens) if input_tokens is not None else None,
        "output_tokens": int(output_tokens) if output_tokens is not None else None,
        "total_tokens": int(total_tokens) if total_tokens is not None else None,
        "reasoning_tokens": details.get("reasoning_tokens"),
        "cached_tokens": input_details.get("cached_tokens"),
        "cost": float(cost) if cost is not None else None,
    }


def extract_output_text(payload: dict[str, Any]) -> str:
    if payload.get("output_text"):
        return str(payload["output_text"])
    output = payload.get("output") or []
    chunks: list[str] = []
    for item in output:
        if not isinstance(item, dict):
            continue
        for content in item.get("content") or []:
            if isinstance(content, dict) and content.get("text"):
                chunks.append(str(content["text"]))
            elif isinstance(content, str):
                chunks.append(content)
    if chunks:
        return "".join(chunks)
    choices = payload.get("choices") or []
    if choices:
        message = choices[0].get("message") or {}
        content = message.get("content")
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            return "".join(
                part.get("text", "") if isinstance(part, dict) else str(part) for part in content
            )
    return ""
