"""LLM call and episode traces."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

from closer import TRACE_SCHEMA_VERSION


class TokenUsage(BaseModel):
    input_tokens: int | None = None
    output_tokens: int | None = None
    total_tokens: int | None = None
    reasoning_tokens: int | None = None
    cached_tokens: int | None = None
    cost: float | None = None


class LLMCallTrace(BaseModel):
    schema_version: str = TRACE_SCHEMA_VERSION
    call_id: str
    stage: str
    model_requested: str
    model_returned: str | None = None
    prompt_hash: str
    raw_request: dict[str, Any] | None = None
    raw_response: dict[str, Any] | None = None
    parsed_response: dict[str, Any] | None = None
    usage: TokenUsage = Field(default_factory=TokenUsage)
    latency_ms: float
    retry_count: int = 0
    repair_count: int = 0
    status: Literal["ok", "error"] = "ok"
    error: str | None = None
