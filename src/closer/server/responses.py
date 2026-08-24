"""OpenAI-compatible /v1/responses implementation."""

from __future__ import annotations

import time
import uuid
from typing import Any

from closer.benchmark.pgllm_parser import parse_pgllm_prompt
from closer.benchmark.pgllm_renderer import render_benchmark_response
from closer.errors import BenchmarkParseError
from closer.orchestrator.engine import CloserEngine
from closer.server.request_parse import (
    extract_max_output_tokens,
    extract_system_user,
    is_connectivity_probe,
)


def responses_envelope(
    *,
    model: str,
    text: str,
    usage: dict[str, int | None],
    response_id: str | None = None,
    status: str = "completed",
) -> dict[str, Any]:
    rid = response_id or f"resp_{uuid.uuid4().hex}"
    msg_id = f"msg_{uuid.uuid4().hex}"
    created = int(time.time())
    input_tokens = int(usage.get("input_tokens") or 0)
    output_tokens = int(usage.get("output_tokens") or 0)
    total_tokens = int(usage.get("total_tokens") or (input_tokens + output_tokens))
    return {
        "id": rid,
        "object": "response",
        "created_at": created,
        "status": status,
        "model": model,
        "output_text": text,
        "output": [
            {
                "id": msg_id,
                "type": "message",
                "status": "completed",
                "role": "assistant",
                "content": [{"type": "output_text", "text": text}],
            }
        ],
        "usage": {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": total_tokens,
            "output_tokens_details": {
                "reasoning_tokens": int(usage.get("reasoning_tokens") or 0),
            },
            "input_tokens_details": {
                "cached_tokens": int(usage.get("cached_tokens") or 0),
            },
        },
        "incomplete_details": None,
        "error": None,
    }


async def handle_responses(engine: CloserEngine, body: dict[str, Any]) -> dict[str, Any]:
    system, user = extract_system_user(body)
    model_id = engine.config.server_model_id()
    if is_connectivity_probe(system, user):
        return responses_envelope(
            model=model_id,
            text="OK",
            usage={
                "input_tokens": max(1, len(system + user) // 4),
                "output_tokens": 1,
                "total_tokens": max(2, len(system + user) // 4 + 1),
            },
        )
    try:
        episode = parse_pgllm_prompt(user, system_prompt=system or None)
    except BenchmarkParseError:
        raise
    result = await engine.run_episode(
        episode,
        max_output_tokens=extract_max_output_tokens(body),
    )
    text = render_benchmark_response(result.ranking, include_preamble=False)
    return responses_envelope(model=model_id, text=text, usage=result.usage)
