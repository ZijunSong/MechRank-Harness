"""OpenAI-compatible /v1/chat/completions implementation."""

from __future__ import annotations

import time
import uuid
from typing import Any

from closer.benchmark.pgllm_parser import parse_pgllm_prompt
from closer.benchmark.pgllm_renderer import render_benchmark_response
from closer.errors import BenchmarkParseError
from closer.orchestrator.engine import CloserEngine
from closer.server.request_parse import extract_system_user, is_connectivity_probe


def chat_envelope(*, model: str, text: str, usage: dict[str, int | None]) -> dict[str, Any]:
    input_tokens = int(usage.get("input_tokens") or 0)
    output_tokens = int(usage.get("output_tokens") or 0)
    total_tokens = int(usage.get("total_tokens") or (input_tokens + output_tokens))
    return {
        "id": f"chatcmpl_{uuid.uuid4().hex}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": text},
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": input_tokens,
            "completion_tokens": output_tokens,
            "total_tokens": total_tokens,
        },
    }


async def handle_chat(engine: CloserEngine, body: dict[str, Any]) -> dict[str, Any]:
    system, user = extract_system_user(body)
    model_id = engine.config.server_model_id()
    if is_connectivity_probe(system, user):
        return chat_envelope(
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
    result = await engine.run_episode(episode)
    text = render_benchmark_response(result.ranking, include_preamble=False)
    return chat_envelope(model=model_id, text=text, usage=result.usage)
