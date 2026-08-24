#!/usr/bin/env python3
"""Render a PG-LLM model registry JSON from environment variables.

Official PG-LLM registries reference env var *names*, not secret values.
Set PGLLM_OPENAI_BASE_URL and PGLLM_OPENAI_API_KEY in .env, then:

  python scripts/pgllm/render_model_registry.py > configs/pgllm_active_model.json
  export PGLLM_REGISTRY=configs/pgllm_active_model.json
  export PGLLM_MODELS=gpt55
"""

from __future__ import annotations

import json
import os
import re
import sys


def _alias(model_id: str) -> str:
    alias = os.environ.get("PGLLM_MODEL_ALIAS", "").strip()
    if alias:
        return alias
    slug = re.sub(r"[^a-zA-Z0-9]+", "_", model_id).strip("_").lower()
    return slug or "model"


def main() -> int:
    model_id = os.environ.get("PGLLM_MODEL_ID", os.environ.get("CLOSER_MODEL", "")).strip()
    if not model_id:
        print("Set PGLLM_MODEL_ID (provider model id) before rendering.", file=sys.stderr)
        return 1

    alias = _alias(model_id)
    display = os.environ.get("PGLLM_MODEL_DISPLAY_NAME", model_id).strip() or model_id
    reasoning = os.environ.get("PGLLM_REASONING", "high").strip() or "high"
    max_tokens = int(os.environ.get("PGLLM_MAX_TOKENS", "4096"))
    ctx = int(os.environ.get("PGLLM_CTX", "262144"))
    api_key_env = os.environ.get("PGLLM_API_KEY_ENV", "PGLLM_OPENAI_API_KEY").strip()
    base_url_env = os.environ.get("PGLLM_BASE_URL_ENV", "PGLLM_OPENAI_BASE_URL").strip()
    response_ids_raw = os.environ.get("PGLLM_RESPONSE_MODEL_IDS", model_id).strip()
    response_ids = [item.strip() for item in response_ids_raw.split(",") if item.strip()]
    leaderboard_preset = os.environ.get("PGLLM_LEADERBOARD_PRESET", "false").lower() in {
        "1",
        "true",
        "yes",
    }

    registry = {
        "models": {
            alias: {
                "display_name": display,
                "provider": "openai-compatible",
                "api_style": "responses",
                "model_id": model_id,
                "response_model_ids": response_ids or [model_id],
                "api_key_env": api_key_env,
                "base_url_env": base_url_env,
                "reasoning": reasoning,
                "send_reasoning": True,
                "leaderboard_preset": leaderboard_preset,
                "require_usage": True,
                "max_tokens": max_tokens,
                "ctx": ctx,
            }
        }
    }
    json.dump(registry, sys.stdout, indent=2)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
