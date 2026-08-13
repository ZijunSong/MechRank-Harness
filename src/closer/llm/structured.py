"""Structured JSON generation with schema validation and real-model repair."""

from __future__ import annotations

import json
import re
from typing import Any

from pydantic import BaseModel, ValidationError

from closer.errors import StructuredOutputError


def model_json_schema(schema: type[BaseModel]) -> dict[str, Any]:
    raw = schema.model_json_schema()
    return _strictify(raw)


def _strictify(schema: dict[str, Any]) -> dict[str, Any]:
    out = dict(schema)
    if out.get("type") == "object" or "properties" in out:
        out["additionalProperties"] = False
        properties = out.get("properties") or {}
        out["required"] = list(properties.keys())
        out["properties"] = {key: _strictify(value) if isinstance(value, dict) else value for key, value in properties.items()}
    if "items" in out and isinstance(out["items"], dict):
        out["items"] = _strictify(out["items"])
    if "$defs" in out:
        out["$defs"] = {key: _strictify(value) for key, value in out["$defs"].items()}
    if "anyOf" in out:
        out["anyOf"] = [_strictify(item) if isinstance(item, dict) else item for item in out["anyOf"]]
    return out


def extract_json_object(text: str) -> dict[str, Any]:
    if not text or not text.strip():
        raise StructuredOutputError("empty model output")
    stripped = text.strip()
    fence = re.search(r"```(?:json)?\s*(\{.*\})\s*```", stripped, re.DOTALL)
    if fence:
        stripped = fence.group(1)
    try:
        parsed = json.loads(stripped)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        parsed = None
    if isinstance(parsed, dict):
        return parsed
    start = stripped.find("{")
    end = stripped.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise StructuredOutputError("no JSON object found in model output")
    try:
        parsed = json.loads(stripped[start : end + 1])
    except json.JSONDecodeError as exc:
        raise StructuredOutputError(f"malformed JSON: {exc}") from exc
    if not isinstance(parsed, dict):
        raise StructuredOutputError("JSON payload is not an object")
    return parsed


def parse_model(schema: type[BaseModel], text: str) -> BaseModel:
    payload = extract_json_object(text)
    try:
        return schema.model_validate(payload)
    except ValidationError as exc:
        raise StructuredOutputError(str(exc)) from exc


def json_only_system(schema: type[BaseModel], extra_system: str) -> str:
    schema_text = json.dumps(model_json_schema(schema), indent=2)
    return (
        f"{extra_system}\n\n"
        "Return ONLY a JSON object that matches this schema. "
        "Do not wrap it in markdown. Do not add commentary.\n"
        f"{schema_text}"
    )


def repair_user_prompt(schema: type[BaseModel], previous: str, error: str) -> str:
    schema_text = json.dumps(model_json_schema(schema), indent=2)
    return (
        "Your previous output failed schema validation.\n"
        f"Validation error:\n{error}\n\n"
        f"Previous output:\n{previous}\n\n"
        "Re-emit ONLY valid JSON matching this schema. Do not invent missing biological "
        "facts; if a required field is unknown, use the schema's uncertain/unknown value "
        "and explain that uncertainty in a rationale field if present.\n"
        f"{schema_text}"
    )
