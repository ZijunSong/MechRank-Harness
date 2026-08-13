"""Extract system/user text from OpenAI-compatible request bodies."""

from __future__ import annotations

from typing import Any


def _content_to_text(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                if item.get("text"):
                    parts.append(str(item["text"]))
                elif item.get("content"):
                    parts.append(_content_to_text(item["content"]))
        return "\n".join(parts)
    if isinstance(content, dict) and "text" in content:
        return str(content["text"])
    return str(content)


def extract_system_user(body: dict[str, Any]) -> tuple[str, str]:
    system = _content_to_text(body.get("instructions") or body.get("system") or "")
    if body.get("input") is not None:
        user = _content_to_text(body.get("input"))
        if isinstance(body.get("input"), list):
            sys_parts: list[str] = []
            user_parts: list[str] = []
            for item in body["input"]:
                if not isinstance(item, dict):
                    user_parts.append(_content_to_text(item))
                    continue
                role = item.get("role", "user")
                text = _content_to_text(item.get("content") or item.get("text") or item)
                if role == "system":
                    sys_parts.append(text)
                else:
                    user_parts.append(text)
            if sys_parts:
                system = "\n".join(sys_parts)
            user = "\n".join(user_parts)
        return system, user
    messages = body.get("messages") or []
    sys_parts = []
    user_parts = []
    for message in messages:
        role = message.get("role")
        text = _content_to_text(message.get("content"))
        if role == "system":
            sys_parts.append(text)
        elif role in {"user", "assistant"}:
            user_parts.append(text)
    if sys_parts:
        system = "\n".join(sys_parts)
    return system, "\n".join(user_parts)


def is_connectivity_probe(system: str, user: str) -> bool:
    sys_l = system.strip().lower()
    user_l = user.strip().lower()
    return "connectivity test" in sys_l or user_l in {
        "reply with the single word: ok",
        "ok",
    }
