import pytest

from closer.config import load_config
from closer.server.request_parse import extract_system_user, is_connectivity_probe
from closer.server.responses import handle_responses, responses_envelope


class _Engine:
    def __init__(self) -> None:
        self.config = load_config()


@pytest.mark.asyncio
async def test_probe_returns_ok():
    body = {
        "model": "closer-v1",
        "instructions": "You are a connectivity test.",
        "input": "Reply with the single word: OK",
    }
    payload = await handle_responses(_Engine(), body)
    assert payload["output_text"] == "OK"
    assert payload["model"] == "closer-v1"
    assert payload["status"] == "completed"
    assert isinstance(payload["usage"]["output_tokens"], int)


def test_extract_chat_messages():
    system, user = extract_system_user(
        {
            "messages": [
                {"role": "system", "content": "sys"},
                {"role": "user", "content": "hello"},
            ]
        }
    )
    assert system == "sys"
    assert user == "hello"


def test_probe_detection():
    assert is_connectivity_probe("You are a connectivity test.", "Reply with the single word: OK")


def test_envelope_has_usage_and_output_text():
    env = responses_envelope(model="closer-v1", text="OK", usage={"input_tokens": 3, "output_tokens": 1})
    assert env["output_text"] == "OK"
    assert env["usage"]["total_tokens"] == 4
