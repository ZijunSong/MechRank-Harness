import pytest

from closer.config import load_config
from closer.server.chat import handle_chat
from closer.server.request_parse import (
    extract_max_output_tokens,
    extract_system_user,
    is_connectivity_probe,
)
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


def test_extract_max_output_tokens():
    assert extract_max_output_tokens({"max_output_tokens": 4096}) == 4096
    assert extract_max_output_tokens({"max_completion_tokens": 8192}) == 8192
    assert extract_max_output_tokens({"max_tokens": 2048}) == 2048
    assert extract_max_output_tokens({}) is None
    assert extract_max_output_tokens({"max_output_tokens": 0}) is None


def test_extract_max_output_tokens_rejects_illegal_and_conflicts():
    from closer.errors import BenchmarkParseError

    with pytest.raises(BenchmarkParseError, match="positive integer"):
        extract_max_output_tokens({"max_tokens": True})
    with pytest.raises(BenchmarkParseError, match="positive integer"):
        extract_max_output_tokens({"max_tokens": "4096"})
    with pytest.raises(BenchmarkParseError, match="conflicting"):
        extract_max_output_tokens({"max_tokens": 100, "max_output_tokens": 200})


@pytest.mark.asyncio
async def test_chat_and_responses_forward_output_limits():
    from tests.fixtures.episodes import ranking_v1_prompt

    from closer.assay.interpreter import deterministic_ablation_contract
    from closer.benchmark.pgllm_renderer import SYSTEM_PROMPT
    from closer.schemas.ranking import CloserResult

    class _RecordingEngine:
        def __init__(self) -> None:
            self.config = load_config()
            self.max_output_tokens = None

        async def run_episode(self, episode, *, max_output_tokens=None):
            self.max_output_tokens = max_output_tokens
            ranking = list(episode.variant_ids())
            return CloserResult(
                ranking=ranking,
                scores={vid: 1.0 for vid in ranking},
                assay_contract=deterministic_ablation_contract(episode),
                diagnostics={},
                usage={"input_tokens": 4, "output_tokens": 2, "total_tokens": 6},
                trace_path="",
            )

    engine = _RecordingEngine()
    prompt = ranking_v1_prompt(3)
    await handle_responses(
        engine,
        {
            "model": "closer-v1",
            "instructions": SYSTEM_PROMPT,
            "input": prompt,
            "max_output_tokens": 3333,
        },
    )
    assert engine.max_output_tokens == 3333
    await handle_chat(
        engine,
        {
            "model": "closer-v1",
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            "max_completion_tokens": 2222,
        },
    )
    assert engine.max_output_tokens == 2222


def test_envelope_has_usage_and_output_text():
    env = responses_envelope(model="closer-v1", text="OK", usage={"input_tokens": 3, "output_tokens": 1})
    assert env["output_text"] == "OK"
    assert env["usage"]["total_tokens"] == 4
