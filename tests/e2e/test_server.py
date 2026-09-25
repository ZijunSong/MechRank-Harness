"""Server E2E. Probe does not need an inner LLM. Ranking path requires a real endpoint."""

from __future__ import annotations

import os

import pytest
from fastapi.testclient import TestClient
from tests.fixtures.episodes import ranking_v1_prompt

from closer.benchmark.pgllm_renderer import SYSTEM_PROMPT, ranking_json_line
from closer.server.app import create_app

skip_full_e2e = pytest.mark.skipif(
    os.environ.get("CLOSER_RUN_E2E") != "1"
    or not all(os.environ.get(name) for name in ("CLOSER_BASE_URL", "CLOSER_API_KEY", "CLOSER_MODEL")),
    reason="set CLOSER_RUN_E2E=1 and inner LLM env to run full ranking e2e",
)


@pytest.fixture
def client():
    with TestClient(create_app()) as test_client:
        yield test_client


def test_health_and_probe(client):
    health = client.get("/health")
    assert health.status_code == 200
    probe = client.post(
        "/v1/responses",
        json={
            "model": "closer-v1",
            "instructions": "You are a connectivity test.",
            "input": "Reply with the single word: OK",
        },
    )
    assert probe.status_code == 200
    body = probe.json()
    assert body["output_text"] == "OK"
    assert body["model"] == "closer-v1"
    assert isinstance(body["usage"]["output_tokens"], int)


def test_chat_probe(client):
    probe = client.post(
        "/v1/chat/completions",
        json={
            "model": "closer-v1",
            "messages": [
                {"role": "system", "content": "You are a connectivity test."},
                {"role": "user", "content": "Reply with the single word: OK"},
            ],
        },
    )
    assert probe.status_code == 200
    assert probe.json()["choices"][0]["message"]["content"] == "OK"


def test_bad_prompt_is_400(client):
    resp = client.post(
        "/v1/responses",
        json={"model": "closer-v1", "instructions": "rank", "input": "not a pgllm prompt"},
    )
    assert resp.status_code == 400


def test_full_ranking_requests_pass_token_limits(monkeypatch):
    """Offline Chat/Responses ranking path: no real model, not just the OK probe."""
    from tests.fixtures.episodes import synthetic_episode

    from closer.assay.interpreter import deterministic_ablation_contract
    from closer.schemas.ranking import CloserResult

    captured: dict[str, object] = {}

    async def fake_run(self, episode, *, max_output_tokens=None):
        captured["max_output_tokens"] = max_output_tokens
        captured["n_variants"] = len(episode.variants)
        ranking = list(episode.variant_ids())
        return CloserResult(
            ranking=ranking,
            scores={vid: float(len(ranking) - i) for i, vid in enumerate(ranking)},
            assay_contract=deterministic_ablation_contract(episode),
            diagnostics={},
            usage={"input_tokens": 11, "output_tokens": 7, "total_tokens": 18},
            trace_path="",
        )

    monkeypatch.setattr("closer.server.app.LazyEngine.run_episode", fake_run)
    prompt = ranking_v1_prompt(4)
    with TestClient(create_app()) as test_client:
        responses = test_client.post(
            "/v1/responses",
            json={
                "model": "closer-v1",
                "instructions": SYSTEM_PROMPT,
                "input": prompt,
                "max_output_tokens": 4096,
            },
        )
        assert responses.status_code == 200, responses.text
        assert captured["max_output_tokens"] == 4096
        assert captured["n_variants"] == 4
        chat = test_client.post(
            "/v1/chat/completions",
            json={
                "model": "closer-v1",
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                "max_tokens": 2048,
            },
        )
        assert chat.status_code == 200, chat.text
        assert captured["max_output_tokens"] == 2048
        assert "ranking" in chat.json()["choices"][0]["message"]["content"]
    assert len(synthetic_episode(4).variants) == 4


@skip_full_e2e
def test_synthetic_ranking_v1_e2e(client):
    prompt = ranking_v1_prompt(10)
    resp = client.post(
        "/v1/responses",
        json={
            "model": "closer-v1",
            "instructions": SYSTEM_PROMPT,
            "input": prompt,
            "max_output_tokens": 4096,
        },
    )
    assert resp.status_code == 200, resp.text
    text = resp.json()["output_text"].strip().splitlines()[-1]
    expected_ids = [f"M{i:02d}" for i in range(1, 11)]
    import json

    payload = json.loads(text)
    assert set(payload["ranking"]) == set(expected_ids)
    assert len(payload["ranking"]) == 10
    usage = resp.json()["usage"]
    assert (usage["input_tokens"] or 0) + (usage["output_tokens"] or 0) > 0
    assert ranking_json_line(payload["ranking"]) == text or text.endswith("}")
