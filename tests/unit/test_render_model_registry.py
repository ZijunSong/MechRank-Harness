import json
import subprocess
import sys
from pathlib import Path


def test_render_model_registry_from_env(tmp_path, monkeypatch):
    root = Path(__file__).resolve().parents[2]
    script = root / "scripts/pgllm/render_model_registry.py"
    monkeypatch.setenv("PGLLM_MODEL_ID", "gpt-5.5")
    monkeypatch.setenv("PGLLM_MODEL_ALIAS", "gpt55")
    monkeypatch.setenv("PGLLM_OPENAI_BASE_URL", "https://example.test/v1")
    monkeypatch.setenv("PGLLM_OPENAI_API_KEY", "secret")
    proc = subprocess.run(
        [sys.executable, str(script)],
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(proc.stdout)
    model = payload["models"]["gpt55"]
    assert model["model_id"] == "gpt-5.5"
    assert model["base_url_env"] == "PGLLM_OPENAI_BASE_URL"
    assert model["api_key_env"] == "PGLLM_OPENAI_API_KEY"
