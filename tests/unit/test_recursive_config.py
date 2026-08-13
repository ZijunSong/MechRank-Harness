from closer.config import load_config


def test_recursive_base_url_rejected(monkeypatch):
    cfg = load_config()
    monkeypatch.setenv(cfg.base_model.base_url_env, "http://127.0.0.1:8099/v1")
    from closer.errors import ConfigError

    try:
        cfg.assert_not_recursive(host="127.0.0.1", port=8099)
    except ConfigError:
        return
    raise AssertionError("recursive CLOSER_BASE_URL must be rejected")
