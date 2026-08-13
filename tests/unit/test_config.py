from closer.config import load_config


def test_default_config_loads_and_hashes():
    cfg = load_config()
    assert cfg.project.name == "MechRank-Harness"
    assert cfg.ranking.method == "bradley_terry"
    assert cfg.budget.hard_stop_on_exceed is True
    digest = cfg.config_hash()
    assert len(digest) == 64
    assert cfg.config_hash() == digest
