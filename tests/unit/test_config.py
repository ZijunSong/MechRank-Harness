from pathlib import Path

import pytest

from closer.config import PACKAGE_ROOT, load_config


def test_default_config_loads_and_hashes():
    cfg = load_config()
    assert cfg.project.name == "MechRank-Harness"
    assert cfg.ranking.method == "bradley_terry"
    assert cfg.budget.hard_stop_on_exceed is True
    assert cfg.execution.mode == "legacy_graph"
    digest = cfg.config_hash()
    assert len(digest) == 64
    assert cfg.config_hash() == digest


def test_v2_pilot_configs_load():
    minimum = load_config(PACKAGE_ROOT / "configs" / "closer.v2_min.pilot.yaml")
    compact = load_config(PACKAGE_ROOT / "configs" / "closer.v2_compact.pilot.yaml")
    assert minimum.execution.mode == "listwise_min"
    assert minimum.ablation.mutation_compiler is True
    assert minimum.ablation.comparative_reasoning is False
    assert minimum.budget.max_total_llm_calls == 1
    assert compact.execution.mode == "listwise_compact"
    assert compact.server.default_model_id == "closer-v2-compact"


def test_unknown_execution_field_is_rejected():
    from pydantic import ValidationError

    from closer.config import ExecutionConfig

    with pytest.raises(ValidationError):
        ExecutionConfig(mode="legacy_graph", ignored_knob=1)  # type: ignore[arg-type]


def test_incompatible_ablation_is_rejected(tmp_path: Path):
    from closer.errors import ConfigError

    path = tmp_path / "bad.yaml"
    path.write_text(
        """
project: {name: x, version: "0"}
ablation:
  comparative_reasoning: true
  global_solver: false
  adaptive_audit: true
audit:
  enabled: true
execution:
  mode: legacy_graph
""",
        encoding="utf-8",
    )
    from pydantic import ValidationError

    with pytest.raises((ConfigError, ValidationError, ValueError)):
        load_config(path)
