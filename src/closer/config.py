"""YAML configuration. Thresholds live here, not in scattered constants."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Literal
from urllib.parse import urlsplit

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from closer.errors import ConfigError

PACKAGE_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = PACKAGE_ROOT / "configs" / "closer.default.yaml"


class ProjectConfig(BaseModel):
    name: str = "MechRank-Harness"
    version: str = "0.1.0"


class BaseModelConfig(BaseModel):
    api_style: Literal["responses", "chat"] = "responses"
    base_url_env: str = "CLOSER_BASE_URL"
    api_key_env: str = "CLOSER_API_KEY"
    model_env: str = "CLOSER_MODEL"
    timeout_s: float = 180
    max_retries: int = 3
    temperature: float | None = None
    reasoning_effort: str = "high"
    require_usage: bool = False
    chat_extra_body: dict[str, Any] = Field(default_factory=dict)


class ServerConfig(BaseModel):
    model_id_env: str = "CLOSER_SERVER_MODEL_ID"
    host_env: str = "CLOSER_SERVER_HOST"
    port_env: str = "CLOSER_SERVER_PORT"
    api_key_env: str = "CLOSER_SERVER_API_KEY"
    default_model_id: str = "closer-v1"
    default_host: str = "127.0.0.1"
    default_port: int = 8099
    allow_dummy_api_key: bool = True


class StructuredOutputConfig(BaseModel):
    max_parse_repairs: int = 2
    native_json_schema: bool = True
    json_only_fallback: bool = True


class AssayConfig(BaseModel):
    enabled: bool = True


class EvidenceConfig(BaseModel):
    enabled: bool = True
    parallelism: int = 8
    max_variants_per_call: int = 5
    local_context_window: int = 10


class ComparisonConfig(BaseModel):
    group_size: int = 5
    overlap: int = 2
    pair_batch_size: int = 8
    initial_rounds: int = 2


class RankingConfig(BaseModel):
    method: Literal["bradley_terry"] = "bradley_terry"
    l2_regularization: float = 0.001
    confidence_clamp_min: float = 0.55
    confidence_clamp_max: float = 0.95
    weight_mode: Literal["neglog", "confidence"] = "neglog"
    tie_mode: Literal["ignore", "weak"] = "ignore"
    max_optimize_iter: int = 500


class AuditConfig(BaseModel):
    enabled: bool = True
    max_rounds: int = 3
    max_items_per_round: int = 12
    cycle_priority: float = 1.0
    low_margin_priority: float = 1.0
    evidence_conflict_priority: float = 1.0
    epistasis_priority: float = 1.0
    low_margin_threshold: float = 0.15
    min_priority: float = 0.25
    evidence_conflict_min_confidence: float = 0.75

    @field_validator("low_margin_threshold")
    @classmethod
    def _threshold_positive(cls, value: float) -> float:
        if value <= 0:
            raise ValueError("audit.low_margin_threshold must be > 0")
        return value

    @field_validator("min_priority")
    @classmethod
    def _min_priority_bounded(cls, value: float) -> float:
        if value < 0 or value > 1:
            raise ValueError("audit.min_priority must be in [0, 1]")
        return value


class BudgetConfig(BaseModel):
    max_total_llm_calls: int = 80
    max_total_input_tokens: int = 250000
    max_total_output_tokens: int = 80000
    hard_stop_on_exceed: bool = True


class LoggingConfig(BaseModel):
    root: str = "artifacts/runs"
    save_raw_provider_payloads: bool = True


class AblationConfig(BaseModel):
    mutation_compiler: bool = True
    assay_contract: bool = True
    evidence_state: bool = True
    comparative_reasoning: bool = True
    global_solver: bool = True
    adaptive_audit: bool = True


class ExecutionConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    mode: Literal[
        "direct_proxy",
        "listwise_min",
        "listwise_compact",
        "listwise_refine",
        "legacy_graph",
    ] = "legacy_graph"


class RefinementConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    enabled: bool = False
    max_batches: int = 1
    max_pairs_per_batch: int = 8
    max_repeats_per_pair: int = 0


class CloserConfig(BaseModel):
    project: ProjectConfig = Field(default_factory=ProjectConfig)
    base_model: BaseModelConfig = Field(default_factory=BaseModelConfig)
    server: ServerConfig = Field(default_factory=ServerConfig)
    structured_output: StructuredOutputConfig = Field(default_factory=StructuredOutputConfig)
    assay: AssayConfig = Field(default_factory=AssayConfig)
    evidence: EvidenceConfig = Field(default_factory=EvidenceConfig)
    comparison: ComparisonConfig = Field(default_factory=ComparisonConfig)
    ranking: RankingConfig = Field(default_factory=RankingConfig)
    audit: AuditConfig = Field(default_factory=AuditConfig)
    budget: BudgetConfig = Field(default_factory=BudgetConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    ablation: AblationConfig = Field(default_factory=AblationConfig)
    execution: ExecutionConfig = Field(default_factory=ExecutionConfig)
    refinement: RefinementConfig = Field(default_factory=RefinementConfig)
    source_path: str | None = None

    @model_validator(mode="after")
    def _validate_ablation_combinations(self) -> CloserConfig:
        if (
            self.ablation.adaptive_audit
            and self.audit.enabled
            and not self.ablation.global_solver
            and self.execution.mode == "legacy_graph"
        ):
            raise ValueError(
                "adaptive_audit requires global_solver; otherwise the auditor "
                "would silently switch from Borda to Bradley-Terry"
            )
        if self.execution.mode == "listwise_refine" and not self.refinement.enabled:
            raise ValueError("execution.mode=listwise_refine requires refinement.enabled=true")
        return self

    @field_validator("evidence")
    @classmethod
    def _validate_evidence_batch(cls, value: EvidenceConfig) -> EvidenceConfig:
        if value.max_variants_per_call < 1 or value.max_variants_per_call > 5:
            raise ValueError("evidence.max_variants_per_call must be in [1, 5]")
        return value

    def config_hash(self) -> str:
        payload = self.model_dump(mode="json", exclude={"source_path"})
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        return hashlib.sha256(encoded).hexdigest()

    def base_url(self) -> str:
        return os.environ.get(self.base_model.base_url_env, "").rstrip("/")

    def api_key(self) -> str:
        return os.environ.get(self.base_model.api_key_env, "")

    def model_id(self) -> str:
        return os.environ.get(self.base_model.model_env, "")

    def server_model_id(self) -> str:
        return os.environ.get(self.server.model_id_env, self.server.default_model_id)

    def server_host(self) -> str:
        return os.environ.get(self.server.host_env, self.server.default_host)

    def server_port(self) -> int:
        raw = os.environ.get(self.server.port_env)
        return int(raw) if raw else self.server.default_port

    def server_api_key(self) -> str:
        return os.environ.get(self.server.api_key_env, "dummy-local-key")

    def require_base_model(self) -> None:
        if not self.base_url():
            raise ConfigError(f"{self.base_model.base_url_env} is not set")
        if not self.api_key():
            raise ConfigError(f"{self.base_model.api_key_env} is not set")
        if not self.model_id():
            raise ConfigError(f"{self.base_model.model_env} is not set")

    def assert_not_recursive(self, host: str | None = None, port: int | None = None) -> None:
        base = self.base_url()
        if not base:
            return
        parsed = urlsplit(base)
        host = host or self.server_host()
        port = port if port is not None else self.server_port()
        netloc = parsed.netloc.lower()
        forbidden = {
            f"{host}:{port}".lower(),
            f"127.0.0.1:{port}",
            f"localhost:{port}",
            f"0.0.0.0:{port}",
        }
        if netloc in forbidden:
            raise ConfigError(
                f"{self.base_model.base_url_env}={base} points at the CLOSER server "
                f"{host}:{port}. Inner base LLM and outer CLOSER endpoint must differ."
            )


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    out = dict(base)
    for key, value in override.items():
        if key in out and isinstance(out[key], dict) and isinstance(value, dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = value
    return out


def load_config(path: str | Path | None = None) -> CloserConfig:
    config_path = Path(path) if path else DEFAULT_CONFIG_PATH
    if not config_path.exists():
        raise ConfigError(f"config file not found: {config_path}")
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        raise ConfigError("config file must be a YAML object")
    cfg = CloserConfig.model_validate(raw)
    cfg.source_path = str(config_path.resolve())
    return cfg
