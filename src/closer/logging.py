"""Run logging, trace directories, and manifests."""

from __future__ import annotations

import hashlib
import logging as stdlib_logging
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import orjson

from closer import TRACE_SCHEMA_VERSION, __version__
from closer.config import CloserConfig

SCHEMA_VERSION = TRACE_SCHEMA_VERSION


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def short_uuid() -> str:
    return uuid.uuid4().hex[:12]


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def dump_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(orjson.dumps(payload, option=orjson.OPT_INDENT_2 | orjson.OPT_SORT_KEYS))


def load_json(path: Path) -> Any:
    return orjson.loads(path.read_bytes())


def append_jsonl(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("ab") as handle:
        handle.write(orjson.dumps(payload))
        handle.write(b"\n")


def setup_stdlib_logging(level: str = "INFO") -> None:
    stdlib_logging.basicConfig(
        level=getattr(stdlib_logging, level.upper(), stdlib_logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )


class RunStore:
    """Filesystem store for one CLOSER run."""

    def __init__(self, root: Path, run_id: str) -> None:
        self.root = root
        self.run_id = run_id
        self.run_dir = root / run_id
        self.episodes_dir = self.run_dir / "episodes"
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.episodes_dir.mkdir(parents=True, exist_ok=True)

    def episode_dir(self, episode_id: str) -> Path:
        path = self.episodes_dir / episode_id
        path.mkdir(parents=True, exist_ok=True)
        (path / "comparisons").mkdir(exist_ok=True)
        (path / "evidence").mkdir(exist_ok=True)
        (path / "llm_calls").mkdir(exist_ok=True)
        return path

    def write_manifest(self, payload: dict[str, Any]) -> Path:
        path = self.run_dir / "manifest.json"
        dump_json(path, payload)
        return path


def create_run_store(config: CloserConfig, run_id: str | None = None) -> RunStore:
    root = Path(config.logging.root)
    if not root.is_absolute():
        root = Path.cwd() / root
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    rid = run_id or f"{stamp}_{short_uuid()}"
    return RunStore(root=root, run_id=rid)


def git_commit() -> str | None:
    import subprocess

    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError:
        return None
    if result.returncode != 0:
        return None
    return result.stdout.strip() or None


def git_status_dirty() -> bool:
    import subprocess

    try:
        result = subprocess.run(
            ["git", "status", "--porcelain"],
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError:
        return False
    return bool(result.stdout.strip())


def build_run_manifest(
    config: CloserConfig,
    *,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    prompt_hashes = hash_prompt_files()
    payload = {
        "schema_version": SCHEMA_VERSION,
        "project": config.project.model_dump(),
        "closer_version": __version__,
        "timestamp": utc_now(),
        "config_path": config.source_path,
        "config_hash": config.config_hash(),
        "git_commit": git_commit(),
        "git_dirty": git_status_dirty(),
        "base_model_requested": config.model_id() or None,
        "base_model_env": {
            "base_url_env": config.base_model.base_url_env,
            "model_env": config.base_model.model_env,
            "api_style": config.base_model.api_style,
            "require_usage": config.base_model.require_usage,
        },
        "prompt_hashes": prompt_hashes,
        "pid": os.getpid(),
    }
    if extra:
        payload.update(extra)
    return payload


def hash_prompt_files() -> dict[str, str]:
    prompt_modules = [
        Path(__file__).resolve().parent / "assay" / "prompts.py",
        Path(__file__).resolve().parent / "evidence" / "prompts.py",
        Path(__file__).resolve().parent / "comparison" / "prompts.py",
        Path(__file__).resolve().parent / "audit" / "prompts.py",
    ]
    hashes: dict[str, str] = {}
    for path in prompt_modules:
        if path.exists():
            hashes[str(path.relative_to(Path(__file__).resolve().parent.parent))] = sha256_file(path)
    return hashes
