#!/usr/bin/env python3
"""Freeze CLOSER config, prompt hashes, and git identity before PG-LLM evaluation."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

from closer.config import load_config
from closer.logging import git_commit, git_status_dirty, hash_prompt_files


def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def lock_hash() -> str | None:
    for name in ("uv.lock", "poetry.lock", "requirements.lock"):
        path = Path(name)
        if path.exists():
            return file_hash(path)
    return None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    cfg = load_config(args.config)
    payload = {
        "project": cfg.project.model_dump(),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "config_path": str(Path(args.config).resolve()),
        "config_hash": cfg.config_hash(),
        "git_commit": git_commit(),
        "git_dirty": git_status_dirty(),
        "prompt_hashes": hash_prompt_files(),
        "base_model_id_env": cfg.base_model.model_env,
        "base_model_id": cfg.model_id() or None,
        "dependency_lock_hash": lock_hash(),
        "python": sys.version,
        "config": cfg.model_dump(mode="json", exclude={"source_path"}),
    }
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
