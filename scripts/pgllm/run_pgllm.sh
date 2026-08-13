#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
export PGLLM_CLOSER_BASE_URL="${PGLLM_CLOSER_BASE_URL:-http://127.0.0.1:8099/v1}"
export PGLLM_CLOSER_API_KEY="${PGLLM_CLOSER_API_KEY:-dummy-local-key}"
cd "$ROOT/external/proteingym-llm"
pgllm-run \
  --registry "$ROOT/configs/pgllm_closer_model.json" \
  --models closer-v1 \
  --sizes 50
