#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PGLLM_ROOT="${PGLLM_ROOT:-$ROOT/external/proteingym-llm}"

if [[ -f "$ROOT/.env" ]]; then
  set -a
  # shellcheck disable=SC1091
  source "$ROOT/.env"
  set +a
fi

export PGLLM_CLOSER_BASE_URL="${PGLLM_CLOSER_BASE_URL:-http://127.0.0.1:${CLOSER_SERVER_PORT:-8099}/v1}"
export PGLLM_CLOSER_API_KEY="${PGLLM_CLOSER_API_KEY:-${CLOSER_SERVER_API_KEY:-dummy-local-key}}"

pgllm_require() {
  if [[ ! -d "$PGLLM_ROOT" ]]; then
    echo "ProteinGym-LLM not found at $PGLLM_ROOT. Run: bash scripts/pgllm/setup_pgllm.sh" >&2
    exit 1
  fi
}

pgllm_cd() {
  pgllm_require
  cd "$PGLLM_ROOT"
}

pgllm_extra_args() {
  if [[ -n "${PGLLM_EXTRA_ARGS:-}" ]]; then
    # shellcheck disable=SC2206
    local extra=($PGLLM_EXTRA_ARGS)
    printf '%s\n' "${extra[@]}"
  fi
}
