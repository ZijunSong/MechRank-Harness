#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
# shellcheck disable=SC1091
source "$ROOT/scripts/pgllm/pgllm_common.sh"

REGISTRY="${PGLLM_REGISTRY:-$ROOT/configs/pgllm_gpt55.json}"
MODELS="${PGLLM_MODELS:-gpt55}"
SIZES="${PGLLM_SIZES:-50}"
RUN_LABEL="${PGLLM_RUN_LABEL:-mechrank_direct_n50}"

export PGLLM_EXTRA_ARGS="${PGLLM_EXTRA_ARGS:---retry-errors --retry-truncated}"

pgllm_cd
pgllm-models --registry "$REGISTRY" --models $MODELS
exec "$ROOT/scripts/pgllm/run_pgllm.sh"
