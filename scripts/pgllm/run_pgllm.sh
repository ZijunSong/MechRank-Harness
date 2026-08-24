#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
# shellcheck disable=SC1091
source "$ROOT/scripts/pgllm/pgllm_common.sh"

REGISTRY="${PGLLM_REGISTRY:-$ROOT/configs/pgllm_closer_model.json}"
MODELS="${PGLLM_MODELS:-closer-v1}"
SIZES="${PGLLM_SIZES:-50}"
RUN_LABEL="${PGLLM_RUN_LABEL:-}"
TIMEOUT="${PGLLM_TIMEOUT:-900}"
RETRIES="${PGLLM_RETRIES:-5}"
CONCURRENCY="${PGLLM_CONCURRENCY:-1}"

pgllm_cd

args=(
  pgllm-run
  --registry "$REGISTRY"
  --models $MODELS
  --sizes $SIZES
  --timeout "$TIMEOUT"
  --retries "$RETRIES"
  --concurrency "$CONCURRENCY"
)

if [[ -n "$RUN_LABEL" ]]; then
  args+=(--run-label "$RUN_LABEL")
fi

mapfile -t extra < <(pgllm_extra_args)
if ((${#extra[@]})); then
  args+=("${extra[@]}")
fi

echo "registry=$REGISTRY models=$MODELS sizes=$SIZES run_label=${RUN_LABEL:-<default>} extra=${PGLLM_EXTRA_ARGS:-}" >&2
"${args[@]}"
