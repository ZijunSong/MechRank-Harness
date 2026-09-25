#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
# shellcheck disable=SC1091
source "$ROOT/scripts/pgllm/pgllm_common.sh"

MODELS="${PGLLM_MODELS:-closer-v1}"
SIZES="${PGLLM_SIZES:-50}"
SEEDS="${PGLLM_SEEDS:-}"
RUN_LABEL="${PGLLM_RUN_LABEL:-}"

pgllm_cd

args=(
  pgllm-status
  --models $MODELS
  --sizes $SIZES
)

if [[ -n "$SEEDS" ]]; then
  # shellcheck disable=SC2206
  args+=(--seeds $SEEDS)
fi

if [[ -n "$RUN_LABEL" ]]; then
  args+=(--run-label "$RUN_LABEL")
fi

mapfile -t extra < <(pgllm_extra_args)
if ((${#extra[@]})); then
  args+=("${extra[@]}")
fi

"${args[@]}"
