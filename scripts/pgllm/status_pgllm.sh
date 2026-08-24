#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
# shellcheck disable=SC1091
source "$ROOT/scripts/pgllm/pgllm_common.sh"

REGISTRY="${PGLLM_REGISTRY:-$ROOT/configs/pgllm_closer_model.json}"
MODELS="${PGLLM_MODELS:-closer-v1}"
SIZES="${PGLLM_SIZES:-50}"
RUN_LABEL="${PGLLM_RUN_LABEL:-}"
OUTPUT="${PGLLM_STATUS_OUTPUT:-$ROOT/artifacts/pgllm_status.json}"

pgllm_cd

args=(
  pgllm-status
  --registry "$REGISTRY"
  --models $MODELS
  --sizes $SIZES
  --output "$OUTPUT"
)

if [[ -n "$RUN_LABEL" ]]; then
  args+=(--run-label "$RUN_LABEL")
fi

mapfile -t extra < <(pgllm_extra_args)
if ((${#extra[@]})); then
  args+=("${extra[@]}")
fi

"${args[@]}"
echo "Wrote $OUTPUT"
