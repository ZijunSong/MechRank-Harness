#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
# shellcheck disable=SC1091
source "$ROOT/scripts/pgllm/pgllm_common.sh"

REGISTRY="${PGLLM_REGISTRY:-$ROOT/configs/pgllm_closer_model.json}"
MODELS="${PGLLM_MODELS:-closer-v1}"
SIZES="${PGLLM_SIZES:-50}"
SEEDS="${PGLLM_SEEDS:-}"
RUN_LABEL="${PGLLM_RUN_LABEL:-}"
TIMEOUT="${PGLLM_TIMEOUT:-900}"
RETRIES="${PGLLM_RETRIES:-5}"
CONCURRENCY="${PGLLM_CONCURRENCY:-1}"
assay_args=()

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

if [[ -n "$SEEDS" ]]; then
  # shellcheck disable=SC2206
  args+=(--seeds $SEEDS)
fi

if [[ -n "$RUN_LABEL" ]]; then
  args+=(--run-label "$RUN_LABEL")
fi

if [[ -n "${PGLLM_ASSAYS_FILE:-}" ]]; then
  if [[ ! -f "$PGLLM_ASSAYS_FILE" ]]; then
    echo "assay list not found: $PGLLM_ASSAYS_FILE" >&2
    exit 1
  fi
  mapfile -t assay_list < "$PGLLM_ASSAYS_FILE"
  assay_args=()
  for assay in "${assay_list[@]}"; do
    assay="${assay//$'\r'/}"
    if [[ -n "$assay" ]]; then
      assay_args+=("$assay")
    fi
  done
  if ((${#assay_args[@]} == 0)); then
    echo "assay list is empty: $PGLLM_ASSAYS_FILE" >&2
    exit 1
  fi
  args+=(--assays "${assay_args[@]}")
fi

mapfile -t extra < <(pgllm_extra_args)
if ((${#extra[@]})); then
  args+=("${extra[@]}")
fi

echo "registry=$REGISTRY models=$MODELS sizes=$SIZES seeds=${SEEDS:-<all>} assays=${#assay_args[@]} run_label=${RUN_LABEL:-<default>} extra=${PGLLM_EXTRA_ARGS:-}" >&2
"${args[@]}"
