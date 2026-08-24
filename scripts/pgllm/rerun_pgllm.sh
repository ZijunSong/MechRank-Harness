#!/usr/bin/env bash
set -euo pipefail

# Re-run cells whose saved provenance no longer matches the registry/endpoint.
# Use after changing base_url, model_id, max_tokens, or server port.
export PGLLM_EXTRA_ARGS="${PGLLM_EXTRA_ARGS:---overwrite --retry-errors --retry-truncated}"
exec "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/run_pgllm.sh" "$@"
