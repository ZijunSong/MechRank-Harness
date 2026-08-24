#!/usr/bin/env bash
set -euo pipefail

# Resume incomplete cells after transient API/server failures.
export PGLLM_EXTRA_ARGS="${PGLLM_EXTRA_ARGS:---retry-errors --retry-truncated}"
exec "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/run_pgllm.sh" "$@"
