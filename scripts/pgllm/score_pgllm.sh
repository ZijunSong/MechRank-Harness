#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT/external/proteingym-llm"
pgllm-score \
  --models closer-v1 \
  --sizes 50 \
  --breakdown
