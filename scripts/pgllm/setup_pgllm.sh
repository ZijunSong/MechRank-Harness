#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
mkdir -p "$ROOT/external"
if [[ ! -d "$ROOT/external/proteingym-llm/.git" ]]; then
  git clone https://github.com/rohitarorayyc/proteingym-llm.git "$ROOT/external/proteingym-llm"
fi
cd "$ROOT/external/proteingym-llm"
PATCH="$ROOT/patches/proteingym-llm/opensource-thinking.patch"
if [[ -f "$PATCH" ]] && ! git apply --reverse --check "$PATCH" >/dev/null 2>&1; then
  git apply "$PATCH"
fi
python -m pip install -e .
echo "ProteinGym-LLM installed. Run pgllm-data before official evaluation."
