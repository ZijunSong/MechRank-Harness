#!/usr/bin/env bash
# Retry P53_HUMAN_Kotler_2018 seed 2 with maximum completion budget under ctx=262144.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
VENV_BIN="${ROOT}/.venv/bin"
export PATH="${VENV_BIN}:${PATH}"

export PGLLM_DATA_ROOT="${PGLLM_DATA_ROOT:-/data/ppnm/data/proteingym-llm/extracted}"
export PGLLM_RESULTS_ROOT="${PGLLM_RESULTS_ROOT:-$ROOT/artifacts/pgllm_results}"
export PGLLM_WORK_ROOT="${PGLLM_WORK_ROOT:-$ROOT/external/proteingym-llm}"
export PGLLM_OPENAI_BASE_URL="${PGLLM_OPENAI_BASE_URL:-http://127.0.0.1:8044/v1}"
export PGLLM_OPENAI_API_KEY="${PGLLM_OPENAI_API_KEY:-EMPTY}"
export PGLLM_REGISTRY="${PGLLM_REGISTRY:-$ROOT/configs/pgllm_qwen38_27b_vllm_think.json}"

RUN_LABEL="mechrank_qwen38_27b_think_p53_b2"
MAIN_RUN_LABEL="mechrank_qwen38_27b_think_n50"
MAX_OUTPUT_TOKENS=241000
LOG_DIR="${ROOT}/logs/pgllm_qwen38_27b_gpu4_think"
mkdir -p "$LOG_DIR"
LOG="${LOG_DIR}/retry_p53_kotler_b2.log"
RESULTS="${PGLLM_RESULTS_ROOT:-$ROOT/artifacts/pgllm_results}"

python3 - <<'PY'
import json
from pathlib import Path
path = Path("/data/ppnm/MechRank-Harness/artifacts/pgllm_results/_runs/mechrank_qwen38_27b_think_n50/_run.json")
manifest = json.loads(path.read_text())
key = "qwen38-27b/n50"
manifest["conditions"][key]["max_output_tokens"] = 86016
manifest["conditions"][key]["request_descriptor"]["max_output_tokens"] = 86016
path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
PY

echo "[$(date '+%F %T')] Retrying P53_HUMAN_Kotler_2018 b2 with max_output_tokens=${MAX_OUTPUT_TOKENS}" | tee "$LOG"
pgllm-run \
  --registry "$PGLLM_REGISTRY" \
  --models qwen38-27b \
  --sizes 50 \
  --seeds 2 \
  --assays P53_HUMAN_Kotler_2018 \
  --run-label "$RUN_LABEL" \
  --max-output-tokens "$MAX_OUTPUT_TOKENS" \
  --timeout 7200 \
  --retries 5 \
  --concurrency 1 \
  --retry-errors \
  --retry-truncated \
  2>&1 | tee -a "$LOG"

SRC="${RESULTS}/_runs/${RUN_LABEL}/qwen38-27b/n50/b2/P53_HUMAN_Kotler_2018.json"
DST="${RESULTS}/_runs/${MAIN_RUN_LABEL}/qwen38-27b/n50/b2/P53_HUMAN_Kotler_2018.json"
if [[ -f "$SRC" ]]; then
  python3 - <<PY
import json
from pathlib import Path
src = Path("$SRC")
dst = Path("$DST")
record = json.loads(src.read_text())
if record.get("parsed") and not record.get("error") and record.get("spearman") is not None:
    record["run_label"] = "$MAIN_RUN_LABEL"
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(json.dumps(record, indent=2, ensure_ascii=False) + "\n")
    print(f"merged result -> {dst}")
else:
    raise SystemExit(f"retry did not succeed: error={record.get('error')!r} parsed={record.get('parsed')}")
PY
fi
