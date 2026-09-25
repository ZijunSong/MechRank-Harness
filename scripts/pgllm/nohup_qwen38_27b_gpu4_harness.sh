#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
VENV_BIN="${ROOT}/.venv/bin"
export PATH="${VENV_BIN}:${PATH}"

LOG_DIR="${ROOT}/logs/pgllm_qwen38_27b_gpu4_harness"
CLOSER_CONFIG="${CLOSER_CONFIG:-$ROOT/configs/closer.qwen38.harness.yaml}"
VLLM_PORT="${VLLM_PORT:-8044}"
CLOSER_PORT="${CLOSER_PORT:-8099}"
MODEL_PATH="${MODEL_PATH:-/data/ppnm/models/Qwen3.8-27B}"
SERVED_NAME="${SERVED_NAME:-Qwen/Qwen3.8-27B}"
VLLM_ENV="${VLLM_ENV:-/data/ppnm/miniconda3/envs/bishop}"
CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-12.4}"
VLLM_GPU="${VLLM_GPU:-4}"

mkdir -p "$LOG_DIR"

export CLOSER_BASE_URL="http://127.0.0.1:${VLLM_PORT}/v1"
export CLOSER_API_KEY="${CLOSER_API_KEY:-EMPTY}"
export CLOSER_MODEL="${SERVED_NAME}"
export CLOSER_SERVER_MODEL_ID="${CLOSER_SERVER_MODEL_ID:-closer-v1}"
export CLOSER_SERVER_HOST="${CLOSER_SERVER_HOST:-127.0.0.1}"
export CLOSER_SERVER_PORT="${CLOSER_PORT}"
export CLOSER_SERVER_API_KEY="${CLOSER_SERVER_API_KEY:-dummy-local-key}"
export PGLLM_CLOSER_BASE_URL="http://127.0.0.1:${CLOSER_PORT}/v1"
export PGLLM_CLOSER_API_KEY="${PGLLM_CLOSER_API_KEY:-dummy-local-key}"

export PGLLM_DATA_ROOT="${PGLLM_DATA_ROOT:-/data/ppnm/data/proteingym-llm/extracted}"
export PGLLM_RESULTS_ROOT="${PGLLM_RESULTS_ROOT:-$ROOT/artifacts/pgllm_results}"
export PGLLM_WORK_ROOT="${PGLLM_WORK_ROOT:-$ROOT/external/proteingym-llm}"
export PGLLM_REGISTRY="$ROOT/configs/pgllm_closer_model.json"
export PGLLM_MODELS="${PGLLM_MODELS:-closer-v1}"
export PGLLM_SIZES="${PGLLM_SIZES:-50}"
export PGLLM_RUN_LABEL="${PGLLM_RUN_LABEL:-mechrank_closer_qwen38_n50}"
export PGLLM_CONCURRENCY="${PGLLM_CONCURRENCY:-1}"
export PGLLM_TIMEOUT="${PGLLM_TIMEOUT:-14400}"
export PGLLM_RETRIES="${PGLLM_RETRIES:-5}"
export PGLLM_EXTRA_ARGS="${PGLLM_EXTRA_ARGS:---retry-errors --retry-truncated}"

CLOSER_LOG="${LOG_DIR}/closer_server.log"
PGLLM_LOG="${LOG_DIR}/pgllm_${PGLLM_RUN_LABEL}.log"
VLLM_LOG="${LOG_DIR}/vllm_gpu${VLLM_GPU}.log"
FROZEN_MANIFEST="${ROOT}/artifacts/frozen/closer_v1_qwen38_harness_manifest.json"

wait_for_url() {
  local url="$1"
  local label="$2"
  local max_tries="${3:-60}"
  local ok=0
  for _ in $(seq 1 "$max_tries"); do
    if curl -sf "$url" >/dev/null 2>&1; then
      ok=1
      break
    fi
    sleep 5
  done
  if [[ "$ok" -ne 1 ]]; then
    echo "${label} did not become ready; see logs under ${LOG_DIR}" >&2
    return 1
  fi
}

vllm_ready=0
if curl -sf "http://127.0.0.1:${VLLM_PORT}/health" >/dev/null 2>&1 \
  || curl -sf "http://127.0.0.1:${VLLM_PORT}/v1/models" >/dev/null 2>&1; then
  vllm_ready=1
  echo "[$(date '+%F %T')] Reusing existing vLLM on port ${VLLM_PORT}" | tee -a "$PGLLM_LOG"
else
  echo "[$(date '+%F %T')] vLLM not ready on port ${VLLM_PORT}; start it first." | tee -a "$PGLLM_LOG"
  exit 1
fi

if ss -tln | grep -q ":${CLOSER_PORT} "; then
  old_pid=""
  if [[ -f "${LOG_DIR}/closer.pid" ]]; then
    old_pid="$(cat "${LOG_DIR}/closer.pid" 2>/dev/null || true)"
  fi
  if [[ -n "${old_pid}" ]] && kill -0 "${old_pid}" 2>/dev/null; then
    echo "Stopping previous closer-server PID ${old_pid}" | tee -a "$CLOSER_LOG"
    kill "${old_pid}" 2>/dev/null || true
    sleep 3
  else
    pkill -f "closer-server.*--port ${CLOSER_PORT}" 2>/dev/null || true
    sleep 2
  fi
fi

pkill -f "pgllm-run.*${PGLLM_RUN_LABEL}" 2>/dev/null || true
sleep 2

echo "[$(date '+%F %T')] Starting closer-server (full harness) on port ${CLOSER_PORT}" | tee -a "$CLOSER_LOG"
cd "$ROOT"
nohup closer-server \
  --config "$CLOSER_CONFIG" \
  --host "${CLOSER_SERVER_HOST}" \
  --port "${CLOSER_PORT}" \
  >> "$CLOSER_LOG" 2>&1 &
echo $! > "${LOG_DIR}/closer.pid"
echo "closer-server PID=$(cat "${LOG_DIR}/closer.pid")" | tee -a "$CLOSER_LOG"

wait_for_url "http://127.0.0.1:${CLOSER_PORT}/health" "closer-server" 60

echo "[$(date '+%F %T')] Freezing harness config" | tee -a "$PGLLM_LOG"
python "$ROOT/scripts/freeze_config.py" \
  --config "$CLOSER_CONFIG" \
  --output "$FROZEN_MANIFEST" >> "$PGLLM_LOG" 2>&1

echo "[$(date '+%F %T')] Launching PG-LLM full harness eval (run_label=${PGLLM_RUN_LABEL})" | tee -a "$PGLLM_LOG"
nohup bash "$ROOT/scripts/pgllm/run_pgllm.sh" >> "$PGLLM_LOG" 2>&1 &
echo $! > "${LOG_DIR}/pgllm.pid"
echo "PG-LLM PID=$(cat "${LOG_DIR}/pgllm.pid")" | tee -a "$PGLLM_LOG"
echo "Logs: ${CLOSER_LOG} , ${PGLLM_LOG}"
echo "Monitor: PGLLM_RUN_LABEL=${PGLLM_RUN_LABEL} bash $ROOT/scripts/pgllm/status_pgllm.sh"
