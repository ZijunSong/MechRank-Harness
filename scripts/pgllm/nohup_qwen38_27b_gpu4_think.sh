#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
VENV_BIN="${ROOT}/.venv/bin"
export PATH="${VENV_BIN}:${PATH}"
LOG_DIR="${ROOT}/logs/pgllm_qwen38_27b_gpu4_think"
VLLM_PORT="${VLLM_PORT:-8044}"
VLLM_GPU="${VLLM_GPU:-4}"
MODEL_PATH="${MODEL_PATH:-/data/ppnm/models/Qwen3.8-27B}"
SERVED_NAME="${SERVED_NAME:-Qwen/Qwen3.8-27B}"
VLLM_ENV="${VLLM_ENV:-/data/ppnm/miniconda3/envs/bishop}"
CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-12.4}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-262144}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.82}"

mkdir -p "$LOG_DIR"

export PGLLM_DATA_ROOT="${PGLLM_DATA_ROOT:-/data/ppnm/data/proteingym-llm/extracted}"
export PGLLM_RESULTS_ROOT="${PGLLM_RESULTS_ROOT:-$ROOT/artifacts/pgllm_results}"
export PGLLM_WORK_ROOT="${PGLLM_WORK_ROOT:-$ROOT/external/proteingym-llm}"
export PGLLM_OPENAI_BASE_URL="http://127.0.0.1:${VLLM_PORT}/v1"
export PGLLM_OPENAI_API_KEY="${PGLLM_OPENAI_API_KEY:-EMPTY}"
export PGLLM_REGISTRY="$ROOT/configs/pgllm_qwen38_27b_vllm_think.json"
export PGLLM_MODELS="${PGLLM_MODELS:-qwen38-27b}"
export PGLLM_SIZES="${PGLLM_SIZES:-50}"
export PGLLM_RUN_LABEL="mechrank_qwen38_27b_think_n50"
export PGLLM_CONCURRENCY="${PGLLM_CONCURRENCY:-1}"
export PGLLM_TIMEOUT="7200"
export PGLLM_RETRIES="${PGLLM_RETRIES:-5}"
export PGLLM_EXTRA_ARGS="${PGLLM_EXTRA_ARGS:---retry-errors --retry-truncated}"

VLLM_LOG="${LOG_DIR}/vllm_gpu${VLLM_GPU}.log"
PGLLM_LOG="${LOG_DIR}/pgllm_${PGLLM_RUN_LABEL}.log"

if ss -tln | grep -q ":${VLLM_PORT} "; then
  old_pid=""
  if [[ -f "${LOG_DIR}/vllm.pid" ]]; then
    old_pid="$(cat "${LOG_DIR}/vllm.pid" 2>/dev/null || true)"
  fi
  if [[ -n "${old_pid}" ]] && kill -0 "${old_pid}" 2>/dev/null; then
    echo "Stopping previous vLLM PID ${old_pid} on port ${VLLM_PORT}" | tee -a "$VLLM_LOG"
    kill "${old_pid}" 2>/dev/null || true
    sleep 5
    kill -9 "${old_pid}" 2>/dev/null || true
  else
    pkill -f "vllm serve ${MODEL_PATH}.*--port ${VLLM_PORT}" 2>/dev/null || true
    sleep 3
  fi
fi

pkill -f "pgllm-run.*mechrank_qwen38_27b" 2>/dev/null || true
sleep 2

if ss -tln | grep -q ":${VLLM_PORT} "; then
  echo "Port ${VLLM_PORT} still in use; abort." | tee -a "$VLLM_LOG"
  exit 1
fi

echo "[$(date '+%F %T')] Starting vLLM (thinking) on GPU${VLLM_GPU}, port ${VLLM_PORT}" | tee -a "$VLLM_LOG"
nohup env CUDA_VISIBLE_DEVICES="${VLLM_GPU}" \
    CUDA_HOME="${CUDA_HOME}" \
    PATH="${CUDA_HOME}/bin:${VLLM_ENV}/bin:${PATH}" \
    LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH:-}" \
    "${VLLM_ENV}/bin/vllm" serve "${MODEL_PATH}" \
    --host 0.0.0.0 \
    --port "${VLLM_PORT}" \
    --tensor-parallel-size 1 \
    --served-model-name "${SERVED_NAME}" \
    --reasoning-parser qwen3 \
    --enable-auto-tool-choice \
    --tool-call-parser qwen3_coder \
    --mm-encoder-tp-mode data \
    --max-model-len "${MAX_MODEL_LEN}" \
    --language-model-only \
    --gpu-memory-utilization "${GPU_MEM_UTIL}" \
    >> "$VLLM_LOG" 2>&1 &
echo $! > "${LOG_DIR}/vllm.pid"
echo "vLLM PID=$(cat "${LOG_DIR}/vllm.pid")" | tee -a "$VLLM_LOG"

echo "[$(date '+%F %T')] Waiting for vLLM health on :${VLLM_PORT}" | tee -a "$PGLLM_LOG"
ready=0
for _ in $(seq 1 180); do
  if curl -sf "http://127.0.0.1:${VLLM_PORT}/health" >/dev/null 2>&1 \
    || curl -sf "http://127.0.0.1:${VLLM_PORT}/v1/models" >/dev/null 2>&1; then
    ready=1
    break
  fi
  sleep 10
done
if [[ "$ready" -ne 1 ]]; then
  echo "vLLM did not become ready within 30 minutes; see ${VLLM_LOG}" | tee -a "$PGLLM_LOG"
  exit 1
fi

echo "[$(date '+%F %T')] vLLM ready; launching PG-LLM thinking eval (store_reasoning=false)" | tee -a "$PGLLM_LOG"
cd "$ROOT"
nohup bash "$ROOT/scripts/pgllm/run_openai_model.sh" >> "$PGLLM_LOG" 2>&1 &
echo $! > "${LOG_DIR}/pgllm.pid"
echo "PG-LLM PID=$(cat "${LOG_DIR}/pgllm.pid")" | tee -a "$PGLLM_LOG"
echo "Logs: ${VLLM_LOG} , ${PGLLM_LOG}"
