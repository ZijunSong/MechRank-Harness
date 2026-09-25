#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
VENV_BIN="${ROOT}/.venv/bin"
export PATH="${VENV_BIN}:${PATH}"

VLLM_GPU="${VLLM_GPU:?VLLM_GPU is required}"
PGLLM_SEEDS="${PGLLM_SEEDS:?PGLLM_SEEDS is required}"
VLLM_PORT="${VLLM_PORT:?VLLM_PORT is required}"
CLOSER_PORT="${CLOSER_PORT:?CLOSER_PORT is required}"

CLOSER_CONFIG="${CLOSER_CONFIG:-$ROOT/configs/closer.qwen38.harness.yaml}"
MODEL_PATH="${MODEL_PATH:-/data/ppnm/models/Qwen3.8-27B}"
SERVED_NAME="${SERVED_NAME:-Qwen/Qwen3.8-27B}"
VLLM_ENV="${VLLM_ENV:-/data/ppnm/miniconda3/envs/bishop}"
CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-12.4}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-131072}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-16}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.70}"
GPU_MEM_HEADROOM_MIB="${GPU_MEM_HEADROOM_MIB:-6144}"
MIN_GPU_MEM_UTIL="${MIN_GPU_MEM_UTIL:-0.55}"

LOG_DIR="${LOG_DIR:-$ROOT/logs/pgllm_qwen38_27b_gpu${VLLM_GPU}_harness}"
PGLLM_RUN_LABEL="${PGLLM_RUN_LABEL:-mechrank_closer_qwen38_n50_b${PGLLM_SEEDS}}"

mkdir -p "$LOG_DIR"

free_total="$(nvidia-smi -i "${VLLM_GPU}" --query-gpu=memory.free,memory.total --format=csv,noheader,nounits | head -n1)"
free_mib="${free_total%%,*}"
total_mib="${free_total##*,}"
free_mib="${free_mib// /}"
total_mib="${total_mib// /}"
avail_util="$(python3 -c "print(max(0.0, (${free_mib}-${GPU_MEM_HEADROOM_MIB})/${total_mib}))")"
GPU_MEM_UTIL="$(python3 -c "print(f'{min(float(\"${GPU_MEM_UTIL}\"), float(\"${avail_util}\")):.3f}')")"
if python3 -c "raise SystemExit(0 if float('${GPU_MEM_UTIL}') >= float('${MIN_GPU_MEM_UTIL}') else 1)"; then
  :
else
  echo "GPU${VLLM_GPU} free=${free_mib}MiB cannot fit Qwen3.8-27B at util=${GPU_MEM_UTIL} (min=${MIN_GPU_MEM_UTIL})" | tee -a "${LOG_DIR}/launcher.log"
  exit 1
fi

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
export PGLLM_REGISTRY="${PGLLM_REGISTRY:-$ROOT/configs/pgllm_closer_model.json}"
export PGLLM_MODELS="${PGLLM_MODELS:-closer-v1}"
export PGLLM_SIZES="${PGLLM_SIZES:-50}"
export PGLLM_SEEDS
export PGLLM_RUN_LABEL
export PGLLM_CONCURRENCY="${PGLLM_CONCURRENCY:-1}"
export PGLLM_TIMEOUT="${PGLLM_TIMEOUT:-14400}"
export PGLLM_RETRIES="${PGLLM_RETRIES:-5}"
export PGLLM_EXTRA_ARGS="${PGLLM_EXTRA_ARGS:---retry-errors --retry-truncated}"

CLOSER_LOG="${LOG_DIR}/closer_server.log"
PGLLM_LOG="${LOG_DIR}/pgllm_${PGLLM_RUN_LABEL}.log"
VLLM_LOG="${LOG_DIR}/vllm_gpu${VLLM_GPU}.log"
LAUNCH_LOG="${LOG_DIR}/launcher.log"
FROZEN_MANIFEST="${FROZEN_MANIFEST:-$ROOT/artifacts/frozen/closer_v1_qwen38_harness_b${PGLLM_SEEDS}_manifest.json}"

port_in_use() {
  ss -tln | grep -Eq ":${1}[[:space:]]"
}

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

stop_pid_file() {
  local pid_file="$1"
  local label="$2"
  if [[ -f "$pid_file" ]]; then
    local old_pid
    old_pid="$(cat "$pid_file" 2>/dev/null || true)"
    if [[ -n "${old_pid}" ]] && kill -0 "${old_pid}" 2>/dev/null; then
      echo "Stopping previous ${label} PID ${old_pid}" | tee -a "$LAUNCH_LOG"
      kill "${old_pid}" 2>/dev/null || true
      sleep 3
      kill -9 "${old_pid}" 2>/dev/null || true
    fi
  fi
}

echo "[$(date '+%F %T')] GPU${VLLM_GPU} harness worker seeds=${PGLLM_SEEDS} vllm=:${VLLM_PORT} closer=:${CLOSER_PORT} util=${GPU_MEM_UTIL} max_model_len=${MAX_MODEL_LEN} max_num_seqs=${MAX_NUM_SEQS} free=${free_mib}MiB" | tee -a "$LAUNCH_LOG"

if [[ "${PGLLM_START_ONLY:-0}" != "1" ]]; then
  stop_pid_file "${LOG_DIR}/pgllm.pid" "pgllm-run"
  pkill -f "pgllm-run.*${PGLLM_RUN_LABEL}" 2>/dev/null || true
fi

stop_pid_file "${LOG_DIR}/closer.pid" "closer-server"
if port_in_use "${CLOSER_PORT}"; then
  pkill -f "closer-server.*--port ${CLOSER_PORT}" 2>/dev/null || true
  sleep 2
fi

stop_pid_file "${LOG_DIR}/vllm.pid" "vLLM"
if port_in_use "${VLLM_PORT}"; then
  fuser -k "${VLLM_PORT}/tcp" 2>/dev/null || true
  pkill -f "vllm serve ${MODEL_PATH}.*--port ${VLLM_PORT}" 2>/dev/null || true
  sleep 3
  fuser -k -9 "${VLLM_PORT}/tcp" 2>/dev/null || true
  sleep 2
fi

if port_in_use "${VLLM_PORT}"; then
  echo "Port ${VLLM_PORT} still in use; abort." | tee -a "$LAUNCH_LOG"
  exit 1
fi
if port_in_use "${CLOSER_PORT}"; then
  echo "Port ${CLOSER_PORT} still in use; abort." | tee -a "$LAUNCH_LOG"
  exit 1
fi

# Host /dev/shm is 0755, so vLLM's multiprocessing.Lock() fails as a normal user.
# Give the server a private writable shm without touching host mounts.
BWRAP=(
  bwrap
  --bind / /
  --dev-bind /dev /dev
  --proc /proc
  --tmpfs /dev/shm
  --share-net
)

echo "[$(date '+%F %T')] Starting vLLM on GPU${VLLM_GPU} port ${VLLM_PORT}" | tee -a "$VLLM_LOG"
nohup "${BWRAP[@]}" env -u VLLM_GPU \
    CUDA_VISIBLE_DEVICES="${VLLM_GPU}" \
    CUDA_HOME="${CUDA_HOME}" \
    PATH="${CUDA_HOME}/bin:${VLLM_ENV}/bin:${PATH}" \
    LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH:-}" \
    "${VLLM_ENV}/bin/vllm" serve "${MODEL_PATH}" \
    --host 127.0.0.1 \
    --port "${VLLM_PORT}" \
    --tensor-parallel-size 1 \
    --served-model-name "${SERVED_NAME}" \
    --reasoning-parser qwen3 \
    --enable-auto-tool-choice \
    --tool-call-parser qwen3_coder \
    --mm-encoder-tp-mode data \
    --max-model-len "${MAX_MODEL_LEN}" \
    --max-num-seqs "${MAX_NUM_SEQS}" \
    --language-model-only \
    --gpu-memory-utilization "${GPU_MEM_UTIL}" \
    >> "$VLLM_LOG" 2>&1 &
echo $! > "${LOG_DIR}/vllm.pid"
echo "vLLM PID=$(cat "${LOG_DIR}/vllm.pid")" | tee -a "$LAUNCH_LOG"

echo "[$(date '+%F %T')] Waiting for vLLM health on :${VLLM_PORT}" | tee -a "$LAUNCH_LOG"
if ! wait_for_url "http://127.0.0.1:${VLLM_PORT}/v1/models" "vLLM" 360; then
  echo "vLLM did not become ready within 30 minutes; see ${VLLM_LOG}" | tee -a "$LAUNCH_LOG"
  exit 1
fi

echo "[$(date '+%F %T')] Starting closer-server on port ${CLOSER_PORT}" | tee -a "$CLOSER_LOG"
cd "$ROOT"
nohup closer-server \
  --config "$CLOSER_CONFIG" \
  --host "${CLOSER_SERVER_HOST}" \
  --port "${CLOSER_PORT}" \
  >> "$CLOSER_LOG" 2>&1 &
echo $! > "${LOG_DIR}/closer.pid"
echo "closer-server PID=$(cat "${LOG_DIR}/closer.pid")" | tee -a "$LAUNCH_LOG"

wait_for_url "http://127.0.0.1:${CLOSER_PORT}/health" "closer-server" 60

echo "[$(date '+%F %T')] Freezing harness config -> ${FROZEN_MANIFEST}" | tee -a "$PGLLM_LOG"
python "$ROOT/scripts/freeze_config.py" \
  --config "$CLOSER_CONFIG" \
  --output "$FROZEN_MANIFEST" >> "$PGLLM_LOG" 2>&1

if [[ "${PGLLM_START_ONLY:-0}" == "1" ]]; then
  echo "[$(date '+%F %T')] PGLLM_START_ONLY=1; vLLM :${VLLM_PORT} and closer :${CLOSER_PORT} are ready" | tee -a "$LAUNCH_LOG"
  exit 0
fi

echo "[$(date '+%F %T')] Launching PG-LLM harness eval run_label=${PGLLM_RUN_LABEL} seeds=${PGLLM_SEEDS}" | tee -a "$PGLLM_LOG"
nohup bash "$ROOT/scripts/pgllm/run_pgllm.sh" >> "$PGLLM_LOG" 2>&1 &
echo $! > "${LOG_DIR}/pgllm.pid"
echo "PG-LLM PID=$(cat "${LOG_DIR}/pgllm.pid")" | tee -a "$LAUNCH_LOG"
echo "Logs: ${VLLM_LOG} , ${CLOSER_LOG} , ${PGLLM_LOG}"
echo "Monitor: PGLLM_RUN_LABEL=${PGLLM_RUN_LABEL} PGLLM_SEEDS=${PGLLM_SEEDS} bash $ROOT/scripts/pgllm/status_pgllm.sh"
