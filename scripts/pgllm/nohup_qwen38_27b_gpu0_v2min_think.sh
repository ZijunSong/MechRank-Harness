#!/usr/bin/env bash
set -euo pipefail

# H-min with thinking ON, matched to bare qwen38 think registry:
# enable_thinking=true, max_tokens=86016, timeout=7200s.
# Reuses GPU0 after the no-think H-min run; does not touch GPU3.
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
WORKER="$ROOT/scripts/pgllm/nohup_qwen38_27b_harness.sh"
VENV_BIN="${ROOT}/.venv/bin"
export PATH="${VENV_BIN}:${PATH}"

VLLM_GPU="${VLLM_GPU:-0}"
VLLM_PORT="${VLLM_PORT:-8540}"
CLOSER_PORT="${CLOSER_PORT:-8590}"
LOG_DIR="${LOG_DIR:-$ROOT/logs/pgllm_qwen38_27b_gpu${VLLM_GPU}_v2min_think}"
SEEDS_TO_RUN="${SEEDS_TO_RUN:-1 2 3}"
OLD_VLLM_PORT="${OLD_VLLM_PORT:-8340}"
OLD_CLOSER_PORT="${OLD_CLOSER_PORT:-8390}"
SEQ_LOG="${LOG_DIR}/sequential.log"

mkdir -p "$LOG_DIR"
exec >>"$SEQ_LOG" 2>&1

wait_pid() {
  local pid="$1"
  local label="$2"
  if [[ -z "$pid" ]]; then
    echo "[$(date '+%F %T')] missing pid for ${label}"
    return 1
  fi
  echo "[$(date '+%F %T')] waiting for ${label} PID ${pid}"
  while kill -0 "$pid" 2>/dev/null; do
    sleep 20
  done
  echo "[$(date '+%F %T')] ${label} PID ${pid} exited"
}

stop_port_service() {
  local port="$1"
  local pattern="$2"
  if ss -tln | grep -Eq ":${port}[[:space:]]"; then
    echo "[$(date '+%F %T')] freeing :${port} (${pattern})"
    fuser -k "${port}/tcp" 2>/dev/null || true
    pkill -f "${pattern}" 2>/dev/null || true
    sleep 3
    fuser -k -9 "${port}/tcp" 2>/dev/null || true
    pkill -9 -f "${pattern}" 2>/dev/null || true
    sleep 2
  fi
}

kill_gpu_engine_leftovers() {
  local gpu="$1"
  local leftover
  leftover="$(nvidia-smi -i "${gpu}" --query-compute-apps=pid,process_name --format=csv,noheader 2>/dev/null | awk -F', ' '$2 ~ /EngineCore/ {print $1}')"
  if [[ -n "${leftover}" ]]; then
    echo "[$(date '+%F %T')] killing leftover EngineCore on GPU${gpu}: ${leftover}"
    # shellcheck disable=SC2086
    kill ${leftover} 2>/dev/null || true
    sleep 3
    # shellcheck disable=SC2086
    kill -9 ${leftover} 2>/dev/null || true
    sleep 2
  fi
}

run_seed_eval() {
  local seed="$1"
  export PGLLM_SEEDS="$seed"
  export PGLLM_RUN_LABEL="mechrank_closer_v2min_think_qwen38_n50_b${seed}"
  export PGLLM_REGISTRY="$ROOT/configs/pgllm_closer_v2_min_think.json"
  export PGLLM_MODELS="closer-v2-min-think"
  export PGLLM_SIZES=50
  export PGLLM_CONCURRENCY=1
  export PGLLM_TIMEOUT=7200
  export PGLLM_RETRIES=5
  export PGLLM_EXTRA_ARGS="--retry-errors --retry-truncated"
  export PGLLM_DATA_ROOT="${PGLLM_DATA_ROOT:-/data/ppnm/data/proteingym-llm/extracted}"
  export PGLLM_RESULTS_ROOT="${PGLLM_RESULTS_ROOT:-$ROOT/artifacts/pgllm_results}"
  export PGLLM_WORK_ROOT="${PGLLM_WORK_ROOT:-$ROOT/external/proteingym-llm}"
  export PGLLM_CLOSER_BASE_URL="http://127.0.0.1:${CLOSER_PORT}/v1"
  export PGLLM_CLOSER_API_KEY="${PGLLM_CLOSER_API_KEY:-dummy-local-key}"
  local pgllm_log="${LOG_DIR}/pgllm_${PGLLM_RUN_LABEL}.log"
  echo "[$(date '+%F %T')] starting seed ${seed} run_label=${PGLLM_RUN_LABEL}" | tee -a "$pgllm_log"
  bash "$ROOT/scripts/pgllm/run_pgllm.sh" >>"$pgllm_log" 2>&1
  echo "[$(date '+%F %T')] seed ${seed} finished exit=$?"
}

# Do not inherit leftover no-think env from the parent shell.
export CLOSER_CONFIG="$ROOT/configs/closer.v2_min.qwen38.think.yaml"
export CLOSER_SERVER_MODEL_ID="closer-v2-min-think"
export PGLLM_REGISTRY="$ROOT/configs/pgllm_closer_v2_min_think.json"
export PGLLM_MODELS="closer-v2-min-think"
export PGLLM_TIMEOUT=7200
export PGLLM_SIZES=50
export PGLLM_CONCURRENCY=1
export PGLLM_RETRIES=5
export PGLLM_EXTRA_ARGS="--retry-errors --retry-truncated"
export VLLM_GPU VLLM_PORT CLOSER_PORT LOG_DIR

echo "[$(date '+%F %T')] GPU${VLLM_GPU} v2-min THINK sequential seeds=${SEEDS_TO_RUN} vllm=:${VLLM_PORT} closer=:${CLOSER_PORT}"
echo "[$(date '+%F %T')] registry=${PGLLM_REGISTRY} models=${PGLLM_MODELS} timeout=${PGLLM_TIMEOUT} config=${CLOSER_CONFIG}"

SKIP_BOOTSTRAP="${SKIP_BOOTSTRAP:-0}"
if [[ "${SKIP_BOOTSTRAP}" == "1" ]]; then
  echo "[$(date '+%F %T')] SKIP_BOOTSTRAP=1; reusing existing vLLM/closer"
  if ! curl -sf "http://127.0.0.1:${VLLM_PORT}/v1/models" >/dev/null; then
    echo "[$(date '+%F %T')] vLLM :${VLLM_PORT} is down; cannot skip bootstrap"
    exit 1
  fi
  if ! curl -sf "http://127.0.0.1:${CLOSER_PORT}/health" >/dev/null; then
    echo "[$(date '+%F %T')] closer :${CLOSER_PORT} is down; cannot skip bootstrap"
    exit 1
  fi
  for seed in ${SEEDS_TO_RUN}; do
    if ! curl -sf "http://127.0.0.1:${CLOSER_PORT}/health" >/dev/null; then
      echo "[$(date '+%F %T')] closer :${CLOSER_PORT} is down; abort before seed ${seed}"
      exit 1
    fi
    run_seed_eval "$seed"
  done
  echo "[$(date '+%F %T')] thinking sequential seeds finished"
  exit 0
fi

# Finished no-think H-min on GPU0: release that vLLM so we can retune context.
stop_port_service "${OLD_CLOSER_PORT}" "closer-server.*--port ${OLD_CLOSER_PORT}"
stop_port_service "${OLD_VLLM_PORT}" "vllm serve .*--port ${OLD_VLLM_PORT}"
stop_port_service "${CLOSER_PORT}" "closer-server.*--port ${CLOSER_PORT}"
stop_port_service "${VLLM_PORT}" "vllm serve .*--port ${VLLM_PORT}"
kill_gpu_engine_leftovers "${VLLM_GPU}"
sleep 5
echo "[$(date '+%F %T')] GPU${VLLM_GPU} after release:"
nvidia-smi -i "${VLLM_GPU}" --query-gpu=memory.free,memory.used --format=csv,noheader

first_seed="${SEEDS_TO_RUN%% *}"
export PGLLM_SEEDS="$first_seed"
export PGLLM_RUN_LABEL="mechrank_closer_v2min_think_qwen38_n50_b${first_seed}"
export GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.85}"
export MIN_GPU_MEM_UTIL="${MIN_GPU_MEM_UTIL:-0.40}"
export GPU_MEM_HEADROOM_MIB="${GPU_MEM_HEADROOM_MIB:-8192}"
export MAX_MODEL_LEN="${MAX_MODEL_LEN:-131072}"
export MAX_NUM_SEQS="${MAX_NUM_SEQS:-1}"
export FROZEN_MANIFEST="$ROOT/artifacts/frozen/closer_v2min_think_qwen38_harness_b${first_seed}_manifest.json"

echo "[$(date '+%F %T')] bootstrapping thinking vLLM/closer via worker for seed ${first_seed} max_model_len=${MAX_MODEL_LEN}"
if ! bash "$WORKER"; then
  echo "[$(date '+%F %T')] 131k worker failed; retrying max_model_len=65536"
  stop_port_service "${CLOSER_PORT}" "closer-server.*--port ${CLOSER_PORT}"
  stop_port_service "${VLLM_PORT}" "vllm serve .*--port ${VLLM_PORT}"
  kill_gpu_engine_leftovers "${VLLM_GPU}"
  export MAX_MODEL_LEN=65536
  export MAX_NUM_SEQS=2
  export GPU_MEM_UTIL=0.70
  bash "$WORKER"
fi
echo "[$(date '+%F %T')] worker returned; waiting for first-seed pgllm"

first_pid="$(cat "${LOG_DIR}/pgllm.pid" 2>/dev/null || true)"
if [[ -n "$first_pid" ]] && kill -0 "$first_pid" 2>/dev/null; then
  wait_pid "$first_pid" "seed ${first_seed} pgllm"
else
  echo "[$(date '+%F %T')] first-seed pgllm pid not running; check logs"
fi

remaining="${SEEDS_TO_RUN#${first_seed}}"
remaining="${remaining#"${remaining%%[![:space:]]*}"}"
for seed in $remaining; do
  if ! curl -sf "http://127.0.0.1:${CLOSER_PORT}/health" >/dev/null; then
    echo "[$(date '+%F %T')] closer :${CLOSER_PORT} is down; abort before seed ${seed}"
    exit 1
  fi
  run_seed_eval "$seed"
done

echo "[$(date '+%F %T')] thinking sequential seeds finished"
