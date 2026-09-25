#!/usr/bin/env bash
set -euo pipefail

# Sequential H-min seeds 2 then 3 on GPU3. One vLLM + closer, do not touch GPU0 seed 1.
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
WORKER="$ROOT/scripts/pgllm/nohup_qwen38_27b_harness.sh"
VENV_BIN="${ROOT}/.venv/bin"
export PATH="${VENV_BIN}:${PATH}"

VLLM_GPU="${VLLM_GPU:-3}"
VLLM_PORT="${VLLM_PORT:-8440}"
CLOSER_PORT="${CLOSER_PORT:-8490}"
LOG_DIR="${LOG_DIR:-$ROOT/logs/pgllm_qwen38_27b_gpu${VLLM_GPU}_v2min}"
SEEDS_TO_RUN="${SEEDS_TO_RUN:-2 3}"
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
    sleep 15
  done
  echo "[$(date '+%F %T')] ${label} PID ${pid} exited"
}

run_seed_eval() {
  local seed="$1"
  export PGLLM_SEEDS="$seed"
  export PGLLM_RUN_LABEL="mechrank_closer_v2min_qwen38_n50_b${seed}"
  export PGLLM_REGISTRY="$ROOT/configs/pgllm_closer_v2_min.json"
  export PGLLM_MODELS="closer-v2-min"
  export PGLLM_SIZES=50
  export PGLLM_CONCURRENCY=1
  export PGLLM_TIMEOUT=14400
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

echo "[$(date '+%F %T')] GPU${VLLM_GPU} sequential v2-min seeds=${SEEDS_TO_RUN} vllm=:${VLLM_PORT} closer=:${CLOSER_PORT}"

first_seed="${SEEDS_TO_RUN%% *}"
export VLLM_GPU VLLM_PORT CLOSER_PORT LOG_DIR
export CLOSER_CONFIG="${CLOSER_CONFIG:-$ROOT/configs/closer.v2_min.qwen38.yaml}"
export CLOSER_SERVER_MODEL_ID="${CLOSER_SERVER_MODEL_ID:-closer-v2-min}"
export PGLLM_REGISTRY="${PGLLM_REGISTRY:-$ROOT/configs/pgllm_closer_v2_min.json}"
export PGLLM_MODELS="${PGLLM_MODELS:-closer-v2-min}"
export PGLLM_SEEDS="$first_seed"
export PGLLM_RUN_LABEL="mechrank_closer_v2min_qwen38_n50_b${first_seed}"
export GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.48}"
export MIN_GPU_MEM_UTIL="${MIN_GPU_MEM_UTIL:-0.40}"
export GPU_MEM_HEADROOM_MIB="${GPU_MEM_HEADROOM_MIB:-8192}"
export MAX_MODEL_LEN="${MAX_MODEL_LEN:-65536}"
export MAX_NUM_SEQS="${MAX_NUM_SEQS:-2}"
export FROZEN_MANIFEST="${FROZEN_MANIFEST:-$ROOT/artifacts/frozen/closer_v2min_qwen38_harness_b${first_seed}_manifest.json}"

echo "[$(date '+%F %T')] bootstrapping vLLM/closer via worker for seed ${first_seed}"
bash "$WORKER"
echo "[$(date '+%F %T')] worker returned; waiting for first-seed pgllm"

first_pid="$(cat "${LOG_DIR}/pgllm.pid" 2>/dev/null || true)"
if [[ -n "$first_pid" ]] && kill -0 "$first_pid" 2>/dev/null; then
  wait_pid "$first_pid" "seed ${first_seed} pgllm"
else
  # Worker already launched and the wrapper may have exited after a quick failure;
  # still continue remaining seeds only if closer is healthy.
  echo "[$(date '+%F %T')] first-seed pgllm pid not running; check ${LOG_DIR}/pgllm_${PGLLM_RUN_LABEL}.log"
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

echo "[$(date '+%F %T')] sequential seeds finished. vLLM/closer left running on :${VLLM_PORT}/:${CLOSER_PORT}"
