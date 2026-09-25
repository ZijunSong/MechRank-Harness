#!/usr/bin/env bash
set -euo pipefail

# One-seed H-min eval on GPU0. Reuses the shared harness worker but does not
# overwrite the old closer-v1 graph experiment labels or ports.
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
WORKER="$ROOT/scripts/pgllm/nohup_qwen38_27b_harness.sh"

VLLM_GPU="${VLLM_GPU:-0}"
PGLLM_SEEDS="${PGLLM_SEEDS:-1}"
VLLM_PORT="${VLLM_PORT:-8340}"
CLOSER_PORT="${CLOSER_PORT:-8390}"
LOG_DIR="${LOG_DIR:-$ROOT/logs/pgllm_qwen38_27b_gpu${VLLM_GPU}_v2min}"

mkdir -p "$LOG_DIR"

echo "[$(date '+%F %T')] Launching Qwen3.8-27B + closer-v2-min on GPU${VLLM_GPU} seed=${PGLLM_SEEDS}"
echo "  vLLM :${VLLM_PORT}  closer :${CLOSER_PORT}  logs=${LOG_DIR}"

nohup env \
  VLLM_GPU="$VLLM_GPU" \
  PGLLM_SEEDS="$PGLLM_SEEDS" \
  VLLM_PORT="$VLLM_PORT" \
  CLOSER_PORT="$CLOSER_PORT" \
  LOG_DIR="$LOG_DIR" \
  CLOSER_CONFIG="${CLOSER_CONFIG:-$ROOT/configs/closer.v2_min.qwen38.yaml}" \
  CLOSER_SERVER_MODEL_ID="${CLOSER_SERVER_MODEL_ID:-closer-v2-min}" \
  PGLLM_REGISTRY="${PGLLM_REGISTRY:-$ROOT/configs/pgllm_closer_v2_min.json}" \
  PGLLM_MODELS="${PGLLM_MODELS:-closer-v2-min}" \
  PGLLM_RUN_LABEL="${PGLLM_RUN_LABEL:-mechrank_closer_v2min_qwen38_n50_b${PGLLM_SEEDS}}" \
  GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.48}" \
  MIN_GPU_MEM_UTIL="${MIN_GPU_MEM_UTIL:-0.40}" \
  GPU_MEM_HEADROOM_MIB="${GPU_MEM_HEADROOM_MIB:-8192}" \
  MAX_MODEL_LEN="${MAX_MODEL_LEN:-65536}" \
  MAX_NUM_SEQS="${MAX_NUM_SEQS:-2}" \
  FROZEN_MANIFEST="${FROZEN_MANIFEST:-$ROOT/artifacts/frozen/closer_v2min_qwen38_harness_b${PGLLM_SEEDS}_manifest.json}" \
  bash "$WORKER" \
  >> "${LOG_DIR}/orchestrator.log" 2>&1 &
echo $! > "${LOG_DIR}/worker.pid"
echo "worker PID=$(cat "${LOG_DIR}/worker.pid")"
echo "Monitor: tail -f ${LOG_DIR}/launcher.log"
echo "Status:  PGLLM_REGISTRY=$ROOT/configs/pgllm_closer_v2_min.json PGLLM_MODELS=closer-v2-min PGLLM_RUN_LABEL=mechrank_closer_v2min_qwen38_n50_b${PGLLM_SEEDS} PGLLM_SEEDS=${PGLLM_SEEDS} bash $ROOT/scripts/pgllm/status_pgllm.sh"
