#!/usr/bin/env bash
set -euo pipefail

# Seed 2 only, thinking-on H-min. Replaces the idle no-think stack on GPU3.
# Does not touch GPU0 (seed 1) or GPU6 (seed 3).
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
WORKER="$ROOT/scripts/pgllm/nohup_qwen38_27b_harness.sh"

export CLOSER_CONFIG="$ROOT/configs/closer.v2_min.qwen38.think.yaml"
export CLOSER_SERVER_MODEL_ID="closer-v2-min-think"
export PGLLM_REGISTRY="$ROOT/configs/pgllm_closer_v2_min_think.json"
export PGLLM_MODELS="closer-v2-min-think"
export PGLLM_TIMEOUT=7200
export PGLLM_SIZES=50
export PGLLM_CONCURRENCY=1
export PGLLM_RETRIES=5
export PGLLM_EXTRA_ARGS="--retry-errors --retry-truncated"
export PGLLM_SEEDS=2
export PGLLM_RUN_LABEL="mechrank_closer_v2min_think_qwen38_n50_b2"

export VLLM_GPU="${VLLM_GPU:-3}"
export VLLM_PORT="${VLLM_PORT:-8440}"
export CLOSER_PORT="${CLOSER_PORT:-8490}"
export LOG_DIR="${LOG_DIR:-$ROOT/logs/pgllm_qwen38_27b_gpu${VLLM_GPU}_v2min_think}"
export GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.85}"
export MIN_GPU_MEM_UTIL="${MIN_GPU_MEM_UTIL:-0.40}"
export GPU_MEM_HEADROOM_MIB="${GPU_MEM_HEADROOM_MIB:-8192}"
export MAX_MODEL_LEN="${MAX_MODEL_LEN:-131072}"
export MAX_NUM_SEQS="${MAX_NUM_SEQS:-1}"
export FROZEN_MANIFEST="$ROOT/artifacts/frozen/closer_v2min_think_qwen38_harness_b2_manifest.json"

mkdir -p "$LOG_DIR"
echo "[$(date '+%F %T')] GPU${VLLM_GPU} v2-min THINK seed=2 vllm=:${VLLM_PORT} closer=:${CLOSER_PORT} max_model_len=${MAX_MODEL_LEN}"
echo "  registry=${PGLLM_REGISTRY} model=${PGLLM_MODELS} timeout=${PGLLM_TIMEOUT}"

if ! bash "$WORKER"; then
  echo "[$(date '+%F %T')] 131k worker failed; retrying max_model_len=65536"
  if ss -tln | grep -Eq ":${CLOSER_PORT}[[:space:]]"; then
    fuser -k "${CLOSER_PORT}/tcp" 2>/dev/null || true
    sleep 2
  fi
  if ss -tln | grep -Eq ":${VLLM_PORT}[[:space:]]"; then
    fuser -k "${VLLM_PORT}/tcp" 2>/dev/null || true
    sleep 3
    fuser -k -9 "${VLLM_PORT}/tcp" 2>/dev/null || true
  fi
  leftover="$(nvidia-smi -i "${VLLM_GPU}" --query-compute-apps=pid,process_name --format=csv,noheader 2>/dev/null | awk -F', ' '$2 ~ /EngineCore/ {print $1}')"
  if [[ -n "${leftover}" ]]; then
    # shellcheck disable=SC2086
    kill ${leftover} 2>/dev/null || true
    sleep 2
    # shellcheck disable=SC2086
    kill -9 ${leftover} 2>/dev/null || true
  fi
  export MAX_MODEL_LEN=65536
  export MAX_NUM_SEQS=2
  export GPU_MEM_UTIL=0.70
  bash "$WORKER"
fi
