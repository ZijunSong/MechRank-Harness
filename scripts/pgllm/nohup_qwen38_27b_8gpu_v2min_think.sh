#!/usr/bin/env bash
# Eight single-GPU vLLM+closer servers, then three seeds one after another.
# Each seed's 217 assays are split across the GPUs (default: balance total
# sequence length) and written into one shared run label, so pgllm-score can
# read the seed as a single result.
#
# Does not kill foreign GPU jobs. Aborts if a selected GPU is too full or if
# the chosen ports are already taken.
#
#   nohup bash scripts/pgllm/nohup_qwen38_27b_8gpu_v2min_think.sh >/dev/null 2>&1 &
#   tail -f logs/pgllm_qwen38_27b_8gpu_v2min_think/orchestrator.log
#
# Resume an interrupted eval on servers that are already healthy:
#   REUSE_SERVICES=1 bash scripts/pgllm/nohup_qwen38_27b_8gpu_v2min_think.sh
# Plan only:
#   DRY_RUN=1 bash scripts/pgllm/nohup_qwen38_27b_8gpu_v2min_think.sh
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
WORKER="$ROOT/scripts/pgllm/nohup_qwen38_27b_harness.sh"
VENV_BIN="${ROOT}/.venv/bin"
export PATH="${VENV_BIN}:${PATH}"

GPUS="${GPUS:-0 1 2 3 4 5 6 7}"
SEEDS="${SEEDS:-1 2 3}"
SHARD_MODE="${SHARD_MODE:-length}" # length | roundrobin
DRY_RUN="${DRY_RUN:-0}"
REUSE_SERVICES="${REUSE_SERVICES:-0}"

VLLM_PORT_BASE="${VLLM_PORT_BASE:-8700}"
CLOSER_PORT_BASE="${CLOSER_PORT_BASE:-8800}"
MIN_FREE_MIB="${MIN_FREE_MIB:-90000}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.85}"
MIN_GPU_MEM_UTIL="${MIN_GPU_MEM_UTIL:-0.60}"
GPU_MEM_HEADROOM_MIB="${GPU_MEM_HEADROOM_MIB:-8192}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-131072}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-1}"

RUN_PREFIX="${RUN_PREFIX:-mechrank_closer_v2min_think8_qwen38_n50}"
CLOSER_CONFIG="${CLOSER_CONFIG:-$ROOT/configs/closer.v2_min.qwen38.think.yaml}"
PGLLM_REGISTRY="${PGLLM_REGISTRY:-$ROOT/configs/pgllm_closer_v2_min_think.json}"
PGLLM_MODELS="${PGLLM_MODELS:-closer-v2-min-think}"
PGLLM_TIMEOUT="${PGLLM_TIMEOUT:-7200}"
PGLLM_SIZES="${PGLLM_SIZES:-50}"
PGLLM_CONCURRENCY="${PGLLM_CONCURRENCY:-1}"
PGLLM_RETRIES="${PGLLM_RETRIES:-5}"
PGLLM_EXTRA_ARGS="${PGLLM_EXTRA_ARGS:---retry-errors --retry-truncated}"
PGLLM_DATA_ROOT="${PGLLM_DATA_ROOT:-/data/ppnm/data/proteingym-llm/extracted}"
PGLLM_RESULTS_ROOT="${PGLLM_RESULTS_ROOT:-$ROOT/artifacts/pgllm_results}"
PGLLM_WORK_ROOT="${PGLLM_WORK_ROOT:-$ROOT/external/proteingym-llm}"
PGLLM_CLOSER_API_KEY="${PGLLM_CLOSER_API_KEY:-dummy-local-key}"

LOG_DIR="${LOG_DIR:-$ROOT/logs/pgllm_qwen38_27b_8gpu_v2min_think}"
SHARD_DIR="${LOG_DIR}/shards"
mkdir -p "$LOG_DIR" "$SHARD_DIR"

echo "[$(date '+%F %T')] 8-GPU think entry log=${LOG_DIR}/orchestrator.log"
exec >>"${LOG_DIR}/orchestrator.log" 2>&1
echo "[$(date '+%F %T')] GPUS='${GPUS}' SEEDS='${SEEDS}' SHARD_MODE=${SHARD_MODE} DRY_RUN=${DRY_RUN} REUSE_SERVICES=${REUSE_SERVICES}"
echo "[$(date '+%F %T')] prefix=${RUN_PREFIX} model=${PGLLM_MODELS} timeout=${PGLLM_TIMEOUT} max_model_len=${MAX_MODEL_LEN} ports vllm=${VLLM_PORT_BASE}+gpu closer=${CLOSER_PORT_BASE}+gpu"

read -r -a GPU_LIST <<< "$GPUS"
read -r -a SEED_LIST <<< "$SEEDS"
if ((${#GPU_LIST[@]} < 1)); then
  echo "GPUS is empty"
  exit 1
fi
if ((${#SEED_LIST[@]} < 1)); then
  echo "SEEDS is empty"
  exit 1
fi

declare -A GPU_SEEN=()
for gpu in "${GPU_LIST[@]}"; do
  if [[ ! "$gpu" =~ ^[0-9]+$ ]]; then
    echo "GPU id must be an integer: ${gpu}"
    exit 1
  fi
  if [[ -n "${GPU_SEEN[$gpu]:-}" ]]; then
    echo "duplicate GPU id: ${gpu}"
    exit 1
  fi
  GPU_SEEN[$gpu]=1
done

vllm_port_for() { echo $((VLLM_PORT_BASE + $1)); }
closer_port_for() { echo $((CLOSER_PORT_BASE + $1)); }

write_shards() {
  PGLLM_DATA_ROOT="$PGLLM_DATA_ROOT" \
  PGLLM_WORK_ROOT="$PGLLM_WORK_ROOT" \
  SHARD_DIR="$SHARD_DIR" \
  SHARD_MODE="$SHARD_MODE" \
  GPU_IDS="${GPU_LIST[*]}" \
  "$VENV_BIN/python" - <<'PY'
import os
from pathlib import Path
import sys

sys.path.insert(0, os.environ["PGLLM_WORK_ROOT"])
from src.assays import load_assay_meta

meta = load_assay_meta()
if not meta:
    raise SystemExit("no runnable assays; check PGLLM_DATA_ROOT")
gpus = [int(x) for x in os.environ["GPU_IDS"].split()]
mode = os.environ["SHARD_MODE"]
shard_dir = Path(os.environ["SHARD_DIR"])
items = sorted(((name, int(info.get("seq_len") or 0)) for name, info in meta.items()), key=lambda x: (-x[1], x[0]))
buckets = {gpu: [] for gpu in gpus}
totals = {gpu: 0 for gpu in gpus}
if mode == "roundrobin":
    ordered = sorted(name for name, _ in items)
    for i, name in enumerate(ordered):
        gpu = gpus[i % len(gpus)]
        length = dict(items)[name]
        buckets[gpu].append(name)
        totals[gpu] += length
elif mode == "length":
    for name, length in items:
        gpu = min(gpus, key=lambda g: (totals[g], len(buckets[g]), g))
        buckets[gpu].append(name)
        totals[gpu] += length
else:
    raise SystemExit(f"unknown SHARD_MODE {mode}")
seen = []
for gpu in gpus:
    names = sorted(buckets[gpu])
    seen.extend(names)
    path = shard_dir / f"gpu{gpu}.assays"
    path.write_text("".join(f"{name}\n" for name in names), encoding="utf-8")
    print(f"shard gpu{gpu} n={len(names):3d} seq_len={totals[gpu]:6d} file={path}")
if len(seen) != len(set(seen)) or set(seen) != set(meta):
    raise SystemExit("shard assignment does not cover each assay once")
print(f"sharded {len(seen)} assays across {len(gpus)} gpus mode={mode}")
PY
}

port_in_use() {
  ss -tln | grep -Eq ":${1}[[:space:]]"
}

preflight() {
  local ok=1
  local gpu free port_v port_c
  for gpu in "${GPU_LIST[@]}"; do
    free="$(nvidia-smi -i "$gpu" --query-gpu=memory.free --format=csv,noheader,nounits | head -n1 | tr -d ' ')"
    port_v="$(vllm_port_for "$gpu")"
    port_c="$(closer_port_for "$gpu")"
    echo "[$(date '+%F %T')] GPU${gpu} free=${free}MiB vllm=:${port_v} closer=:${port_c}"
    if [[ "$REUSE_SERVICES" != "1" && "$free" -lt "$MIN_FREE_MIB" ]]; then
      echo "GPU${gpu} free=${free}MiB < ${MIN_FREE_MIB}MiB; not starting (no processes killed)"
      ok=0
    fi
    if [[ "$REUSE_SERVICES" == "1" ]]; then
      curl -sf "http://127.0.0.1:${port_v}/v1/models" >/dev/null || { echo "vLLM :${port_v} is not healthy"; ok=0; }
      curl -sf "http://127.0.0.1:${port_c}/health" >/dev/null || { echo "closer :${port_c} is not healthy"; ok=0; }
    else
      if port_in_use "$port_v" || port_in_use "$port_c"; then
        echo "port :${port_v} or :${port_c} is in use; not starting"
        ok=0
      fi
    fi
  done
  [[ "$ok" -eq 1 ]]
}

start_services() {
  local gpu port_v port_c log_dir pid
  local -a pids=()
  for gpu in "${GPU_LIST[@]}"; do
    port_v="$(vllm_port_for "$gpu")"
    port_c="$(closer_port_for "$gpu")"
    log_dir="${LOG_DIR}/gpu${gpu}"
    mkdir -p "$log_dir"
    echo "[$(date '+%F %T')] starting GPU${gpu} vLLM :${port_v} closer :${port_c}"
    env \
      PGLLM_START_ONLY=1 \
      VLLM_GPU="$gpu" \
      PGLLM_SEEDS="${SEED_LIST[0]}" \
      VLLM_PORT="$port_v" \
      CLOSER_PORT="$port_c" \
      LOG_DIR="$log_dir" \
      CLOSER_CONFIG="$CLOSER_CONFIG" \
      CLOSER_SERVER_MODEL_ID="$PGLLM_MODELS" \
      PGLLM_REGISTRY="$PGLLM_REGISTRY" \
      PGLLM_MODELS="$PGLLM_MODELS" \
      PGLLM_RUN_LABEL="${RUN_PREFIX}_services" \
      PGLLM_TIMEOUT="$PGLLM_TIMEOUT" \
      PGLLM_SIZES="$PGLLM_SIZES" \
      PGLLM_CONCURRENCY="$PGLLM_CONCURRENCY" \
      PGLLM_RETRIES="$PGLLM_RETRIES" \
      PGLLM_EXTRA_ARGS="$PGLLM_EXTRA_ARGS" \
      PGLLM_DATA_ROOT="$PGLLM_DATA_ROOT" \
      PGLLM_RESULTS_ROOT="$PGLLM_RESULTS_ROOT" \
      PGLLM_WORK_ROOT="$PGLLM_WORK_ROOT" \
      GPU_MEM_UTIL="$GPU_MEM_UTIL" \
      MIN_GPU_MEM_UTIL="$MIN_GPU_MEM_UTIL" \
      GPU_MEM_HEADROOM_MIB="$GPU_MEM_HEADROOM_MIB" \
      MAX_MODEL_LEN="$MAX_MODEL_LEN" \
      MAX_NUM_SEQS="$MAX_NUM_SEQS" \
      FROZEN_MANIFEST="$ROOT/artifacts/frozen/${RUN_PREFIX}_gpu${gpu}_manifest.json" \
      bash "$WORKER" >>"${log_dir}/service.log" 2>&1 &
    pids+=("$!")
  done
  local failed=0
  for pid in "${pids[@]}"; do
    if ! wait "$pid"; then
      echo "[$(date '+%F %T')] service worker PID ${pid} failed"
      failed=1
    fi
  done
  if [[ "$failed" -ne 0 ]]; then
    echo "[$(date '+%F %T')] one or more GPU services failed; leaving the ones that came up. See ${LOG_DIR}/gpu*/service.log"
    exit 1
  fi
  echo "[$(date '+%F %T')] all vLLM/closer servers are ready"
}

run_seed() {
  local seed="$1"
  local run_label="${RUN_PREFIX}_b${seed}"
  local gpu port_c log pid
  local -a pids=()
  echo "[$(date '+%F %T')] seed ${seed} run_label=${run_label} shards=${#GPU_LIST[@]}"
  for gpu in "${GPU_LIST[@]}"; do
    port_c="$(closer_port_for "$gpu")"
    if ! curl -sf "http://127.0.0.1:${port_c}/health" >/dev/null; then
      echo "[$(date '+%F %T')] closer :${port_c} is down before seed ${seed}"
      exit 1
    fi
    log="${LOG_DIR}/pgllm_${run_label}_gpu${gpu}.log"
    echo "[$(date '+%F %T')] seed ${seed} GPU${gpu} assays=$(wc -l < "${SHARD_DIR}/gpu${gpu}.assays") log=${log}"
    env \
      PGLLM_REGISTRY="$PGLLM_REGISTRY" \
      PGLLM_MODELS="$PGLLM_MODELS" \
      PGLLM_SIZES="$PGLLM_SIZES" \
      PGLLM_SEEDS="$seed" \
      PGLLM_RUN_LABEL="$run_label" \
      PGLLM_CONCURRENCY="$PGLLM_CONCURRENCY" \
      PGLLM_TIMEOUT="$PGLLM_TIMEOUT" \
      PGLLM_RETRIES="$PGLLM_RETRIES" \
      PGLLM_EXTRA_ARGS="$PGLLM_EXTRA_ARGS" \
      PGLLM_DATA_ROOT="$PGLLM_DATA_ROOT" \
      PGLLM_RESULTS_ROOT="$PGLLM_RESULTS_ROOT" \
      PGLLM_WORK_ROOT="$PGLLM_WORK_ROOT" \
      PGLLM_CLOSER_BASE_URL="http://127.0.0.1:${port_c}/v1" \
      PGLLM_CLOSER_API_KEY="$PGLLM_CLOSER_API_KEY" \
      PGLLM_ASSAYS_FILE="${SHARD_DIR}/gpu${gpu}.assays" \
      bash "$ROOT/scripts/pgllm/run_pgllm.sh" >>"$log" 2>&1 &
    pids+=("$!")
  done
  local failed=0
  for pid in "${pids[@]}"; do
    if ! wait "$pid"; then
      echo "[$(date '+%F %T')] seed ${seed} shard PID ${pid} exited non-zero"
      failed=1
    fi
  done
  echo "[$(date '+%F %T')] seed ${seed} shards finished failed=${failed}"
  return "$failed"
}

echo "[$(date '+%F %T')] writing assay shards"
write_shards

if ! preflight; then
  if [[ "$DRY_RUN" == "1" ]]; then
    echo "[$(date '+%F %T')] DRY_RUN=1; preflight failed, not starting"
    exit 0
  fi
  exit 1
fi

if [[ "$DRY_RUN" == "1" ]]; then
  echo "[$(date '+%F %T')] DRY_RUN=1; plan only, not starting"
  exit 0
fi

if [[ "$REUSE_SERVICES" != "1" ]]; then
  start_services
fi

seed_failed=0
for seed in "${SEED_LIST[@]}"; do
  if ! run_seed "$seed"; then
    seed_failed=1
  fi
done

echo "[$(date '+%F %T')] 8-GPU think seeds finished failed=${seed_failed}"
echo "Score: PGLLM_REGISTRY=${PGLLM_REGISTRY} PGLLM_MODELS=${PGLLM_MODELS} PGLLM_RUN_LABEL=${RUN_PREFIX}_b<seed> bash $ROOT/scripts/pgllm/score_pgllm.sh"
exit "$seed_failed"
