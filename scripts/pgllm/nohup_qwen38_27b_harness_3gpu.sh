#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
WORKER="$ROOT/scripts/pgllm/nohup_qwen38_27b_harness.sh"

# GPU4/5/7 each own one ProteinGym-LLM batch so wall-clock tracks a single 217-assay seed.
# Separate run labels avoid pgllm fingerprint clashes from different closer ports.
# Space vLLM ports: EngineCore also binds nearby TCP/ZMQ ports.
LAUNCHES=(
  "4 1 8044 8094"
  "5 2 8145 8195"
  "7 3 8247 8297"
)

echo "[$(date '+%F %T')] Launching Qwen3.8-27B + harness on GPU 4/5/7"

for spec in "${LAUNCHES[@]}"; do
  # shellcheck disable=SC2086
  set -- $spec
  gpu="$1"
  seed="$2"
  vllm_port="$3"
  closer_port="$4"
  log_dir="$ROOT/logs/pgllm_qwen38_27b_gpu${gpu}_harness"
  mkdir -p "$log_dir"
  echo "[$(date '+%F %T')] GPU${gpu} -> seed ${seed}, vLLM :${vllm_port}, closer :${closer_port}"
  nohup env \
    VLLM_GPU="$gpu" \
    PGLLM_SEEDS="$seed" \
    VLLM_PORT="$vllm_port" \
    CLOSER_PORT="$closer_port" \
    GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.70}" \
    MAX_MODEL_LEN="${MAX_MODEL_LEN:-131072}" \
    MAX_NUM_SEQS="${MAX_NUM_SEQS:-16}" \
    bash "$WORKER" \
    >> "${log_dir}/orchestrator.log" 2>&1 &
  echo $! > "${log_dir}/worker.pid"
  echo "  worker PID=$(cat "${log_dir}/worker.pid")  logs=${log_dir}"
done

echo
echo "Workers are loading vLLM in parallel. Check:"
echo "  tail -f $ROOT/logs/pgllm_qwen38_27b_gpu{4,5,7}_harness/launcher.log"
echo "Status after results start appearing:"
echo "  PGLLM_RUN_LABEL=mechrank_closer_qwen38_n50_b1 PGLLM_SEEDS=1 bash $ROOT/scripts/pgllm/status_pgllm.sh"
echo "  PGLLM_RUN_LABEL=mechrank_closer_qwen38_n50_b2 PGLLM_SEEDS=2 bash $ROOT/scripts/pgllm/status_pgllm.sh"
echo "  PGLLM_RUN_LABEL=mechrank_closer_qwen38_n50_b3 PGLLM_SEEDS=3 bash $ROOT/scripts/pgllm/status_pgllm.sh"
