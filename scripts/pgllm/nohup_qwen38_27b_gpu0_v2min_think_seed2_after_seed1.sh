#!/usr/bin/env bash
set -euo pipefail

# After GPU0 seed 1 finishes, start seed 2 on the same think stack.
# Seed 3 is running separately on GPU6; do not start it here.
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SEQ="$ROOT/scripts/pgllm/nohup_qwen38_27b_gpu0_v2min_think.sh"
LOG_DIR="${LOG_DIR:-$ROOT/logs/pgllm_qwen38_27b_gpu0_v2min_think}"
WAIT_LOG="${LOG_DIR}/seed2_after_seed1.log"
mkdir -p "$LOG_DIR"
exec >>"$WAIT_LOG" 2>&1

echo "[$(date '+%F %T')] waiting for seed 1 pgllm-run to finish before starting seed 2"
while pgrep -f 'pgllm-run.*mechrank_closer_v2min_think_qwen38_n50_b1' >/dev/null; do
  sleep 30
done
echo "[$(date '+%F %T')] seed 1 pgllm-run exited; starting seed 2 on GPU0 think stack"
export SKIP_BOOTSTRAP=1
export SEEDS_TO_RUN=2
export CLOSER_CONFIG="$ROOT/configs/closer.v2_min.qwen38.think.yaml"
export CLOSER_SERVER_MODEL_ID="closer-v2-min-think"
export PGLLM_REGISTRY="$ROOT/configs/pgllm_closer_v2_min_think.json"
export PGLLM_MODELS="closer-v2-min-think"
export PGLLM_TIMEOUT=7200
exec bash "$SEQ"
