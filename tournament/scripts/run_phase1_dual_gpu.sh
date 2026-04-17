#!/usr/bin/env bash
set -euo pipefail

# Run the fuller phase-1 workflow across two GPUs by parallelizing experiments.
#
# Assumes:
# - repo is already bootstrapped
# - HF_TOKEN and WANDB_API_KEY are exported
# - working directory is the repo root
#
# This script:
#   1. verifies access
#   2. builds shared data
#   3. runs M2 and M1 in parallel
#   4. evaluates both
#   5. stages lightweight results into tournament/results/phase1
#
# Optional:
#   export PUSH_RESULTS_ON_COMPLETE=1
#   bash tournament/scripts/run_phase1_dual_gpu.sh

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")"/../.. && pwd)"
cd "$REPO_ROOT"

if [[ -z "${HF_TOKEN:-}" ]]; then
  echo "[error] HF_TOKEN is not set"
  exit 1
fi

if [[ -z "${WANDB_API_KEY:-}" ]]; then
  echo "[error] WANDB_API_KEY is not set"
  exit 1
fi

if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "[error] nvidia-smi not found"
  exit 1
fi

GPU_COUNT="$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l | tr -d ' ')"
if [[ "${GPU_COUNT}" -lt 2 ]]; then
  echo "[error] run_phase1_dual_gpu.sh expects at least 2 GPUs, found ${GPU_COUNT}"
  exit 1
fi

mkdir -p tournament/logs tournament/results/phase1

echo "[step] verify access"
python tournament/scripts/verify_access.py | tee tournament/logs/verify_access.log

echo "[step] download and validate sources"
python tournament/scripts/download_sources.py --config tournament/configs/data/source_pool.yaml | tee tournament/logs/download_sources.log

echo "[step] build source pool"
python tournament/scripts/build_source_pool.py --config tournament/configs/data/source_pool.yaml | tee tournament/logs/build_source_pool.log

echo "[step] build P0"
python tournament/scripts/build_p0_pairs.py | tee tournament/logs/build_p0.log

echo "[step] build P1"
python tournament/scripts/build_p1_curriculum.py | tee tournament/logs/build_p1.log

echo "[step] build P3"
python tournament/scripts/build_p3_preferences.py | tee tournament/logs/build_p3.log

echo "[step] train M2 on GPU 0 and M1 on GPU 1"
CUDA_VISIBLE_DEVICES=0 python tournament/scripts/train_sft.py --config tournament/configs/train/m2_full_lora.yaml \
  2>&1 | tee tournament/logs/m2_train.log &
PID_M2=$!

CUDA_VISIBLE_DEVICES=1 python tournament/scripts/train_sft.py --config tournament/configs/train/m1_attn_lora.yaml \
  2>&1 | tee tournament/logs/m1_train.log &
PID_M1=$!

wait "$PID_M2"
wait "$PID_M1"

echo "[step] evaluate M2"
CUDA_VISIBLE_DEVICES=0 python tournament/scripts/eval_checkpoint.py --config tournament/configs/eval/leaderboard.yaml \
  2>&1 | tee tournament/logs/m2_eval.log

echo "[step] evaluate M1"
python - <<'PY'
from pathlib import Path
import yaml
cfg_path = Path("tournament/configs/eval/leaderboard.yaml")
cfg = yaml.safe_load(cfg_path.read_text())
cfg["run_dir"] = "tournament/outputs/runs/m1_attn_lora"
cfg["output_path"] = "tournament/outputs/leaderboards/m1_eval.json"
tmp = Path("tournament/configs/eval/m1_temp_eval.yaml")
tmp.write_text(yaml.safe_dump(cfg, sort_keys=False))
print(tmp)
PY
CUDA_VISIBLE_DEVICES=1 python tournament/scripts/eval_checkpoint.py --config tournament/configs/eval/m1_temp_eval.yaml \
  2>&1 | tee tournament/logs/m1_eval.log

echo "[step] build leaderboard"
python tournament/scripts/eval_leaderboard.py \
  --eval-json tournament/outputs/leaderboards/m2_eval.json tournament/outputs/leaderboards/m1_eval.json \
  2>&1 | tee tournament/logs/leaderboard.log

echo "[step] stage lightweight results"
mkdir -p tournament/results/phase1/logs
cp tournament/outputs/leaderboards/leaderboard.csv tournament/results/phase1/leaderboard.csv || true
cp tournament/outputs/leaderboards/m1_eval.json tournament/results/phase1/m1_eval.json || true
cp tournament/outputs/leaderboards/m2_eval.json tournament/results/phase1/m2_eval.json || true
cp tournament/logs/*.log tournament/results/phase1/logs/ || true

if [[ "${PUSH_RESULTS_ON_COMPLETE:-0}" == "1" ]]; then
  bash tournament/scripts/push_results_to_github.sh
fi

cat <<EOF

[success] Dual-GPU phase-1 run completed.

Tracked in realtime via:
  - tournament/logs/*.log
  - W&B runs
  - nvidia-smi / tmux on the instance

Lightweight GitHub-pushable results were staged in:
  tournament/results/phase1

If you enabled push-on-complete and GitHub SSH is configured, those results were pushed automatically.

EOF
