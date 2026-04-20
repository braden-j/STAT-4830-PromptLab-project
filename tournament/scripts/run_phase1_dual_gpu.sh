#!/usr/bin/env bash
set -euo pipefail

# Run the tournament bracket across two GPUs:
#   stage 1 -> M1 vs M2
#   stage 2 -> M3 (AdamW) vs M4 (REINFORCE)
# Then evaluate all four and build a leaderboard.

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")"/../.. && pwd)"
cd "$REPO_ROOT"

if command -v python3 >/dev/null 2>&1; then
  PYTHON_BIN="python3"
elif command -v python >/dev/null 2>&1; then
  PYTHON_BIN="python"
else
  echo "[error] Neither python3 nor python is available"
  exit 1
fi

if [[ -z "${HF_TOKEN:-}" ]]; then
  echo "[error] HF_TOKEN is not set"
  exit 1
fi

if [[ -z "${WANDB_API_KEY:-}" ]]; then
  echo "[error] WANDB_API_KEY is not set"
  exit 1
fi

GPU_COUNT="$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l | tr -d ' ')"
if [[ "${GPU_COUNT}" -lt 2 ]]; then
  echo "[warn] run_phase1_dual_gpu.sh expects at least 2 GPUs, found ${GPU_COUNT}"
  echo "[warn] Falling back to the single-GPU runner: tournament/scripts/run_phase1_remote.sh"
  exec bash tournament/scripts/run_phase1_remote.sh
fi

export MPLCONFIGDIR="${MPLCONFIGDIR:-${WORK_ROOT:-$REPO_ROOT/tournament}/mplconfig}"
mkdir -p "$MPLCONFIGDIR" tournament/logs tournament/results/phase1/logs tournament/outputs/leaderboards

push_results() {
  local message="$1"
  if [[ "${PUSH_RESULTS_ON_COMPLETE:-0}" == "1" ]]; then
    COMMIT_MESSAGE="$message" bash tournament/scripts/push_results_to_github.sh
  fi
}

make_eval_config() {
  local run_dir="$1"
  local out_path="$2"
  local tmp_path="$3"
  "$PYTHON_BIN" - <<PY
from pathlib import Path
import yaml
cfg_path = Path("tournament/configs/eval/leaderboard.yaml")
cfg = yaml.safe_load(cfg_path.read_text())
cfg["run_dir"] = "${run_dir}"
cfg["output_path"] = "${out_path}"
Path("${tmp_path}").write_text(yaml.safe_dump(cfg, sort_keys=False))
print("${tmp_path}")
PY
}

stage_artifacts() {
  cp tournament/outputs/leaderboards/*.json tournament/results/phase1/ 2>/dev/null || true
  cp tournament/outputs/leaderboards/*.csv tournament/results/phase1/ 2>/dev/null || true
  cp tournament/logs/*.log tournament/results/phase1/logs/ 2>/dev/null || true
}

if [[ ! -f tournament/data/raw/kaggle/aeon_essays.csv ]]; then
  echo "[warn] Kaggle essays CSV missing; creating empty placeholder so Pangram backfill can proceed"
  mkdir -p tournament/data/raw/kaggle
  : > tournament/data/raw/kaggle/aeon_essays.csv
fi

echo "[step] verify access"
"$PYTHON_BIN" tournament/scripts/verify_access.py | tee tournament/logs/verify_access.log

echo "[step] download and validate sources"
"$PYTHON_BIN" tournament/scripts/download_sources.py --config tournament/configs/data/source_pool.yaml | tee tournament/logs/download_sources.log

echo "[step] build source pool"
"$PYTHON_BIN" tournament/scripts/build_source_pool.py --config tournament/configs/data/source_pool.yaml | tee tournament/logs/build_source_pool.log

echo "[step] build P0"
"$PYTHON_BIN" tournament/scripts/build_p0_pairs.py | tee tournament/logs/build_p0.log

echo "[step] build P1"
"$PYTHON_BIN" tournament/scripts/build_p1_curriculum.py | tee tournament/logs/build_p1.log

stage_artifacts
push_results "Update tournament data packs"

echo "[stage 1] train M2 on GPU 0 and M1 on GPU 1"
CUDA_VISIBLE_DEVICES=0 WANDB_NAME="m2_full_lora" WANDB_RUN_GROUP="phase1_m1_m2" \
  "$PYTHON_BIN" tournament/scripts/train_sft.py --config tournament/configs/train/m2_full_lora.yaml \
  2>&1 | tee tournament/logs/m2_train.log &
PID_M2=$!

CUDA_VISIBLE_DEVICES=1 WANDB_NAME="m1_attn_lora" WANDB_RUN_GROUP="phase1_m1_m2" \
  "$PYTHON_BIN" tournament/scripts/train_sft.py --config tournament/configs/train/m1_attn_lora.yaml \
  2>&1 | tee tournament/logs/m1_train.log &
PID_M1=$!

wait "$PID_M2"
wait "$PID_M1"

make_eval_config "tournament/outputs/runs/m2_full_lora" "tournament/outputs/leaderboards/m2_eval.json" "tournament/configs/eval/m2_temp_eval.yaml"
make_eval_config "tournament/outputs/runs/m1_attn_lora" "tournament/outputs/leaderboards/m1_eval.json" "tournament/configs/eval/m1_temp_eval.yaml"

echo "[stage 1] evaluate M2 on GPU 0 and M1 on GPU 1"
CUDA_VISIBLE_DEVICES=0 "$PYTHON_BIN" tournament/scripts/eval_checkpoint.py --config tournament/configs/eval/m2_temp_eval.yaml \
  2>&1 | tee tournament/logs/m2_eval.log &
PID_M2_EVAL=$!

CUDA_VISIBLE_DEVICES=1 "$PYTHON_BIN" tournament/scripts/eval_checkpoint.py --config tournament/configs/eval/m1_temp_eval.yaml \
  2>&1 | tee tournament/logs/m1_eval.log &
PID_M1_EVAL=$!

wait "$PID_M2_EVAL"
wait "$PID_M1_EVAL"

stage_artifacts
push_results "Update M1 and M2 tournament results"

echo "[stage 2] train M3 AdamW on GPU 0 and M4 REINFORCE on GPU 1"
CUDA_VISIBLE_DEVICES=0 WANDB_NAME="m3_adamw" WANDB_RUN_GROUP="phase1_m3_m4" \
  "$PYTHON_BIN" tournament/scripts/train_sft.py --config tournament/configs/train/m3_adamw.yaml \
  2>&1 | tee tournament/logs/m3_train.log &
PID_M3=$!

CUDA_VISIBLE_DEVICES=1 WANDB_NAME="m4_reinforce" WANDB_RUN_GROUP="phase1_m3_m4" \
  "$PYTHON_BIN" tournament/scripts/train_reinforce.py --config tournament/configs/train/m4_reinforce.yaml \
  2>&1 | tee tournament/logs/m4_train.log &
PID_M4=$!

wait "$PID_M3"
wait "$PID_M4"

make_eval_config "tournament/outputs/runs/m3_adamw" "tournament/outputs/leaderboards/m3_eval.json" "tournament/configs/eval/m3_temp_eval.yaml"
make_eval_config "tournament/outputs/runs/m4_reinforce" "tournament/outputs/leaderboards/m4_eval.json" "tournament/configs/eval/m4_temp_eval.yaml"

echo "[stage 2] evaluate M3 on GPU 0 and M4 on GPU 1"
CUDA_VISIBLE_DEVICES=0 "$PYTHON_BIN" tournament/scripts/eval_checkpoint.py --config tournament/configs/eval/m3_temp_eval.yaml \
  2>&1 | tee tournament/logs/m3_eval.log &
PID_M3_EVAL=$!

CUDA_VISIBLE_DEVICES=1 "$PYTHON_BIN" tournament/scripts/eval_checkpoint.py --config tournament/configs/eval/m4_temp_eval.yaml \
  2>&1 | tee tournament/logs/m4_eval.log &
PID_M4_EVAL=$!

wait "$PID_M3_EVAL"
wait "$PID_M4_EVAL"

stage_artifacts
push_results "Update M3 and M4 tournament results"

echo "[step] build final leaderboard"
"$PYTHON_BIN" tournament/scripts/eval_leaderboard.py \
  --eval-json \
    tournament/outputs/leaderboards/m1_eval.json \
    tournament/outputs/leaderboards/m2_eval.json \
    tournament/outputs/leaderboards/m3_eval.json \
    tournament/outputs/leaderboards/m4_eval.json \
  2>&1 | tee tournament/logs/leaderboard.log

stage_artifacts
push_results "Update final tournament leaderboard"

cat <<EOF

[success] Dual-GPU tournament run completed.

Tracked via:
  - tournament/logs/*.log
  - W&B runs and groups
  - nvidia-smi / tmux on the instance

Lightweight GitHub-pushable results were staged in:
  tournament/results/phase1

EOF
