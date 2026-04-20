#!/usr/bin/env bash
set -euo pipefail

# Run the phase-2 tournament bracket across two GPUs:
#   stage 1 -> build P2/P3 from round-1 artifacts
#   stage 2 -> M5 (curriculum) vs M6 (hard negatives)
#   stage 3 -> M7 (DPO)
# Then build a combined leaderboard from any available round-1 + phase-2 evals.

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
  echo "[error] run_phase2_dual_gpu.sh expects at least 2 GPUs, found ${GPU_COUNT}"
  exit 1
fi

export MPLCONFIGDIR="${MPLCONFIGDIR:-${WORK_ROOT:-$REPO_ROOT/tournament}/mplconfig}"
mkdir -p "$MPLCONFIGDIR" tournament/logs tournament/results/phase2/logs tournament/outputs/leaderboards

RESULTS_DIR="tournament/results/phase2"
P3_LIMIT="${P3_LIMIT:-600}"
WINNER_FAILURES_JSONL="${WINNER_FAILURES_JSONL:-}"
if [[ -z "$WINNER_FAILURES_JSONL" ]]; then
  if [[ -f tournament/outputs/leaderboards/m3_eval.jsonl ]]; then
    WINNER_FAILURES_JSONL="tournament/outputs/leaderboards/m3_eval.jsonl"
  elif [[ -f tournament/outputs/leaderboards/m2_eval.jsonl ]]; then
    WINNER_FAILURES_JSONL="tournament/outputs/leaderboards/m2_eval.jsonl"
  else
    WINNER_FAILURES_JSONL=""
  fi
fi

push_results() {
  local message="$1"
  if [[ "${PUSH_RESULTS_ON_COMPLETE:-0}" == "1" ]]; then
    RESULTS_DIR="$RESULTS_DIR" COMMIT_MESSAGE="$message" bash tournament/scripts/push_results_to_github.sh
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
  cp tournament/outputs/leaderboards/*.json "$RESULTS_DIR"/ 2>/dev/null || true
  cp tournament/outputs/leaderboards/*.csv "$RESULTS_DIR"/ 2>/dev/null || true
  cp tournament/logs/*.log "$RESULTS_DIR"/logs/ 2>/dev/null || true
}

echo "[step] verify round-1 prerequisites"
test -f tournament/data/packs/p0_pairs.jsonl || { echo "[error] Missing P0 pack"; exit 1; }
test -f tournament/data/packs/p1_curriculum.jsonl || { echo "[error] Missing P1 pack"; exit 1; }

echo "[step] build P2 hard pack"
if [[ -n "$WINNER_FAILURES_JSONL" ]]; then
  "$PYTHON_BIN" tournament/scripts/build_p2_hard_pack.py --failure-evals "$WINNER_FAILURES_JSONL" \
    2>&1 | tee tournament/logs/build_p2.log
else
  echo "[warn] No winner eval jsonl found; building P2 without winner failures" | tee tournament/logs/build_p2.log
  "$PYTHON_BIN" tournament/scripts/build_p2_hard_pack.py \
    2>&1 | tee -a tournament/logs/build_p2.log
fi

echo "[step] build P3 preference pack"
"$PYTHON_BIN" tournament/scripts/build_p3_preferences.py --limit "$P3_LIMIT" \
  2>&1 | tee tournament/logs/build_p3.log

stage_artifacts
push_results "Update phase-2 data packs"

echo "[stage 1] train M5 on GPU 0 and M6 on GPU 1"
CUDA_VISIBLE_DEVICES=0 WANDB_NAME="m5_curriculum" WANDB_RUN_GROUP="phase2_m5_m6" \
  "$PYTHON_BIN" tournament/scripts/train_sft.py --config tournament/configs/train/m5_curriculum.yaml \
  2>&1 | tee tournament/logs/m5_train.log &
PID_M5=$!

CUDA_VISIBLE_DEVICES=1 WANDB_NAME="m6_hard_negative" WANDB_RUN_GROUP="phase2_m5_m6" \
  "$PYTHON_BIN" tournament/scripts/train_sft.py --config tournament/configs/train/m6_hard_negative.yaml \
  2>&1 | tee tournament/logs/m6_train.log &
PID_M6=$!

wait "$PID_M5"
wait "$PID_M6"

make_eval_config "tournament/outputs/runs/m5_curriculum" "tournament/outputs/leaderboards/m5_eval.json" "tournament/configs/eval/m5_temp_eval.yaml"
make_eval_config "tournament/outputs/runs/m6_hard_negative" "tournament/outputs/leaderboards/m6_eval.json" "tournament/configs/eval/m6_temp_eval.yaml"

echo "[stage 1] evaluate M5 on GPU 0 and M6 on GPU 1"
CUDA_VISIBLE_DEVICES=0 "$PYTHON_BIN" tournament/scripts/eval_checkpoint.py --config tournament/configs/eval/m5_temp_eval.yaml \
  2>&1 | tee tournament/logs/m5_eval.log &
PID_M5_EVAL=$!

CUDA_VISIBLE_DEVICES=1 "$PYTHON_BIN" tournament/scripts/eval_checkpoint.py --config tournament/configs/eval/m6_temp_eval.yaml \
  2>&1 | tee tournament/logs/m6_eval.log &
PID_M6_EVAL=$!

wait "$PID_M5_EVAL"
wait "$PID_M6_EVAL"

stage_artifacts
push_results "Update M5 and M6 phase-2 results"

echo "[stage 2] train M7 DPO on GPU 0"
CUDA_VISIBLE_DEVICES=0 WANDB_NAME="m7_dpo" WANDB_RUN_GROUP="phase2_m7" \
  "$PYTHON_BIN" tournament/scripts/train_dpo.py --config tournament/configs/train/m7_dpo.yaml \
  2>&1 | tee tournament/logs/m7_train.log

make_eval_config "tournament/outputs/runs/m7_dpo" "tournament/outputs/leaderboards/m7_eval.json" "tournament/configs/eval/m7_temp_eval.yaml"

echo "[stage 2] evaluate M7 on GPU 0"
CUDA_VISIBLE_DEVICES=0 "$PYTHON_BIN" tournament/scripts/eval_checkpoint.py --config tournament/configs/eval/m7_temp_eval.yaml \
  2>&1 | tee tournament/logs/m7_eval.log

stage_artifacts
push_results "Update M7 phase-2 results"

echo "[step] build combined leaderboard"
EVAL_JSONS=()
for candidate in \
  tournament/outputs/leaderboards/m1_eval.json \
  tournament/outputs/leaderboards/m2_eval.json \
  tournament/outputs/leaderboards/m3_eval.json \
  tournament/outputs/leaderboards/m4_eval.json \
  tournament/outputs/leaderboards/m5_eval.json \
  tournament/outputs/leaderboards/m6_eval.json \
  tournament/outputs/leaderboards/m7_eval.json
do
  if [[ -f "$candidate" ]]; then
    EVAL_JSONS+=("$candidate")
  fi
done

if [[ "${#EVAL_JSONS[@]}" -lt 2 ]]; then
  echo "[error] Need at least two eval summaries to build a leaderboard"
  exit 1
fi

"$PYTHON_BIN" tournament/scripts/eval_leaderboard.py --eval-json "${EVAL_JSONS[@]}" \
  2>&1 | tee tournament/logs/phase2_leaderboard.log

stage_artifacts
push_results "Update combined tournament leaderboard after phase 2"

cat <<EOF

[success] Phase-2 dual-GPU tournament run completed.

Results staged in:
  $RESULTS_DIR

EOF
