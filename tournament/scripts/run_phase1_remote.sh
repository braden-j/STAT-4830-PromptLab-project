#!/usr/bin/env bash
set -euo pipefail

# Run the fuller phase-1 workflow on a remote Prime H100.
#
# Assumes:
# - repo has already been cloned
# - dependencies are installed
# - HF_TOKEN and WANDB_API_KEY are exported
# - the working directory is the repo root
#
# This script intentionally follows the "fuller phase 1" plan:
# - build data
# - run the M2 smoke/anchor path first
# - build P3
# - stop at the point where the user chooses how many promoted full runs to fund

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

echo "[step] verify access"
python tournament/scripts/verify_access.py

echo "[step] download and validate sources"
python tournament/scripts/download_sources.py --config tournament/configs/data/source_pool.yaml

echo "[step] build source pool"
python tournament/scripts/build_source_pool.py --config tournament/configs/data/source_pool.yaml

echo "[step] build P0"
python tournament/scripts/build_p0_pairs.py

echo "[step] build P1"
python tournament/scripts/build_p1_curriculum.py

echo "[step] build P3"
python tournament/scripts/build_p3_preferences.py

echo "[step] train M2 full-LoRA anchor candidate"
python tournament/scripts/train_sft.py --config tournament/configs/train/m2_full_lora.yaml

echo "[step] evaluate M2"
python tournament/scripts/eval_checkpoint.py --config tournament/configs/eval/leaderboard.yaml

cat <<EOF

[success] Fuller phase-1 remote bootstrap completed.

Completed:
  - source download/validation
  - source pool build
  - P0/P1/P3 build
  - M2 training
  - M2 evaluation

Suggested next manual steps:
  1. Run M1 pilot:
     python tournament/scripts/train_sft.py --config tournament/configs/train/m1_attn_lora.yaml

  2. Evaluate M1 and compare against M2.

  3. If funding the broader bracket, continue with:
     - M3
     - optionally M4
     - build P2 after reviewing M2 failures
     - M5 / M6 / M7

EOF
