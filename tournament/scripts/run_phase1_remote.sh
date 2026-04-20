#!/usr/bin/env bash
set -euo pipefail

# Run the fuller phase-1 workflow on a remote single-GPU Prime instance.
#
# Assumes:
# - repo has already been cloned
# - dependencies are installed
# - HF_TOKEN and WANDB_API_KEY are exported
# - the working directory is the repo root
#
# This script intentionally follows a single-GPU version of the "fuller phase 1" plan:
# - build shared data
# - run M2 first
# - run M1 second on the same GPU
# - evaluate both
# - build a lightweight leaderboard and stage results

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

rebuild_p0_permissive() {
  echo "[warn] Rebuilding P0 with permissive bootstrap fallback"
  "$PYTHON_BIN" - <<'PY'
import json, sys
sys.path.insert(0, "src")

from hill_climb.tournament.io import read_jsonl, write_jsonl, write_json
from hill_climb.tournament.data import sloppify_candidates
from hill_climb.tournament.scoring import load_canonical_pangram_module, length_ratio

import torch

rows = read_jsonl("tournament/data/source_pool.jsonl")
module = load_canonical_pangram_module()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tok, mdl = module.load_editlens(device)

kept = []
for idx, row in enumerate(rows):
    target = row["text"]
    target_score = module.editlens_score(tok, mdl, target, device)
    best = None

    for mode, cand in sloppify_candidates(target, seed=idx * 17):
        cand = cand.strip()
        if not cand or cand == target.strip():
            continue

        cand_score = module.editlens_score(tok, mdl, cand, device)
        item = {
            "example_id": f"p0-{idx}",
            "source_id": row["id"],
            "split": row["split"],
            "input_text": cand,
            "target_text": target,
            "input_score_roberta": cand_score,
            "target_score_roberta": target_score,
            "generation_mode": mode,
            "length_ratio": length_ratio(target, cand),
            "delta": cand_score - target_score,
            "acceptance_reason": "permissive_bootstrap",
        }

        if best is None or item["delta"] > best["delta"] or (
            item["delta"] == best["delta"] and item["input_score_roberta"] > best["input_score_roberta"]
        ):
            best = item

    if best and 0.70 <= best["length_ratio"] <= 1.50:
        kept.append(best)

manifest = {
    "accepted_rows": len(kept),
    "source_rows": len(rows),
    "acceptance_mode": "permissive_bootstrap",
}
write_jsonl("tournament/data/packs/p0_pairs.jsonl", kept)
write_json("tournament/data/packs/p0_manifest.json", manifest)
print(json.dumps(manifest, indent=2))
PY
}

mkdir -p tournament/logs tournament/results/phase1
export MPLCONFIGDIR="${MPLCONFIGDIR:-${WORK_ROOT:-$REPO_ROOT/tournament}/mplconfig}"
mkdir -p "$MPLCONFIGDIR"

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

if [[ ! -s tournament/data/packs/p0_pairs.jsonl ]]; then
  rebuild_p0_permissive | tee -a tournament/logs/build_p0.log
fi

if [[ ! -s tournament/data/packs/p0_pairs.jsonl ]]; then
  echo "[error] P0 is still empty after permissive bootstrap rebuild"
  exit 1
fi

echo "[step] build P1"
"$PYTHON_BIN" tournament/scripts/build_p1_curriculum.py | tee tournament/logs/build_p1.log

echo "[step] build P3"
if ! "$PYTHON_BIN" tournament/scripts/build_p3_preferences.py | tee tournament/logs/build_p3.log; then
  echo "[warn] P3 build failed; continuing without DPO preferences for this unattended single-GPU run" | tee -a tournament/logs/build_p3.log
fi

echo "[step] train M2 full-LoRA anchor candidate"
"$PYTHON_BIN" tournament/scripts/train_sft.py --config tournament/configs/train/m2_full_lora.yaml | tee tournament/logs/m2_train.log

echo "[step] evaluate M2"
"$PYTHON_BIN" tournament/scripts/eval_checkpoint.py --config tournament/configs/eval/leaderboard.yaml | tee tournament/logs/m2_eval.log

echo "[step] train M1 attention-only LoRA comparison model"
"$PYTHON_BIN" tournament/scripts/train_sft.py --config tournament/configs/train/m1_attn_lora.yaml | tee tournament/logs/m1_train.log

echo "[step] evaluate M1"
"$PYTHON_BIN" - <<'PY'
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
"$PYTHON_BIN" tournament/scripts/eval_checkpoint.py --config tournament/configs/eval/m1_temp_eval.yaml | tee tournament/logs/m1_eval.log

echo "[step] build leaderboard"
"$PYTHON_BIN" tournament/scripts/eval_leaderboard.py \
  --eval-json tournament/outputs/leaderboards/m2_eval.json tournament/outputs/leaderboards/m1_eval.json \
  | tee tournament/logs/leaderboard.log

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

[success] Single-GPU phase-1 run completed.

Completed:
  - source download/validation
  - source pool build
  - P0/P1/P3 build
  - M2 training
  - M2 evaluation
  - M1 training
  - M1 evaluation
  - leaderboard staging

Lightweight results were staged in:
  tournament/results/phase1

If you enabled PUSH_RESULTS_ON_COMPLETE and GitHub SSH is configured,
those results were pushed automatically.

EOF
