#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")"/../.. && pwd)"
cd "$REPO_ROOT"

SESSION_NAME="${SESSION_NAME:-tournament}"
MASTER_LOG="${MASTER_LOG:-tournament/logs/run_phase1_master.log}"

if ! command -v tmux >/dev/null 2>&1; then
  echo "[error] tmux is not installed"
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

mkdir -p "$(dirname "$MASTER_LOG")"

TMUX_CMD=$(cat <<EOF
cd "$REPO_ROOT" && \
export PATH="$HOME/bin:\$PATH" && \
export WORK_ROOT="${WORK_ROOT:-$HOME/stat4830}" && \
export HF_HOME="${HF_HOME:-${WORK_ROOT:-$HOME/stat4830}/hf}" && \
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${WORK_ROOT:-$HOME/stat4830}/hf/transformers}" && \
export WANDB_DIR="${WANDB_DIR:-${WORK_ROOT:-$HOME/stat4830}/wandb}" && \
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-${WORK_ROOT:-$HOME/stat4830}/pip-cache}" && \
export TMPDIR="${TMPDIR:-${WORK_ROOT:-$HOME/stat4830}/tmp}" && \
export MPLCONFIGDIR="${MPLCONFIGDIR:-${WORK_ROOT:-$HOME/stat4830}/mplconfig}" && \
mkdir -p "\$HF_HOME" "\$TRANSFORMERS_CACHE" "\$WANDB_DIR" "\$PIP_CACHE_DIR" "\$TMPDIR" "\$MPLCONFIGDIR" && \
bash tournament/scripts/run_phase1_remote.sh 2>&1 | tee "$MASTER_LOG"
EOF
)

if tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
  echo "[error] tmux session '$SESSION_NAME' already exists"
  echo "[hint] Attach with: tmux attach -t $SESSION_NAME"
  exit 1
fi

tmux new-session -d -s "$SESSION_NAME" "$TMUX_CMD"

cat <<EOF
[success] Detached single-GPU phase-1 run started.

tmux session:
  $SESSION_NAME

Master log:
  $MASTER_LOG

Useful commands:
  tmux attach -t $SESSION_NAME
  tail -f $MASTER_LOG
  tail -f tournament/logs/m2_train.log
  tail -f tournament/logs/m1_train.log
  watch -n 2 nvidia-smi
EOF
