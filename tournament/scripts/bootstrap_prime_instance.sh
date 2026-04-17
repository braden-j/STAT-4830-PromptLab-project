#!/usr/bin/env bash
set -euo pipefail

# Bootstrap a fresh Prime H100 instance for the tournament workflow.
#
# Usage:
#   export HF_TOKEN=...
#   export WANDB_API_KEY=...
#   export WORK_ROOT=/mnt/shared/stat4830
#   export REPO_URL=https://github.com/braden-j/STAT-4830-PromptLab-project.git
#   export REPO_BRANCH=tournament-phase1
#   export GENERATE_GITHUB_PUSH_KEY=1
#   bash tournament/scripts/bootstrap_prime_instance.sh

WORK_ROOT="${WORK_ROOT:-/mnt/shared/stat4830}"
REPO_URL="${REPO_URL:-https://github.com/braden-j/STAT-4830-PromptLab-project.git}"
REPO_BRANCH="${REPO_BRANCH:-tournament-phase1}"
REPO_DIR="${REPO_DIR:-$WORK_ROOT/STAT-4830-PromptLab-project}"
GENERATE_GITHUB_PUSH_KEY="${GENERATE_GITHUB_PUSH_KEY:-1}"
GITHUB_PUSH_KEY_PATH="${GITHUB_PUSH_KEY_PATH:-$HOME/.ssh/id_ed25519_stat4830_results}"
GITHUB_PUSH_KEY_COMMENT="${GITHUB_PUSH_KEY_COMMENT:-prime-stat4830-results}"
GITHUB_PUSH_REMOTE="${GITHUB_PUSH_REMOTE:-git@github.com:braden-j/STAT-4830-PromptLab-project.git}"

if [[ -z "${HF_TOKEN:-}" ]]; then
  echo "[error] HF_TOKEN is not set"
  exit 1
fi

if [[ -z "${WANDB_API_KEY:-}" ]]; then
  echo "[error] WANDB_API_KEY is not set"
  exit 1
fi

echo "[info] Using WORK_ROOT=$WORK_ROOT"
mkdir -p "$WORK_ROOT"

export HF_HOME="${HF_HOME:-$WORK_ROOT/hf}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$WORK_ROOT/hf/transformers}"
export WANDB_DIR="${WANDB_DIR:-$WORK_ROOT/wandb}"
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-$WORK_ROOT/pip-cache}"
export TMPDIR="${TMPDIR:-$WORK_ROOT/tmp}"

mkdir -p "$HF_HOME" "$TRANSFORMERS_CACHE" "$WANDB_DIR" "$PIP_CACHE_DIR" "$TMPDIR"

if command -v apt-get >/dev/null 2>&1; then
  sudo apt-get update
  sudo apt-get install -y git tmux
fi

python3 -m pip install --upgrade pip

mkdir -p "$HOME/.ssh"
chmod 700 "$HOME/.ssh"

if [[ "$GENERATE_GITHUB_PUSH_KEY" == "1" ]]; then
  if [[ ! -f "$GITHUB_PUSH_KEY_PATH" ]]; then
    ssh-keygen -t ed25519 -C "$GITHUB_PUSH_KEY_COMMENT" -f "$GITHUB_PUSH_KEY_PATH" -N ""
  fi

  if [[ ! -f "$HOME/.ssh/config" ]] || ! grep -q "$GITHUB_PUSH_KEY_PATH" "$HOME/.ssh/config"; then
    cat <<EOF >> "$HOME/.ssh/config"
Host github.com
  HostName github.com
  User git
  IdentityFile $GITHUB_PUSH_KEY_PATH
  IdentitiesOnly yes

EOF
  fi
  chmod 600 "$HOME/.ssh/config"
fi

if [[ ! -d "$REPO_DIR/.git" ]]; then
  git clone "$REPO_URL" "$REPO_DIR"
fi

cd "$REPO_DIR"
git fetch --all --prune
git checkout "$REPO_BRANCH"
git pull --ff-only origin "$REPO_BRANCH" || true

if [[ "$GENERATE_GITHUB_PUSH_KEY" == "1" ]]; then
  git remote set-url origin "$GITHUB_PUSH_REMOTE"
fi

python3 -m pip install -e .[tournament]

python tournament/scripts/verify_access.py

cat <<EOF

[success] Prime instance bootstrap complete.

Repo dir: $REPO_DIR
Work root: $WORK_ROOT

EOF

if [[ "$GENERATE_GITHUB_PUSH_KEY" == "1" ]]; then
  cat <<EOF
GitHub deploy/public key path:
  ${GITHUB_PUSH_KEY_PATH}.pub

Add this public key to GitHub with write access:
-----BEGIN STAT4830 GITHUB PUBLIC KEY-----
$(cat "${GITHUB_PUSH_KEY_PATH}.pub")
-----END STAT4830 GITHUB PUBLIC KEY-----

After you add it on GitHub, test:
  ssh -T git@github.com
  git -C "$REPO_DIR" push --dry-run origin "$REPO_BRANCH"

EOF
fi

cat <<EOF
Next:
  cd "$REPO_DIR"
  bash tournament/scripts/run_phase1_dual_gpu.sh

EOF
