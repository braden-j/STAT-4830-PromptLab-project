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

DEFAULT_WORK_ROOT="/mnt/shared/stat4830"
FALLBACK_WORK_ROOT="$HOME/stat4830"
WORK_ROOT="${WORK_ROOT:-}"
REPO_URL="${REPO_URL:-https://github.com/braden-j/STAT-4830-PromptLab-project.git}"
REPO_BRANCH="${REPO_BRANCH:-tournament-phase1}"
REPO_DIR="${REPO_DIR:-}"
GENERATE_GITHUB_PUSH_KEY="${GENERATE_GITHUB_PUSH_KEY:-1}"
GITHUB_PUSH_KEY_PATH="${GITHUB_PUSH_KEY_PATH:-$HOME/.ssh/id_ed25519_stat4830_results}"
GITHUB_PUSH_KEY_COMMENT="${GITHUB_PUSH_KEY_COMMENT:-prime-stat4830-results}"
GITHUB_PUSH_REMOTE="${GITHUB_PUSH_REMOTE:-git@github.com:braden-j/STAT-4830-PromptLab-project.git}"

if command -v python3 >/dev/null 2>&1; then
  PYTHON_BIN="python3"
elif command -v python >/dev/null 2>&1; then
  PYTHON_BIN="python"
else
  echo "[error] Neither python3 nor python is available" >&2
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

choose_work_root() {
  local candidate

  if [[ -n "$WORK_ROOT" ]]; then
    if mkdir -p "$WORK_ROOT" 2>/dev/null; then
      echo "$WORK_ROOT"
      return 0
    fi
    echo "[error] WORK_ROOT is set to '$WORK_ROOT' but is not writable" >&2
    echo "[error] Choose a writable path, for example: export WORK_ROOT=\$HOME/stat4830" >&2
    return 1
  fi

  for candidate in "$DEFAULT_WORK_ROOT" "$FALLBACK_WORK_ROOT" "/tmp/stat4830"; do
    if mkdir -p "$candidate" 2>/dev/null; then
      echo "$candidate"
      return 0
    fi
  done

  echo "[error] Could not find a writable working directory" >&2
  return 1
}

WORK_ROOT="$(choose_work_root)"
REPO_DIR="${REPO_DIR:-$WORK_ROOT/STAT-4830-PromptLab-project}"
echo "[info] Using WORK_ROOT=$WORK_ROOT"

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

"$PYTHON_BIN" -m pip install --upgrade pip

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
CURRENT_ORIGIN_URL="$(git remote get-url origin 2>/dev/null || true)"
if [[ "$CURRENT_ORIGIN_URL" == git@github.com:* ]]; then
  echo "[warn] origin currently uses SSH; switching to HTTPS for bootstrap fetch/pull"
  git remote set-url origin "$REPO_URL"
fi

git fetch --all --prune
git checkout "$REPO_BRANCH"
git pull --ff-only origin "$REPO_BRANCH" || true

if [[ "$GENERATE_GITHUB_PUSH_KEY" == "1" ]]; then
  git remote set-url origin "$GITHUB_PUSH_REMOTE"
fi

"$PYTHON_BIN" -m pip install -e .[tournament]

"$PYTHON_BIN" tournament/scripts/verify_access.py

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
