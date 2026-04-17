#!/usr/bin/env bash
set -euo pipefail

# Push lightweight phase-1 results back to GitHub.
#
# Usage:
#   export REPO_BRANCH=tournament-phase1
#   export RESULTS_DIR=tournament/results/phase1
#   bash tournament/scripts/push_results_to_github.sh

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")"/../.. && pwd)"
cd "$REPO_ROOT"

REPO_BRANCH="${REPO_BRANCH:-tournament-phase1}"
RESULTS_DIR="${RESULTS_DIR:-tournament/results/phase1}"
COMMIT_MESSAGE="${COMMIT_MESSAGE:-Update tournament phase-1 results}"

if [[ ! -d .git ]]; then
  echo "[error] Must run inside the git repository"
  exit 1
fi

mkdir -p "$RESULTS_DIR"

if [[ -z "$(find "$RESULTS_DIR" -type f -print -quit 2>/dev/null)" ]]; then
  echo "[error] No files found under $RESULTS_DIR"
  exit 1
fi

git add "$RESULTS_DIR"

if git diff --cached --quiet; then
  echo "[info] No staged results changes to push"
  exit 0
fi

git commit -m "$COMMIT_MESSAGE"
git push origin "$REPO_BRANCH"

echo "[success] Results pushed to GitHub"
