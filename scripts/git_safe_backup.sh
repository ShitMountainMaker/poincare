#!/usr/bin/env bash

set -euo pipefail

REMOTE_NAME="${GIT_BACKUP_REMOTE_NAME:-backup}"
REMOTE_URL="${GIT_BACKUP_REMOTE_URL:-https://github.com/ShitMountainMaker/poincare.git}"
TARGET_BRANCH="${GIT_BACKUP_BRANCH:-main}"

repo_root="$(git rev-parse --show-toplevel)"
cd "${repo_root}"

timestamp="$(date -u '+%Y-%m-%d %H:%M:%SZ')"
commit_message="${1:-backup: ${timestamp}}"

# Keep backup commits focused on code and docs, not local data or generated artifacts.
git add -u -- . ':(exclude)data' ':(exclude)history' ':(exclude)models' ':(exclude)outputs'

while IFS= read -r -d '' path; do
  git add -- "${path}"
done < <(git ls-files -o --exclude-standard -z -- . ':(exclude)data' ':(exclude)history' ':(exclude)models' ':(exclude)outputs')

if git diff --cached --quiet; then
  echo "No non-artifact changes to commit."
  exit 0
fi

git commit -m "${commit_message}"

if git remote get-url "${REMOTE_NAME}" >/dev/null 2>&1; then
  git push "${REMOTE_NAME}" "HEAD:${TARGET_BRANCH}"
else
  git push "${REMOTE_URL}" "HEAD:${TARGET_BRANCH}"
fi

git rev-parse --short HEAD
