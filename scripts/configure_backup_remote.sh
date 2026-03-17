#!/usr/bin/env bash

set -euo pipefail

REMOTE_NAME="${1:-backup}"
REMOTE_URL="${2:-https://github.com/ShitMountainMaker/poincare.git}"

repo_root="$(git rev-parse --show-toplevel)"
cd "${repo_root}"

if git remote get-url "${REMOTE_NAME}" >/dev/null 2>&1; then
  git remote set-url "${REMOTE_NAME}" "${REMOTE_URL}"
else
  git remote add "${REMOTE_NAME}" "${REMOTE_URL}"
fi

git remote -v
