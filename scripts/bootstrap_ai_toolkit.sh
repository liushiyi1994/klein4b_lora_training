#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOCK_JSON="${REPO_ROOT}/configs/ai_toolkit.lock.json"

readarray -t LOCK_VALUES < <(python3.12 - "${LOCK_JSON}" <<'PY'
from pathlib import Path
import json
import sys

payload = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
print(payload["repo_url"])
print(payload["commit"])
PY
)

REPO_URL="${LOCK_VALUES[0]}"
COMMIT="${LOCK_VALUES[1]}"
TARGET_DIR="${REPO_ROOT}/vendor/ai-toolkit"
VENV_ACTIVATE="${REPO_ROOT}/.venv/bin/activate"

if [ ! -f "${VENV_ACTIVATE}" ]; then
  echo "Missing virtual environment: ${VENV_ACTIVATE}. Run scripts/setup_local_env.sh first." >&2
  exit 1
fi

mkdir -p "${REPO_ROOT}/vendor"
if [ ! -d "${TARGET_DIR}/.git" ]; then
  git clone "${REPO_URL}" "${TARGET_DIR}"
fi

git -C "${TARGET_DIR}" fetch --all --tags
git -C "${TARGET_DIR}" checkout "${COMMIT}"

cd "${REPO_ROOT}"
source "${VENV_ACTIVATE}"
pip install -r "${TARGET_DIR}/requirements.txt"
