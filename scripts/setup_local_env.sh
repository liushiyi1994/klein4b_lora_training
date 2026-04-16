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
print(payload["torch"]["version"])
print(payload["torch"]["torchvision"])
print(payload["torch"]["torchaudio"])
print(payload["torch"]["index_url"])
PY
)

REPO_URL="${LOCK_VALUES[0]}"
COMMIT="${LOCK_VALUES[1]}"
TORCH_VERSION="${LOCK_VALUES[2]}"
TORCHVISION_VERSION="${LOCK_VALUES[3]}"
TORCHAUDIO_VERSION="${LOCK_VALUES[4]}"
TORCH_INDEX_URL="${LOCK_VALUES[5]}"

cd "${REPO_ROOT}"
if ! python3.12 -m venv .venv; then
  echo "python3.12 -m venv failed; falling back to virtualenv." >&2
  # Fallback equivalent to: pip install virtualenv
  python3 -m pip install --user virtualenv
  rm -rf .venv
  python3 -m virtualenv -p "$(command -v python3.12)" .venv
fi
source "${REPO_ROOT}/.venv/bin/activate"
python -m pip install --upgrade pip setuptools wheel
pip install --no-cache-dir \
  "torch==${TORCH_VERSION}" \
  "torchvision==${TORCHVISION_VERSION}" \
  "torchaudio==${TORCHAUDIO_VERSION}" \
  --index-url "${TORCH_INDEX_URL}"
pip install -r requirements.txt
pip install -e .
