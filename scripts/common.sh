#!/usr/bin/env bash
set -euo pipefail
PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"
if [[ -n "${XRAY_PYTHON:-}" ]]; then
    PYTHON_BIN="$XRAY_PYTHON"
elif [[ -n "${VIRTUAL_ENV:-}" ]]; then
    PYTHON_BIN="$VIRTUAL_ENV/bin/python"
elif [[ -x "$PROJECT_ROOT/.venv/bin/python" ]]; then
    PYTHON_BIN="$PROJECT_ROOT/.venv/bin/python"
else
    PYTHON_BIN="python3"
fi
export PYTHONPATH="$PROJECT_ROOT"
"$PYTHON_BIN" -c 'import torch, monai, fastapi' || { echo 'Dependencies missing. See README installation instructions.' >&2; exit 1; }
