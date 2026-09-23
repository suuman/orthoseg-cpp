#!/usr/bin/env bash
set -euo pipefail

ORTHOSEG_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
MONAI_REPO="${MONAI_REPO:-/run/media/suman/Data/monai}"
MONAI_CONFIG="${MONAI_CONFIG:-$MONAI_REPO/configs/management.yaml}"
MONAI_BACKEND_URL="${MONAI_BACKEND_URL:-http://127.0.0.1:8000}"
ORTHOSEG_BIN="${ORTHOSEG_BIN:-$ORTHOSEG_ROOT/build/orthoseg}"
MONAI_STARTUP_TIMEOUT="${MONAI_STARTUP_TIMEOUT:-90}"

if [[ "${1:-}" == "--help" ]]; then
    cat <<'HELP'
Usage: scripts/launch_with_monai.sh [OrthoSeg arguments]

Starts the local MONAI backend if needed, waits for /health, and opens OrthoSeg.
The launcher stops only a backend that it started.

Environment overrides:
  MONAI_REPO             Backend checkout (default: /run/media/suman/Data/monai)
  MONAI_CONFIG           Backend config (default: MONAI_REPO/configs/management.yaml)
  MONAI_BACKEND_URL      Local URL (default: http://127.0.0.1:8000)
  ORTHOSEG_BIN           OrthoSeg executable (default: build/orthoseg)
  MONAI_STARTUP_TIMEOUT  Seconds to wait for the backend (default: 90)
HELP
    exit 0
fi

for command_name in curl python3; do
    command -v "$command_name" >/dev/null 2>&1 || {
        echo "Missing required command: $command_name" >&2
        exit 1
    }
done
[[ -x "$ORTHOSEG_BIN" ]] || { echo "OrthoSeg executable not found: $ORTHOSEG_BIN. Build it with CMake first." >&2; exit 1; }
[[ -x "$MONAI_REPO/scripts/run_server_models.sh" ]] || { echo "MONAI models server script not found in $MONAI_REPO" >&2; exit 1; }
[[ -f "$MONAI_CONFIG" ]] || { echo "MONAI config not found: $MONAI_CONFIG" >&2; exit 1; }
[[ "$MONAI_STARTUP_TIMEOUT" =~ ^[1-9][0-9]*$ ]] || { echo "MONAI_STARTUP_TIMEOUT must be a positive integer" >&2; exit 1; }

python3 - "$MONAI_BACKEND_URL" <<'PY'
import sys
from urllib.parse import urlsplit

url = urlsplit(sys.argv[1])
try:
    valid = url.scheme == 'http' and url.hostname in {'localhost', '127.0.0.1', '::1'} and url.port is not None
except ValueError:
    valid = False
if not valid or url.username or url.password or url.query or url.fragment:
    sys.exit('MONAI_BACKEND_URL must be a local http URL with an explicit port')
PY

MONAI_BACKEND_URL="${MONAI_BACKEND_URL%/}"
export MONAI_BACKEND_URL

health() {
    curl --silent --show-error --fail --noproxy '*' --connect-timeout 1 --max-time 3 \
        "$MONAI_BACKEND_URL/health" 2>/dev/null |
        python3 -c 'import json,sys; value=json.load(sys.stdin); status=value.get("status"); print("production=" + str(bool(value.get("model_loaded"))).lower() + ", SAM2=" + str(bool(value.get("sam2_model_loaded"))).lower() + ", nnUNet=" + str(bool(value.get("nnunet_model_loaded"))).lower()); sys.exit(0 if status == "ok" else 1)' 2>/dev/null
}

server_pid=""
cleanup() {
    if [[ -n "$server_pid" ]]; then
        kill "$server_pid" 2>/dev/null || true
        wait "$server_pid" 2>/dev/null || true
    fi
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

if model_state="$(health)"; then
    echo "Using existing MONAI backend at $MONAI_BACKEND_URL ($model_state)."
else
    echo "Starting MONAI backend from $MONAI_REPO with $MONAI_CONFIG..."
    "$MONAI_REPO/scripts/run_server_models.sh" --config "$MONAI_CONFIG" &
    server_pid=$!
    ready=false
    for ((attempt=0; attempt<MONAI_STARTUP_TIMEOUT; attempt++)); do
        if model_state="$(health)"; then
            ready=true
            break
        fi
        if ! kill -0 "$server_pid" 2>/dev/null; then
            echo "MONAI backend exited during startup. Check its output above." >&2
            exit 1
        fi
        sleep 1
    done
    if [[ "$ready" != true ]]; then
        echo "MONAI backend did not become ready within $MONAI_STARTUP_TIMEOUT seconds." >&2
        exit 1
    fi
    echo "MONAI backend ready at $MONAI_BACKEND_URL ($model_state)."
fi

if [[ "$model_state" == "production=false, SAM2=false, nnUNet=false" ]]; then
    echo "Neither MONAI inference model is loaded; check the backend output." >&2
fi
if ! curl --silent --show-error --fail --noproxy '*' --connect-timeout 1 --max-time 3 \
    "$MONAI_BACKEND_URL/management/status" >/dev/null 2>&1; then
    echo "This backend does not grant model management access; its UI button will stay disabled." >&2
fi

export ORTHOSEG_ENABLE_MODEL_MANAGEMENT=1
"$ORTHOSEG_BIN" "$@"
