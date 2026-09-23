#!/usr/bin/env bash
set -euo pipefail

# The workstation's deepnet environment has both SAM2 and nnUNet v2 installed.
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
if [[ -z "${XRAY_PYTHON:-}" && -x /home/suman/deepnet/bin/python ]]; then
    export XRAY_PYTHON=/home/suman/deepnet/bin/python
fi
exec "$SCRIPT_DIR/run_server.sh" "$@"
