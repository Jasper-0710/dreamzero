#!/usr/bin/env bash
# Use the active server conda environment; never activate a machine-specific env here.
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
exec "${PYTHON_BIN:-python}" -u "$SCRIPT_DIR/launch_pillow_training.py" "$@"
