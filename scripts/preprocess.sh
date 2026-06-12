#!/usr/bin/env bash
# Generate toy tensor-file splits for testing data=tensor_file.
# Usage: bash scripts/preprocess.sh [--force]
set -euo pipefail
export PYTHONPATH="$(pwd):${PYTHONPATH:-}"
uv run python src/data/preprocess.py "$@"
