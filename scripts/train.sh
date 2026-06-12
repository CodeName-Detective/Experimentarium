#!/usr/bin/env bash
# Run training through the Hydra entrypoint.
# Usage: bash scripts/train.sh +experiment=sanity_cpu optimizer.lr=3e-4
set -euo pipefail
export PYTHONPATH="$(pwd):${PYTHONPATH:-}"
uv run python src/main.py "$@"
