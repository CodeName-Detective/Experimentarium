#!/usr/bin/env bash
# Run training through the Hydra entrypoint.
# Usage: bash scripts/train.sh +experiment=sanity_cpu optimizer.lr=3e-4
# Replay: bash scripts/train.sh --config-file outputs/run_configs/<run_id>.yaml --run-id replayed_run
set -euo pipefail
export PYTHONPATH="$(pwd):${PYTHONPATH:-}"
uv run python src/main.py "$@"
