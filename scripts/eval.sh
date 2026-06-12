#!/usr/bin/env bash
# Evaluate a checkpoint through the Hydra entrypoint.
# Usage: bash scripts/eval.sh outputs/checkpoints/best.pt
set -euo pipefail
export PYTHONPATH="$(pwd):${PYTHONPATH:-}"
CHECKPOINT=${1:?Provide checkpoint path as first argument}
shift || true
uv run python src/main.py run.mode=eval checkpoint.resume="$CHECKPOINT" "$@"
