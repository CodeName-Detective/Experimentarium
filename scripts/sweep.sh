#!/usr/bin/env bash
# Launch a W&B sweep from configs/sweep.yaml.
# Usage: bash scripts/sweep.sh
set -euo pipefail
export PYTHONPATH="$(pwd):${PYTHONPATH:-}"
command -v wandb >/dev/null 2>&1 || { echo "wandb is not installed; run: uv sync --extra tracking" >&2; exit 1; }
OUTPUT=$(wandb sweep configs/sweep.yaml)
echo "$OUTPUT"
SWEEP_ID=$(SWEEP_OUTPUT="$OUTPUT" python - <<'SWEEPPY'
import os
import re
text = os.environ['SWEEP_OUTPUT']
match = re.search(r'sweep ID:\s*(\S+)', text)
if not match:
    raise SystemExit('Could not parse sweep ID from wandb output')
print(match.group(1))
SWEEPPY
)
echo "Sweep ID: $SWEEP_ID"
wandb agent "$SWEEP_ID"
