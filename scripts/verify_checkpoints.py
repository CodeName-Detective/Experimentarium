"""Verify checkpoint manifest entries and best/last selector checksums."""
# ruff: noqa: E402

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.utils.checkpoint import CheckpointManager


def main() -> None:
    """Run the checkpoint verification CLI."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('checkpoint_dir', type=Path, help='Directory containing manifest.json and checkpoint files')
    args = parser.parse_args()
    checkpoint_dir = args.checkpoint_dir
    if not checkpoint_dir.exists():
        print(json.dumps({'ok': False, 'issues': [f'missing checkpoint directory: {checkpoint_dir}']}, indent=2))
        raise SystemExit(1)
    manager = CheckpointManager(checkpoint_dir)
    issues = manager.verify()
    print(json.dumps({'ok': not issues, 'issues': issues}, indent=2))
    raise SystemExit(1 if issues else 0)


if __name__ == '__main__':
    main()
