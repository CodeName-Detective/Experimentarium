"""Console-script wrappers for script-style Hydra entrypoints."""

from __future__ import annotations

import runpy
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _run_script(path: Path) -> None:
    """Execute a repository script with argv[0] set to that script path."""
    sys.argv[0] = str(path)
    runpy.run_path(str(path), run_name='__main__')


def train() -> None:
    """Run the main training/evaluation entrypoint as a console script."""
    _run_script(ROOT / 'src' / 'main.py')


def sanity() -> None:
    """Run the sanity-check entrypoint as a console script."""
    _run_script(ROOT / 'scripts' / 'run_sanity.py')
