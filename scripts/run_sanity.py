"""Run canonical framework sanity checks from the command line.

Use this script immediately after cloning the repo or moving it to a new machine.
It composes the Hydra config, imports all registry modules, checks dependency and
runtime health, validates directories, and runs a tiny data/model/optimizer smoke
test before you spend time on a real experiment.

Examples:
    uv run python scripts/run_sanity.py
    uv run python scripts/run_sanity.py +experiment=sanity_cpu
    uv run python scripts/run_sanity.py run.device=cuda sanity.strict=true
"""
# ruff: noqa: E402

from __future__ import annotations

import sys
from pathlib import Path
from typing import TYPE_CHECKING

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import hydra

if TYPE_CHECKING:
    from omegaconf import DictConfig

from src.utils.config import cfg_get
from src.utils.paths import make_output_dirs
from src.utils.run import prepare_run
from src.utils.sanity import run_sanity_checks


@hydra.main(config_path='../configs', config_name='config', version_base='1.3')
def main(cfg: DictConfig) -> None:
    """Run the configured training or evaluation workflow."""
    prepare_run(cfg)
    make_output_dirs(cfg)
    strict = bool(cfg_get(cfg, 'sanity.strict', True))
    run_sanity_checks(cfg, strict=strict)


if __name__ == '__main__':
    main()
