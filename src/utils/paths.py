"""Repository path helpers.

Config owns runtime paths, while this module provides defaults and directory
creation helpers used by entrypoints and tests.

Typical usage:
    from src.utils.paths import make_output_dirs
    make_output_dirs(cfg)
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from src.utils.config import cfg_get

if TYPE_CHECKING:
    from src.utils.types import ConfigType

BASE_DIR = Path(__file__).resolve().parents[2]
SRC_DIR = BASE_DIR / 'src'
CONFIG_DIR = BASE_DIR / 'configs'
DATA_DIR = BASE_DIR / 'data'
OUTPUT_DIR = BASE_DIR / 'outputs'
CHECKPOINT_DIR = OUTPUT_DIR / 'checkpoints'
LOG_DIR = OUTPUT_DIR / 'logs'
PREDICTION_DIR = OUTPUT_DIR / 'predictions'
PROFILE_DIR = OUTPUT_DIR / 'profiles'


def make_output_dirs(cfg: ConfigType | None = None) -> None:
    """Create configured output, checkpoint, log, and prediction directories."""
    for key, default in (
        ('run.output_dir', OUTPUT_DIR),
        ('run.run_dir', cfg_get(cfg, 'run.run_dir', OUTPUT_DIR)),
        ('run.config_dir', cfg_get(cfg, 'run.config_dir', OUTPUT_DIR / 'run_configs')),
        ('checkpoint.dir', CHECKPOINT_DIR),
        ('run.log_dir', LOG_DIR),
        ('run.prediction_dir', PREDICTION_DIR),
        ('run.profile_dir', PROFILE_DIR),
    ):
        Path(cfg_get(cfg, key, default)).mkdir(parents=True, exist_ok=True)


def get_checkpoint_path(run_id: str, epoch: int, val_loss: float) -> Path:
    """Return the conventional checkpoint path for an epoch."""
    run_dir = CHECKPOINT_DIR / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir / f'epoch_{epoch:03d}_val_loss_{val_loss:.4f}.pt'


def get_log_path(run_id: str) -> Path:
    """Return the conventional log path for a run."""
    return LOG_DIR / f'{run_id}.log'


def get_prediction_path(run_id: str, split: str = 'test') -> Path:
    """Return the conventional prediction path for a dataset split."""
    return PREDICTION_DIR / f'{run_id}_{split}.json'
