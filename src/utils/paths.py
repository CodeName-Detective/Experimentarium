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
RUNS_DIR = OUTPUT_DIR / 'runs'
EVALUATIONS_DIR = OUTPUT_DIR / 'evaluations'
RUN_CONFIG_DIR = OUTPUT_DIR / 'run_configs'
CHECKPOINT_DIR = OUTPUT_DIR / 'checkpoints'
LOG_DIR = OUTPUT_DIR / 'logs'
PREDICTION_DIR = OUTPUT_DIR / 'predictions'
PROFILE_DIR = OUTPUT_DIR / 'profiles'


def make_output_dirs(cfg: ConfigType | None = None) -> None:
    """Create configured output, checkpoint, log, and prediction directories."""
    for key, default in (
        ('run.output_dir', OUTPUT_DIR),
        ('run.run_dir', OUTPUT_DIR),
        ('run.evaluations_dir', EVALUATIONS_DIR),
        ('run.config_dir', RUN_CONFIG_DIR),
        ('checkpoint.dir', CHECKPOINT_DIR),
        ('run.log_dir', LOG_DIR),
        ('run.prediction_dir', PREDICTION_DIR),
        ('run.profile_dir', PROFILE_DIR),
    ):
        configured = cfg_get(cfg, key, default)
        Path(default if configured is None else configured).mkdir(parents=True, exist_ok=True)


def get_checkpoint_path(run_id: str, epoch: int, val_loss: float | None = None, trial_id: int = 1) -> Path:
    """Return the current run-scoped checkpoint path for an epoch."""
    del val_loss
    run_dir = RUNS_DIR / run_id / f'trial_{trial_id}' / 'checkpoints'
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir / f'epoch_{epoch:04d}.pt'


def get_log_path(run_id: str, trial_id: int = 1) -> Path:
    """Return the conventional log path for a run."""
    return RUNS_DIR / run_id / f'trial_{trial_id}' / 'logs' / 'train.log'


def get_prediction_path(run_id: str, split: str = 'test', trial_id: int = 1) -> Path:
    """Return the conventional prediction path for a dataset split."""
    return RUNS_DIR / run_id / f'trial_{trial_id}' / 'predictions' / f'{split}_predictions.json'
