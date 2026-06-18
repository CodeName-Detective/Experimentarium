"""Built-in callbacks registered for config-driven trainer extensions."""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from pathlib import Path

from src.callbacks.base import Callback
from src.utils.config import cfg_get
from src.utils.registry import register_callback


@register_callback('learning_rate_logger')
class LearningRateLogger(Callback):
    """Log optimizer learning rates every N optimizer steps."""

    def __init__(self, cfg: Any | None = None) -> None:
        self.every_n_steps = max(1, int(cfg_get(cfg, 'every_n_steps', 1)))
        self.metric_prefix = str(cfg_get(cfg, 'metric_prefix', 'train/lr'))

    def on_batch_end(self, trainer: Any, batch_idx: int, metrics: dict[str, float]) -> None:
        """Add current optimizer learning rates to step metrics."""
        if trainer.global_step % self.every_n_steps != 0:
            return
        for index, group in enumerate(trainer.optimizer.param_groups):
            metrics[f'{self.metric_prefix}/group_{index}'] = float(group.get('lr', 0.0))


@register_callback('grad_norm_logger')
class GradNormLogger(Callback):
    """Log gradient norm values computed by the trainer."""

    def __init__(self, cfg: Any | None = None) -> None:
        self.every_n_steps = max(1, int(cfg_get(cfg, 'every_n_steps', 1)))
        self.metric_name = str(cfg_get(cfg, 'metric_name', 'train/grad_norm'))

    def on_batch_end(self, trainer: Any, batch_idx: int, metrics: dict[str, float]) -> None:
        """Log the trainer-computed gradient norm when available."""
        if trainer.global_step % self.every_n_steps != 0:
            return
        value = metrics.get(self.metric_name)
        if value is not None:
            trainer.loggers.log_metrics({self.metric_name: value}, step=trainer.global_step)


@register_callback('training_timer')
class TrainingTimer(Callback):
    """Log wall-clock training time at the end of training."""

    def __init__(self, cfg: Any | None = None) -> None:
        self.metric_name = str(cfg_get(cfg, 'metric_name', 'runtime/train_seconds'))
        self._started_at = 0.0

    def on_train_start(self, trainer: Any) -> None:
        """Start wall-clock timing."""
        self._started_at = time.perf_counter()

    def on_train_end(self, trainer: Any) -> None:
        """Log elapsed wall-clock training time."""
        elapsed = time.perf_counter() - self._started_at
        trainer.loggers.log_metrics({self.metric_name: elapsed}, step=trainer.global_step)


@register_callback('checkpoint_artifact_logger')
class CheckpointArtifactLogger(Callback):
    """Log every saved checkpoint as an artifact through configured loggers."""

    def __init__(self, cfg: Any | None = None) -> None:
        self.artifact_type = str(cfg_get(cfg, 'artifact_type', 'checkpoint'))

    def on_checkpoint_saved(self, trainer: Any, path: Path) -> None:
        """Log a saved checkpoint through configured logger backends."""
        trainer.loggers.log_artifact(
            path,
            name=f'checkpoint-epoch-{trainer.current_epoch}',
            artifact_type=self.artifact_type,
            metadata={'epoch': trainer.current_epoch, 'global_step': trainer.global_step},
        )
