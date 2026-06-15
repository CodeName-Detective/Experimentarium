"""Callback interfaces for extending training without editing the trainer.

Callbacks receive lifecycle events from the trainer and can add behavior such as
custom visual logging, profiler ranges, extra checkpoint artifacts, exception
reporting, or task-specific diagnostics.

Copy a small example into your project and pass an instance to ``Trainer`` with
``callbacks=[...]``. Current ``src/main.py`` does not build callbacks from config.

Example: log the learning rate after each optimizer step.

    from src.callbacks import Callback


    class LearningRateLogger(Callback):
        def on_batch_end(self, trainer, batch_idx: int, metrics: dict[str, float]) -> None:
            lr = float(trainer.optimizer.param_groups[0]['lr'])
            trainer.loggers.log_metrics({'train/lr': lr}, step=trainer.global_step)

Example: time a full training run.

    import time

    from src.callbacks import Callback


    class TrainingTimer(Callback):
        def on_train_start(self, trainer) -> None:
            self.started_at = time.perf_counter()

        def on_train_end(self, trainer) -> None:
            elapsed = time.perf_counter() - self.started_at
            trainer.loggers.log_metrics({'runtime/train_seconds': elapsed}, step=trainer.global_step)

Example: log every checkpoint as an artifact.

    from pathlib import Path

    from src.callbacks import Callback


    class CheckpointArtifactLogger(Callback):
        def on_checkpoint_saved(self, trainer, path: Path) -> None:
            trainer.loggers.log_artifact(
                path,
                name=f'checkpoint-epoch-{trainer.current_epoch}',
                artifact_type='checkpoint',
                metadata={'epoch': trainer.current_epoch, 'global_step': trainer.global_step},
            )

Example: record hook order in a test.

    from src.callbacks import Callback


    class RecordingCallback(Callback):
        def __init__(self) -> None:
            self.events = []

        def on_train_start(self, trainer) -> None:
            self.events.append('train_start')

        def on_validation_end(self, trainer, metrics: dict[str, float]) -> None:
            self.events.append(('validation_end', dict(metrics)))

        def on_train_end(self, trainer) -> None:
            self.events.append('train_end')

Timing notes:
    - ``on_batch_end`` runs after an optimizer step, not after every dataloader batch.
    - With gradient accumulation, ``on_batch_end`` runs only when the optimizer steps.
    - Gradients have already been cleared before ``on_batch_end``; do not use that hook
      to inspect gradients from the completed step.
    - ``on_validation_end`` runs before the trainer updates ``best_metric``.
    - ``on_train_end`` does not run when training exits through an exception.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from pathlib import Path


class Callback:
    """Base callback with no-op lifecycle hooks.

    Override only the hooks your callback needs. Hook methods receive the live
    trainer object, so they can read state such as ``trainer.model``,
    ``trainer.optimizer``, ``trainer.loggers``, ``trainer.current_epoch``, and
    ``trainer.global_step``.
    """

    def on_train_start(self, trainer: Any) -> None:
        """Handle the start of training."""

    def on_epoch_start(self, trainer: Any, epoch: int) -> None:
        """Handle the start of an epoch."""

    def on_batch_end(self, trainer: Any, batch_idx: int, metrics: dict[str, float]) -> None:
        """Handle completion of an optimization step."""

    def on_validation_end(self, trainer: Any, metrics: dict[str, float]) -> None:
        """Handle completion of validation."""

    def on_checkpoint_saved(self, trainer: Any, path: Path) -> None:
        """Handle a newly saved checkpoint."""

    def on_exception(self, trainer: Any, exc: BaseException) -> None:
        """Handle an exception raised during training."""

    def on_train_end(self, trainer: Any) -> None:
        """Handle the end of training."""


class CallbackList:
    """Small dispatcher for a list of callbacks."""

    def __init__(self, callbacks: list[Callback] | None = None) -> None:
        self.callbacks = callbacks or []

    def call(self, hook: str, *args: Any, **kwargs: Any) -> None:
        """Invoke a lifecycle hook on every callback that implements it."""
        for callback in self.callbacks:
            method = getattr(callback, hook, None)
            if method is not None:
                method(*args, **kwargs)
