"""Callback interfaces for extending training without editing the trainer.

Callbacks receive lifecycle events from the trainer and can add behavior such as
custom visual logging, EMA updates, profiler ranges, extra checkpoint artifacts,
or task-specific diagnostics.

Typical usage:
    from src.callbacks import Callback

    class MyCallback(Callback):
        def on_validation_end(self, trainer, metrics):
            trainer.loggers.log_metrics({'custom/value': 1.0}, trainer.global_step)
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from pathlib import Path


class Callback:
    """Base callback with no-op lifecycle hooks."""

    def on_train_start(self, trainer: Any) -> None:
        """Handle the start of training."""

    def on_epoch_start(self, trainer: Any, epoch: int) -> None:
        """Handle the start of an epoch."""

    def on_batch_end(self, trainer: Any, batch_idx: int, metrics: dict[str, float]) -> None:
        """Handle completion of an optimization batch."""

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
