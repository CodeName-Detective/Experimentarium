"""Reusable evaluator independent of the trainer.

Evaluator runs a task over any dataloader for validation, test, or standalone
checkpoint evaluation. It keeps evaluation free of optimizer/training concerns
and reduces weighted metric totals across DDP ranks when distributed execution is active.

Typical usage:
    evaluator = Evaluator(model, task, device='cpu')
    metrics = evaluator.evaluate(loaders['val'], prefix='val')
    records = evaluator.predict(loaders['test'])
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any

import torch

from src.engine.precision import precision_autocast
from src.runtime.distributed import reduce_sum_count, unwrap_model

if TYPE_CHECKING:
    from collections.abc import Iterable

    from src.tasks import BaseTask

Batch = dict[str, Any]


def _move_value_to_device(value: Any, device: torch.device | str) -> Any:
    if torch.is_tensor(value):
        return value.to(device, non_blocking=True)
    if isinstance(value, dict):
        return {key: _move_value_to_device(item, device) for key, item in value.items()}
    if isinstance(value, list):
        return [_move_value_to_device(item, device) for item in value]
    if isinstance(value, tuple):
        return tuple(_move_value_to_device(item, device) for item in value)
    return value


def move_to_device(batch: Batch, device: torch.device | str) -> Batch:
    """Recursively move tensors in a batch dictionary to a device."""
    return {key: _move_value_to_device(value, device) for key, value in batch.items()}


class Evaluator:
    """Task-aware evaluator for validation, test, and prediction."""

    def __init__(
        self,
        model: torch.nn.Module,
        task: BaseTask,
        device: torch.device | str = 'cpu',
        precision: str = 'fp32',
    ) -> None:
        self.model = model
        self.task = task
        self.device = torch.device(device)
        self.precision = precision

    @torch.no_grad()
    def evaluate(self, loader: Iterable[Batch], prefix: str = 'val', limit_batches: Any = None) -> dict[str, float]:
        """Evaluate a loader and return prefixed aggregate metrics without mutating training state."""
        was_training = self.model.training
        metric_state = self.task.metric_state_dict()
        evaluation_model = unwrap_model(self.model)
        try:
            self.model.eval()
            self.task.reset_metrics()
            total_loss = 0.0
            loss_count = 0.0
            max_batches = _batch_limit(loader, limit_batches)
            for batch_idx, batch in enumerate(loader, start=1):
                if max_batches is not None and batch_idx > max_batches:
                    break
                batch = move_to_device(batch, self.device)
                with precision_autocast(self.device, self.precision):
                    result = self.task.step(evaluation_model, batch, stage=prefix)
                batch_size = int(result.targets.shape[0]) if result.targets is not None else 1
                loss_weight = float(result.loss_weight if result.loss_weight is not None else batch_size)
                if result.loss is not None:
                    total_loss += float(result.loss.detach().cpu()) * loss_weight
                    loss_count += loss_weight
            metrics = self.task.compute_metrics_distributed(device=self.device)
            global_loss, global_loss_count = reduce_sum_count(total_loss, loss_count, device=self.device)
            if global_loss_count:
                metrics['loss'] = global_loss / global_loss_count
            return {f'{prefix}/{key}': value for key, value in metrics.items()}
        finally:
            self.task.load_metric_state_dict(metric_state)
            self.model.train(was_training)

    @torch.no_grad()
    def predict(
        self, loader: Iterable[Batch], limit: int | None = None, limit_batches: Any = None
    ) -> list[dict[str, Any]]:
        """Generate serializable prediction records without changing model mode."""
        was_training = self.model.training
        prediction_model = unwrap_model(self.model)
        try:
            self.model.eval()
            records: list[dict[str, Any]] = []
            max_batches = _batch_limit(loader, limit_batches)
            for batch_idx, batch in enumerate(loader, start=1):
                if max_batches is not None and batch_idx > max_batches:
                    break
                batch = move_to_device(batch, self.device)
                with precision_autocast(self.device, self.precision):
                    outputs = prediction_model(batch)
                records.extend(self.task.predict_records(outputs, batch))
                if limit is not None and len(records) >= limit:
                    return records[:limit]
            return records
        finally:
            self.model.train(was_training)


def _batch_limit(loader: Any, configured: Any) -> int | None:
    if configured is None:
        return None
    if isinstance(configured, str):
        normalized = configured.lower()
        if normalized in {'none', 'null'}:
            return None
        if any(marker in normalized for marker in ('.', 'e')):
            return _fractional_batch_limit(loader, float(configured))
        return max(0, int(configured))
    if isinstance(configured, int) and not isinstance(configured, bool):
        return max(0, configured)
    return _fractional_batch_limit(loader, float(configured))


def _fractional_batch_limit(loader: Any, value: float) -> int:
    if 0.0 < value <= 1.0:
        if not hasattr(loader, '__len__'):
            raise ValueError('Fractional batch limits require a sized dataloader')
        return max(1, math.ceil(len(loader) * value))
    return max(0, int(value))
