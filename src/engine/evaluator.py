"""Reusable evaluator independent of the trainer.

Evaluator runs a task over any dataloader for validation, test, or standalone
checkpoint evaluation. It keeps evaluation free of optimizer/training concerns
and averages numeric metrics across DDP ranks when distributed training is active.

Typical usage:
    evaluator = Evaluator(model, task, device='cpu')
    metrics = evaluator.evaluate(loaders['val'], prefix='val')
    records = evaluator.predict(loaders['test'])
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch

from src.runtime.distributed import mean_dict

if TYPE_CHECKING:
    from collections.abc import Iterable

    from src.tasks import BaseTask

Batch = dict[str, Any]


def move_to_device(batch: Batch, device: torch.device | str) -> Batch:
    """Move a flat tensor batch dictionary to a device."""
    moved: Batch = {}
    for key, value in batch.items():
        moved[key] = value.to(device, non_blocking=True) if torch.is_tensor(value) else value
    return moved


class Evaluator:
    """Task-aware evaluator for validation, test, and prediction."""

    def __init__(self, model: torch.nn.Module, task: BaseTask, device: torch.device | str = 'cpu') -> None:
        self.model = model
        self.task = task
        self.device = torch.device(device)

    @torch.no_grad()
    def evaluate(self, loader: Iterable[Batch], prefix: str = 'val') -> dict[str, float]:
        """Evaluate a loader and return prefixed aggregate metrics."""
        self.model.eval()
        self.task.reset_metrics()
        total_loss = 0.0
        total_count = 0
        for batch in loader:
            batch = move_to_device(batch, self.device)
            result = self.task.step(self.model, batch, stage=prefix)
            batch_size = int(result.targets.shape[0]) if result.targets is not None else 1
            if result.loss is not None:
                total_loss += float(result.loss.detach().cpu()) * batch_size
            total_count += batch_size
        metrics = self.task.compute_metrics()
        if total_count:
            metrics['loss'] = total_loss / total_count
        metrics = mean_dict(metrics, device=self.device)
        return {f'{prefix}/{key}': value for key, value in metrics.items()}

    @torch.no_grad()
    def predict(self, loader: Iterable[Batch], limit: int | None = None) -> list[dict[str, Any]]:
        """Generate serializable prediction records from a loader."""
        self.model.eval()
        records: list[dict[str, Any]] = []
        for batch in loader:
            batch = move_to_device(batch, self.device)
            outputs = self.model(batch)
            records.extend(self.task.predict_records(outputs, batch))
            if limit is not None and len(records) >= limit:
                return records[:limit]
        return records
