"""Learning-to-rank task implementation."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch

from src.losses import build_loss
from src.tasks.task import BaseTask, StepResult
from src.utils.config import cfg_get
from src.utils.registry import register_task

if TYPE_CHECKING:
    from torch import nn

    from src.utils.types import ConfigType


@register_task('ranking')
class RankingTask(BaseTask):
    """Pointwise ranking task using scores and continuous relevance labels."""

    def __init__(self, cfg: ConfigType) -> None:
        super().__init__(cfg)
        self.loss_fn = build_loss(cfg_get(cfg, 'loss', {'name': 'mse'}))

    def step(self, model: nn.Module, batch: dict[str, Any], stage: str) -> StepResult:
        """Compute ranking loss and configured score metrics."""
        self.validate_batch(batch, self.target_key)
        outputs = model(batch)
        self.validate_outputs(outputs, self.output_key)
        scores = outputs[self.output_key].float()
        targets = batch[self.target_key].float()
        if scores.shape != targets.shape:
            raise ValueError(f'Ranking scores and targets must have the same shape: {scores.shape} vs {targets.shape}')
        loss = outputs.get('loss')
        if loss is None:
            loss = self.loss_fn(scores, targets)
        self.metrics.update(scores.detach(), targets.detach(), n=targets.shape[0])
        return StepResult(loss=loss, outputs=outputs, targets=targets, loss_weight=targets.shape[0])

    def predict_records(self, outputs: dict[str, Any], batch: dict[str, Any]) -> list[dict[str, Any]]:
        """Export scores, relevance labels, and descending item order."""
        scores = outputs[self.output_key].detach().cpu()
        targets = batch.get(self.target_key)
        relevance = targets.detach().cpu() if torch.is_tensor(targets) else None
        records: list[dict[str, Any]] = []
        for index in range(scores.shape[0]):
            sample_scores = scores[index]
            record: dict[str, Any]
            if sample_scores.ndim == 0:
                record = {'score': float(sample_scores)}
                if relevance is not None:
                    record['relevance'] = float(relevance[index])
            else:
                record = {
                    'scores': sample_scores.tolist(),
                    'ranking': sample_scores.argsort(descending=True).tolist(),
                }
                if relevance is not None:
                    record['relevance'] = relevance[index].tolist()
            records.append(record)
        return records
