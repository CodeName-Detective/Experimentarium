"""Semantic-segmentation task implementation."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch

from src.losses import build_loss
from src.tasks.task import BaseTask, StepResult
from src.utils.config import cfg_get
from src.utils.registry import register_task

if TYPE_CHECKING:
    from torch import Tensor, nn

    from src.utils.types import ConfigType


@register_task('segmentation')
class SegmentationTask(BaseTask):
    """Multi-class semantic segmentation with per-pixel logits and masks."""

    def __init__(self, cfg: ConfigType) -> None:
        super().__init__(cfg)
        loss_cfg = cfg_get(cfg, 'loss', {'name': 'cross_entropy'})
        self.loss_fn = build_loss(loss_cfg)
        self.ignore_index = int(cfg_get(cfg, 'ignore_index', cfg_get(loss_cfg, 'ignore_index', 255)))
        if hasattr(self.loss_fn, 'ignore_index'):
            self.loss_fn.ignore_index = self.ignore_index

    def step(self, model: nn.Module, batch: dict[str, Any], stage: str) -> StepResult:
        """Compute per-pixel loss and configured metrics."""
        self.validate_batch(batch, self.target_key)
        outputs = model(batch)
        self.validate_outputs(outputs, self.output_key)
        logits = outputs[self.output_key]
        targets = batch[self.target_key].long()
        if logits.ndim < 3:
            raise ValueError(f'Segmentation logits must have shape [B, C, ...], got {tuple(logits.shape)}')
        if logits.shape[0] != targets.shape[0] or logits.shape[2:] != targets.shape[1:]:
            raise ValueError(
                f'Segmentation logits/target shapes are incompatible: {tuple(logits.shape)} vs {tuple(targets.shape)}'
            )
        loss = outputs.get('loss')
        if loss is None:
            loss = self.loss_fn(logits, targets)
        metric_logits, metric_targets = self._metric_inputs(logits, targets)
        if metric_targets.numel():
            self.metrics.update(metric_logits.detach(), metric_targets.detach(), n=metric_targets.numel())
        return StepResult(loss=loss, outputs=outputs, targets=targets)

    def _metric_inputs(self, logits: Tensor, targets: Tensor) -> tuple[Tensor, Tensor]:
        num_classes = logits.shape[1]
        flat_logits = logits.movedim(1, -1).reshape(-1, num_classes)
        flat_targets = targets.reshape(-1)
        valid = flat_targets != self.ignore_index
        return flat_logits[valid], flat_targets[valid]

    def predict_records(self, outputs: dict[str, Any], batch: dict[str, Any]) -> list[dict[str, Any]]:
        """Export one predicted and target mask per sample."""
        predictions = outputs[self.output_key].argmax(dim=1).detach().cpu()
        targets = batch.get(self.target_key)
        target_masks = targets.detach().cpu() if torch.is_tensor(targets) else None
        records: list[dict[str, Any]] = []
        for index, prediction in enumerate(predictions):
            record: dict[str, Any] = {'pred_mask': prediction.tolist()}
            if target_masks is not None:
                record['target_mask'] = target_masks[index].tolist()
            records.append(record)
        return records
