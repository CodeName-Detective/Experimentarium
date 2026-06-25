"""Semantic-segmentation task implementation."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

import torch

from src.losses import build_loss
from src.runtime.distributed import all_gather_objects
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
        self.metric_names = set(cfg_get(cfg, 'metrics', []) or [])
        for name in ('accuracy', 'iou', 'dice'):
            self.metrics.metrics.pop(name, None)
        self._confusion = torch.zeros((0, 0), dtype=torch.float64)
        if hasattr(self.loss_fn, 'ignore_index'):
            cast('Any', self.loss_fn).ignore_index = self.ignore_index

    def reset_metrics(self) -> None:
        """Reset generic metrics and the dataset-level confusion matrix."""
        super().reset_metrics()
        self._confusion = torch.zeros((0, 0), dtype=torch.float64)

    def compute_metrics(self) -> dict[str, float]:
        """Compute segmentation metrics from the accumulated confusion matrix."""
        metrics = super().compute_metrics()
        metrics.update(self._confusion_metrics(self._confusion))
        return metrics

    def compute_metrics_distributed(self, device: Any = 'cpu') -> dict[str, float]:
        """Compute segmentation metrics from a globally reduced confusion matrix."""
        metrics = super().compute_metrics_distributed(device=device)
        gathered = all_gather_objects(self._confusion)
        size = max((int(confusion.shape[0]) for confusion in gathered), default=0)
        confusion = torch.zeros((size, size), dtype=torch.float64)
        for rank_confusion in gathered:
            rank_size = int(rank_confusion.shape[0])
            confusion[:rank_size, :rank_size] += rank_confusion
        metrics.update(self._confusion_metrics(confusion))
        return metrics

    def metric_state_dict(self) -> dict[str, Any]:
        """Snapshot generic metrics and the confusion matrix."""
        state = super().metric_state_dict()
        state['confusion'] = self._confusion.clone()
        return state

    def load_metric_state_dict(self, state: dict[str, Any]) -> None:
        """Restore generic metrics and the confusion matrix."""
        super().load_metric_state_dict(state)
        self._confusion = state.get('confusion', torch.zeros((0, 0), dtype=torch.float64)).clone()

    def _update_confusion(self, logits: Tensor, targets: Tensor) -> None:
        predictions = logits.argmax(dim=1)
        valid = targets != self.ignore_index
        if not valid.any():
            return
        num_classes = int(logits.shape[1])
        encoded = targets[valid].long() * num_classes + predictions[valid].long()
        update = torch.bincount(encoded, minlength=num_classes * num_classes).reshape(num_classes, num_classes).cpu()
        if self._confusion.shape != update.shape:
            self._confusion = torch.zeros_like(update, dtype=torch.float64)
        self._confusion += update.to(torch.float64)

    def _confusion_metrics(self, confusion: Tensor) -> dict[str, float]:
        if confusion.numel() == 0 or confusion.sum() == 0:
            return dict.fromkeys(self.metric_names & {'accuracy', 'iou', 'dice'}, 0.0)
        diagonal = confusion.diag()
        metrics: dict[str, float] = {}
        if 'accuracy' in self.metric_names:
            metrics['accuracy'] = float((diagonal.sum() / confusion.sum()).item())
        predicted = confusion.sum(dim=0)
        target = confusion.sum(dim=1)
        if 'iou' in self.metric_names:
            union = predicted + target - diagonal
            valid = union > 0
            metrics['iou'] = float((diagonal[valid] / union[valid]).mean().item()) if valid.any() else 0.0
        if 'dice' in self.metric_names:
            denominator = predicted + target
            valid = denominator > 0
            metrics['dice'] = (
                float(((2.0 * diagonal[valid]) / denominator[valid]).mean().item()) if valid.any() else 0.0
            )
        return metrics

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
            if self.metrics.metrics:
                self.metrics.update(metric_logits.detach(), metric_targets.detach(), n=metric_targets.numel())
            self._update_confusion(logits.detach(), targets.detach())
        return StepResult(loss=loss, outputs=outputs, targets=targets, loss_weight=metric_targets.numel())

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
