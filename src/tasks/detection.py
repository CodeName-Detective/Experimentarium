"""Object-detection task implementation."""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

import torch

from src.tasks.task import BaseTask, StepResult
from src.utils.config import cfg_get
from src.utils.registry import register_task

if TYPE_CHECKING:
    from torch import Tensor, nn

    from src.utils.types import ConfigType


@register_task('detection')
class DetectionTask(BaseTask):
    """Detection task for models using the framework batch-dictionary contract."""

    def __init__(self, cfg: ConfigType) -> None:
        super().__init__(cfg)
        self.score_threshold = float(cfg_get(cfg, 'score_threshold', 0.05))
        self.nms_iou_threshold = float(cfg_get(cfg, 'nms_iou_threshold', 0.5))
        self.metric_names = set(cfg_get(cfg, 'metrics', []) or [])
        self._ap50_total = 0.0
        self._ap50_count = 0

    def reset_metrics(self) -> None:
        """Reset detection-specific metrics."""
        super().reset_metrics()
        self._ap50_total = 0.0
        self._ap50_count = 0

    def compute_metrics(self) -> dict[str, float]:
        """Return detection metrics computed from structured predictions."""
        metrics = super().compute_metrics()
        if 'map50' in self.metric_names and self._ap50_count:
            metrics['map50'] = self._ap50_total / self._ap50_count
        return metrics

    def step(self, model: nn.Module, batch: dict[str, Any], stage: str) -> StepResult:
        """Aggregate model-provided detection losses for training."""
        outputs = model(batch)
        if not isinstance(outputs, Mapping):
            raise TypeError('Detection models must return a mapping of losses and/or detections')
        output_dict = dict(outputs)
        loss, loss_components = self._extract_loss(output_dict)
        batch_size = self._batch_size(batch)
        device = loss.device if loss is not None else self._find_tensor_device(output_dict, batch)
        count_targets = torch.empty(batch_size, device=device or torch.device('cpu'))
        artifacts = {'loss_components': loss_components} if loss_components else {}
        self._update_detection_metrics(output_dict, batch)
        return StepResult(loss=loss, outputs=output_dict, targets=count_targets, artifacts=artifacts)

    def _update_detection_metrics(self, outputs: dict[str, Any], batch: dict[str, Any]) -> None:
        if 'map50' not in self.metric_names:
            return
        detections = outputs.get(self.output_key)
        targets = batch.get(self.target_key)
        if detections is None or targets is None:
            return
        if isinstance(detections, Mapping):
            detections = [detections]
        if not isinstance(detections, (list, tuple)) or not isinstance(targets, (list, tuple)):
            return
        for detection, target in zip(detections, targets, strict=False):
            if isinstance(detection, Mapping) and isinstance(target, Mapping):
                self._ap50_total += self._ap50(detection, target)
                self._ap50_count += 1

    def _ap50(self, detection: Mapping[str, Any], target: Mapping[str, Any]) -> float:
        pred_boxes = detection.get('boxes')
        pred_scores = detection.get('scores')
        pred_labels = detection.get('labels')
        target_boxes = target.get('boxes')
        target_labels = target.get('labels')
        if not all(torch.is_tensor(value) for value in (pred_boxes, pred_scores, target_boxes)):
            return 0.0
        if pred_boxes.numel() == 0 or target_boxes.numel() == 0:
            return 0.0
        if not torch.is_tensor(pred_labels):
            pred_labels = torch.zeros(pred_boxes.shape[0], dtype=torch.long, device=pred_boxes.device)
        if not torch.is_tensor(target_labels):
            target_labels = torch.zeros(target_boxes.shape[0], dtype=torch.long, device=target_boxes.device)
        order = pred_scores.argsort(descending=True)
        matched: set[int] = set()
        precisions: list[float] = []
        true_positives = 0
        for rank, pred_idx in enumerate(order, start=1):
            same_label = torch.nonzero(target_labels == pred_labels[pred_idx], as_tuple=False).flatten()
            best_iou = 0.0
            best_target = -1
            for target_idx in same_label.tolist():
                if target_idx in matched:
                    continue
                iou = float(self._box_iou(pred_boxes[pred_idx], target_boxes[target_idx]).item())
                if iou > best_iou:
                    best_iou = iou
                    best_target = target_idx
            if best_iou >= 0.5 and best_target >= 0:
                matched.add(best_target)
                true_positives += 1
                precisions.append(true_positives / rank)
        return sum(precisions) / max(1, int(target_boxes.shape[0]))

    def _box_iou(self, box: Tensor, other: Tensor) -> Tensor:
        top_left = torch.maximum(box[:2], other[:2])
        bottom_right = torch.minimum(box[2:], other[2:])
        intersection = (bottom_right - top_left).clamp(min=0).prod()
        area = (box[2:] - box[:2]).clamp(min=0).prod()
        other_area = (other[2:] - other[:2]).clamp(min=0).prod()
        return intersection / (area + other_area - intersection).clamp(min=torch.finfo(box.float().dtype).eps)

    def _extract_loss(self, outputs: dict[str, Any]) -> tuple[Tensor | None, dict[str, float]]:
        direct_loss = outputs.get('loss')
        if torch.is_tensor(direct_loss):
            return direct_loss, {'loss': float(direct_loss.detach().cpu())}

        candidates: dict[str, Tensor] = {}
        nested = outputs.get('losses')
        if isinstance(nested, Mapping):
            candidates.update({str(key): value for key, value in nested.items() if torch.is_tensor(value)})
        candidates.update({
            key: value for key, value in outputs.items() if key.startswith('loss_') and torch.is_tensor(value)
        })
        if not candidates:
            return None, {}
        loss = torch.stack([value.reshape(()) for value in candidates.values()]).sum()
        components = {key: float(value.detach().cpu()) for key, value in candidates.items()}
        return loss, components

    def _batch_size(self, batch: dict[str, Any]) -> int:
        targets = batch.get(self.target_key)
        if torch.is_tensor(targets):
            return int(targets.shape[0])
        if isinstance(targets, (list, tuple)):
            return len(targets)
        inputs = batch.get('input')
        if torch.is_tensor(inputs):
            return int(inputs.shape[0])
        if isinstance(inputs, (list, tuple)):
            return len(inputs)
        return 1

    def _find_tensor_device(self, *values: Any) -> torch.device | None:
        for value in values:
            if torch.is_tensor(value):
                return value.device
            if isinstance(value, Mapping):
                device = self._find_tensor_device(*value.values())
                if device is not None:
                    return device
            elif isinstance(value, (list, tuple)):
                device = self._find_tensor_device(*value)
                if device is not None:
                    return device
        return None

    def predict_records(self, outputs: dict[str, Any], batch: dict[str, Any]) -> list[dict[str, Any]]:
        """Filter and serialize model detections."""
        detections = outputs.get(self.output_key, outputs)
        if isinstance(detections, Mapping):
            detections = [detections]
        if not isinstance(detections, (list, tuple)):
            raise TypeError(f'Detection output {self.output_key!r} must be a mapping or sequence of mappings')
        return [self._serialize_detection(detection) for detection in detections]

    def _serialize_detection(self, detection: Any) -> dict[str, Any]:
        if not isinstance(detection, Mapping):
            raise TypeError('Each detection result must be a mapping')
        filtered = self._filter_detection(dict(detection))
        return {str(key): self._to_serializable(value) for key, value in filtered.items()}

    def _filter_detection(self, detection: dict[str, Any]) -> dict[str, Any]:
        scores = detection.get('scores')
        if not torch.is_tensor(scores):
            return detection
        original_count = scores.shape[0]
        keep = torch.nonzero(scores >= self.score_threshold, as_tuple=False).flatten()
        boxes = detection.get('boxes')
        if torch.is_tensor(boxes) and boxes.ndim == 2 and boxes.shape[-1] == 4 and keep.numel():
            keep = keep[self._nms(boxes[keep], scores[keep], self.nms_iou_threshold)]
        return {
            key: value[keep]
            if torch.is_tensor(value) and value.ndim > 0 and value.shape[0] == original_count
            else value
            for key, value in detection.items()
        }

    def _nms(self, boxes: Tensor, scores: Tensor, iou_threshold: float) -> Tensor:
        boxes = boxes.float()
        scores = scores.float()
        order = scores.argsort(descending=True)
        kept: list[Tensor] = []
        while order.numel():
            current = order[0]
            kept.append(current)
            if order.numel() == 1:
                break
            remaining = order[1:]
            current_box = boxes[current]
            other_boxes = boxes[remaining]
            top_left = torch.maximum(current_box[:2], other_boxes[:, :2])
            bottom_right = torch.minimum(current_box[2:], other_boxes[:, 2:])
            intersection = (bottom_right - top_left).clamp(min=0).prod(dim=1)
            current_area = (current_box[2:] - current_box[:2]).clamp(min=0).prod()
            other_area = (other_boxes[:, 2:] - other_boxes[:, :2]).clamp(min=0).prod(dim=1)
            union = current_area + other_area - intersection
            iou = intersection / union.clamp(min=torch.finfo(boxes.dtype).eps)
            order = remaining[iou <= iou_threshold]
        return torch.stack(kept) if kept else torch.empty(0, dtype=torch.long, device=boxes.device)

    def _to_serializable(self, value: Any) -> Any:
        if torch.is_tensor(value):
            value = value.detach().cpu()
            return value.item() if value.ndim == 0 else value.tolist()
        if isinstance(value, Mapping):
            return {str(key): self._to_serializable(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [self._to_serializable(item) for item in value]
        return value
