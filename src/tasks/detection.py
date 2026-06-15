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
        return StepResult(loss=loss, outputs=output_dict, targets=count_targets, artifacts=artifacts)

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
