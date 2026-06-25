"""Object-detection task implementation."""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, cast

import torch

from src.runtime.distributed import all_gather_objects
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
        self.metrics.metrics.pop('map50', None)
        self._map_records: list[dict[str, torch.Tensor]] = []

    def reset_metrics(self) -> None:
        """Reset detection-specific metrics."""
        super().reset_metrics()
        self._map_records = []

    def compute_metrics(self) -> dict[str, float]:
        """Return dataset-level detection metrics."""
        metrics = super().compute_metrics()
        if 'map50' in self.metric_names:
            metrics['map50'] = self._mean_ap50(self._map_records)
        return metrics

    def compute_metrics_distributed(self, device: Any = 'cpu') -> dict[str, float]:
        """Compute mAP@50 after gathering detection records from every rank."""
        metrics = super().compute_metrics_distributed(device=device)
        if 'map50' in self.metric_names:
            gathered = all_gather_objects(self._map_records)
            records = [record for rank_records in gathered for record in rank_records]
            metrics['map50'] = self._mean_ap50(records)
        return metrics

    def metric_state_dict(self) -> dict[str, Any]:
        """Snapshot generic and detection-specific metric accumulators."""
        state = super().metric_state_dict()
        state['map_records'] = list(self._map_records)
        return state

    def load_metric_state_dict(self, state: dict[str, Any]) -> None:
        """Restore generic and detection-specific metric accumulators."""
        super().load_metric_state_dict(state)
        self._map_records = list(state.get('map_records', []))

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
        return StepResult(
            loss=loss, outputs=output_dict, targets=count_targets, artifacts=artifacts, loss_weight=batch_size
        )

    def _update_detection_metrics(self, outputs: dict[str, Any], batch: dict[str, Any]) -> None:
        if 'map50' not in self.metric_names:
            return
        detections = outputs.get(self.output_key)
        targets = batch.get(self.target_key)
        if isinstance(detections, Mapping):
            detections = [detections]
        if not isinstance(detections, (list, tuple)) or not isinstance(targets, (list, tuple)):
            return
        for detection, target in zip(detections, targets, strict=False):
            if not isinstance(detection, Mapping) or not isinstance(target, Mapping):
                continue
            pred_boxes = detection.get('boxes')
            pred_scores = detection.get('scores')
            target_boxes = target.get('boxes')
            if not all(torch.is_tensor(value) for value in (pred_boxes, pred_scores, target_boxes)):
                continue
            pred_boxes = cast('torch.Tensor', pred_boxes)
            pred_scores = cast('torch.Tensor', pred_scores)
            target_boxes = cast('torch.Tensor', target_boxes)
            pred_labels = detection.get('labels')
            target_labels = target.get('labels')
            if not torch.is_tensor(pred_labels):
                pred_labels = torch.zeros(pred_boxes.shape[0], dtype=torch.long, device=pred_boxes.device)
            if not torch.is_tensor(target_labels):
                target_labels = torch.zeros(target_boxes.shape[0], dtype=torch.long, device=target_boxes.device)
            self._map_records.append({
                'pred_boxes': pred_boxes.detach().float().cpu(),
                'pred_scores': pred_scores.detach().float().cpu(),
                'pred_labels': pred_labels.detach().long().cpu(),
                'target_boxes': target_boxes.detach().float().cpu(),
                'target_labels': target_labels.detach().long().cpu(),
            })

    def _mean_ap50(self, records: list[dict[str, torch.Tensor]]) -> float:
        classes = sorted({int(label) for record in records for label in record['target_labels'].tolist()})
        if not classes:
            return 0.0
        return sum(self._class_ap50(records, class_id) for class_id in classes) / len(classes)

    def _class_ap50(self, records: list[dict[str, torch.Tensor]], class_id: int) -> float:
        targets: dict[int, torch.Tensor] = {}
        predictions: list[tuple[float, int, torch.Tensor]] = []
        target_count = 0
        for image_index, record in enumerate(records):
            target_boxes = record['target_boxes'][record['target_labels'] == class_id]
            targets[image_index] = target_boxes
            target_count += int(target_boxes.shape[0])
            mask = record['pred_labels'] == class_id
            for score, box in zip(record['pred_scores'][mask], record['pred_boxes'][mask], strict=False):
                predictions.append((float(score), image_index, box))
        if target_count == 0:
            return 0.0
        predictions.sort(key=lambda item: item[0], reverse=True)
        matched = {index: torch.zeros(len(boxes), dtype=torch.bool) for index, boxes in targets.items()}
        true_positive: list[float] = []
        false_positive: list[float] = []
        for _, image_index, box in predictions:
            target_boxes = targets[image_index]
            available = ~matched[image_index]
            if target_boxes.numel() == 0 or not available.any():
                true_positive.append(0.0)
                false_positive.append(1.0)
                continue
            ious = self._box_iou_many(box, target_boxes)
            ious[~available] = -1.0
            best_iou, best_index = ious.max(dim=0)
            if float(best_iou) >= 0.5:
                matched[image_index][best_index] = True
                true_positive.append(1.0)
                false_positive.append(0.0)
            else:
                true_positive.append(0.0)
                false_positive.append(1.0)
        if not predictions:
            return 0.0
        tp = torch.tensor(true_positive).cumsum(0)
        fp = torch.tensor(false_positive).cumsum(0)
        recall = tp / target_count
        precision = tp / (tp + fp).clamp_min(torch.finfo(torch.float32).eps)
        recall = torch.cat([torch.tensor([0.0]), recall, torch.tensor([1.0])])
        precision = torch.cat([torch.tensor([0.0]), precision, torch.tensor([0.0])])
        for index in range(precision.numel() - 2, -1, -1):
            precision[index] = torch.maximum(precision[index], precision[index + 1])
        changing = torch.nonzero(recall[1:] != recall[:-1], as_tuple=False).flatten()
        return float(((recall[changing + 1] - recall[changing]) * precision[changing + 1]).sum().item())

    def _box_iou_many(self, box: Tensor, others: Tensor) -> Tensor:
        top_left = torch.maximum(box[:2], others[:, :2])
        bottom_right = torch.minimum(box[2:], others[:, 2:])
        intersection = (bottom_right - top_left).clamp(min=0).prod(dim=1)
        area = (box[2:] - box[:2]).clamp(min=0).prod()
        other_area = (others[:, 2:] - others[:, :2]).clamp(min=0).prod(dim=1)
        return intersection / (area + other_area - intersection).clamp(min=torch.finfo(torch.float32).eps)

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
