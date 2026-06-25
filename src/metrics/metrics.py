"""Task-agnostic metric functions and metric collections.

Tasks build ``MetricCollection`` from names listed in config. Add new metrics with
``@register_metric('name')`` and include the name in ``configs/task/*.yaml``.

Typical usage:
    metrics = MetricCollection.from_names(['accuracy'])
    metrics.update(logits, targets)
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import torch
from torch import Tensor

from src.utils.registry import METRIC_REGISTRY, register_metric

MetricFn = Callable[[Tensor, Tensor], float]


@register_metric('accuracy')
def accuracy(logits: Tensor, targets: Tensor) -> float:
    """Compute single-label classification accuracy."""
    preds = logits.argmax(dim=-1)
    return (preds == targets.long()).float().mean().item()


@register_metric('mse')
def mse(preds: Tensor, targets: Tensor) -> float:
    """Compute mean squared error."""
    return torch.nn.functional.mse_loss(preds.float(), targets.float()).item()


@register_metric('mae')
def mae(preds: Tensor, targets: Tensor) -> float:
    """Compute mean absolute error."""
    return torch.nn.functional.l1_loss(preds.float(), targets.float()).item()


@register_metric('top5_accuracy')
def top5_accuracy(logits: Tensor, targets: Tensor) -> float:
    """Compute top-5 categorical accuracy."""
    k = min(5, logits.shape[-1])
    topk = logits.topk(k, dim=-1).indices
    return topk.eq(targets.long().unsqueeze(-1)).any(dim=-1).float().mean().item()


@register_metric('iou')
def mean_iou(logits: Tensor, targets: Tensor) -> float:
    """Compute macro IoU from logits and integer targets."""
    preds = logits.argmax(dim=-1).reshape(-1)
    targets = targets.long().reshape(-1)
    classes = torch.unique(torch.cat([preds, targets]))
    scores = []
    for cls in classes:
        pred_mask = preds == cls
        target_mask = targets == cls
        union = torch.logical_or(pred_mask, target_mask).sum().item()
        if union:
            intersection = torch.logical_and(pred_mask, target_mask).sum().item()
            scores.append(intersection / union)
    return float(sum(scores) / max(1, len(scores)))


@register_metric('dice')
def dice_score(logits: Tensor, targets: Tensor) -> float:
    """Compute macro Dice score from logits and integer targets."""
    preds = logits.argmax(dim=-1).reshape(-1)
    targets = targets.long().reshape(-1)
    classes = torch.unique(torch.cat([preds, targets]))
    scores = []
    for cls in classes:
        pred_mask = preds == cls
        target_mask = targets == cls
        denominator = pred_mask.sum().item() + target_mask.sum().item()
        if denominator:
            intersection = torch.logical_and(pred_mask, target_mask).sum().item()
            scores.append((2.0 * intersection) / denominator)
    return float(sum(scores) / max(1, len(scores)))


@register_metric('ndcg')
def ndcg(scores: Tensor, relevance: Tensor) -> float:
    """Compute mean normalized discounted cumulative gain."""
    scores = scores.float()
    relevance = relevance.float()
    if scores.ndim == 1:
        scores = scores.unsqueeze(0)
        relevance = relevance.unsqueeze(0)
    values = []
    discounts = 1.0 / torch.log2(torch.arange(scores.shape[1], device=scores.device, dtype=torch.float32) + 2.0)
    for row_scores, row_relevance in zip(scores, relevance, strict=False):
        order = row_scores.argsort(descending=True)
        ideal = row_relevance.argsort(descending=True)
        dcg = ((2.0 ** row_relevance[order] - 1.0) * discounts).sum()
        idcg = ((2.0 ** row_relevance[ideal] - 1.0) * discounts).sum()
        values.append(float((dcg / idcg.clamp_min(torch.finfo(torch.float32).eps)).item()))
    return float(sum(values) / max(1, len(values)))


@register_metric('mrr')
def mean_reciprocal_rank(scores: Tensor, relevance: Tensor) -> float:
    """Compute mean reciprocal rank using positive relevance as relevant."""
    scores = scores.float()
    relevance = relevance.float()
    if scores.ndim == 1:
        scores = scores.unsqueeze(0)
        relevance = relevance.unsqueeze(0)
    values = []
    for row_scores, row_relevance in zip(scores, relevance, strict=False):
        order = row_scores.argsort(descending=True)
        relevant = row_relevance[order] > 0
        indices = torch.nonzero(relevant, as_tuple=False).flatten()
        values.append(0.0 if indices.numel() == 0 else 1.0 / float(indices[0].item() + 1))
    return float(sum(values) / max(1, len(values)))


@register_metric('precision_at_1')
def precision_at_1(scores: Tensor, relevance: Tensor) -> float:
    """Compute precision at rank 1 using positive relevance as relevant."""
    scores = scores.float()
    relevance = relevance.float()
    if scores.ndim == 1:
        scores = scores.unsqueeze(0)
        relevance = relevance.unsqueeze(0)
    top = scores.argmax(dim=-1)
    return (relevance.gather(1, top.unsqueeze(1)).squeeze(1) > 0).float().mean().item()


@register_metric('map50')
def map50_placeholder(preds: Tensor, targets: Tensor) -> float:
    """Placeholder registry entry; DetectionTask computes structured mAP@50."""
    return 0.0


@dataclass
class MetricAccumulator:
    """Accumulate weighted scalar metric values across batches."""

    name: str
    fn: MetricFn
    total: float = 0.0
    count: int = 0

    def update(self, preds: Tensor, targets: Tensor, n: int | None = None) -> None:
        """Accumulate one batch of predictions and targets."""
        batch_n = int(n if n is not None else targets.shape[0])
        self.total += float(self.fn(preds.detach().cpu(), targets.detach().cpu())) * batch_n
        self.count += batch_n

    def compute(self) -> float:
        """Return the weighted mean metric value."""
        return self.total / max(1, self.count)

    def reset(self) -> None:
        """Clear accumulated metric state."""
        self.total = 0.0
        self.count = 0

    def state_dict(self) -> dict[str, float | int]:
        """Return serializable accumulator state."""
        return {'total': self.total, 'count': self.count}

    def load_state_dict(self, state: dict[str, float | int]) -> None:
        """Restore accumulator state."""
        self.total = float(state.get('total', 0.0))
        self.count = int(state.get('count', 0))


@dataclass
class MetricCollection:
    """Coordinate a named collection of metric accumulators."""

    metrics: dict[str, MetricAccumulator] = field(default_factory=dict)

    @classmethod
    def from_names(cls, names: list[str]) -> MetricCollection:
        """Build a collection from registered metric names."""
        return cls({name: MetricAccumulator(name, METRIC_REGISTRY.get(name)) for name in names})

    def update(self, preds: Tensor, targets: Tensor, n: int | None = None) -> None:
        """Update every metric with one batch."""
        for metric in self.metrics.values():
            metric.update(preds, targets, n=n)

    def compute(self) -> dict[str, float]:
        """Compute all metrics by name."""
        return {name: metric.compute() for name, metric in self.metrics.items()}

    def reset(self) -> None:
        """Reset every metric accumulator."""
        for metric in self.metrics.values():
            metric.reset()

    def totals(self) -> dict[str, tuple[float, int]]:
        """Return weighted numerators and denominators for distributed reduction."""
        return {name: (metric.total, metric.count) for name, metric in self.metrics.items()}

    def state_dict(self) -> dict[str, dict[str, float | int]]:
        """Return serializable state for every metric."""
        return {name: metric.state_dict() for name, metric in self.metrics.items()}

    def load_state_dict(self, state: dict[str, dict[str, float | int]]) -> None:
        """Restore state for metrics present in this collection."""
        for name, metric_state in state.items():
            metric = self.metrics.get(name)
            if metric is not None:
                metric.load_state_dict(metric_state)


def compute_all_metrics(preds: Tensor, targets: Tensor) -> dict[str, float]:
    """Compute the framework's default metric set."""
    return {'accuracy': accuracy(preds, targets)}
