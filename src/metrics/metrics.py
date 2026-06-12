"""Task-agnostic metric functions and metric collections.

Tasks build ``MetricCollection`` from names listed in config. Add new metrics with
``@register_metric('name')`` and include the name in ``configs/task/*.yaml``.

Typical usage:
    metrics = MetricCollection.from_names(['accuracy'])
    metrics.update(logits, targets)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import torch
from torch import Tensor

from src.utils.registry import METRIC_REGISTRY, register_metric

MetricFn = Callable[[Tensor, Tensor], float]


@register_metric('accuracy')
def accuracy(logits: Tensor, targets: Tensor) -> float:
    preds = logits.argmax(dim=-1)
    return (preds == targets.long()).float().mean().item()


@register_metric('mse')
def mse(preds: Tensor, targets: Tensor) -> float:
    return torch.nn.functional.mse_loss(preds.float(), targets.float()).item()


@register_metric('mae')
def mae(preds: Tensor, targets: Tensor) -> float:
    return torch.nn.functional.l1_loss(preds.float(), targets.float()).item()


@dataclass
class MetricAccumulator:
    name: str
    fn: MetricFn
    total: float = 0.0
    count: int = 0

    def update(self, preds: Tensor, targets: Tensor, n: int | None = None) -> None:
        batch_n = int(n if n is not None else targets.shape[0])
        self.total += float(self.fn(preds.detach().cpu(), targets.detach().cpu())) * batch_n
        self.count += batch_n

    def compute(self) -> float:
        return self.total / max(1, self.count)

    def reset(self) -> None:
        self.total = 0.0
        self.count = 0


@dataclass
class MetricCollection:
    metrics: dict[str, MetricAccumulator] = field(default_factory=dict)

    @classmethod
    def from_names(cls, names: list[str]) -> 'MetricCollection':
        return cls({name: MetricAccumulator(name, METRIC_REGISTRY.get(name)) for name in names})

    def update(self, preds: Tensor, targets: Tensor, n: int | None = None) -> None:
        for metric in self.metrics.values():
            metric.update(preds, targets, n=n)

    def compute(self) -> dict[str, float]:
        return {name: metric.compute() for name, metric in self.metrics.items()}

    def reset(self) -> None:
        for metric in self.metrics.values():
            metric.reset()


def compute_all_metrics(preds: Tensor, targets: Tensor) -> dict[str, float]:
    return {'accuracy': accuracy(preds, targets)}
