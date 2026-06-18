"""Task abstractions define problem-specific training behavior.

Use tasks to keep the trainer generic. A task owns the loss, metrics, prediction
records, and output/target key semantics for one workload family.

Typical usage:
    from src.tasks import build_task
    task = build_task(cfg)
    result = task.step(model, batch, stage='train')
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from src.losses import build_loss
from src.metrics import MetricCollection
from src.utils.config import cfg_get
from src.utils.registry import TASK_REGISTRY, register_task

if TYPE_CHECKING:
    from torch import Tensor, nn

    from src.utils.types import ConfigType


@dataclass
class StepResult:
    """Structured output from a task step."""

    loss: Tensor | None = None
    metrics: dict[str, float] = field(default_factory=dict)
    outputs: dict[str, Any] = field(default_factory=dict)
    targets: Tensor | None = None
    artifacts: dict[str, Any] = field(default_factory=dict)


class BaseTask:
    """Base class for task-specific loss, metrics, and prediction logic."""

    def __init__(self, cfg: ConfigType) -> None:
        self.cfg = cfg
        self.output_key = str(cfg_get(cfg, 'output_key', 'logits'))
        self.target_key = str(cfg_get(cfg, 'target_key', 'label'))
        metric_names = list(cfg_get(cfg, 'metrics', []) or [])
        self.metrics = MetricCollection.from_names(metric_names)

    def step(self, model: nn.Module, batch: dict[str, Any], stage: str) -> StepResult:
        """Compute loss, outputs, and metric inputs for one batch."""
        raise NotImplementedError

    def validate_batch(self, batch: dict[str, Any], *keys: str) -> None:
        """Validate that a batch contains required keys."""
        missing = [key for key in keys if key not in batch]
        if missing:
            raise KeyError(f'{type(self).__name__} batch missing required key(s): {missing}')

    def validate_outputs(self, outputs: dict[str, Any], *keys: str) -> None:
        """Validate that model outputs contain required keys."""
        missing = [key for key in keys if key not in outputs]
        if missing:
            raise KeyError(f'{type(self).__name__} model output missing required key(s): {missing}')

    def reset_metrics(self) -> None:
        """Reset task metric state."""
        self.metrics.reset()

    def compute_metrics(self) -> dict[str, float]:
        """Compute accumulated task metrics."""
        return self.metrics.compute()

    def predict_records(self, outputs: dict[str, Any], batch: dict[str, Any]) -> list[dict[str, Any]]:
        """Convert model outputs into serializable prediction records."""
        return []


@register_task('classification')
class ClassificationTask(BaseTask):
    """Single-label classification task using logits and class targets."""

    def __init__(self, cfg: ConfigType) -> None:
        super().__init__(cfg)
        self.loss_fn = build_loss(cfg_get(cfg, 'loss', {'name': 'cross_entropy'}))

    def step(self, model: nn.Module, batch: dict[str, Tensor], stage: str) -> StepResult:
        """Run one classification step."""
        self.validate_batch(batch, self.target_key)
        outputs = model(batch)
        self.validate_outputs(outputs, self.output_key)
        logits = outputs[self.output_key]
        targets = batch[self.target_key].long()
        loss = outputs.get('loss')
        if loss is None:
            loss = self.loss_fn(logits, targets)
        self.metrics.update(logits.detach(), targets.detach(), n=targets.shape[0])
        return StepResult(loss=loss, outputs=outputs, targets=targets)

    def predict_records(self, outputs: dict[str, Tensor], batch: dict[str, Tensor]) -> list[dict[str, Any]]:
        """Create classification prediction records."""
        logits = outputs[self.output_key]
        probs = logits.softmax(dim=-1).detach().cpu()
        preds = probs.argmax(dim=-1)
        targets = batch[self.target_key].detach().cpu()
        return [{'pred': int(preds[i]), 'label': int(targets[i]), 'prob': probs[i].tolist()} for i in range(len(preds))]


@register_task('regression')
class RegressionTask(BaseTask):
    """Regression task using continuous predictions and targets."""

    def __init__(self, cfg: ConfigType) -> None:
        super().__init__(cfg)
        self.loss_fn = build_loss(cfg_get(cfg, 'loss', {'name': 'mse'}))

    def step(self, model: nn.Module, batch: dict[str, Tensor], stage: str) -> StepResult:
        """Run one regression step."""
        self.validate_batch(batch, self.target_key)
        outputs = model(batch)
        self.validate_outputs(outputs, self.output_key)
        preds = outputs[self.output_key]
        targets = batch[self.target_key].float()
        loss = outputs.get('loss')
        if loss is None:
            loss = self.loss_fn(preds.float(), targets)
        self.metrics.update(preds.detach(), targets.detach(), n=targets.shape[0])
        return StepResult(loss=loss, outputs=outputs, targets=targets)


def build_task(cfg: ConfigType) -> BaseTask:
    """Build the task selected by configuration."""
    task_cfg = cfg_get(cfg, 'task', cfg)
    name = cfg_get(task_cfg, 'name', 'classification')
    return TASK_REGISTRY.build(name, task_cfg)
