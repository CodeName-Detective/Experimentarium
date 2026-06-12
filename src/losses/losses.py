"""Loss functions registered for task-level construction.

Tasks call ``build_loss(cfg.task.loss)`` to instantiate a configured loss. Add
new losses with ``@register_loss('name')`` and reference them from task YAML.

Typical usage:
    from src.losses import build_loss
    loss_fn = build_loss({'name': 'cross_entropy'})
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.utils.config import cfg_get
from src.utils.registry import LOSS_REGISTRY, register_loss


@register_loss('cross_entropy')
class CrossEntropyLoss(nn.CrossEntropyLoss):
    def __init__(self, cfg=None) -> None:
        label_smoothing = float(cfg_get(cfg, 'label_smoothing', 0.0))
        super().__init__(label_smoothing=label_smoothing)


@register_loss('mse')
class MSELoss(nn.MSELoss):
    def __init__(self, cfg=None) -> None:
        super().__init__()


@register_loss('bce_with_logits')
class BCEWithLogitsLoss(nn.BCEWithLogitsLoss):
    def __init__(self, cfg=None) -> None:
        super().__init__()


@register_loss('focal')
class FocalLoss(nn.Module):
    """Focal loss for class-imbalanced classification."""

    def __init__(self, cfg=None) -> None:
        super().__init__()
        self.alpha = float(cfg_get(cfg, 'alpha', 0.25))
        self.gamma = float(cfg_get(cfg, 'gamma', 2.0))

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce = F.cross_entropy(logits, targets.long(), reduction='none')
        pt = torch.exp(-ce)
        return (self.alpha * (1 - pt) ** self.gamma * ce).mean()


@register_loss('label_smoothing')
class LabelSmoothingLoss(CrossEntropyLoss):
    pass


def build_loss(cfg) -> nn.Module:
    name = cfg_get(cfg, 'name', 'cross_entropy')
    return LOSS_REGISTRY.build(name, cfg)
