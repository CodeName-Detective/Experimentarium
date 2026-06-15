"""Optimizer registry and factory.

Optimizers are selected by config through ``optimizer.name``. The factory also
supports no-weight-decay parameter groups for common research practice: biases
and normalization parameters can be excluded from weight decay without editing
model code.

Typical usage:
    from src.optim import build_optimizer
    optimizer = build_optimizer(model, cfg)
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from src.utils.config import cfg_get
from src.utils.registry import OPTIMIZER_REGISTRY, register_optimizer

if TYPE_CHECKING:
    from collections.abc import Iterable

    from torch.optim import Optimizer

    from src.utils.types import ConfigType


def parameter_groups(model: torch.nn.Module, cfg: ConfigType) -> list[dict[str, object]]:
    """Build parameter groups with optional norm/bias weight-decay exclusion."""
    weight_decay = float(cfg_get(cfg, 'weight_decay', 0.0))
    no_decay = bool(cfg_get(cfg, 'no_decay_norm_bias', True))
    if not no_decay or not weight_decay:
        return [{'params': [p for p in model.parameters() if p.requires_grad], 'weight_decay': weight_decay}]
    decay_params = []
    no_decay_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if name.endswith('bias') or 'norm' in name.lower() or 'bn' in name.lower():
            no_decay_params.append(param)
        else:
            decay_params.append(param)
    groups: list[dict[str, object]] = []
    if decay_params:
        groups.append({'params': decay_params, 'weight_decay': weight_decay})
    if no_decay_params:
        groups.append({'params': no_decay_params, 'weight_decay': 0.0})
    return groups


@register_optimizer('adamw')
def build_adamw(params: Iterable[torch.Tensor | dict[str, object]], cfg: ConfigType) -> Optimizer:
    """Build an AdamW optimizer from configuration."""
    return torch.optim.AdamW(
        params,
        lr=float(cfg_get(cfg, 'lr', 1e-3)),
        weight_decay=float(cfg_get(cfg, 'weight_decay', 0.0)),
        betas=tuple(cfg_get(cfg, 'betas', (0.9, 0.999))),
        foreach=cfg_get(cfg, 'foreach', None),
        fused=cfg_get(cfg, 'fused', None),
    )


@register_optimizer('adam')
def build_adam(params: Iterable[torch.Tensor | dict[str, object]], cfg: ConfigType) -> Optimizer:
    """Build an Adam optimizer from configuration."""
    return torch.optim.Adam(
        params,
        lr=float(cfg_get(cfg, 'lr', 1e-3)),
        weight_decay=float(cfg_get(cfg, 'weight_decay', 0.0)),
    )


@register_optimizer('sgd')
def build_sgd(params: Iterable[torch.Tensor | dict[str, object]], cfg: ConfigType) -> Optimizer:
    """Build an SGD optimizer from configuration."""
    return torch.optim.SGD(
        params,
        lr=float(cfg_get(cfg, 'lr', 0.1)),
        momentum=float(cfg_get(cfg, 'momentum', 0.9)),
        weight_decay=float(cfg_get(cfg, 'weight_decay', 0.0)),
        nesterov=bool(cfg_get(cfg, 'nesterov', False)),
    )


def build_optimizer(model: torch.nn.Module, cfg: ConfigType) -> Optimizer:
    """Build the configured optimizer for a model."""
    opt_cfg = cfg_get(cfg, 'optimizer', cfg)
    name = cfg_get(opt_cfg, 'name', 'adamw')
    params = parameter_groups(model, opt_cfg)
    return OPTIMIZER_REGISTRY.build(name, params, opt_cfg)
