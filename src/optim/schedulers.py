"""Scheduler registry and factory with behavior notes.

Schedulers are selected by ``scheduler.name`` and may step per epoch or per batch
via ``scheduler.interval``. Warmup can wrap most schedulers with a linear ramp.

Scheduler behavior guide:
    none: Leaves learning rate unchanged; use for debugging and very small runs.
    constant: Keeps LR at a fixed factor for a set number of iterations.
    linear: Linearly interpolates LR from start_factor to end_factor.
    step: Drops LR by gamma every fixed step_size epochs or steps.
    multistep: Drops LR at specified milestone epochs or steps.
    exponential: Multiplies LR by gamma every step; smooth monotonic decay.
    cosine: Smoothly anneals LR toward eta_min over the full run.
    cosine_restart: Cosine schedule with periodic warm restarts.
    plateau: Reduces LR when a monitored metric stops improving.
    polynomial: Polynomial decay over the configured training horizon.
    onecycle: Ramps LR up then down within one run; usually step-based.

Typical usage:
    from src.optim import build_scheduler, scheduler_step
    bundle = build_scheduler(cfg, optimizer, steps_per_epoch=len(train_loader))
    scheduler_step(bundle, metric=val_loss)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from torch.optim.lr_scheduler import (
    ConstantLR,
    CosineAnnealingLR,
    CosineAnnealingWarmRestarts,
    ExponentialLR,
    LinearLR,
    MultiStepLR,
    OneCycleLR,
    PolynomialLR,
    ReduceLROnPlateau,
    SequentialLR,
    StepLR,
)

from src.utils.config import cfg_get
from src.utils.registry import SCHEDULER_REGISTRY, register_scheduler

if TYPE_CHECKING:
    from torch.optim import Optimizer
    from torch.optim.lr_scheduler import LRScheduler

    from src.utils.types import ConfigType

SCHEDULER_DESCRIPTIONS = {
    'none': 'Leaves learning rate unchanged; useful for debugging and tiny sanity runs.',
    'constant': 'Keeps LR at a fixed factor for total_iters before returning to base LR behavior.',
    'linear': 'Linearly interpolates LR from start_factor to end_factor over total_iters.',
    'step': 'Drops LR by gamma every fixed step_size epochs or steps.',
    'multistep': 'Drops LR by gamma at specific milestone epochs or steps.',
    'exponential': 'Multiplies LR by gamma every step for smooth monotonic decay.',
    'cosine': 'Smoothly anneals LR toward eta_min over the training horizon.',
    'cosine_restart': 'Runs cosine cycles with warm restarts controlled by T_0 and T_mult.',
    'plateau': 'Reduces LR when a monitored metric stops improving.',
    'polynomial': 'Decays LR polynomially over the training horizon.',
    'onecycle': 'Ramps LR up and then down within one run; generally batch-step based.',
}


@dataclass
class SchedulerBundle:
    """Scheduler instance plus the metadata required to step it correctly."""

    scheduler: LRScheduler | ReduceLROnPlateau | None
    interval: str = 'epoch'
    monitor: str = 'val/loss'
    name: str = 'none'
    description: str = SCHEDULER_DESCRIPTIONS['none']


@register_scheduler('none')
def build_none(
    optimizer: Optimizer,
    cfg: ConfigType,
    total_steps: int,
    total_epochs: int,
) -> None:
    """Disable learning-rate scheduling."""
    return


@register_scheduler('constant')
def build_constant(
    optimizer: Optimizer,
    cfg: ConfigType,
    total_steps: int,
    total_epochs: int,
) -> ConstantLR:
    """Build a constant-factor scheduler."""
    return ConstantLR(
        optimizer,
        factor=float(cfg_get(cfg, 'factor', 1.0)),
        total_iters=int(cfg_get(cfg, 'total_iters', 1)),
    )


@register_scheduler('linear')
def build_linear(
    optimizer: Optimizer,
    cfg: ConfigType,
    total_steps: int,
    total_epochs: int,
) -> LinearLR:
    """Build a linear learning-rate scheduler."""
    return LinearLR(
        optimizer,
        start_factor=float(cfg_get(cfg, 'start_factor', 0.1)),
        end_factor=float(cfg_get(cfg, 'end_factor', 1.0)),
        total_iters=int(cfg_get(cfg, 'total_iters', 5)),
    )


@register_scheduler('step')
def build_step(
    optimizer: Optimizer,
    cfg: ConfigType,
    total_steps: int,
    total_epochs: int,
) -> StepLR:
    """Build a fixed-step decay scheduler."""
    return StepLR(
        optimizer,
        step_size=int(cfg_get(cfg, 'step_size', 10)),
        gamma=float(cfg_get(cfg, 'gamma', 0.1)),
    )


@register_scheduler('multistep')
def build_multistep(
    optimizer: Optimizer,
    cfg: ConfigType,
    total_steps: int,
    total_epochs: int,
) -> MultiStepLR:
    """Build a milestone-based decay scheduler."""
    return MultiStepLR(
        optimizer,
        milestones=list(cfg_get(cfg, 'milestones', [30, 60, 90])),
        gamma=float(cfg_get(cfg, 'gamma', 0.1)),
    )


@register_scheduler('exponential')
def build_exponential(
    optimizer: Optimizer,
    cfg: ConfigType,
    total_steps: int,
    total_epochs: int,
) -> ExponentialLR:
    """Build an exponential decay scheduler."""
    return ExponentialLR(optimizer, gamma=float(cfg_get(cfg, 'gamma', 0.95)))


@register_scheduler('cosine')
def build_cosine(
    optimizer: Optimizer,
    cfg: ConfigType,
    total_steps: int,
    total_epochs: int,
) -> CosineAnnealingLR:
    """Build a cosine annealing scheduler."""
    interval = cfg_get(cfg, 'interval', 'epoch')
    t_max = total_steps if interval == 'step' else total_epochs
    return CosineAnnealingLR(
        optimizer,
        T_max=max(1, int(t_max)),
        eta_min=float(cfg_get(cfg, 'eta_min', 0.0)),
    )


@register_scheduler('cosine_restart')
def build_cosine_restart(
    optimizer: Optimizer,
    cfg: ConfigType,
    total_steps: int,
    total_epochs: int,
) -> CosineAnnealingWarmRestarts:
    """Build a cosine scheduler with warm restarts."""
    return CosineAnnealingWarmRestarts(
        optimizer,
        T_0=int(cfg_get(cfg, 'T_0', 10)),
        T_mult=int(cfg_get(cfg, 'T_mult', 2)),
        eta_min=float(cfg_get(cfg, 'eta_min', 0.0)),
    )


@register_scheduler('plateau')
def build_plateau(
    optimizer: Optimizer,
    cfg: ConfigType,
    total_steps: int,
    total_epochs: int,
) -> ReduceLROnPlateau:
    """Build a metric-driven plateau scheduler."""
    return ReduceLROnPlateau(
        optimizer,
        mode=str(cfg_get(cfg, 'mode', 'min')),
        factor=float(cfg_get(cfg, 'factor', 0.5)),
        patience=int(cfg_get(cfg, 'patience', 5)),
        threshold=float(cfg_get(cfg, 'threshold', 1e-4)),
        min_lr=float(cfg_get(cfg, 'min_lr', 0.0)),
    )


@register_scheduler('polynomial')
def build_polynomial(
    optimizer: Optimizer,
    cfg: ConfigType,
    total_steps: int,
    total_epochs: int,
) -> PolynomialLR:
    """Build a polynomial decay scheduler."""
    interval = cfg_get(cfg, 'interval', 'epoch')
    total_iters = total_steps if interval == 'step' else total_epochs
    return PolynomialLR(
        optimizer,
        total_iters=max(1, int(total_iters)),
        power=float(cfg_get(cfg, 'power', 1.0)),
    )


@register_scheduler('onecycle')
def build_onecycle(
    optimizer: Optimizer,
    cfg: ConfigType,
    total_steps: int,
    total_epochs: int,
) -> OneCycleLR:
    """Build a one-cycle scheduler."""
    max_lr = cfg_get(cfg, 'max_lr', None)
    if max_lr is None:
        max_lr = max(group.get('lr', 1e-3) for group in optimizer.param_groups)
    return OneCycleLR(
        optimizer,
        max_lr=float(max_lr),
        total_steps=max(1, total_steps),
        pct_start=float(cfg_get(cfg, 'pct_start', 0.3)),
        anneal_strategy=str(cfg_get(cfg, 'anneal_strategy', 'cos')),
    )


def _with_warmup(
    optimizer: Optimizer,
    scheduler: LRScheduler | ReduceLROnPlateau | None,
    cfg: ConfigType,
    total_steps: int,
) -> LRScheduler | ReduceLROnPlateau | None:
    """Prepend a linear warmup schedule when configured."""
    warmup = cfg_get(cfg, 'warmup', {})
    if scheduler is None or not bool(cfg_get(warmup, 'enabled', False)):
        return scheduler
    warmup_steps = int(cfg_get(warmup, 'steps', 100))
    warmup_scheduler = LinearLR(
        optimizer,
        start_factor=float(cfg_get(warmup, 'start_factor', 1e-4)),
        total_iters=warmup_steps,
    )
    return SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, scheduler],
        milestones=[warmup_steps],
    )


def build_scheduler(cfg: ConfigType, optimizer: Optimizer, steps_per_epoch: int) -> SchedulerBundle:
    """Build a scheduler bundle from the experiment configuration."""
    scheduler_cfg = cfg_get(cfg, 'scheduler', cfg)
    name = str(cfg_get(scheduler_cfg, 'name', 'none'))
    interval = str(cfg_get(scheduler_cfg, 'interval', 'epoch'))
    if interval not in {'epoch', 'step'}:
        raise ValueError(f"scheduler.interval must be 'epoch' or 'step', got {interval!r}")
    epochs = int(cfg_get(cfg, 'trainer.max_epochs', cfg_get(cfg, 'trainer.epochs', 1)))
    total_steps = max(1, steps_per_epoch * epochs)
    scheduler = SCHEDULER_REGISTRY.build(name, optimizer, scheduler_cfg, total_steps, epochs)
    if name != 'onecycle':
        scheduler = _with_warmup(optimizer, scheduler, scheduler_cfg, total_steps)
    return SchedulerBundle(
        scheduler=scheduler,
        interval=interval,
        monitor=str(cfg_get(scheduler_cfg, 'monitor', 'val/loss')),
        name=name,
        description=SCHEDULER_DESCRIPTIONS.get(name, 'Custom scheduler registered by the user.'),
    )


def scheduler_step(bundle: SchedulerBundle, metric: float | None = None) -> None:
    """Advance a scheduler using its configured stepping semantics."""
    scheduler = bundle.scheduler
    if scheduler is None:
        return
    if isinstance(scheduler, ReduceLROnPlateau):
        if metric is None:
            raise ValueError(f'ReduceLROnPlateau requires monitor metric {bundle.monitor!r}, but it was missing')
        scheduler.step(metric)
    else:
        scheduler.step()
