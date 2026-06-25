"""Shared type aliases and protocols used across the framework.

These aliases keep function signatures readable and provide structural protocols
for models, datasets, metrics, and schedulers. They are intentionally lightweight;
when a concrete component needs richer validation, prefer a dataclass or config
schema over a broad alias.

Typical usage:
    from src.utils.types import BatchDict, MetricDict, AnyModel
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol, TypeAlias

if TYPE_CHECKING:
    from collections.abc import Iterator

import torch
import torch.nn as nn
from torch import Tensor
from torch.optim.lr_scheduler import LRScheduler

PathLike = str | Path
Device = str | torch.device
DType = torch.dtype
Scalar = int | float
Seed = int

FloatTensor = Tensor
LongTensor = Tensor
BoolMask = Tensor
BatchTensor = Tensor

BatchDict = dict[str, Tensor]
MetricDict = dict[str, float]
ConfigDict = dict[str, Any]
LossDict = dict[str, Tensor]

StepOutput = dict[str, Tensor | float]
TrainState = tuple[int, int]
Split = str

CheckpointDict = dict[str, Any]
PredRecord = dict[str, Any]
PredList = list[PredRecord]


class ModelProtocol(Protocol):
    """Structural interface required from trainable models."""

    def forward(self, batch: BatchDict) -> StepOutput:
        """Compute model outputs for a batch."""

    def parameters(self) -> Iterator[nn.Parameter]:
        """Iterate over trainable model parameters."""

    def train(self, mode: bool = True) -> ModelProtocol:
        """Set training mode and return the model."""

    def eval(self) -> ModelProtocol:
        """Set evaluation mode and return the model."""

    def to(self, device: Device) -> ModelProtocol:
        """Move the model to a device and return it."""


class DatasetProtocol(Protocol):
    """Structural interface required from indexed datasets."""

    def __len__(self) -> int: ...

    def __getitem__(self, idx: int) -> BatchDict: ...


class MetricProtocol(Protocol):
    """Structural interface required from stateful metrics."""

    def update(self, preds: Tensor, targets: Tensor) -> None:
        """Accumulate predictions and targets."""

    def compute(self) -> Tensor:
        """Compute the current metric value."""

    def reset(self) -> None:
        """Clear accumulated metric state."""


class SchedulerProtocol(Protocol):
    """Structural interface required from learning-rate schedulers."""

    def step(self, metrics: float | None = None) -> None:
        """Advance the scheduler by one step."""

    def state_dict(self) -> ConfigDict:
        """Return serializable scheduler state."""

    def load_state_dict(self, state: ConfigDict) -> None:
        """Restore scheduler state."""


AnyScheduler = LRScheduler | SchedulerProtocol
AnyModel = nn.Module | nn.DataParallel | nn.parallel.DistributedDataParallel

if TYPE_CHECKING:
    from omegaconf import DictConfig

    ConfigType: TypeAlias = DictConfig | ConfigDict
else:
    ConfigType: TypeAlias = ConfigDict
