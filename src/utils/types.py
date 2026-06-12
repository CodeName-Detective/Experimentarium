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
from typing import Any, Dict, Iterator, List, Optional, Protocol, Tuple, Union

import torch
import torch.nn as nn
from torch import Tensor
from torch.optim.lr_scheduler import LRScheduler

PathLike = Union[str, Path]
Device = Union[str, torch.device]
DType = torch.dtype
Scalar = Union[int, float]
Seed = int

FloatTensor = Tensor
LongTensor = Tensor
BoolMask = Tensor
BatchTensor = Tensor

BatchDict = Dict[str, Tensor]
MetricDict = Dict[str, float]
ConfigDict = Dict[str, Any]
LossDict = Dict[str, Tensor]

StepOutput = Dict[str, Union[Tensor, float]]
TrainState = Tuple[int, int]
Split = str

CheckpointDict = Dict[str, Any]
PredRecord = Dict[str, Any]
PredList = List[PredRecord]


class ModelProtocol(Protocol):
    def forward(self, batch: BatchDict) -> StepOutput: ...
    def parameters(self) -> Iterator[nn.Parameter]: ...
    def train(self, mode: bool = True) -> 'ModelProtocol': ...
    def eval(self) -> 'ModelProtocol': ...
    def to(self, device: Device) -> 'ModelProtocol': ...


class DatasetProtocol(Protocol):
    def __len__(self) -> int: ...
    def __getitem__(self, idx: int) -> BatchDict: ...


class MetricProtocol(Protocol):
    def update(self, preds: Tensor, targets: Tensor) -> None: ...
    def compute(self) -> Tensor: ...
    def reset(self) -> None: ...


class SchedulerProtocol(Protocol):
    def step(self, metrics: Optional[float] = None) -> None: ...
    def state_dict(self) -> ConfigDict: ...
    def load_state_dict(self, state: ConfigDict) -> None: ...


AnyScheduler = Union[LRScheduler, SchedulerProtocol]
AnyModel = Union[nn.Module, nn.DataParallel, nn.parallel.DistributedDataParallel]

try:
    from omegaconf import DictConfig
    ConfigType = Union[DictConfig, ConfigDict]
except ImportError:
    ConfigType = ConfigDict  # type: ignore[misc]
