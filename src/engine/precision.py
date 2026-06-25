"""Precision policy shared by training and evaluation."""

from __future__ import annotations

from contextlib import nullcontext
from typing import Any

import torch

_MIXED_PRECISIONS = {'amp', 'fp16', 'bf16'}
SUPPORTED_PRECISIONS = {'fp32', *_MIXED_PRECISIONS}


def normalize_precision(precision: Any) -> str:
    """Normalize and validate configured precision names."""
    mode = str(precision or 'fp32').lower()
    if mode not in SUPPORTED_PRECISIONS:
        raise ValueError(f'Unsupported precision {mode!r}. Expected one of {sorted(SUPPORTED_PRECISIONS)}')
    return mode


def amp_dtype(precision: Any) -> torch.dtype:
    """Return the autocast dtype for a configured mixed-precision mode."""
    return torch.bfloat16 if normalize_precision(precision) == 'bf16' else torch.float16


def amp_enabled(device: torch.device | str, precision: Any) -> bool:
    """Return whether autocast should run for this device and precision."""
    return torch.device(device).type == 'cuda' and normalize_precision(precision) in _MIXED_PRECISIONS


def scaler_enabled(device: torch.device | str, precision: Any) -> bool:
    """Return whether GradScaler should be enabled for this precision mode."""
    mode = normalize_precision(precision)
    return amp_enabled(device, mode) and mode != 'bf16'


def precision_autocast(device: torch.device | str, precision: Any) -> Any:
    """Create the train/eval autocast context for the configured precision."""
    resolved_device = torch.device(device)
    mode = normalize_precision(precision)
    if not amp_enabled(resolved_device, mode):
        return nullcontext()
    return torch.amp.autocast(device_type=resolved_device.type, dtype=amp_dtype(mode))
