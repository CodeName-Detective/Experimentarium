"""Distributed runtime helpers for CPU, single-GPU, and DDP workflows.

This module centralizes rank checks, barriers, rank-zero behavior, and simple
metric reductions. It does not force DDP on users; it only reflects an already
initialized process group or environment variables from ``torchrun``.

Typical usage:
    from src.runtime.distributed import is_rank0, barrier, broadcast_object, mean_dict
    if is_rank0():
        print('only rank 0 logs this')
    run_id = broadcast_object(run_id)
    metrics = mean_dict(metrics)
"""

from __future__ import annotations

import os
from typing import Any

try:
    import torch.distributed as dist
except Exception:  # pragma: no cover - torch missing or without distributed support
    dist = None  # type: ignore[assignment]


def is_available() -> bool:
    """Return whether torch.distributed is available."""
    return dist is not None and dist.is_available()


def is_initialized() -> bool:
    """Return whether a distributed process group is initialized."""
    return is_available() and dist.is_initialized()


def world_size() -> int:
    """Return the distributed world size or one outside DDP."""
    return dist.get_world_size() if is_initialized() else 1


def rank() -> int:
    """Return the global process rank."""
    return dist.get_rank() if is_initialized() else int(os.environ.get('RANK', '0'))


def local_rank() -> int:
    """Return the node-local process rank."""
    return int(os.environ.get('LOCAL_RANK', '0'))


def is_rank0() -> bool:
    """Return whether the current process is rank zero."""
    return rank() == 0


def barrier() -> None:
    """Synchronize initialized distributed processes."""
    if is_initialized():
        dist.barrier()


def broadcast_object(value: Any, src: int = 0) -> Any:
    """Broadcast a small Python object from one rank to every rank."""
    if not is_initialized():
        return value
    values = [value if rank() == src else None]
    dist.broadcast_object_list(values, src=src)
    return values[0]


def setup_from_env(backend: str = 'nccl') -> bool:
    """Initialize DDP from torchrun environment variables when available."""
    if is_initialized() or 'RANK' not in os.environ:
        return is_initialized()
    try:
        import torch
    except ModuleNotFoundError as exc:
        raise RuntimeError('torch is required for distributed setup') from exc
    if dist is None or not dist.is_available():
        raise RuntimeError('torch.distributed is not available')
    backend = backend.lower()
    if not torch.cuda.is_available() and backend == 'nccl':
        backend = 'gloo'
    dist.init_process_group(backend=backend)
    if backend == 'nccl':
        torch.cuda.set_device(local_rank())
    return True


def cleanup() -> None:
    """Destroy the active distributed process group."""
    if is_initialized():
        dist.destroy_process_group()


def mean_scalar(value: float, device: Any = 'cpu') -> float:
    """Average a scalar value across distributed ranks."""
    if not is_initialized():
        return float(value)
    try:
        import torch
    except ModuleNotFoundError as exc:
        raise RuntimeError('torch is required for distributed metric reduction') from exc
    tensor = torch.tensor(float(value), device=device)
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    tensor /= world_size()
    return float(tensor.item())


def mean_dict(metrics: dict[str, Any], device: Any = 'cpu') -> dict[str, float]:
    """Average numeric metric values across distributed ranks."""
    if not is_initialized():
        return {key: float(value) for key, value in metrics.items()}
    return {key: mean_scalar(float(value), device=device) for key, value in metrics.items()}
