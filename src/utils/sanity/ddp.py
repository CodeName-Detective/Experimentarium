"""Distributed sanity helpers.

Use this module when sanity checks need to know whether the current process is a
DDP rank. It intentionally has no side effects and does not initialize process
groups by itself.
"""

from __future__ import annotations

import torch.distributed as dist


def is_ddp() -> bool:
    return dist.is_available() and dist.is_initialized()


def rank() -> int:
    return dist.get_rank() if is_ddp() else 0


def is_rank0() -> bool:
    return rank() == 0


def barrier(enabled: bool = True) -> None:
    if enabled and is_ddp():
        dist.barrier()
