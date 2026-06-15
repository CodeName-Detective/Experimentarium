"""Reproducibility utilities for CPU, single-GPU, DDP, and future FSDP runs.

Use ``setup_reproducibility`` once per process after distributed initialization
and before building dataloaders/models. Worker seeding helpers are used by the
dataloader factory.

Typical usage:
    from src.utils.seed import setup_reproducibility
    setup_reproducibility(seed=42, strict=False)
"""

import os
import random

import numpy as np
import torch

try:
    import torch.distributed as dist
except Exception:
    dist = None


# =========================================================
# DISTRIBUTED HELPERS
# =========================================================


def get_rank() -> int:
    """Return the current process rank."""
    if dist is not None and dist.is_available() and dist.is_initialized():
        return dist.get_rank()
    return 0


def get_world_size() -> int:
    """Return the total number of distributed processes."""
    if dist is not None and dist.is_available() and dist.is_initialized():
        return dist.get_world_size()
    return 1


# =========================================================
# ENVIRONMENT CHECK
# =========================================================


def check_environment(require_cuda: bool = False) -> None:
    """Validate requested runtime capabilities."""
    if require_cuda and not torch.cuda.is_available():
        print('[WARN] CUDA requested but not available; falling back must be handled by the caller.')


# =========================================================
# SEEDING (DDP SAFE)
# =========================================================


def set_seed(seed: int, add_rank_offset: bool = True, seed_cuda: bool = False) -> None:
    """Set seeds across all randomness sources.

    Args:
        seed: Base random seed.
        add_rank_offset: Whether to create a distinct RNG stream for each rank.
        seed_cuda: Whether to seed CUDA random-number generators when available.
    """
    rank = get_rank()
    final_seed = seed + rank if add_rank_offset else seed
    os.environ['PYTHONHASHSEED'] = str(final_seed)
    random.seed(final_seed)
    np.random.seed(final_seed)
    torch.manual_seed(final_seed)

    if seed_cuda and torch.cuda.is_available():
        torch.cuda.manual_seed(final_seed)
        torch.cuda.manual_seed_all(final_seed)


# =========================================================
# DETERMINISTIC BACKEND
# =========================================================


def set_deterministic_mode(strict: bool = False) -> None:
    """Force deterministic execution on a best-effort basis."""
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    if strict:
        torch.use_deterministic_algorithms(True)


# =========================================================
# DATALOADER WORKER SEED
# =========================================================


def seed_worker(worker_id: int) -> None:
    """Seed a DataLoader worker deterministically."""
    worker_seed = torch.initial_seed() % 2**32 + get_rank()
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def get_generator(seed: int) -> torch.Generator:
    """Return a seeded generator for DataLoader shuffling."""
    generator = torch.Generator()
    generator.manual_seed(seed + get_rank())
    return generator


# =========================================================
# MAIN ENTRYPOINT
# =========================================================


def setup_reproducibility(seed: int, strict: bool = False, require_cuda: bool = False) -> None:
    """Configure process-local seeding and deterministic execution."""
    check_environment(require_cuda=require_cuda)
    set_seed(seed, add_rank_offset=True, seed_cuda=require_cuda)
    set_deterministic_mode(strict=strict)
