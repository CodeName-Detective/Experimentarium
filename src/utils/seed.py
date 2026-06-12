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
    """Returns current process rank (DDP/FSDP safe)."""
    if dist is not None and dist.is_available() and dist.is_initialized():
        return dist.get_rank()
    return 0


def get_world_size() -> int:
    """Returns total number of distributed processes."""
    if dist is not None and dist.is_available() and dist.is_initialized():
        return dist.get_world_size()
    return 1


# =========================================================
# ENVIRONMENT CHECK
# =========================================================

def check_environment(require_cuda: bool = False) -> None:
    """Basic runtime sanity checks."""

    import sys

    if sys.version_info < (3, 9):
        raise RuntimeError(f"Python >= 3.9 required, found {sys.version}")

    if require_cuda and not torch.cuda.is_available():
        print("[WARN] CUDA requested but not available; falling back must be handled by the caller.")


# =========================================================
# SEEDING (DDP SAFE)
# =========================================================

def set_seed(seed: int, add_rank_offset: bool = True, seed_cuda: bool = False) -> None:
    """
    Sets seeds across all randomness sources.

    Args:
        seed: base seed
        add_rank_offset: ensures different RNG streams per rank
    """

    rank = get_rank()

    final_seed = seed + rank if add_rank_offset else seed

    os.environ["PYTHONHASHSEED"] = str(final_seed)

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
    """
    Forces deterministic execution (best effort).

    Args:
        strict: if True, raises errors for nondeterministic ops
    """

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Improve determinism in matmul / conv paths
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    if strict:
        torch.use_deterministic_algorithms(True)


# =========================================================
# DATALOADER WORKER SEED
# =========================================================

def seed_worker(worker_id: int) -> None:
    """
    Ensures deterministic DataLoader workers in DDP/FSDP.
    """

    worker_seed = torch.initial_seed() % 2**32 + get_rank()

    random.seed(worker_seed)
    np.random.seed(worker_seed)


def get_generator(seed: int):
    """
    Returns a seeded torch Generator for DataLoader shuffle.
    """
    g = torch.Generator()
    g.manual_seed(seed + get_rank())
    return g


# =========================================================
# MAIN ENTRYPOINT
# =========================================================

def setup_reproducibility(seed: int, strict: bool = False, require_cuda: bool = False) -> None:
    """
    Call once per process AFTER init_process_group().

    Works for:
        - single GPU
        - DDP
        - FSDP
    """

    check_environment(require_cuda=require_cuda)
    set_seed(seed, add_rank_offset=True, seed_cuda=require_cuda)
    set_deterministic_mode(strict=strict)


# =========================================================
# USAGE EXAMPLE
# =========================================================

"""
USAGE:

1. Initialize distributed first:
    torch.distributed.init_process_group(...)

2. Then call:
    setup_reproducibility(seed=42)

3. DataLoader:
    loader = DataLoader(
        dataset,
        shuffle=True,
        num_workers=4,
        worker_init_fn=seed_worker,
        generator=get_generator(42)
    )

------------------------------------------------------------

RECOMMENDED ORDER:

init_process_group()
setup_reproducibility()
build_dataloader()
build_model()
train()

------------------------------------------------------------

STRICT MODE:

setup_reproducibility(seed=42, strict=True)

- forces deterministic ops
- may throw errors if unsupported ops exist
"""