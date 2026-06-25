"""Runtime helpers for distributed training and process-level behavior.

Use this package for rank checks, DDP setup/cleanup, barriers, and distributed
metric reductions. The training engine imports these helpers so rank-zero logic
stays consistent across checkpointing and logging.
"""

from src.runtime.distributed import (
    all_gather_objects,
    barrier,
    cleanup,
    is_initialized,
    is_rank0,
    local_rank,
    mean_dict,
    mean_scalar,
    rank,
    reduce_sum_count,
    setup_from_env,
    sum_scalar,
    sum_tensor,
    unwrap_model,
    world_size,
    wrap_model_for_distributed,
)

__all__ = [
    'all_gather_objects',
    'barrier',
    'cleanup',
    'is_initialized',
    'is_rank0',
    'local_rank',
    'mean_dict',
    'mean_scalar',
    'rank',
    'reduce_sum_count',
    'setup_from_env',
    'sum_scalar',
    'sum_tensor',
    'unwrap_model',
    'world_size',
    'wrap_model_for_distributed',
]
