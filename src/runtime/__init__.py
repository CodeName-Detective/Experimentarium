"""Runtime helpers for distributed training and process-level behavior.

Use this package for rank checks, DDP setup/cleanup, barriers, and distributed
metric reductions. The training engine imports these helpers so rank-zero logic
stays consistent across checkpointing and logging.
"""

from src.runtime.distributed import barrier, cleanup, is_initialized, is_rank0, local_rank, mean_dict, mean_scalar, rank, setup_from_env, world_size

__all__ = ['barrier', 'cleanup', 'is_initialized', 'is_rank0', 'local_rank', 'mean_dict', 'mean_scalar', 'rank', 'setup_from_env', 'world_size']
