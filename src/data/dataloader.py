"""Dataloader factory with registry-based datasets and reproducible workers.

Use ``build_dataloaders(cfg)`` after Hydra config composition to construct
``train``, ``val``, and ``test`` dataloaders. The dataset class is selected by
``data.name`` from ``DATASET_REGISTRY``.

Typical usage:
    from src.data import build_dataloaders
    loaders = build_dataloaders(cfg)
"""

from __future__ import annotations

from torch.utils.data import DataLoader, DistributedSampler

from src.utils.config import cfg_get
from src.utils.registry import DATASET_REGISTRY
from src.utils.seed import get_generator, seed_worker


def _is_distributed() -> bool:
    try:
        import torch.distributed as dist
        return dist.is_available() and dist.is_initialized()
    except Exception:
        return False


def build_dataloaders(cfg) -> dict[str, DataLoader]:
    data_cfg = cfg_get(cfg, 'data')
    name = cfg_get(data_cfg, 'name', 'toy_classification')
    batch_size = int(cfg_get(data_cfg, 'batch_size', 32))
    num_workers = int(cfg_get(data_cfg, 'num_workers', 0))
    requested_pin_memory = bool(cfg_get(data_cfg, 'pin_memory', False))
    pin_memory = requested_pin_memory and str(cfg_get(cfg, 'run.device', cfg_get(cfg, 'device', 'cpu'))).startswith('cuda')
    seed = int(cfg_get(cfg, 'run.seed', cfg_get(cfg, 'seed', 42)))
    loaders: dict[str, DataLoader] = {}

    for split in ('train', 'val', 'test'):
        dataset = DATASET_REGISTRY.build(name, data_cfg, split=split)
        sampler = DistributedSampler(dataset, shuffle=(split == 'train')) if _is_distributed() else None
        loaders[split] = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=(split == 'train' and sampler is None),
            sampler=sampler,
            num_workers=num_workers,
            collate_fn=getattr(dataset, 'collate_fn', None),
            pin_memory=pin_memory,
            drop_last=bool(cfg_get(data_cfg, 'drop_last', False) and split == 'train'),
            worker_init_fn=seed_worker if num_workers > 0 else None,
            generator=get_generator(seed) if sampler is None else None,
        )
    return loaders
