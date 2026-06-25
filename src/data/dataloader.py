"""Dataloader factory with registry-based datasets and reproducible workers.

Use ``build_dataloaders(cfg)`` after Hydra config composition to construct
``train``, ``val``, and ``test`` dataloaders. The dataset class is selected by
``data.name`` from ``DATASET_REGISTRY``.

Typical usage:
    from src.data import build_dataloaders
    loaders = build_dataloaders(cfg)
"""

from __future__ import annotations

from collections.abc import Iterator, Sized
from typing import Any, cast

from torch.utils.data import DataLoader, Dataset, DistributedSampler, Sampler
from torch.utils.data._utils.collate import default_collate

from src.data.transforms import build_transform
from src.utils.config import cfg_get
from src.utils.registry import DATASET_REGISTRY
from src.utils.seed import get_generator, seed_worker


def _is_distributed() -> bool:
    try:
        import torch.distributed as dist

        return dist.is_available() and dist.is_initialized()
    except Exception:
        return False


class DistributedEvalSampler(Sampler[int]):
    """Shard evaluation data without padding or duplicating samples."""

    def __init__(self, dataset: Dataset) -> None:
        import torch.distributed as dist

        self.dataset = dataset
        self.dataset_size = len(cast(Sized, dataset))
        self.num_replicas = dist.get_world_size()
        self.rank = dist.get_rank()

    def __iter__(self) -> Iterator[int]:
        return iter(range(self.rank, self.dataset_size, self.num_replicas))

    def __len__(self) -> int:
        remaining = max(0, self.dataset_size - self.rank)
        return (remaining + self.num_replicas - 1) // self.num_replicas


class TransformDataset(Dataset):
    """Apply a transform to sample['input'] without changing the source dataset."""

    def __init__(self, dataset: Dataset, transform: Any) -> None:
        self.dataset = dataset
        self.transform = transform
        self._collate_fn = getattr(dataset, 'collate_fn', None)

    def __len__(self) -> int:
        return len(cast(Sized, self.dataset))

    def __getitem__(self, index: int) -> Any:
        sample = self.dataset[index]
        if isinstance(sample, dict) and 'input' in sample:
            sample = dict(sample)
            sample['input'] = self.transform(sample['input'])
        return sample

    def collate_fn(self, batch):
        return self._collate_fn(batch) if self._collate_fn is not None else default_collate(batch)


def _split_value(data_cfg, split: str, key: str, default: Any = None) -> Any:
    return cfg_get(data_cfg, f'splits.{split}.{key}', cfg_get(data_cfg, key, default))


def _limit_prefetch(num_workers: int, prefetch_factor: Any) -> Any:
    return int(prefetch_factor) if num_workers > 0 and prefetch_factor is not None else None


def build_dataloaders(cfg) -> dict[str, DataLoader]:
    data_cfg = cfg_get(cfg, 'data')
    name = cfg_get(data_cfg, 'name', 'toy_classification')
    seed = int(cfg_get(cfg, 'run.seed', cfg_get(cfg, 'seed', 42)))
    device_name = str(cfg_get(cfg, 'run.device', cfg_get(cfg, 'device', 'cpu')))
    loaders: dict[str, DataLoader] = {}

    for split in ('train', 'val', 'test'):
        dataset = DATASET_REGISTRY.build(name, data_cfg, split=split)
        transform = build_transform(data_cfg, split)
        if transform is not None:
            dataset = TransformDataset(dataset, transform)
        batch_size = int(_split_value(data_cfg, split, 'batch_size', 32))
        num_workers = int(_split_value(data_cfg, split, 'num_workers', 0))
        requested_pin_memory = bool(_split_value(data_cfg, split, 'pin_memory', False))
        pin_memory = requested_pin_memory and device_name.startswith('cuda')
        shuffle = bool(_split_value(data_cfg, split, 'shuffle', split == 'train'))
        sampler = None
        if _is_distributed():
            sampler = (
                DistributedSampler(dataset, shuffle=shuffle) if split == 'train' else DistributedEvalSampler(dataset)
            )
        prefetch_factor = _limit_prefetch(num_workers, _split_value(data_cfg, split, 'prefetch_factor', None))
        loaders[split] = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle and sampler is None,
            sampler=sampler,
            num_workers=num_workers,
            collate_fn=getattr(dataset, 'collate_fn', None),
            pin_memory=pin_memory,
            drop_last=bool(_split_value(data_cfg, split, 'drop_last', False) and split == 'train'),
            worker_init_fn=seed_worker if num_workers > 0 else None,
            generator=get_generator(seed) if sampler is None else None,
            persistent_workers=bool(_split_value(data_cfg, split, 'persistent_workers', False)) and num_workers > 0,
            prefetch_factor=prefetch_factor,
        )
    return loaders
