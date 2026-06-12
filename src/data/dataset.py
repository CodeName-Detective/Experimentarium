"""Reference datasets and dataset registry entries.

These datasets keep the framework runnable from a clean clone and demonstrate the
expected sample contract: each item returns a dictionary containing at least
``input`` and ``label``. Real datasets should follow the same contract or provide
a custom task that understands their batch keys.

Typical usage:
    from src.data import build_dataloaders
    loaders = build_dataloaders(cfg)

Registered datasets:
    - ``toy_classification``: deterministic vector classification.
    - ``toy_regression``: deterministic vector regression.
    - ``toy_sequence``: token sequences for transformer smoke tests.
    - ``tensor_file``: file-backed ``.pt`` samples with schema validation.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset

from src.utils.config import cfg_get
from src.utils.registry import register_dataset


def _split_offset(split: str) -> int:
    return {'train': 0, 'val': 1000, 'test': 2000}.get(split, 3000)


@register_dataset('toy_classification')
class ToyClassificationDataset(Dataset):
    """Deterministic synthetic classification dataset with shared split boundary."""

    def __init__(self, cfg, split: str = 'train') -> None:
        self.split = split
        self.input_dim = int(cfg_get(cfg, 'input_dim', 16))
        self.num_classes = int(cfg_get(cfg, 'num_classes', 2))
        n_default = {'train': 256, 'val': 64, 'test': 64}.get(split, 64)
        self.num_samples = int(cfg_get(cfg, f'splits.{split}.num_samples', n_default))
        base_seed = int(cfg_get(cfg, 'seed', 42))
        data_generator = torch.Generator().manual_seed(base_seed + _split_offset(split))
        weight_generator = torch.Generator().manual_seed(base_seed + 9999)
        self.x = torch.randn(self.num_samples, self.input_dim, generator=data_generator)
        weights = torch.randn(self.input_dim, self.num_classes, generator=weight_generator)
        self.y = (self.x @ weights).argmax(dim=-1).long()

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return {'input': self.x[idx], 'label': self.y[idx]}


@register_dataset('toy_regression')
class ToyRegressionDataset(Dataset):
    """Deterministic synthetic regression dataset for regression task validation."""

    def __init__(self, cfg, split: str = 'train') -> None:
        self.split = split
        self.input_dim = int(cfg_get(cfg, 'input_dim', 16))
        self.output_dim = int(cfg_get(cfg, 'output_dim', 1))
        n_default = {'train': 256, 'val': 64, 'test': 64}.get(split, 64)
        self.num_samples = int(cfg_get(cfg, f'splits.{split}.num_samples', n_default))
        base_seed = int(cfg_get(cfg, 'seed', 42))
        data_generator = torch.Generator().manual_seed(base_seed + _split_offset(split))
        weight_generator = torch.Generator().manual_seed(base_seed + 9999)
        self.x = torch.randn(self.num_samples, self.input_dim, generator=data_generator)
        weights = torch.randn(self.input_dim, self.output_dim, generator=weight_generator)
        noise = 0.05 * torch.randn(self.num_samples, self.output_dim, generator=data_generator)
        self.y = self.x @ weights + noise

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return {'input': self.x[idx], 'label': self.y[idx]}


@register_dataset('toy_sequence')
class ToySequenceDataset(Dataset):
    """Token-sequence classification dataset for transformer smoke tests."""

    def __init__(self, cfg, split: str = 'train') -> None:
        self.split = split
        self.seq_len = int(cfg_get(cfg, 'seq_len', 16))
        self.vocab_size = int(cfg_get(cfg, 'vocab_size', 128))
        self.num_classes = int(cfg_get(cfg, 'num_classes', 2))
        n_default = {'train': 256, 'val': 64, 'test': 64}.get(split, 64)
        self.num_samples = int(cfg_get(cfg, f'splits.{split}.num_samples', n_default))
        generator = torch.Generator().manual_seed(int(cfg_get(cfg, 'seed', 42)) + _split_offset(split))
        self.tokens = torch.randint(0, self.vocab_size, (self.num_samples, self.seq_len), generator=generator)
        self.labels = (self.tokens.sum(dim=1) % self.num_classes).long()

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return {'input': self.tokens[idx], 'label': self.labels[idx]}


@register_dataset('tensor_file')
@register_dataset('default')
class TensorFileDataset(Dataset):
    """Dataset for ``.pt`` files containing samples with ``input`` and ``label`` fields."""

    def __init__(self, cfg, split: str = 'train') -> None:
        path = cfg_get(cfg, f'splits.{split}.path', None) or cfg_get(cfg, f'{split}_path', None)
        if path is None:
            raise ValueError(f"No path configured for split '{split}'")
        self.path = Path(path)
        self.split = split
        if not self.path.exists():
            raise FileNotFoundError(f'Tensor dataset not found: {self.path}')
        self.data = torch.load(self.path, map_location='cpu', weights_only=True)
        self._validate_schema()

    def _validate_schema(self) -> None:
        if not hasattr(self.data, '__len__'):
            raise TypeError(f'{self.path} must contain a sized sequence of samples')
        if len(self.data) == 0:
            raise ValueError(f'{self.path} contains no samples')
        sample: Any = self.data[0]
        if not isinstance(sample, dict) or 'input' not in sample or 'label' not in sample:
            raise ValueError(f'{self.path} samples must be dicts with input and label keys')

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        sample = self.data[idx]
        return {'input': sample['input'], 'label': sample['label']}


MyDataset = TensorFileDataset
