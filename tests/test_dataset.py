import copy

import torch
from torch.utils.data import Dataset

from src.data import build_dataloaders
from src.utils.registry import DATASET_REGISTRY, register_dataset


@register_dataset('variable_detection_test')
class VariableDetectionDataset(Dataset):
    def __init__(self, cfg, split='train'):
        self.num_samples = 4

    def __len__(self):
        return self.num_samples

    def __getitem__(self, index):
        return {
            'input': torch.ones(3, 4 + index, 4 + index),
            'targets': {'boxes': torch.ones(index + 1, 4), 'labels': torch.ones(index + 1, dtype=torch.long)},
        }

    @staticmethod
    def collate_fn(samples):
        return {
            'input': [sample['input'] for sample in samples],
            'targets': [sample['targets'] for sample in samples],
        }


def test_registry_contains_toy_dataset():
    assert 'toy_classification' in DATASET_REGISTRY


def test_build_dataloaders(tiny_cfg):
    loaders = build_dataloaders(tiny_cfg)
    batch = next(iter(loaders['train']))
    assert set(loaders) == {'train', 'val', 'test'}
    assert batch['input'].shape[-1] == 16


def test_build_dataloaders_uses_dataset_collate_fn(tiny_cfg):
    cfg = copy.deepcopy(tiny_cfg)
    cfg['data'] = {
        'name': 'variable_detection_test',
        'batch_size': 2,
        'num_workers': 0,
        'pin_memory': False,
    }

    batch = next(iter(build_dataloaders(cfg)['train']))

    assert isinstance(batch['input'], list)
    assert isinstance(batch['targets'], list)
    assert len(batch['input']) == 2
