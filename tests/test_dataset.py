from src.data import build_dataloaders
from src.utils.registry import DATASET_REGISTRY


def test_registry_contains_toy_dataset():
    assert 'toy_classification' in DATASET_REGISTRY


def test_build_dataloaders(tiny_cfg):
    loaders = build_dataloaders(tiny_cfg)
    batch = next(iter(loaders['train']))
    assert set(loaders) == {'train', 'val', 'test'}
    assert batch['input'].shape[-1] == 16
