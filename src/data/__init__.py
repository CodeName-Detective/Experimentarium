"""Data package exports dataset registrations and dataloader builders.

Importing this package registers reference datasets in ``DATASET_REGISTRY``. Use
``build_dataloaders(cfg)`` in training code and use registry decorators when
adding new datasets.

Typical usage:
    from src.data import build_dataloaders
    loaders = build_dataloaders(cfg)
"""

from src.data.dataloader import build_dataloaders
from src.data.dataset import MyDataset, TensorFileDataset, ToyClassificationDataset, ToyRegressionDataset, ToySequenceDataset

__all__ = ['build_dataloaders', 'MyDataset', 'TensorFileDataset', 'ToyClassificationDataset', 'ToyRegressionDataset', 'ToySequenceDataset']
