"""Data package exports dataset registrations, dataloaders, and transforms.

Importing this package registers reference datasets in ``DATASET_REGISTRY``. Use
``build_dataloaders(cfg)`` in training code and use registry decorators when
adding new datasets or transforms.

Typical usage:
    from src.data import build_dataloaders
    loaders = build_dataloaders(cfg)
"""

from src.data.dataloader import TransformDataset, build_dataloaders
from src.data.dataset import MyDataset, TensorFileDataset, ToyClassificationDataset, ToyRegressionDataset, ToySequenceDataset
from src.data.transforms import build_transform, get_train_transforms, get_val_transforms

__all__ = [
    'MyDataset',
    'TensorFileDataset',
    'ToyClassificationDataset',
    'ToyRegressionDataset',
    'ToySequenceDataset',
    'TransformDataset',
    'build_dataloaders',
    'build_transform',
    'get_train_transforms',
    'get_val_transforms',
]
