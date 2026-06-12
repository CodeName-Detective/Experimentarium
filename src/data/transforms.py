"""Optional transform builders for image experiments.

Use these helpers only for torchvision-backed image datasets. They import
torchvision lazily so vector and sequence experiments do not require it.

Typical usage:
    train_tfms = get_train_transforms(cfg)
    val_tfms = get_val_transforms(cfg)
"""

from __future__ import annotations


def get_train_transforms(cfg):
    try:
        from torchvision import transforms
    except Exception as exc:
        raise ImportError("Install torchvision to use image transforms") from exc
    mean = cfg.data.normalize.mean
    std = cfg.data.normalize.std
    return transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])


def get_val_transforms(cfg):
    try:
        from torchvision import transforms
    except Exception as exc:
        raise ImportError("Install torchvision to use image transforms") from exc
    mean = cfg.data.normalize.mean
    std = cfg.data.normalize.std
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
