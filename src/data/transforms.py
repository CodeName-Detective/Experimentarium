"""Configurable transform builders for dataset wrappers.

Transforms are applied to ``sample['input']`` by ``build_dataloaders`` when a
split has ``data.transforms.<split>`` or ``data.transforms.default`` configured.
Torchvision transforms are imported lazily so vector and sequence experiments do
not pay that import cost.
"""

from __future__ import annotations

from typing import Any

from src.utils.config import cfg_get
from src.utils.registry import TRANSFORM_REGISTRY, register_transform


@register_transform('identity')
def identity_transform(cfg: Any | None = None):
    """Return a no-op transform."""
    return lambda value: value


@register_transform('torchvision_train')
def torchvision_train_transform(cfg: Any | None = None):
    """Build a common torchvision training transform."""
    try:
        from torchvision import transforms
    except Exception as exc:  # pragma: no cover - depends on optional torchvision import health.
        raise ImportError('Install torchvision to use image transforms') from exc
    mean = cfg_get(cfg, 'mean', cfg_get(cfg, 'normalize.mean', (0.4914, 0.4822, 0.4465)))
    std = cfg_get(cfg, 'std', cfg_get(cfg, 'normalize.std', (0.2470, 0.2435, 0.2616)))
    size = int(cfg_get(cfg, 'size', 32))
    padding = int(cfg_get(cfg, 'padding', 4))
    return transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(size, padding=padding),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])


@register_transform('torchvision_val')
def torchvision_val_transform(cfg: Any | None = None):
    """Build a common torchvision validation transform."""
    try:
        from torchvision import transforms
    except Exception as exc:  # pragma: no cover - depends on optional torchvision import health.
        raise ImportError('Install torchvision to use image transforms') from exc
    mean = cfg_get(cfg, 'mean', cfg_get(cfg, 'normalize.mean', (0.4914, 0.4822, 0.4465)))
    std = cfg_get(cfg, 'std', cfg_get(cfg, 'normalize.std', (0.2470, 0.2435, 0.2616)))
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])


def build_transform(cfg: Any, split: str):
    """Build the transform configured for a split, or return None."""
    transform_cfg = cfg_get(cfg, f'transforms.{split}', None) or cfg_get(cfg, 'transforms.default', None)
    if transform_cfg is None:
        return None
    name = cfg_get(transform_cfg, 'name', transform_cfg if isinstance(transform_cfg, str) else 'identity')
    return TRANSFORM_REGISTRY.build(str(name), transform_cfg)


# Backward-compatible helpers for older imports.
def get_train_transforms(cfg):
    return torchvision_train_transform(cfg_get(cfg, 'data.transforms.train', cfg_get(cfg, 'data', cfg)))


def get_val_transforms(cfg):
    return torchvision_val_transform(cfg_get(cfg, 'data.transforms.val', cfg_get(cfg, 'data', cfg)))
