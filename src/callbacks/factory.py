"""Callback factory helpers."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from src.callbacks.base import Callback
from src.utils.config import cfg_get
from src.utils.registry import CALLBACK_REGISTRY


def build_callbacks(cfg: Any) -> list[Callback]:
    """Build enabled callbacks from a top-level ``callbacks`` config list."""
    callback_configs = list(cfg_get(cfg, 'callbacks', []) or [])
    callbacks: list[Callback] = []
    for callback_cfg in callback_configs:
        if not bool(cfg_get(callback_cfg, 'enabled', True)):
            continue
        name = cfg_get(callback_cfg, 'name', None)
        if not name:
            raise ValueError('Each callback config requires a name field')
        callbacks.append(CALLBACK_REGISTRY.build(str(name), callback_cfg))
    return callbacks
