"""Configuration helpers for OmegaConf and plain dictionaries.

Use these helpers instead of directly indexing config objects in framework code.
They work with both Hydra/OmegaConf objects and plain dict fixtures in tests.

Typical usage:
    from src.utils.config import cfg_get
    lr = cfg_get(cfg, 'optimizer.lr', 1e-3)
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

try:
    from omegaconf import DictConfig, OmegaConf
except Exception:  # pragma: no cover - fallback for minimal environments
    DictConfig = None  # type: ignore[assignment]
    OmegaConf = None  # type: ignore[assignment]


def cfg_get(obj: Any, key: str, default: Any = None) -> Any:
    """Read a nested key from dict, OmegaConf, or object using dot notation."""
    cur = obj
    for part in key.split('.'):
        if cur is None:
            return default
        try:
            cur = cur[part] if isinstance(cur, dict) else getattr(cur, part)
        except (KeyError, AttributeError, TypeError):
            return default
    return cur


def cfg_has(obj: Any, key: str) -> bool:
    """Return whether a nested configuration key exists."""
    sentinel = object()
    return cfg_get(obj, key, sentinel) is not sentinel


def config_to_dict(cfg: Any) -> dict[str, Any]:
    """Convert a supported configuration object into a plain dictionary."""
    if OmegaConf is not None and isinstance(cfg, DictConfig):
        return OmegaConf.to_container(cfg, resolve=True)  # type: ignore[return-value]
    if isinstance(cfg, dict):
        return dict(cfg)
    return dict(vars(cfg))


def load_config(path: str | Path) -> Any:
    """Load configuration data from a YAML file."""
    if OmegaConf is not None:
        return OmegaConf.load(path)
    import yaml

    with Path(path).open('r', encoding='utf-8') as handle:
        return yaml.safe_load(handle)


def merge_configs(*cfgs: Any) -> Any:
    """Merge configuration objects from left to right."""
    if OmegaConf is not None:
        return OmegaConf.merge(*cfgs)
    merged: dict[str, Any] = {}
    for cfg in cfgs:
        merged.update(config_to_dict(cfg))
    return merged
