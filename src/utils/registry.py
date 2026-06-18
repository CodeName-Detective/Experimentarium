"""Named registries used by framework factories.

Use registries to add components without editing the trainer or entrypoint. Each
registry maps a stable config name to a callable.

Typical usage:
    from src.utils.registry import register_model, MODEL_REGISTRY
    @register_model('my_model')
    class MyModel(nn.Module): ...
    model = MODEL_REGISTRY.build('my_model', cfg.model)
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import Any, TypeVar

RegistryItem = TypeVar('RegistryItem', bound=Callable[..., Any])


class Registry:
    """Simple name-to-callable registry with useful error messages."""

    def __init__(self, register_name: str) -> None:
        self._register_name = register_name
        self._store: dict[str, Callable[..., Any]] = {}

    def register(self, item_name: str) -> Callable[[RegistryItem], RegistryItem]:
        """Create a decorator that registers a callable under ``item_name``."""

        def decorator(item: RegistryItem) -> RegistryItem:
            if item_name in self._store:
                raise KeyError(f"{self._register_name} registry already has '{item_name}'")
            self._store[item_name] = item
            return item

        return decorator

    def build(self, item_name: str, *args: Any, **kwargs: Any) -> Any:
        """Construct a registered item with the supplied arguments."""
        return self.get(item_name)(*args, **kwargs)

    def get(self, item_name: str) -> Callable[..., Any]:
        """Return the callable registered under ``item_name``."""
        if item_name not in self._store:
            available = sorted(self._store.keys())
            raise KeyError(f"'{item_name}' not in {self._register_name} registry. Available: {available}")
        return self._store[item_name]

    def keys(self) -> Iterable[str]:
        """Return the registered names."""
        return self._store.keys()

    def __contains__(self, item_name: str) -> bool:
        return item_name in self._store

    def __repr__(self) -> str:
        return f'Registry({self._register_name}): {sorted(self._store.keys())}'


MODEL_REGISTRY = Registry('model')
DATASET_REGISTRY = Registry('dataset')
TRANSFORM_REGISTRY = Registry('transform')
LOSS_REGISTRY = Registry('loss')
METRIC_REGISTRY = Registry('metric')
OPTIMIZER_REGISTRY = Registry('optimizer')
SCHEDULER_REGISTRY = Registry('scheduler')
TASK_REGISTRY = Registry('task')
CALLBACK_REGISTRY = Registry('callback')
LOGGER_REGISTRY = Registry('logger')


def register_model(name: str) -> Callable[[RegistryItem], RegistryItem]:
    """Register a model factory or class."""
    return MODEL_REGISTRY.register(name)


def register_dataset(name: str) -> Callable[[RegistryItem], RegistryItem]:
    """Register a dataset factory or class."""
    return DATASET_REGISTRY.register(name)


def register_transform(name: str) -> Callable[[RegistryItem], RegistryItem]:
    """Register a transform factory or callable."""
    return TRANSFORM_REGISTRY.register(name)


def register_loss(name: str) -> Callable[[RegistryItem], RegistryItem]:
    """Register a loss factory or class."""
    return LOSS_REGISTRY.register(name)


def register_metric(name: str) -> Callable[[RegistryItem], RegistryItem]:
    """Register a metric function or class."""
    return METRIC_REGISTRY.register(name)


def register_optimizer(name: str) -> Callable[[RegistryItem], RegistryItem]:
    """Register an optimizer factory."""
    return OPTIMIZER_REGISTRY.register(name)


def register_scheduler(name: str) -> Callable[[RegistryItem], RegistryItem]:
    """Register a scheduler factory."""
    return SCHEDULER_REGISTRY.register(name)


def register_task(name: str) -> Callable[[RegistryItem], RegistryItem]:
    """Register a task factory or class."""
    return TASK_REGISTRY.register(name)


def register_callback(name: str) -> Callable[[RegistryItem], RegistryItem]:
    """Register a callback factory or class."""
    return CALLBACK_REGISTRY.register(name)


def register_logger(name: str) -> Callable[[RegistryItem], RegistryItem]:
    """Register a logger factory or class."""
    return LOGGER_REGISTRY.register(name)
