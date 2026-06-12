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

from typing import Any, Callable, Iterable


class Registry:
    """Simple name-to-callable registry with useful error messages."""

    def __init__(self, register_name: str) -> None:
        self._register_name = register_name
        self._store: dict[str, Any] = {}

    def register(self, item_name: str) -> Callable[[Any], Any]:
        def decorator(item: Any) -> Any:
            if item_name in self._store:
                raise KeyError(f"{self._register_name} registry already has '{item_name}'")
            self._store[item_name] = item
            return item
        return decorator

    def build(self, item_name: str, *args: Any, **kwargs: Any) -> Any:
        return self.get(item_name)(*args, **kwargs)

    def get(self, item_name: str) -> Any:
        if item_name not in self._store:
            available = sorted(self._store.keys())
            raise KeyError(f"'{item_name}' not in {self._register_name} registry. Available: {available}")
        return self._store[item_name]

    def keys(self) -> Iterable[str]:
        return self._store.keys()

    def __contains__(self, item_name: str) -> bool:
        return item_name in self._store

    def __repr__(self) -> str:
        return f"Registry({self._register_name}): {sorted(self._store.keys())}"


MODEL_REGISTRY = Registry('model')
DATASET_REGISTRY = Registry('dataset')
LOSS_REGISTRY = Registry('loss')
METRIC_REGISTRY = Registry('metric')
OPTIMIZER_REGISTRY = Registry('optimizer')
SCHEDULER_REGISTRY = Registry('scheduler')
TASK_REGISTRY = Registry('task')
CALLBACK_REGISTRY = Registry('callback')
LOGGER_REGISTRY = Registry('logger')


def register_model(name: str):
    return MODEL_REGISTRY.register(name)


def register_dataset(name: str):
    return DATASET_REGISTRY.register(name)


def register_loss(name: str):
    return LOSS_REGISTRY.register(name)


def register_metric(name: str):
    return METRIC_REGISTRY.register(name)


def register_optimizer(name: str):
    return OPTIMIZER_REGISTRY.register(name)


def register_scheduler(name: str):
    return SCHEDULER_REGISTRY.register(name)


def register_task(name: str):
    return TASK_REGISTRY.register(name)


def register_callback(name: str):
    return CALLBACK_REGISTRY.register(name)


def register_logger(name: str):
    return LOGGER_REGISTRY.register(name)
