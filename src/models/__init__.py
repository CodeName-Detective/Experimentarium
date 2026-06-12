"""Reference model exports and registration side effects.

Use this package before building models from ``MODEL_REGISTRY`` so built-in models
are registered.

Typical usage:
    import src.models
    from src.utils.registry import MODEL_REGISTRY
    model = MODEL_REGISTRY.build('mlp', cfg.model)
"""

from .model import MLP, MyModel, SmallCNN, SmallTransformer

__all__ = ['MLP', 'MyModel', 'SmallCNN', 'SmallTransformer']
