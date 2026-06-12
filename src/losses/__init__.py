"""Loss registry exports.

Importing this package registers built-in loss functions. Use ``build_loss`` from
tasks or register new losses with ``@register_loss``.
"""

from .losses import BCEWithLogitsLoss, CrossEntropyLoss, FocalLoss, LabelSmoothingLoss, MSELoss, build_loss

__all__ = ['BCEWithLogitsLoss', 'CrossEntropyLoss', 'FocalLoss', 'LabelSmoothingLoss', 'MSELoss', 'build_loss']
