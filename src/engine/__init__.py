"""Training and evaluation engine exports.

Use ``Trainer`` for optimization loops and ``Evaluator`` for validation, testing,
or standalone checkpoint evaluation.
"""

from .evaluator import Evaluator
from .trainer import Trainer

__all__ = ['Evaluator', 'Trainer']
