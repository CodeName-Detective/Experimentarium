"""Training callbacks for framework extension points.

Use callbacks when you need to add logging, visualization, profiling, EMA,
custom artifacts, or task-specific behavior without modifying ``Trainer``.
"""

from src.callbacks.base import Callback, CallbackList

__all__ = ['Callback', 'CallbackList']
