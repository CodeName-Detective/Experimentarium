"""Training callbacks for framework extension points.

Use callbacks when you need to add logging, visualization, profiling, EMA,
custom artifacts, or task-specific behavior without modifying ``Trainer``.
"""

from src.callbacks.base import Callback, CallbackList
from src.callbacks.common import CheckpointArtifactLogger, GradNormLogger, LearningRateLogger, TrainingTimer
from src.callbacks.factory import build_callbacks

__all__ = [
    'Callback',
    'CallbackList',
    'CheckpointArtifactLogger',
    'GradNormLogger',
    'LearningRateLogger',
    'TrainingTimer',
    'build_callbacks',
]
