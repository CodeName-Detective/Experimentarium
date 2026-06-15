"""Task abstraction exports.

Use tasks to define workload-specific loss, metrics, and prediction behavior while
keeping the trainer generic.
"""

from .detection import DetectionTask
from .language_modeling import LanguageModelingTask
from .ranking import RankingTask
from .segmentation import SegmentationTask
from .task import BaseTask, ClassificationTask, RegressionTask, StepResult, build_task

__all__ = [
    'BaseTask',
    'ClassificationTask',
    'DetectionTask',
    'LanguageModelingTask',
    'RankingTask',
    'RegressionTask',
    'SegmentationTask',
    'StepResult',
    'build_task',
]
