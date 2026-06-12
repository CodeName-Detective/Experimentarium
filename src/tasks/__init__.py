"""Task abstraction exports.

Use tasks to define workload-specific loss, metrics, and prediction behavior while
keeping the trainer generic.
"""

from .task import BaseTask, ClassificationTask, RegressionTask, StepResult, build_task

__all__ = ['BaseTask', 'ClassificationTask', 'RegressionTask', 'StepResult', 'build_task']
