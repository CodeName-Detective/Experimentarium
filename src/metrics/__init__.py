"""Metric registry exports.

Use this package to register built-in metrics before constructing tasks. Import
``MetricCollection`` directly when writing custom task tests.

Typical usage:
    from src.metrics import MetricCollection
    metrics = MetricCollection.from_names(['accuracy'])
"""

from .metrics import (
    MetricAccumulator,
    MetricCollection,
    accuracy,
    compute_all_metrics,
    dice_score,
    mae,
    map50_placeholder,
    mean_iou,
    mean_reciprocal_rank,
    mse,
    ndcg,
    precision_at_1,
    top5_accuracy,
)

__all__ = [
    'MetricAccumulator',
    'MetricCollection',
    'accuracy',
    'compute_all_metrics',
    'dice_score',
    'mae',
    'map50_placeholder',
    'mean_iou',
    'mean_reciprocal_rank',
    'mse',
    'ndcg',
    'precision_at_1',
    'top5_accuracy',
]
