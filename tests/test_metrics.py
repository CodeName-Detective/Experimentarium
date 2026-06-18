import pytest
import torch

from src.metrics import (
    accuracy,
    compute_all_metrics,
    dice_score,
    mean_iou,
    mean_reciprocal_rank,
    ndcg,
    precision_at_1,
    top5_accuracy,
)


def test_accuracy_perfect():
    preds = torch.tensor([[0.1, 0.9], [0.8, 0.2]])
    targets = torch.tensor([1, 0])
    assert accuracy(preds, targets) == pytest.approx(1.0)


def test_accuracy_zero():
    preds = torch.tensor([[0.9, 0.1], [0.2, 0.8]])
    targets = torch.tensor([1, 0])
    assert accuracy(preds, targets) == pytest.approx(0.0)


def test_compute_all_metrics_keys():
    preds = torch.randn(8, 2)
    targets = torch.randint(0, 2, (8,))
    metrics = compute_all_metrics(preds, targets)
    assert 'accuracy' in metrics


def test_top5_accuracy_handles_small_class_counts():
    logits = torch.tensor([[0.1, 0.2, 0.9], [0.8, 0.1, 0.0]])
    targets = torch.tensor([2, 0])
    assert top5_accuracy(logits, targets) == pytest.approx(1.0)


def test_segmentation_style_iou_and_dice_from_flat_logits():
    logits = torch.tensor([[4.0, 1.0], [1.0, 4.0], [4.0, 1.0]])
    targets = torch.tensor([0, 1, 0])
    assert mean_iou(logits, targets) == pytest.approx(1.0)
    assert dice_score(logits, targets) == pytest.approx(1.0)


def test_ranking_metrics_reward_correct_top_result():
    scores = torch.tensor([[0.2, 0.8, 0.4]])
    relevance = torch.tensor([[0.0, 1.0, 0.5]])
    assert ndcg(scores, relevance) == pytest.approx(1.0)
    assert mean_reciprocal_rank(scores, relevance) == pytest.approx(1.0)
    assert precision_at_1(scores, relevance) == pytest.approx(1.0)
