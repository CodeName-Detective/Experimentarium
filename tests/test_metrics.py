import pytest
import torch

from src.metrics import accuracy, compute_all_metrics


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
