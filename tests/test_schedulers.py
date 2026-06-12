"""Tests for scheduler registry metadata."""

from src.optim.schedulers import SCHEDULER_DESCRIPTIONS


def test_scheduler_descriptions_are_present():
    expected = {'none', 'constant', 'linear', 'step', 'multistep', 'exponential', 'cosine', 'cosine_restart', 'plateau', 'polynomial', 'onecycle'}
    assert expected <= set(SCHEDULER_DESCRIPTIONS)
    assert all(SCHEDULER_DESCRIPTIONS[name] for name in expected)
