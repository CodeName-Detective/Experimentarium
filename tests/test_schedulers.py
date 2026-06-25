"""Tests for scheduler registry metadata."""

import pytest
import torch

from src.optim import build_scheduler
from src.optim.schedulers import SCHEDULER_DESCRIPTIONS


def test_scheduler_descriptions_are_present():
    expected = {
        'none',
        'constant',
        'linear',
        'step',
        'multistep',
        'exponential',
        'cosine',
        'cosine_restart',
        'plateau',
        'polynomial',
        'onecycle',
    }
    assert expected <= set(SCHEDULER_DESCRIPTIONS)
    assert all(SCHEDULER_DESCRIPTIONS[name] for name in expected)


def test_onecycle_horizon_uses_optimizer_steps_and_max_steps():
    parameter = torch.nn.Parameter(torch.tensor(1.0))
    optimizer = torch.optim.SGD([parameter], lr=0.1)
    cfg = {
        'trainer': {
            'max_epochs': 5,
            'max_steps': 3,
            'accumulate_grad_batches': 4,
            'limit_train_batches': 8,
        },
        'scheduler': {'name': 'onecycle', 'interval': 'step', 'max_lr': 0.1},
    }

    scheduler = build_scheduler(cfg, optimizer, steps_per_epoch=8)

    assert scheduler.scheduler.total_steps == 3


def test_plateau_scheduler_rejects_warmup_wrapper():
    parameter = torch.nn.Parameter(torch.tensor(1.0))
    optimizer = torch.optim.SGD([parameter], lr=0.1)
    cfg = {
        'trainer': {'max_epochs': 5},
        'scheduler': {
            'name': 'plateau',
            'interval': 'epoch',
            'monitor': 'val/loss',
            'warmup': {'enabled': True, 'steps': 1},
        },
    }

    with pytest.raises(ValueError, match='not supported with ReduceLROnPlateau'):
        build_scheduler(cfg, optimizer, steps_per_epoch=4)


def test_warmup_is_included_inside_cosine_training_horizon():
    parameter = torch.nn.Parameter(torch.tensor(1.0))
    optimizer = torch.optim.SGD([parameter], lr=0.1)
    cfg = {
        'trainer': {'max_epochs': 5},
        'scheduler': {
            'name': 'cosine',
            'interval': 'epoch',
            'warmup': {'enabled': True, 'steps': 2, 'start_factor': 0.1},
        },
    }

    bundle = build_scheduler(cfg, optimizer, steps_per_epoch=4)

    assert bundle.scheduler._milestones == [2]
    assert bundle.scheduler._schedulers[1].T_max == 3
