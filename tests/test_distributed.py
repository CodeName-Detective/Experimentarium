from types import SimpleNamespace

import torch

from src.runtime import distributed


def test_setup_from_env_gloo_does_not_bind_cuda(monkeypatch):
    initialized_backends = []
    fake_dist = SimpleNamespace(
        is_available=lambda: True,
        is_initialized=lambda: False,
        init_process_group=lambda *, backend: initialized_backends.append(backend),
    )
    monkeypatch.setattr(distributed, 'dist', fake_dist)
    monkeypatch.setenv('RANK', '0')
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: True)

    def fail_set_device(device):
        raise AssertionError(f'Gloo unexpectedly selected CUDA device {device}')

    monkeypatch.setattr(torch.cuda, 'set_device', fail_set_device)

    assert distributed.setup_from_env('gloo')
    assert initialized_backends == ['gloo']


def test_setup_from_env_nccl_binds_local_rank(monkeypatch):
    initialized_backends = []
    selected_devices = []
    fake_dist = SimpleNamespace(
        is_available=lambda: True,
        is_initialized=lambda: False,
        init_process_group=lambda *, backend: initialized_backends.append(backend),
    )
    monkeypatch.setattr(distributed, 'dist', fake_dist)
    monkeypatch.setenv('RANK', '1')
    monkeypatch.setenv('LOCAL_RANK', '1')
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: True)
    monkeypatch.setattr(torch.cuda, 'set_device', selected_devices.append)

    assert distributed.setup_from_env('NCCL')
    assert initialized_backends == ['nccl']
    assert selected_devices == [1]
