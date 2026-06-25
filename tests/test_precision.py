import torch

from src.engine.precision import amp_dtype, amp_enabled, normalize_precision, precision_autocast, scaler_enabled


def test_fp32_precision_uses_plain_train_and_eval_path():
    assert normalize_precision('fp32') == 'fp32'
    assert not amp_enabled(torch.device('cuda'), 'fp32')
    assert not scaler_enabled(torch.device('cuda'), 'fp32')
    assert not amp_enabled(torch.device('cpu'), 'fp32')


def test_mixed_precision_modes_use_cuda_autocast_policy():
    assert amp_enabled(torch.device('cuda'), 'amp')
    assert amp_enabled(torch.device('cuda'), 'fp16')
    assert amp_enabled(torch.device('cuda'), 'bf16')
    assert amp_dtype('amp') is torch.float16
    assert amp_dtype('fp16') is torch.float16
    assert amp_dtype('bf16') is torch.bfloat16
    assert scaler_enabled(torch.device('cuda'), 'amp')
    assert scaler_enabled(torch.device('cuda'), 'fp16')
    assert not scaler_enabled(torch.device('cuda'), 'bf16')


def test_mixed_precision_falls_back_to_plain_path_off_cuda():
    assert not amp_enabled(torch.device('cpu'), 'amp')
    assert not scaler_enabled(torch.device('cpu'), 'amp')
    with precision_autocast(torch.device('cpu'), 'amp'):
        assert True


def test_invalid_precision_is_rejected():
    import pytest

    with pytest.raises(ValueError, match='Unsupported precision'):
        normalize_precision('fp23_typo')
