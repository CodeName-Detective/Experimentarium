"""Tests for fault-tolerant checkpoint save/load behavior."""

from pathlib import Path

from src.models import MLP
from src.utils.checkpoint import CheckpointManager


def _state(model, epoch=1):
    return {
        'epoch': epoch,
        'global_step': epoch,
        'model_state': model.state_dict(),
        'optimizer_state': None,
        'scheduler_state': None,
        'scaler_state': None,
        'rng_state': None,
        'best_metric': 1.0,
        'metrics': {'val/loss': 1.0},
        'cfg': {},
    }


def test_checkpoint_manifest_and_best(tmp_path):
    model = MLP({'input_dim': 4, 'hidden_dim': 8, 'output_dim': 2})
    manager = CheckpointManager(tmp_path, keep_last_k=2, save_top_k=1)
    path = manager.save(_state(model), epoch=1, metric=0.5, is_best=True)
    assert path is not None
    assert Path(tmp_path, 'manifest.json').exists()
    assert Path(tmp_path, 'best.pt').exists()
    assert Path(tmp_path, 'last.pt').exists()


def test_load_latest_falls_back_from_corrupt_last(tmp_path):
    model = MLP({'input_dim': 4, 'hidden_dim': 8, 'output_dim': 2})
    manager = CheckpointManager(tmp_path, keep_last_k=2, save_top_k=1)
    manager.save(_state(model), epoch=1, metric=0.5, is_best=True)
    Path(tmp_path, 'last.pt').write_text('corrupt', encoding='utf-8')
    reloaded = MLP({'input_dim': 4, 'hidden_dim': 8, 'output_dim': 2})
    state = manager.load_latest(reloaded)
    assert state is not None
    assert state['epoch'] == 1


def test_checkpoint_resume_selector_paths(tmp_path):
    manager = CheckpointManager(tmp_path)

    assert manager.resolve_resume_path('best') == Path(tmp_path, 'best.pt')
    assert manager.resolve_resume_path('last') == Path(tmp_path, 'last.pt')
    assert manager.resolve_resume_path('5') == Path(tmp_path, 'epoch_0005.pt')
    assert manager.resolve_resume_path('epoch_5') == Path(tmp_path, 'epoch_0005.pt')
    assert manager.resolve_resume_path('epoch_0005.pt') == Path(tmp_path, 'epoch_0005.pt')
    assert manager.resolve_resume_path('custom/path.pt') == Path('custom/path.pt')


def test_checkpoint_checksum_uses_latest_manifest_entry(tmp_path):
    model = MLP({'input_dim': 4, 'hidden_dim': 8, 'output_dim': 2})
    manager = CheckpointManager(tmp_path, keep_last_k=10, save_top_k=1)
    manager.save(_state(model, epoch=5), epoch=5, metric=0.5, is_best=False)
    manager.save(_state(model, epoch=5), epoch=5, metric=0.4, is_best=False)

    reloaded = MLP({'input_dim': 4, 'hidden_dim': 8, 'output_dim': 2})
    state = manager.load(Path(tmp_path, 'epoch_0005.pt'), reloaded)

    assert state['epoch'] == 5
