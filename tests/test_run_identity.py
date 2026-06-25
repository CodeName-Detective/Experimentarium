import copy
import json
import shlex
import shutil
import sys
from pathlib import Path

import pytest

from src.utils.run import prepare_run


def test_prepare_run_derives_stable_id_and_allocates_trials(tmp_path, tiny_cfg):
    cfg = copy.deepcopy(tiny_cfg)
    cfg['run']['name'] = 'unit_test'
    cfg['run']['output_dir'] = str(tmp_path)

    info = prepare_run(cfg)

    assert cfg['run']['id'] == info.run_id
    assert cfg['run']['trial_id'] == 1
    assert cfg['run']['config_id'] == info.config_id
    assert cfg['run']['run_dir'] == str(Path(tmp_path, 'runs', info.run_id, 'trial_1'))
    assert cfg['checkpoint']['dir'] == str(Path(cfg['run']['run_dir'], 'checkpoints'))
    assert cfg['logging']['jsonl']['path'] == str(Path(cfg['run']['run_dir'], 'logs', 'metrics.jsonl'))
    assert info.config_path == str(Path(tmp_path, 'run_configs', info.run_id, 'trial_1.yaml'))
    assert Path(info.config_path).exists()
    assert Path(info.config_registry).exists()

    same_cfg = copy.deepcopy(tiny_cfg)
    same_cfg['run']['name'] = 'unit_test'
    same_cfg['run']['output_dir'] = str(tmp_path)
    same_info = prepare_run(same_cfg)

    assert same_info.config_id == info.config_id
    assert same_info.run_id == info.run_id
    assert same_info.trial_id == 2
    assert same_info.reused_existing is False
    assert same_info.warning is None
    assert same_cfg['run']['run_dir'] == str(Path(tmp_path, 'runs', info.run_id, 'trial_2'))

    third_cfg = copy.deepcopy(tiny_cfg)
    third_cfg['run']['name'] = 'unit_test'
    third_cfg['run']['output_dir'] = str(tmp_path)
    third_info = prepare_run(third_cfg)
    assert third_info.run_id == info.run_id
    assert third_info.trial_id == 3

    checkpoint = Path(info.run_dir, 'checkpoints', 'last.pt')
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b'placeholder')
    resume_cfg = copy.deepcopy(tiny_cfg)
    resume_cfg['run']['name'] = 'unit_test'
    resume_cfg['run']['output_dir'] = str(tmp_path)
    resume_cfg['checkpoint']['resume'] = 'latest'
    resume_info = prepare_run(resume_cfg)

    assert resume_info.config_id == info.config_id
    assert resume_info.run_id == info.run_id
    assert resume_info.trial_id == 1
    assert resume_info.reused_existing is True
    assert resume_cfg['checkpoint']['dir'] == str(Path(info.run_dir, 'checkpoints'))


def test_sanity_settings_do_not_change_experiment_identity(tmp_path, tiny_cfg):
    cfg = copy.deepcopy(tiny_cfg)
    cfg['run']['output_dir'] = str(tmp_path)
    first = prepare_run(cfg)

    changed = copy.deepcopy(tiny_cfg)
    changed['run']['output_dir'] = str(tmp_path)
    changed['sanity']['strict'] = not bool(changed['sanity']['strict'])
    changed['sanity']['min_disk_gb'] = 999.0
    second = prepare_run(changed)

    assert second.config_id == first.config_id
    assert second.run_id == first.run_id
    assert second.trial_id == 2


def test_prepare_run_allocates_trials_for_explicit_id_collisions(tmp_path, tiny_cfg):
    cfg = copy.deepcopy(tiny_cfg)
    cfg['run']['id'] = 'manual run'
    cfg['run']['output_dir'] = str(tmp_path)

    info = prepare_run(cfg)
    assert info.run_id == 'manual-run'
    assert info.trial_id == 1

    next_cfg = copy.deepcopy(tiny_cfg)
    next_cfg['run']['id'] = 'manual run'
    next_cfg['run']['output_dir'] = str(tmp_path)
    next_info = prepare_run(next_cfg)

    assert next_info.run_id == 'manual-run'
    assert next_info.trial_id == 2
    assert next_cfg['run']['run_dir'] == str(Path(tmp_path, 'runs', 'manual-run', 'trial_2'))


def test_user_supplied_trial_and_tracking_values_are_overwritten(tmp_path, tiny_cfg):
    cfg = copy.deepcopy(tiny_cfg)
    cfg['run'].update({
        'id': 'managed-identity',
        'trial_id': 99,
        'tracking_id': 'user-tracking-id',
        'output_dir': str(tmp_path),
    })
    cfg['logging']['wandb']['run_name'] = 'user-wandb-name'

    info = prepare_run(cfg)

    assert info.trial_id == 1
    assert cfg['run']['trial_id'] == 1
    assert cfg['run']['tracking_id'] == 'managed-identity-trial-1'
    assert cfg['logging']['wandb']['run_name'] == 'managed-identity-trial-1'


def test_explicit_checkpoint_path_adopts_its_output_root(tmp_path, tiny_cfg):
    source_root = tmp_path / 'source-output'
    checkpoint = source_root / 'runs' / 'source-run' / 'trial_4' / 'checkpoints' / 'best.pt'
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b'checkpoint')

    cfg = copy.deepcopy(tiny_cfg)
    cfg['run']['output_dir'] = str(tmp_path / 'wrong-output-root')
    cfg['run']['mode'] = 'eval'
    cfg['checkpoint']['resume'] = str(checkpoint)

    info = prepare_run(cfg)

    assert info.run_id == 'source-run'
    assert info.trial_id == 4
    assert info.run_dir == str(source_root / 'evaluations' / 'source-run' / 'trial_4' / 'eval_best')
    assert info.config_registry == str(source_root / 'run_registry.jsonl')
    assert cfg['run']['output_dir'] == str(source_root)


def test_trial_numbers_remain_monotonic_after_artifact_cleanup(tmp_path, tiny_cfg):
    cfg = copy.deepcopy(tiny_cfg)
    cfg['run']['id'] = 'cleanup-test'
    cfg['run']['output_dir'] = str(tmp_path)
    first = prepare_run(cfg)
    shutil.rmtree(first.run_dir)

    next_cfg = copy.deepcopy(tiny_cfg)
    next_cfg['run']['id'] = 'cleanup-test'
    next_cfg['run']['output_dir'] = str(tmp_path)
    second = prepare_run(next_cfg)

    assert second.trial_id == 2


def test_prepare_run_eval_uses_checkpoint_identity_and_overwrites_same_target(tmp_path, tiny_cfg):
    training_run_id = 'trained-run'
    training_trial_id = 3
    checkpoint = Path(
        tmp_path,
        'runs',
        training_run_id,
        f'trial_{training_trial_id}',
        'checkpoints',
        'best.pt',
    )
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b'placeholder')

    cfg = copy.deepcopy(tiny_cfg)
    cfg['run']['id'] = 'ignored-config-id'
    cfg['run']['output_dir'] = str(tmp_path)
    cfg['run']['mode'] = 'eval'
    cfg['checkpoint']['resume'] = str(checkpoint)
    info = prepare_run(cfg)

    evaluation_dir = Path(
        tmp_path,
        'evaluations',
        training_run_id,
        f'trial_{training_trial_id}',
        'eval_best',
    )
    assert info.run_id == training_run_id
    assert info.trial_id == training_trial_id
    assert info.checkpoint_label == 'best'
    assert info.run_dir == str(evaluation_dir)
    assert info.config_path == str(evaluation_dir / 'config.yaml')
    assert cfg['run']['source_trial_id'] == training_trial_id
    assert cfg['run']['tracking_id'] == f'{training_run_id}-trial-3-eval-best'
    assert cfg['logging']['wandb']['run_name'] == f'{training_run_id}-trial-3-eval-best'
    assert cfg['checkpoint']['dir'] == str(evaluation_dir / 'checkpoints')
    assert cfg['run']['prediction_dir'] == str(evaluation_dir / 'predictions')

    stale = evaluation_dir / 'stale.txt'
    stale.write_text('remove me', encoding='utf-8')
    repeat_cfg = copy.deepcopy(tiny_cfg)
    repeat_cfg['run']['output_dir'] = str(tmp_path)
    repeat_cfg['run']['mode'] = 'eval'
    repeat_cfg['checkpoint']['resume'] = str(checkpoint)
    repeat_info = prepare_run(repeat_cfg)

    assert repeat_info.run_id == training_run_id
    assert repeat_info.trial_id == training_trial_id
    assert repeat_info.run_dir == str(evaluation_dir)
    assert repeat_info.reused_existing is True
    assert repeat_info.warning is not None
    assert 'OVERWRITING EXISTING EVALUATION OUTPUT' in repeat_info.warning
    assert not stale.exists()


def test_missing_evaluation_checkpoint_does_not_delete_existing_output(tmp_path, tiny_cfg):
    checkpoint = tmp_path / 'runs' / 'trained-run' / 'trial_2' / 'checkpoints' / 'best.pt'
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b'checkpoint')

    cfg = copy.deepcopy(tiny_cfg)
    cfg['run']['output_dir'] = str(tmp_path)
    cfg['run']['mode'] = 'eval'
    cfg['checkpoint']['resume'] = str(checkpoint)
    info = prepare_run(cfg)
    marker_path = Path(info.run_dir, 'keep.txt')
    marker_path.write_text('existing result', encoding='utf-8')
    checkpoint.unlink()

    with pytest.raises(FileNotFoundError, match='Evaluation checkpoint does not exist'):
        prepare_run(copy.deepcopy(cfg))

    assert marker_path.read_text(encoding='utf-8') == 'existing result'


def test_prepare_run_eval_checkpoint_selectors_use_checkpoint_specific_folders(tmp_path, tiny_cfg):
    cfg = copy.deepcopy(tiny_cfg)
    cfg['run']['name'] = 'unit_test'
    cfg['run']['output_dir'] = str(tmp_path)
    train_info = prepare_run(cfg)

    checkpoint_dir = Path(train_info.run_dir, 'checkpoints')
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    for name in ('last.pt', 'best.pt', 'epoch_0003.pt'):
        Path(checkpoint_dir, name).write_bytes(b'placeholder')

    cases = {
        'latest': ('last.pt', 'last'),
        'best': ('best.pt', 'best'),
        '3': ('epoch_0003.pt', 'epoch_0003'),
        'epoch_0003': ('epoch_0003.pt', 'epoch_0003'),
    }
    for selector, (expected_name, expected_label) in cases.items():
        eval_cfg = copy.deepcopy(tiny_cfg)
        eval_cfg['run']['name'] = 'unit_test'
        eval_cfg['run']['output_dir'] = str(tmp_path)
        eval_cfg['run']['mode'] = 'eval'
        eval_cfg['checkpoint']['resume'] = selector

        eval_info = prepare_run(eval_cfg)

        assert eval_info.run_id == train_info.run_id
        assert eval_info.trial_id == train_info.trial_id
        assert eval_info.checkpoint_label == expected_label
        assert eval_info.run_dir == str(
            Path(
                tmp_path,
                'evaluations',
                train_info.run_id,
                f'trial_{train_info.trial_id}',
                f'eval_{expected_label}',
            )
        )
        assert eval_cfg['run']['source_trial_id'] == train_info.trial_id
        assert eval_cfg['checkpoint']['resume'] == str(Path(checkpoint_dir, expected_name))


def test_prepare_run_eval_without_checkpoint_uses_managed_trial(tmp_path, tiny_cfg):
    cfg = copy.deepcopy(tiny_cfg)
    cfg['run']['id'] = 'manual-eval'
    cfg['run']['output_dir'] = str(tmp_path)
    cfg['run']['mode'] = 'eval'

    info = prepare_run(cfg)

    assert info.run_id == 'manual-eval'
    assert info.trial_id == 1
    assert info.run_dir == str(Path(tmp_path, 'evaluations', 'manual-eval', 'trial_1', 'eval_uninitialized'))
    assert info.config_path == str(Path(info.run_dir, 'config.yaml'))


def test_prepare_run_registry_records_trial_and_redacted_command(tmp_path, tiny_cfg, monkeypatch):
    argv = [
        '/workspace/.venv/bin/python',
        'src/main.py',
        '+experiment=baseline',
        'optimizer.lr=3e-4',
        '--api-key',
        'do-not-store',
        'logging.secret=also-do-not-store',
    ]
    monkeypatch.setattr(sys, 'orig_argv', argv)
    cfg = copy.deepcopy(tiny_cfg)
    cfg['run']['output_dir'] = str(tmp_path)

    info = prepare_run(cfg)

    record = json.loads(Path(info.config_registry).read_text(encoding='utf-8').splitlines()[-1])
    assert record['trial_id'] == 1
    assert record['config']['run']['trial_id'] == 1
    assert shlex.split(record['command']) == [
        *argv[:5],
        '<redacted>',
        'logging.secret=<redacted>',
    ]
    assert record['command_cwd'] == str(Path.cwd())


def test_prepare_run_output_dir_roots_all_default_artifacts(tmp_path, tiny_cfg):
    cfg = copy.deepcopy(tiny_cfg)
    cfg['run']['output_dir'] = str(tmp_path / 'custom-output')
    cfg['run']['runs_dir'] = None
    cfg['run']['evaluations_dir'] = None
    cfg['run']['config_dir'] = None
    cfg['run']['config_registry'] = None

    info = prepare_run(cfg)

    root = tmp_path / 'custom-output'
    assert Path(info.run_dir).is_relative_to(root / 'runs')
    assert Path(info.config_path).is_relative_to(root / 'run_configs')
    assert Path(info.config_registry) == root / 'run_registry.jsonl'


def test_legacy_flat_run_is_treated_as_trial_one(tmp_path, tiny_cfg):
    run_id = 'legacy-run'
    checkpoint = Path(tmp_path, 'runs', run_id, 'checkpoints', 'last.pt')
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b'legacy')

    resume_cfg = copy.deepcopy(tiny_cfg)
    resume_cfg['run']['id'] = run_id
    resume_cfg['run']['output_dir'] = str(tmp_path)
    resume_cfg['checkpoint']['resume'] = 'latest'
    resumed = prepare_run(resume_cfg)
    assert resumed.trial_id == 1
    assert resumed.run_dir == str(Path(tmp_path, 'runs', run_id))

    fresh_cfg = copy.deepcopy(tiny_cfg)
    fresh_cfg['run']['id'] = run_id
    fresh_cfg['run']['output_dir'] = str(tmp_path)
    fresh = prepare_run(fresh_cfg)
    assert fresh.trial_id == 2
    assert fresh.run_dir == str(Path(tmp_path, 'runs', run_id, 'trial_2'))
