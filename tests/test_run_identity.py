import copy
from pathlib import Path

from src.utils.run import prepare_run


def test_prepare_run_derives_stable_id_and_paths(tmp_path, tiny_cfg):
    cfg = copy.deepcopy(tiny_cfg)
    cfg['run']['name'] = 'unit_test'
    cfg['run']['trial'] = 1
    cfg['run']['output_dir'] = str(tmp_path)

    info = prepare_run(cfg)

    assert cfg['run']['id'] == info.run_id
    assert cfg['run']['config_id'] == info.config_id
    assert cfg['run']['run_dir'] == str(Path(tmp_path, 'runs', info.run_id))
    assert cfg['checkpoint']['dir'] == str(Path(cfg['run']['run_dir'], 'checkpoints'))
    assert cfg['logging']['jsonl']['path'] == str(Path(cfg['run']['run_dir'], 'logs', 'metrics.jsonl'))
    assert Path(info.config_path).exists()
    assert Path(info.config_registry).exists()

    same_cfg = copy.deepcopy(tiny_cfg)
    same_cfg['run']['name'] = 'unit_test'
    same_cfg['run']['trial'] = 1
    same_cfg['run']['output_dir'] = str(tmp_path)
    same_info = prepare_run(same_cfg)
    assert same_info.config_id == info.config_id
    assert same_info.run_id == info.run_id
    assert same_info.reused_existing is True
    assert same_info.warning is not None
    assert 'WARNING: RUN ID' in same_info.warning
    assert same_cfg['run']['run_dir'] == str(Path(tmp_path, 'runs', same_info.run_id))
    assert Path(same_info.config_path).exists()

    third_cfg = copy.deepcopy(tiny_cfg)
    third_cfg['run']['name'] = 'unit_test'
    third_cfg['run']['trial'] = 1
    third_cfg['run']['output_dir'] = str(tmp_path)
    third_info = prepare_run(third_cfg)
    assert third_info.config_id == info.config_id
    assert third_info.run_id == info.run_id
    assert third_info.reused_existing is True

    resume_cfg = copy.deepcopy(tiny_cfg)
    resume_cfg['run']['name'] = 'unit_test'
    resume_cfg['run']['trial'] = 1
    resume_cfg['run']['output_dir'] = str(tmp_path)
    resume_cfg['checkpoint']['resume'] = 'latest'
    resume_info = prepare_run(resume_cfg)
    assert resume_info.config_id == info.config_id
    assert resume_info.run_id == info.run_id
    assert resume_info.warning is None
    assert resume_cfg['checkpoint']['dir'] == str(Path(tmp_path, 'runs', info.run_id, 'checkpoints'))

    repeat_cfg = copy.deepcopy(tiny_cfg)
    repeat_cfg['run']['name'] = 'unit_test'
    repeat_cfg['run']['trial'] = 2
    repeat_cfg['run']['output_dir'] = str(tmp_path)
    repeat_info = prepare_run(repeat_cfg)
    assert repeat_info.run_id != info.run_id


def test_prepare_run_reuses_explicit_id_collisions(tmp_path, tiny_cfg):
    cfg = copy.deepcopy(tiny_cfg)
    cfg['run']['id'] = 'manual run'
    cfg['run']['output_dir'] = str(tmp_path)

    info = prepare_run(cfg)
    assert info.run_id == 'manual-run'
    assert info.reused_existing is False

    next_cfg = copy.deepcopy(tiny_cfg)
    next_cfg['run']['id'] = 'manual run'
    next_cfg['run']['output_dir'] = str(tmp_path)

    next_info = prepare_run(next_cfg)
    assert next_info.run_id == 'manual-run'
    assert next_info.reused_existing is True
    assert next_info.warning is not None
    assert next_cfg['run']['run_dir'] == str(Path(tmp_path, 'runs', 'manual-run'))


def test_prepare_run_eval_uses_checkpoint_run_dir_with_evaluation_suffix(tmp_path, tiny_cfg):
    training_run_id = 'trained-run'
    eval_run_id = f'{training_run_id}_evaluation'
    checkpoint = Path(tmp_path, 'runs', training_run_id, 'checkpoints', 'best.pt')
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b'placeholder')

    cfg = copy.deepcopy(tiny_cfg)
    cfg['run']['output_dir'] = str(tmp_path)
    cfg['run']['mode'] = 'eval'
    cfg['checkpoint']['resume'] = str(checkpoint)

    info = prepare_run(cfg)

    assert info.run_id == eval_run_id
    assert info.run_dir == str(Path(tmp_path, 'runs', eval_run_id))
    assert cfg['checkpoint']['dir'] == str(Path(tmp_path, 'runs', eval_run_id, 'checkpoints'))
    assert cfg['run']['prediction_dir'] == str(Path(tmp_path, 'runs', eval_run_id, 'predictions'))
    assert info.reused_existing is False

    repeat_cfg = copy.deepcopy(tiny_cfg)
    repeat_cfg['run']['output_dir'] = str(tmp_path)
    repeat_cfg['run']['mode'] = 'eval'
    repeat_cfg['checkpoint']['resume'] = str(checkpoint)
    repeat_info = prepare_run(repeat_cfg)

    assert repeat_info.run_id == eval_run_id
    assert repeat_info.reused_existing is True
    assert repeat_info.warning is not None


def test_prepare_run_eval_checkpoint_selectors_use_evaluation_run_dir(tmp_path, tiny_cfg):
    cfg = copy.deepcopy(tiny_cfg)
    cfg['run']['name'] = 'unit_test'
    cfg['run']['trial'] = 1
    cfg['run']['output_dir'] = str(tmp_path)
    train_info = prepare_run(cfg)

    checkpoint_dir = Path(train_info.run_dir, 'checkpoints')
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    for name in ('last.pt', 'best.pt', 'epoch_0003.pt'):
        Path(checkpoint_dir, name).write_bytes(b'placeholder')

    cases = {
        'latest': 'last.pt',
        'best': 'best.pt',
        '3': 'epoch_0003.pt',
        'epoch_0003': 'epoch_0003.pt',
    }
    for selector, expected_name in cases.items():
        eval_cfg = copy.deepcopy(tiny_cfg)
        eval_cfg['run']['name'] = 'unit_test'
        eval_cfg['run']['trial'] = 1
        eval_cfg['run']['output_dir'] = str(tmp_path)
        eval_cfg['run']['mode'] = 'eval'
        eval_cfg['checkpoint']['resume'] = selector

        eval_info = prepare_run(eval_cfg)

        assert eval_info.run_id == f'{train_info.run_id}_evaluation'
        assert eval_info.run_dir == str(Path(tmp_path, 'runs', f'{train_info.run_id}_evaluation'))
        assert eval_cfg['checkpoint']['resume'] == str(Path(checkpoint_dir, expected_name))
        assert eval_cfg['checkpoint']['dir'] == str(Path(tmp_path, 'runs', f'{train_info.run_id}_evaluation', 'checkpoints'))

