from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from src.main import _extract_config_file_args, load_replay_config
from src.utils.config import cfg_get, config_to_dict, load_config


def test_config_loads():
    cfg = load_config(Path('configs/config.yaml'))
    assert cfg is not None


def test_config_to_dict(tiny_cfg):
    data = config_to_dict(tiny_cfg)
    assert data['run']['seed'] == 42


@pytest.mark.parametrize(
    'task_name',
    ['segmentation', 'detection', 'ranking', 'language_modeling'],
)
def test_builtin_task_configs_compose(task_name):
    config_dir = str(Path('configs').resolve())
    with initialize_config_dir(config_dir=config_dir, version_base='1.3'):
        cfg = compose(config_name='config', overrides=[f'task={task_name}'])

    assert cfg.task.name == task_name


def _write_saved_run_config(path: Path, wandb_run_name: str = 'old-run') -> None:
    cfg = {
        'run': {
            'name': 'saved',
            'trial': 1,
            'id': 'old-run',
            'mode': 'train',
            'seed': 42,
            'device': 'cpu',
            'precision': 'fp32',
            'output_dir': 'outputs',
            'runs_dir': 'outputs/runs',
            'evaluations_dir': 'outputs/evaluations',
            'config_dir': 'outputs/run_configs',
            'config_registry': 'outputs/run_registry.jsonl',
            'run_dir': 'outputs/runs/old-run',
            'config_path': 'outputs/run_configs/old-run.yaml',
            'config_id': 'abc123',
            'tracking_id': 'old-run',
            'log_dir': 'outputs/runs/old-run/logs',
            'prediction_dir': 'outputs/runs/old-run/predictions',
            'profile_dir': 'outputs/runs/old-run/profiles',
        },
        'model': {'name': 'mlp', 'input_dim': 16, 'hidden_dim': 32, 'num_layers': 1, 'num_classes': 2},
        'data': {'name': 'toy_classification', 'input_dim': 16, 'num_classes': 2, 'batch_size': 4},
        'task': {'name': 'classification'},
        'checkpoint': {'dir': 'outputs/runs/old-run/checkpoints', 'resume': None},
        'logging': {
            'jsonl': {'path': 'outputs/runs/old-run/logs/metrics.jsonl'},
            'tensorboard': {'log_dir': 'outputs/runs/old-run/logs/tensorboard'},
            'wandb': {'enabled': False, 'run_name': wandb_run_name},
        },
        'seed': 999,
        'device': 'cuda',
    }
    OmegaConf.save(config=OmegaConf.create(cfg), f=path)


def test_replay_config_arg_extraction():
    config_file, overrides = _extract_config_file_args([
        '--config-file',
        'outputs/run_configs/demo.yaml',
        '--run-id',
        'manual-replay',
        'run.trial=2',
        'optimizer.lr=0.01',
    ])

    assert config_file == 'outputs/run_configs/demo.yaml'
    assert overrides == ['run.trial=2', 'optimizer.lr=0.01', 'run.id=manual-replay']

    config_file, overrides = _extract_config_file_args([
        '--run-config=outputs/run_configs/demo.yaml',
        '--run-id=manual-replay',
    ])
    assert config_file == 'outputs/run_configs/demo.yaml'
    assert overrides == ['run.id=manual-replay']


@pytest.mark.parametrize('flag', ['--config-file', '--run-config', '--replay-config'])
def test_replay_config_arg_requires_value(flag):
    with pytest.raises(ValueError, match='requires a path'):
        _extract_config_file_args([flag])


@pytest.mark.parametrize('flag', ['--run-id', '--replay-run-id'])
def test_replay_run_id_arg_requires_value(flag):
    with pytest.raises(ValueError, match='requires a run id'):
        _extract_config_file_args(['--config-file', 'saved.yaml', flag])


def test_replay_run_id_arg_rejects_duplicates():
    with pytest.raises(ValueError, match='Only one replay run id'):
        _extract_config_file_args(['--config-file', 'saved.yaml', '--run-id', 'a', '--replay-run-id', 'b'])


def test_load_replay_config_scrubs_generated_fields_and_applies_overrides(tmp_path):
    config_path = tmp_path / 'saved.yaml'
    _write_saved_run_config(config_path)

    cfg = load_replay_config(
        config_path,
        ['run.trial=2', 'run.output_dir=outputs/replayed', 'run.id=replayed-run', '++custom.note=replay'],
    )

    assert cfg_get(cfg, 'run.id') == 'replayed-run'
    assert cfg_get(cfg, 'run.trial') == 2
    assert cfg_get(cfg, 'run.output_dir') == 'outputs/replayed'
    assert cfg_get(cfg, 'run.config_id') is None
    assert cfg_get(cfg, 'run.run_dir') is None
    assert cfg_get(cfg, 'run.tracking_id') is None
    assert cfg_get(cfg, 'checkpoint.dir') is None
    assert cfg_get(cfg, 'logging.jsonl.path') is None
    assert cfg_get(cfg, 'logging.tensorboard.log_dir') is None
    assert cfg_get(cfg, 'logging.wandb.run_name') is None
    assert cfg_get(cfg, 'custom.note') == 'replay'
    assert cfg_get(cfg, 'seed') == 42
    assert cfg_get(cfg, 'device') == 'cpu'


def test_load_replay_config_preserves_custom_wandb_run_name(tmp_path):
    config_path = tmp_path / 'saved.yaml'
    _write_saved_run_config(config_path, wandb_run_name='custom-wandb-name')

    cfg = load_replay_config(config_path)

    assert cfg_get(cfg, 'logging.wandb.run_name') == 'custom-wandb-name'


def test_load_replay_config_rejects_non_dotlist_overrides(tmp_path):
    config_path = tmp_path / 'saved.yaml'
    _write_saved_run_config(config_path)

    with pytest.raises(ValueError, match='key=value dotlist overrides'):
        load_replay_config(config_path, ['--multirun'])
