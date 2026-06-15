from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir

from src.utils.config import config_to_dict, load_config


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
