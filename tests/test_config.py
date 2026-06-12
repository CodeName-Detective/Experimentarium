from pathlib import Path

from src.utils.config import config_to_dict, load_config


def test_config_loads():
    cfg = load_config(Path('configs/config.yaml'))
    assert cfg is not None


def test_config_to_dict(tiny_cfg):
    data = config_to_dict(tiny_cfg)
    assert data['run']['seed'] == 42
