from copy import deepcopy

from src.data import build_dataloaders


def test_dataloader_supports_split_overrides_and_configured_transforms(tiny_cfg):
    cfg = deepcopy(tiny_cfg)
    cfg['data']['batch_size'] = 4
    cfg['data']['transforms'] = {'train': {'name': 'identity'}}
    cfg['data']['splits']['train']['batch_size'] = 2
    cfg['data']['splits']['val']['batch_size'] = 3

    loaders = build_dataloaders(cfg)

    assert loaders['train'].batch_size == 2
    assert loaders['val'].batch_size == 3
    assert loaders['test'].batch_size == 4
    assert next(iter(loaders['train']))['input'].shape[0] == 2
