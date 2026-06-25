from copy import deepcopy

import torch

from src.data import build_dataloaders
from src.data.dataloader import DistributedEvalSampler


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


def test_distributed_eval_sampler_shards_without_padding_or_duplicates(monkeypatch):
    monkeypatch.setattr(torch.distributed, 'get_world_size', lambda: 2)
    rank = {'value': 0}
    monkeypatch.setattr(torch.distributed, 'get_rank', lambda: rank['value'])
    dataset = list(range(5))

    rank0 = list(DistributedEvalSampler(dataset))
    rank['value'] = 1
    rank1 = list(DistributedEvalSampler(dataset))

    assert set(rank0).isdisjoint(rank1)
    assert sorted(rank0 + rank1) == list(range(len(dataset)))
