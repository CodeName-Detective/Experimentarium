import torch

from src.models import MLP


def test_output_shape(tiny_cfg, tiny_batch):
    model = MLP(tiny_cfg['model'])
    output = model(tiny_batch)
    assert output['logits'].shape == (4, 2)


def test_no_nan_outputs(tiny_cfg, tiny_batch):
    model = MLP(tiny_cfg['model'])
    output = model(tiny_batch)
    assert not torch.isnan(output['logits']).any()


def test_gradient_flow(tiny_cfg, tiny_batch):
    model = MLP(tiny_cfg['model'])
    output = model(tiny_batch)
    loss = output['logits'].sum()
    loss.backward()
    assert all(p.grad is not None for p in model.parameters() if p.requires_grad)
