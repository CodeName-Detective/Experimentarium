"""Reference model implementations for end-to-end framework validation.

The models here are intentionally small and easy to inspect. They are examples
of the model contract expected by tasks: ``forward(batch)`` returns a dictionary,
usually containing ``logits``.

Typical usage:
    from src.utils.registry import MODEL_REGISTRY
    model = MODEL_REGISTRY.build('mlp', cfg.model)

Registered models:
    - ``mlp`` / ``baseline`` for vector classification or regression.
    - ``cnn`` for CIFAR-like image classification smoke tests.
    - ``small_transformer`` for token-sequence classification.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from src.utils.config import cfg_get
from src.utils.registry import register_model


@register_model('mlp')
@register_model('baseline')
class MLP(nn.Module):
    """Small MLP for vector classification/regression toy experiments."""

    def __init__(self, cfg) -> None:
        super().__init__()
        input_dim = int(cfg_get(cfg, 'input_dim', 16))
        hidden_dim = int(cfg_get(cfg, 'hidden_dim', 64))
        num_layers = int(cfg_get(cfg, 'num_layers', 2))
        output_dim = int(cfg_get(cfg, 'output_dim', cfg_get(cfg, 'num_classes', 2)))
        dropout = float(cfg_get(cfg, 'dropout', 0.0))
        layers: list[nn.Module] = []
        dim = input_dim
        for _ in range(max(1, num_layers)):
            layers.extend([nn.Linear(dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout)])
            dim = hidden_dim
        self.encoder = nn.Sequential(*layers)
        self.head = nn.Linear(dim, output_dim)

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        x = batch['input'].float()
        return {'logits': self.head(self.encoder(x))}


@register_model('cnn')
class SmallCNN(nn.Module):
    """Compact CNN for CIFAR-like image classification smoke tests."""

    def __init__(self, cfg) -> None:
        super().__init__()
        in_channels = int(cfg_get(cfg, 'in_channels', 3))
        width = int(cfg_get(cfg, 'width', 32))
        num_classes = int(cfg_get(cfg, 'num_classes', 10))
        dropout = float(cfg_get(cfg, 'dropout', 0.0))
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, width, 3, padding=1),
            nn.BatchNorm2d(width),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(width, width * 2, 3, padding=1),
            nn.BatchNorm2d(width * 2),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(width * 2, num_classes),
        )

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        return {'logits': self.net(batch['input'].float())}


@register_model('small_transformer')
class SmallTransformer(nn.Module):
    """Minimal transformer encoder for sequence classification experiments."""

    def __init__(self, cfg) -> None:
        super().__init__()
        vocab_size = int(cfg_get(cfg, 'vocab_size', 128))
        d_model = int(cfg_get(cfg, 'd_model', 64))
        num_heads = int(cfg_get(cfg, 'num_heads', 4))
        num_layers = int(cfg_get(cfg, 'num_layers', 2))
        num_classes = int(cfg_get(cfg, 'num_classes', 2))
        max_len = int(cfg_get(cfg, 'max_len', 128))
        dropout = float(cfg_get(cfg, 'dropout', 0.1))
        self.token = nn.Embedding(vocab_size, d_model)
        self.position = nn.Parameter(torch.zeros(1, max_len, d_model))
        layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=num_heads, dropout=dropout, batch_first=True)
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.head = nn.Linear(d_model, num_classes)

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        tokens = batch['input'].long()
        x = self.token(tokens) + self.position[:, : tokens.shape[1]]
        x = self.encoder(x)
        return {'logits': self.head(x[:, 0])}


MyModel = MLP
