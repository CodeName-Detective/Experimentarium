"""Reusable neural-network layers for reference models and experiments.

Use these layers when building custom models in ``src/models``. They are small,
inspectable alternatives to PyTorch's built-ins and are intended for education or
lightweight research prototypes.

Typical usage:
    from src.models.layers import MultiHeadAttention, FeedForward
    block = FeedForward(d_model=128, d_ff=512)
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from src.utils.types import FloatTensor


class MultiHeadAttention(nn.Module):
    """Scaled dot-product multi-head attention with output projection."""

    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1) -> None:
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError(f'd_model ({d_model}) must be divisible by num_heads ({num_heads})')
        self.d_k = d_model // num_heads
        self.num_heads = num_heads
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.out = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: FloatTensor, mask: FloatTensor | None = None) -> FloatTensor:
        batch_size, tokens, channels = x.shape
        qkv = self.qkv(x).reshape(batch_size, tokens, 3, self.num_heads, self.d_k).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        attn = self.dropout(scores.softmax(dim=-1))
        y = (attn @ v).transpose(1, 2).reshape(batch_size, tokens, channels)
        return self.out(y)


class FeedForward(nn.Module):
    """Position-wise feed-forward block used in transformer-style models."""

    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
        )

    def forward(self, x: FloatTensor) -> FloatTensor:
        return self.net(x)


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for sequence models."""

    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(max_len).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x: FloatTensor) -> FloatTensor:
        return self.dropout(x + self.pe[:, : x.size(1)])
