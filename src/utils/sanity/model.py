"""Model sanity helpers.

Use ``run_model_smoke`` for a quick CPU forward/backward check when extending the
framework with a new model or task. The main CLI sanity command already calls a
similar check through ``src.utils.sanity.run_sanity_checks``.
"""

from __future__ import annotations

import torch


def run_model_smoke(model: torch.nn.Module, batch: dict[str, torch.Tensor], loss: torch.Tensor) -> bool:
    """Run backward on a provided loss and return whether finite gradients exist."""

    model.zero_grad(set_to_none=True)
    loss.backward()
    return any(p.grad is not None and torch.isfinite(p.grad).all().item() for p in model.parameters() if p.requires_grad)
