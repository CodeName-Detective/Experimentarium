"""Autoregressive language-modeling task implementation."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any, cast

import torch

from src.losses import build_loss
from src.runtime.distributed import reduce_sum_count
from src.tasks.task import BaseTask, StepResult
from src.utils.config import cfg_get
from src.utils.registry import register_task

if TYPE_CHECKING:
    from torch import Tensor, nn

    from src.utils.types import ConfigType


@register_task('language_modeling')
class LanguageModelingTask(BaseTask):
    """Next-token language modeling with logits shaped ``[B, T, V]``."""

    def __init__(self, cfg: ConfigType) -> None:
        super().__init__(cfg)
        loss_cfg = cfg_get(cfg, 'loss', {'name': 'cross_entropy'})
        self.loss_fn = build_loss(loss_cfg)
        self.ignore_index = int(cfg_get(cfg, 'ignore_index', cfg_get(loss_cfg, 'ignore_index', -100)))
        if hasattr(self.loss_fn, 'ignore_index'):
            cast('Any', self.loss_fn).ignore_index = self.ignore_index
        self._nll_total = 0.0
        self._token_count = 0

    def reset_metrics(self) -> None:
        """Reset token metrics and perplexity accumulators."""
        super().reset_metrics()
        self._nll_total = 0.0
        self._token_count = 0

    def compute_metrics(self) -> dict[str, float]:
        """Return configured token metrics plus perplexity."""
        metrics = super().compute_metrics()
        if self._token_count:
            metrics['perplexity'] = math.exp(min(self._nll_total / self._token_count, 80.0))
        return metrics

    def compute_metrics_distributed(self, device: Any = 'cpu') -> dict[str, float]:
        """Return token metrics and perplexity reduced by global token count."""
        metrics = super().compute_metrics_distributed(device=device)
        nll_total, token_count = reduce_sum_count(self._nll_total, self._token_count, device=device)
        if token_count:
            metrics['perplexity'] = math.exp(min(nll_total / token_count, 80.0))
        return metrics

    def metric_state_dict(self) -> dict[str, Any]:
        """Snapshot generic and language-model metric accumulators."""
        state = super().metric_state_dict()
        state.update({'nll_total': self._nll_total, 'token_count': self._token_count})
        return state

    def load_metric_state_dict(self, state: dict[str, Any]) -> None:
        """Restore generic and language-model metric accumulators."""
        super().load_metric_state_dict(state)
        self._nll_total = float(state.get('nll_total', 0.0))
        self._token_count = int(state.get('token_count', 0))

    def step(self, model: nn.Module, batch: dict[str, Any], stage: str) -> StepResult:
        """Compute flattened next-token cross-entropy and token metrics."""
        self.validate_batch(batch, self.target_key)
        outputs = model(batch)
        self.validate_outputs(outputs, self.output_key)
        logits = outputs[self.output_key]
        targets = batch[self.target_key].long()
        flat_logits, flat_targets = self._flatten(logits, targets)
        loss = outputs.get('loss')
        if loss is None:
            loss = self.loss_fn(flat_logits, flat_targets)
        valid = flat_targets != self.ignore_index
        valid_count = int(valid.sum().item())
        if valid_count:
            self.metrics.update(flat_logits[valid].detach(), flat_targets[valid].detach(), n=valid_count)
            self._nll_total += float(loss.detach().cpu()) * valid_count
            self._token_count += valid_count
        return StepResult(loss=loss, outputs=outputs, targets=targets, loss_weight=valid_count)

    def _flatten(self, logits: Tensor, targets: Tensor) -> tuple[Tensor, Tensor]:
        if logits.ndim != targets.ndim + 1:
            raise ValueError(
                f'Language-model logits must add a vocabulary dimension: {tuple(logits.shape)} vs {tuple(targets.shape)}'
            )
        if logits.shape[:-1] != targets.shape:
            raise ValueError(f'Language-model logits/target shapes are incompatible: {logits.shape} vs {targets.shape}')
        return logits.reshape(-1, logits.shape[-1]), targets.reshape(-1)

    def predict_records(self, outputs: dict[str, Any], batch: dict[str, Any]) -> list[dict[str, Any]]:
        """Export predicted and target token IDs for each sequence."""
        predictions = outputs[self.output_key].argmax(dim=-1).detach().cpu()
        targets = batch.get(self.target_key)
        labels = targets.detach().cpu() if torch.is_tensor(targets) else None
        records: list[dict[str, Any]] = []
        for index, prediction in enumerate(predictions):
            record: dict[str, Any] = {'pred_tokens': prediction.tolist()}
            if labels is not None:
                record['target_tokens'] = labels[index].tolist()
            records.append(record)
        return records
