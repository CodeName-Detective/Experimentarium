"""Generic, fault-tolerant training engine for research experiments.

The trainer owns orchestration only: epochs, optimization, precision, validation,
checkpointing, resume, callbacks, and logging. Task-specific behavior lives in
``src.tasks``. This separation lets you add segmentation, detection, NLP, or
multimodal workloads without editing the engine.

Typical usage:
    trainer = Trainer(cfg, model, task, loaders, optimizer, scheduler, loggers, checkpoints)
    metrics = trainer.fit()
    test_metrics = trainer.test()

Fault tolerance:
    - Saves full state on normal checkpoint intervals.
    - Saves an ``*_exception.pt`` checkpoint on exceptions when enabled.
    - Detects non-finite losses before optimizer steps.
    - Restores model, optimizer, scheduler, scaler, RNG, epoch, step, and early-stopping state.
"""

from __future__ import annotations

import logging
import math
from typing import TYPE_CHECKING, Any

import torch

from src.callbacks import Callback, CallbackList
from src.engine.evaluator import Evaluator, move_to_device
from src.engine.precision import amp_dtype, amp_enabled, precision_autocast, scaler_enabled
from src.optim.schedulers import SchedulerBundle, scheduler_step
from src.runtime.distributed import reduce_sum_count, unwrap_model
from src.utils.checkpoint import CheckpointManager, get_rng_state
from src.utils.config import cfg_get, config_to_dict

if TYPE_CHECKING:
    from src.tasks import BaseTask
    from src.utils.logger import LoggerCollection
    from src.utils.types import ConfigType

_LOG = logging.getLogger(__name__)


class Trainer:
    """Training loop with AMP, accumulation, clipping, validation, resume, and checkpoints."""

    def __init__(
        self,
        cfg: ConfigType,
        model: torch.nn.Module,
        task: BaseTask,
        loaders: dict[str, Any],
        optimizer: torch.optim.Optimizer,
        scheduler: SchedulerBundle | None,
        loggers: LoggerCollection,
        checkpoint_manager: CheckpointManager,
        callbacks: list[Callback] | None = None,
    ) -> None:
        self.cfg = cfg
        self.device = torch.device(cfg_get(cfg, 'run.device', cfg_get(cfg, 'device', 'cpu')))
        if self.device.type == 'cuda' and not torch.cuda.is_available():
            _LOG.warning('CUDA requested but unavailable; falling back to CPU')
            self.device = torch.device('cpu')
        self.model = model.to(self.device)
        self.task = task
        self.loaders = loaders
        self.optimizer = optimizer
        self.scheduler = scheduler or SchedulerBundle(None)
        self.loggers = loggers
        self.checkpoints = checkpoint_manager
        self.callbacks = CallbackList(callbacks)
        self.max_epochs = int(cfg_get(cfg, 'trainer.max_epochs', cfg_get(cfg, 'trainer.epochs', 1)))
        self.max_steps = self._optional_positive_int(cfg_get(cfg, 'trainer.max_steps', None))
        self.accumulate_grad_batches = max(1, int(cfg_get(cfg, 'trainer.accumulate_grad_batches', 1)))
        self.grad_clip = cfg_get(cfg, 'trainer.grad_clip', None)
        self.log_every_n_steps = max(1, int(cfg_get(cfg, 'trainer.log_every_n_steps', 50)))
        self.val_every_n_epochs = max(1, int(cfg_get(cfg, 'trainer.val_every_n_epochs', 1)))
        self.val_every_n_steps = max(0, int(cfg_get(cfg, 'trainer.val_every_n_steps', 0)))
        self.limit_train_batches = cfg_get(cfg, 'trainer.limit_train_batches', None)
        self.limit_val_batches = cfg_get(cfg, 'trainer.limit_val_batches', None)
        self.limit_test_batches = cfg_get(cfg, 'trainer.limit_test_batches', None)
        self.skip_test_after_train = bool(cfg_get(cfg, 'trainer.skip_test_after_train', False))
        self.log_gradient_norm = bool(cfg_get(cfg, 'trainer.log_gradient_norm', False))
        self.log_learning_rate = bool(cfg_get(cfg, 'trainer.log_learning_rate', False))
        self.early_patience = int(
            cfg_get(cfg, 'trainer.early_stopping.patience', cfg_get(cfg, 'trainer.early_stopping_patience', 0))
        )
        self.check_finite_loss = bool(cfg_get(cfg, 'trainer.check_finite_loss', True))
        self.detect_anomaly = bool(cfg_get(cfg, 'trainer.detect_anomaly', False))
        self.save_on_exception = bool(cfg_get(cfg, 'checkpoint.save_on_exception', True))
        self.precision = str(cfg_get(cfg, 'run.precision', 'fp32'))
        self.use_amp = amp_enabled(self.device, self.precision)
        self.amp_dtype = amp_dtype(self.precision)
        self.scaler = torch.amp.GradScaler('cuda', enabled=scaler_enabled(self.device, self.precision))
        self.global_step = 0
        self.start_epoch = 1
        self.current_epoch = 0
        self.best_metric: float | None = None
        self.epochs_without_improvement = 0
        self.trained_epochs = 0
        self.monitor = str(cfg_get(cfg, 'checkpoint.monitor', 'val/loss'))
        self._stop_training = False

    def fit(self) -> dict[str, float]:
        """Run training and return final train/validation metrics."""
        final_metrics: dict[str, float] = {}
        training_started = False
        self.trained_epochs = 0
        self._stop_training = False
        self.resume()
        try:
            training_started = True
            self.callbacks.call('on_train_start', self)
            with torch.autograd.set_detect_anomaly(self.detect_anomaly):
                for epoch in range(self.start_epoch, self.max_epochs + 1):
                    if self._reached_max_steps():
                        break
                    self.trained_epochs += 1
                    self.current_epoch = epoch
                    self.callbacks.call('on_epoch_start', self, epoch)
                    self._set_epoch(epoch)
                    train_metrics = self.train_epoch(epoch)
                    metrics = dict(train_metrics)
                    if epoch % self.val_every_n_epochs == 0 and 'val' in self.loaders:
                        val_metrics = self._evaluate_split('val', prefix='val')
                        metrics.update(val_metrics)
                        self.callbacks.call('on_validation_end', self, metrics)
                    self.loggers.log_metrics(metrics, step=self.global_step)
                    final_metrics = metrics
                    monitor_value = metrics.get(self.monitor)
                    improved = self._update_best(monitor_value)
                    should_stop = self._should_stop(improved, checked=monitor_value is not None)
                    if self.scheduler.scheduler is not None and self.scheduler.interval == 'epoch':
                        scheduler_metric = metrics.get(self.scheduler.monitor)
                        if self.scheduler.name != 'plateau' or scheduler_metric is not None:
                            scheduler_step(self.scheduler, scheduler_metric)
                    self._save_checkpoint(epoch, improved, metrics)
                    if should_stop:
                        _LOG.info('Early stopping at epoch %s', epoch)
                        break
                    if self._stop_training:
                        break
            self.callbacks.call('on_train_end', self)
            return final_metrics
        except BaseException as exc:
            try:
                self.callbacks.call('on_exception', self, exc)
            except Exception as callback_exc:
                _LOG.exception('Exception callback failed while handling %s: %s', type(exc).__name__, callback_exc)
            if self.save_on_exception and training_started:
                self._save_exception_checkpoint(final_metrics)
            raise

    def train_epoch(self, epoch: int) -> dict[str, float]:
        """Run one training epoch and return prefixed metrics."""
        self.model.train()
        self.task.reset_metrics()
        total_loss = 0.0
        total_count = 0.0
        self.optimizer.zero_grad(set_to_none=True)
        max_batches = self._batch_limit(self.loaders['train'], self.limit_train_batches)
        if max_batches == 0:
            return {'train/loss': 0.0}
        accumulated_batches = 0
        last_result: Any = None
        last_batch_idx = 0
        for batch_idx, batch in enumerate(self.loaders['train'], start=1):
            if max_batches is not None and batch_idx > max_batches:
                break
            batch = move_to_device(batch, self.device)
            with precision_autocast(self.device, self.precision):
                result = self.task.step(self.model, batch, stage='train')
                if result.loss is None:
                    raise RuntimeError('Task returned no loss during training')
                if self.check_finite_loss and not torch.isfinite(result.loss.detach()).all().item():
                    raise FloatingPointError(
                        f'Non-finite training loss at epoch={epoch} batch={batch_idx}: {result.loss.detach().cpu()}'
                    )
                loss = result.loss / self.accumulate_grad_batches
            self.scaler.scale(loss).backward()
            accumulated_batches += 1
            last_result = result
            last_batch_idx = batch_idx
            batch_size = int(result.targets.shape[0]) if result.targets is not None else 1
            loss_weight = float(result.loss_weight if result.loss_weight is not None else batch_size)
            total_loss += float(result.loss.detach().cpu()) * loss_weight
            total_count += loss_weight
            if accumulated_batches == self.accumulate_grad_batches:
                self._complete_optimizer_step(batch_idx, result)
                accumulated_batches = 0
                if self._stop_training:
                    break
        if accumulated_batches and not self._stop_training and last_result is not None:
            gradient_scale = self.accumulate_grad_batches / accumulated_batches
            self._complete_optimizer_step(last_batch_idx, last_result, gradient_scale=gradient_scale)
        metrics = self.task.compute_metrics_distributed(device=self.device)
        global_loss, global_count = reduce_sum_count(total_loss, total_count, device=self.device)
        metrics['loss'] = global_loss / max(1.0, global_count)
        return {f'train/{key}': value for key, value in metrics.items()}

    def test(self) -> dict[str, float]:
        """Run test evaluation with the current model state."""
        if 'test' not in self.loaders:
            return {}
        return self._evaluate_split('test', prefix='test')

    def resume(self) -> None:
        """Resume training state or load model-only state for evaluation."""
        resume = cfg_get(self.cfg, 'checkpoint.resume', None)
        if not resume:
            return
        training_resume = str(cfg_get(self.cfg, 'run.mode', 'train')).lower() == 'train'
        optimizer = self.optimizer if training_resume else None
        scheduler = self.scheduler.scheduler if training_resume else None
        scaler = self.scaler if training_resume else None
        if str(resume).lower() == 'latest':
            state = self.checkpoints.load_latest(self.model, optimizer, scheduler, scaler, restore_rng=training_resume)
            if state is None:
                return
        else:
            resume_path = self.checkpoints.resolve_resume_path(resume)
            state = self.checkpoints.load(
                resume_path, self.model, optimizer, scheduler, scaler, restore_rng=training_resume
            )
        self.start_epoch = int(state.get('epoch', 0)) + 1
        self.current_epoch = int(state.get('epoch', 0))
        self.global_step = int(state.get('global_step', 0))
        self.best_metric = state.get('best_metric')
        self.epochs_without_improvement = int(state.get('epochs_without_improvement', 0))
        if training_resume:
            _LOG.info(
                '\x1b[1;32mRESUMING TRAINING FROM EPOCH %s (checkpoint_epoch=%s global_step=%s)\x1b[0m',
                self.start_epoch,
                self.current_epoch,
                self.global_step,
            )
        else:
            _LOG.info('Loaded checkpoint from epoch=%s global_step=%s', self.current_epoch, self.global_step)

    def _state_dict(self, epoch: int, metrics: dict[str, float]) -> dict[str, Any]:
        scheduler_state = self.scheduler.scheduler.state_dict() if self.scheduler.scheduler is not None else None
        return {
            'epoch': epoch,
            'global_step': self.global_step,
            'best_metric': self.best_metric,
            'epochs_without_improvement': self.epochs_without_improvement,
            'model_state': unwrap_model(self.model).state_dict(),
            'optimizer_state': self.optimizer.state_dict(),
            'scheduler_state': scheduler_state,
            'scaler_state': self.scaler.state_dict(),
            'rng_state': get_rng_state(seed_cuda=self.device.type == 'cuda'),
            'metrics': metrics,
            'cfg': config_to_dict(self.cfg),
        }

    def _save_checkpoint(self, epoch: int, is_best: bool, metrics: dict[str, float]) -> None:
        monitor_value = metrics.get(self.monitor)
        path = self.checkpoints.save(
            self._state_dict(epoch, metrics), epoch=epoch, metric=monitor_value, is_best=is_best
        )
        if path is not None:
            self.callbacks.call('on_checkpoint_saved', self, path)
            if is_best and self.checkpoints.save_top_k > 0:
                self.loggers.log_artifact(path, name='best_checkpoint', artifact_type='model', metadata=metrics)

    def _save_exception_checkpoint(self, metrics: dict[str, float]) -> None:
        try:
            state = self._state_dict(epoch=max(0, self.current_epoch), metrics=metrics)
            path = self.checkpoints.save_exception(state, epoch=max(0, self.current_epoch))
            if path is not None:
                _LOG.warning('Saved exception checkpoint: %s', path)
        except Exception as save_exc:
            _LOG.exception('Failed to save exception checkpoint: %s', save_exc)

    def _update_best(self, metric: float | None) -> bool:
        if metric is None or math.isnan(float(metric)):
            return False
        improved = self.checkpoints.is_better(float(metric), self.best_metric)
        if improved:
            self.best_metric = float(metric)
        return improved

    def _should_stop(self, improved: bool, checked: bool = True) -> bool:
        if self.early_patience <= 0 or not checked:
            return False
        self.epochs_without_improvement = 0 if improved else self.epochs_without_improvement + 1
        return self.epochs_without_improvement >= self.early_patience

    def _set_epoch(self, epoch: int) -> None:
        sampler = getattr(self.loaders['train'], 'sampler', None)
        set_epoch = getattr(sampler, 'set_epoch', None)
        if callable(set_epoch):
            set_epoch(epoch)

    def _evaluate_split(self, split: str, prefix: str) -> dict[str, float]:
        evaluator = Evaluator(self.model, self.task, self.device, precision=self.precision)
        limit = self.limit_val_batches if split == 'val' else self.limit_test_batches
        return evaluator.evaluate(self.loaders[split], prefix=prefix, limit_batches=limit)

    def _complete_optimizer_step(self, batch_idx: int, result: Any, gradient_scale: float = 1.0) -> None:
        """Apply one optimizer step and run step-level hooks."""
        grad_norm = self._prepare_gradients_for_step(gradient_scale=gradient_scale)
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.optimizer.zero_grad(set_to_none=True)
        self.global_step += 1
        if self.scheduler.scheduler is not None and self.scheduler.interval == 'step':
            scheduler_step(self.scheduler)
        step_metrics = {'train/loss_step': float(result.loss.detach().cpu())}
        if grad_norm is not None:
            step_metrics['train/grad_norm'] = grad_norm
        if self.log_learning_rate:
            step_metrics.update(self._learning_rate_metrics())
        self.callbacks.call('on_batch_end', self, batch_idx, step_metrics)
        if self.global_step % self.log_every_n_steps == 0:
            self.loggers.log_metrics(step_metrics, step=self.global_step)
        if self._should_run_step_validation():
            val_metrics = self._evaluate_split('val', prefix='val')
            self.callbacks.call('on_validation_end', self, val_metrics)
            self.loggers.log_metrics(val_metrics, step=self.global_step)
        if self._reached_max_steps():
            self._stop_training = True

    def _prepare_gradients_for_step(self, gradient_scale: float = 1.0) -> float | None:
        grad_norm: float | None = None
        rescale_gradients = not math.isclose(gradient_scale, 1.0)
        if self.grad_clip is not None or self.log_gradient_norm or rescale_gradients:
            self.scaler.unscale_(self.optimizer)
            if rescale_gradients:
                for parameter in unwrap_model(self.model).parameters():
                    if parameter.grad is not None:
                        parameter.grad.mul_(gradient_scale)
            if self.log_gradient_norm:
                grad_norm = self._gradient_norm()
            if self.grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(unwrap_model(self.model).parameters(), float(self.grad_clip))
        return grad_norm

    def _gradient_norm(self) -> float:
        norms = []
        for parameter in unwrap_model(self.model).parameters():
            if parameter.grad is None:
                continue
            grad = parameter.grad.detach()
            if grad.is_sparse:
                grad = grad.coalesce().values()
            norms.append(torch.linalg.vector_norm(grad.float(), ord=2))
        if not norms:
            return 0.0
        return float(torch.linalg.vector_norm(torch.stack(norms), ord=2).item())

    def _learning_rate_metrics(self) -> dict[str, float]:
        return {
            f'train/lr/group_{index}': float(group.get('lr', 0.0))
            for index, group in enumerate(self.optimizer.param_groups)
        }

    def _should_run_step_validation(self) -> bool:
        return self.val_every_n_steps > 0 and self.global_step % self.val_every_n_steps == 0 and 'val' in self.loaders

    def _reached_max_steps(self) -> bool:
        return self.max_steps is not None and self.global_step >= self.max_steps

    def _batch_limit(self, loader: Any, configured: Any) -> int | None:
        if configured is None:
            return len(loader) if hasattr(loader, '__len__') else None
        if isinstance(configured, str):
            normalized = configured.lower()
            if normalized in {'none', 'null'}:
                return len(loader) if hasattr(loader, '__len__') else None
            if any(marker in normalized for marker in ('.', 'e')):
                return self._fractional_batch_limit(loader, float(configured))
            return max(0, int(configured))
        if isinstance(configured, int) and not isinstance(configured, bool):
            return max(0, configured)
        return self._fractional_batch_limit(loader, float(configured))

    def _fractional_batch_limit(self, loader: Any, value: float) -> int:
        if 0.0 < value <= 1.0:
            if not hasattr(loader, '__len__'):
                raise ValueError('Fractional batch limits require a sized dataloader')
            return max(1, math.ceil(len(loader) * value))
        return max(0, int(value))

    def _optional_positive_int(self, value: Any) -> int | None:
        if value is None:
            return None
        if isinstance(value, str) and value.lower() in {'none', 'null'}:
            return None
        return max(1, int(value))

    def run(self) -> None:
        """Backward-compatible alias for older code paths."""
        self.fit()
